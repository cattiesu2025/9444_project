"""PMT + CAJ 通道增强（M0_CA / L3_B050_CA）的自检。合成数据只验证流程，不代表训练效果。"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import agw  # noqa: E402
from src import data as D  # noqa: E402
from src.config import load_config, validate  # noqa: E402
from synthetic_sysu import make_synthetic_sysu  # noqa: E402

IMG = Image.fromarray(np.random.default_rng(0).integers(0, 256, (256, 128, 3), dtype=np.uint8))


def single_channel(t):
    """三个通道完全相同（被复制成单通道或转灰度）。"""
    return torch.equal(t[0], t[1]) and torch.equal(t[1], t[2])


def run(transform, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    return transform(IMG)


def test_default_transform_unchanged_and_consumes_no_randomness():
    for modality in ("rgb", "ir"):
        a = run(D.PMTTrainTransform(modality), 3)
        b = run(D.PMTTrainTransform(modality, channel_exchange=0.0), 3)
        assert torch.equal(a, b)
    t = D.PMTTrainTransform("rgb")
    random.seed(1)
    torch.manual_seed(1)
    t(IMG)
    state = random.getstate()
    random.seed(1)
    torch.manual_seed(1)
    D.PMTTrainTransform("rgb", channel_exchange=0.0)(IMG)
    assert random.getstate() == state


def test_rgb_exchange_probability():
    always = D.PMTTrainTransform("rgb", channel_exchange=1.0)
    assert isinstance(always.exchange, agw.ChannelExchange)
    assert all(single_channel(run(always, s)) for s in range(40))
    half = D.PMTTrainTransform("rgb", channel_exchange=0.5)
    rate = np.mean([single_channel(run(half, s)) for s in range(600)])
    assert abs(rate - 0.5) < 0.07
    plain = D.PMTTrainTransform("rgb")
    assert not any(single_channel(run(plain, s)) for s in range(40))


def test_ir_uses_adaptive_gray():
    t = D.PMTTrainTransform("ir", channel_exchange=0.5)
    assert isinstance(t.exchange, agw.ChannelAdapGray)
    # ChannelAdapGray(0.5)：3/4 复制单通道，其余一半转灰度 → 约 7/8 变成单通道
    rate = np.mean([single_channel(run(t, s)) for s in range(800)])
    assert abs(rate - 0.875) < 0.05


def test_rejects_bad_probability():
    with pytest.raises(ValueError):
        D.PMTTrainTransform("rgb", channel_exchange=1.5)
    cfg = load_config("M0_CA")
    cfg["augmentation"]["channel_exchange"] = -0.1
    with pytest.raises(ValueError, match="channel_exchange"):
        validate(cfg)


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out


@pytest.mark.parametrize("new,base", [("M0_CA", "M0"), ("L3_B050_CA", "L3_B050")])
def test_configs_differ_only_in_channel_exchange(new, base):
    a, b = flat(load_config(new)), flat(load_config(base))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff <= {"experiment", "owner", "training.seeds", "augmentation.channel_exchange"}
    assert "augmentation.channel_exchange" in diff and a["augmentation.channel_exchange"] == 0.5


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = make_synthetic_sysu(tmp_path_factory.mktemp("sysu") / "SYSU-MM01", train=range(1, 9),
                               val=range(9, 11), test=range(11, 13), images_per_camera=4, inject_anomalies=False)
    rows, _ = D.build_manifest(root, workers=1)
    return root, rows


def test_config_wires_channel_exchange_into_loader(dataset):
    root, rows = dataset
    loader, sampler, ds, _ = D.build_train_loader_from_config(load_config("M0_CA"), manifest=rows, root=root,
                                                              num_workers=0, pin_memory=False)
    assert ds.transforms["rgb"].channel_exchange == 0.5 and ds.transforms["ir"].channel_exchange == 0.5
    assert ds.transforms["rgb"].band_erase.p == 0.0                         # 不带横条遮挡
    sampler.set_epoch(0)
    batch = next(iter(loader))
    D.check_pmt_layout(batch, sampler.P, sampler.K)                         # PMT 批次结构不变
    _, _, ds0, _ = D.build_train_loader_from_config(load_config("M0"), manifest=rows, root=root,
                                                    num_workers=0, pin_memory=False)
    assert ds0.transforms["rgb"].channel_exchange == 0.0                    # 已有配置不受影响
