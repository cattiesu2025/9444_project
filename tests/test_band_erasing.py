"""横条遮挡增强（BandErasing）与 L3_B050_BE / M0_BE 配置的自检。合成数据只验证流程，不代表训练效果。"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import data as D  # noqa: E402
from src.config import load_config, validate  # noqa: E402
from synthetic_sysu import make_synthetic_sysu  # noqa: E402

FILL = torch.tensor(D.IMAGENET_MEAN).view(3, 1, 1)


def erased_rows(img):
    """返回整行（全宽、三通道）都等于填充值的行号。"""
    full = (img == FILL).all(dim=0).all(dim=1)
    return set(torch.nonzero(full).flatten().tolist())


@pytest.mark.parametrize("j,expected", [(0, (0, 85)), (1, (85, 170)), (2, (170, 256))])
def test_band_geometry_matches_local_regions(j, expected):
    # 256 行分 3 条：floor(j*256/3)，与 LocalHead 在 patch 网格上的整数划分方式相同
    assert D.BandErasing(p=1, regions=3).band(256, j) == expected


def test_p1_erases_exactly_one_full_width_band():
    eraser = D.BandErasing(p=1, regions=3)
    seen = set()
    for seed in range(60):
        random.seed(seed)
        img = torch.randn(3, 256, 128)
        before = img.clone()
        out = eraser(img)
        rows = erased_rows(out)
        bands = [eraser.band(256, j) for j in range(3)]
        hit = [j for j, (t, b) in enumerate(bands) if set(range(t, b)) <= rows]
        assert len(hit) == 1                                      # 恰好一条带被整条擦除
        t, b = bands[hit[0]]
        keep = [r for r in range(256) if not t <= r < b]
        assert torch.equal(out[:, keep], before[:, keep])         # 带外像素不变
        seen.add(hit[0])
    assert seen == {0, 1, 2}                                      # 三条带都会被选到


def test_p0_is_identity_and_consumes_no_randomness():
    random.seed(5)
    state = random.getstate()
    img = torch.randn(3, 256, 128)
    out = D.BandErasing(p=0, regions=3)(img.clone())
    assert torch.equal(out, img)
    assert random.getstate() == state                            # 不改变已有实验的增强随机序列


def test_probability_is_respected():
    eraser = D.BandErasing(p=0.5, regions=3)
    random.seed(0)
    hits = sum(bool(erased_rows(eraser(torch.randn(3, 64, 16))) & set(range(64))) for _ in range(2000))
    assert abs(hits / 2000 - 0.5) < 0.05


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        D.BandErasing(p=1.5)
    with pytest.raises(ValueError):
        D.BandErasing(p=0.5, regions=0)


def test_default_transform_unchanged():
    # band_erasing 默认 0：与原 PMTTrainTransform 的输出逐像素一致（同一随机种子）
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (256, 128, 3), dtype=np.uint8))
    for modality in ("rgb", "ir"):
        outs = []
        for t in (D.PMTTrainTransform(modality), D.PMTTrainTransform(modality, band_erasing=0.0)):
            random.seed(3)
            torch.manual_seed(3)
            outs.append(t(img))
        assert torch.equal(outs[0], outs[1])


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out


def test_configs_differ_only_in_band_erasing():
    be, base = flat(load_config("L3_B050_BE")), flat(load_config("L3_B050"))
    diff = {k for k in set(be) | set(base) if be.get(k) != base.get(k)}
    assert diff == {"experiment", "owner", "training.seeds", "augmentation.band_erasing"}
    assert be["augmentation.band_erasing"] == 0.5 and be["local.regions"] == 3 and be["local.enabled"]
    m0be, m0 = flat(load_config("M0_BE")), flat(load_config("M0"))
    diff = {k for k in set(m0be) | set(m0) if m0be.get(k) != m0.get(k)}
    # M0_BE 的 local.regions 只决定擦除带的位置（局部监督关闭）
    assert diff == {"experiment", "owner", "augmentation.band_erasing", "local.regions"}
    assert not m0be["local.enabled"]


def test_validate_rejects_bad_probability():
    cfg = load_config("L3_B050_BE")
    cfg["augmentation"]["band_erasing"] = 1.5
    with pytest.raises(ValueError, match="band_erasing"):
        validate(cfg)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = make_synthetic_sysu(tmp_path_factory.mktemp("sysu") / "SYSU-MM01", train=range(1, 9),
                               val=range(9, 11), test=range(11, 13), images_per_camera=4, inject_anomalies=False)
    rows, _ = D.build_manifest(root, workers=1)
    return root, rows


def test_config_wires_band_erasing_into_loader(dataset):
    root, rows = dataset
    _, _, ds, _ = D.build_train_loader_from_config(load_config("L3_B050_BE"), manifest=rows, root=root,
                                                   num_workers=0, pin_memory=False)
    for m in ("rgb", "ir"):
        assert (ds.transforms[m].band_erase.p, ds.transforms[m].band_erase.regions) == (0.5, 3)
    _, _, ds0, _ = D.build_train_loader_from_config(load_config("L3_B050"), manifest=rows, root=root,
                                                    num_workers=0, pin_memory=False)
    assert ds0.transforms["ir"].band_erase.p == 0.0               # 已有配置不受影响


def test_loader_batches_carry_bands(dataset):
    root, rows = dataset
    loader, sampler, _, _ = D.build_train_loader(rows, root, identities_per_batch=4, images_per_identity_per_modality=2,
                                                 num_workers=0, pin_memory=False, band_erasing=1.0, band_regions=3)
    sampler.set_epoch(0)
    batch = next(iter(loader))
    eraser = D.BandErasing(p=1, regions=3)
    bands = [set(range(*eraser.band(256, j))) for j in range(3)]
    for img in batch["images"]:
        rows_ = erased_rows(img)
        assert any(b <= rows_ for b in bands)                     # RGB 与 IR 每张都有一整条被擦除
    D.check_pmt_layout(batch, 4, 2)                               # 不影响 PMT 的 batch 排列
