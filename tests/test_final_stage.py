"""最终阶段（train+val 合并训练、只导出测试集）的自检。合成数据/小模型只验证流程，不代表训练结果。"""
import copy
from types import SimpleNamespace

import pytest
import torch

import src.train as training
from src.config import load_config
from src.data import build_manifest, build_train_loader
from src.export_features import check_split
from src.model import PMT
from src.training import load_checkpoint, seed_all
from synthetic_sysu import make_synthetic_sysu

TINY = dict(image_size=[32, 20], patch_size=8, patch_stride=6, global_feature_dim=24,
            depth=2, heads=3, dropout=.1, drop_path=.1)
TRAIN, VAL = range(1, 5), range(5, 7)


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(1)
    seed_all(17)


def test_finalize_merges_train_and_val_only():
    for experiment in ("M0", "L3_B050"):
        dev = load_config(experiment)
        final = training.finalize(dev)
        assert final["data"]["stage"] == "final" and training.is_final(final) and not training.is_final(dev)
        assert final["data"]["train_splits"] == ["train", "val"]
        assert final["data"]["evaluation_split"] == "test"
        assert final["data"]["train_identities"] == 395                     # 296 + 99
        assert dev["data"]["train_identities"] == 296                       # 原配置不被修改
        changed = {k for k in final["data"] if final["data"].get(k) != dev["data"].get(k)}
        assert changed == {"stage", "train_splits", "evaluation_split", "train_identities"}
        assert {k: v for k, v in final.items() if k != "data"} == {k: v for k, v in dev.items() if k != "data"}


@pytest.fixture
def data(tmp_path):
    root = make_synthetic_sysu(tmp_path / "data", train=TRAIN, val=VAL, test=range(7, 9),
                               images_per_camera=2, inject_anomalies=False)
    rows, _ = build_manifest(root, workers=1)
    return root, rows


def prepare(data, tmp_path, monkeypatch, experiment="L3_B050"):
    root, rows = data
    cfg = load_config(experiment)
    cfg["model"].update(TINY)
    cfg["training"].update(epochs=2, stage1_epochs=1, amp=False,
                           identities_per_batch=2, images_per_identity_per_modality=2)
    cfg["data"].update(train_identities=len(TRAIN), validation_identities=len(VAL))
    seen = {}

    def loader(cfg, final=False, **kwargs):
        seen["final"] = final
        splits = ("train", "val") if final else ("train",)
        return build_train_loader(rows, root, train_splits=splits, identities_per_batch=2,
                                  images_per_identity_per_modality=2, image_size=(32, 20),
                                  num_workers=0, batches_per_epoch=2, pin_memory=False)

    monkeypatch.setattr(training, "PROJECT", tmp_path)
    monkeypatch.setattr(training, "load_config", lambda _: copy.deepcopy(cfg))
    monkeypatch.setattr(training, "get_train_loader", loader)
    monkeypatch.setattr(training, "provenance", lambda *a: {"manifest_sha256": "synthetic", "pretrained_sha256": "synthetic"})
    initial = tmp_path / "initial.pt"
    torch.save(PMT(4, TINY).base.state_dict(), initial)
    args = SimpleNamespace(resume=None, experiment=experiment, run_id="final_L3_B050_seed17", seed=17, device="cpu",
                           pretrained=str(initial), root=str(root), num_workers=0, max_steps=1,
                           log_every=20, save_every=1, final=True)
    return args, seen


def test_final_run_trains_on_train_plus_val_and_resumes(data, tmp_path, monkeypatch):
    args, seen = prepare(data, tmp_path, monkeypatch)
    training.run(args)
    assert seen["final"] is True
    path = tmp_path / "checkpoints/final_L3_B050_seed17/last.pt"
    first = load_checkpoint(path)
    assert first["config"]["data"]["stage"] == "final"
    assert first["num_classes"] == len(TRAIN) + len(VAL)                    # 验证集身份也参与训练
    assert set(first["label_map"]) == set(TRAIN) | set(VAL)
    # 恢复训练沿用 checkpoint 的最终阶段配置
    args.resume, args.max_steps = str(path), None
    training.run(args)
    done = load_checkpoint(path)
    assert done["config"]["data"]["stage"] == "final" and done["sampler"]["epoch"] == 2
    # 恢复时漏掉 --final 会被拒绝
    args.final = False
    with pytest.raises(ValueError, match="--final must match"):
        training.run(args)


def test_dev_run_unchanged(data, tmp_path, monkeypatch):
    args, seen = prepare(data, tmp_path, monkeypatch)
    args.final, args.run_id = False, "dev_L3_B050_seed17"
    training.run(args)
    ck = load_checkpoint(tmp_path / "checkpoints/dev_L3_B050_seed17/last.pt")
    assert seen["final"] is False and "stage" not in ck["config"]["data"]
    assert set(ck["label_map"]) == set(TRAIN)


@pytest.mark.parametrize("final,run_id", [(True, "dev_L3_B050_seed17"), (False, "final_L3_B050_seed17")])
def test_run_id_must_match_stage(data, tmp_path, monkeypatch, final, run_id):
    args, _ = prepare(data, tmp_path, monkeypatch)
    args.final, args.run_id = final, run_id
    with pytest.raises(ValueError, match="final_"):
        training.run(args)


def test_export_split_guards():
    check_split("final", "test")
    check_split("dev", "val")
    check_split("dev", "test", allow_dev_checkpoint=True)                   # 降级方案需显式声明
    with pytest.raises(ValueError, match="validation identities"):
        check_split("final", "val")
    with pytest.raises(ValueError, match="allow-dev-checkpoint"):
        check_split("dev", "test")
