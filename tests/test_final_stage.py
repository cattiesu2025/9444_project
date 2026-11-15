"""Final stage (M0_final / M0_CA_final): retrain on train+val (395 identities), export the test split only.

Synthetic data checks the pipeline only, not training quality.
"""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import load_config, validate, train_splits  # noqa: E402
from src.data import build_manifest, build_train_loader, get_train_loader  # noqa: E402
from src.export_features import resolve_split  # noqa: E402
from src.model import PMT  # noqa: E402
from src.training import load_checkpoint  # noqa: E402
from synthetic_sysu import make_synthetic_sysu  # noqa: E402

TINY = {"image_size": [32, 20], "patch_stride": 12, "depth": 1, "heads": 2, "global_feature_dim": 24,
        "dropout": 0.0, "drop_path": 0.0}


def without(cfg, *keys):
    cfg = copy.deepcopy(cfg)
    for key in keys:
        cfg.pop(key, None)
    return cfg


@pytest.mark.parametrize("final, dev", [("M0_final", "M0"), ("M0_CA_final", "M0_CA")])
def test_final_config_is_dev_method_on_train_val(final, dev):
    f, d = load_config(final), load_config(dev)
    assert f["stage"] == "final"
    assert train_splits(f) == ["train", "val"] and f["data"]["train_identities"] == 395
    assert f["data"]["evaluation_split"] == "test"
    # Same method and schedule as development: only identity splits, stage, ID and seeds differ.
    for section in ("model", "loss", "smooth", "local", "augmentation"):
        assert f.get(section) == d.get(section), section
    assert without(f["training"], "seeds") == without(d["training"], "seeds")
    assert f["training"]["seeds"] == [17, 42, 2026]
    assert without(f["data"], "train_splits", "train_identities", "evaluation_split") == \
        without(d["data"], "train_identities", "evaluation_split")


def test_m0_ca_final_differs_from_m0_final_only_in_channel_exchange():
    a, b = load_config("M0_final"), load_config("M0_CA_final")
    assert b.get("augmentation", {}).get("channel_exchange") == 0.5
    assert a.get("augmentation", {}).get("channel_exchange", 0.0) == 0.0
    a.pop("augmentation", None), b.pop("augmentation", None)
    a.pop("experiment"), b.pop("experiment")
    assert a == b


def test_stage_split_rules():
    dev = load_config("M0")
    assert train_splits(dev) == ["train"]
    bad = copy.deepcopy(dev)
    bad["data"]["evaluation_split"] = "test"
    with pytest.raises(ValueError, match="Development"):
        validate(bad)
    final = load_config("M0_final")
    for key, value in (("evaluation_split", "val"), ("train_splits", ["train"]), ("train_identities", 296)):
        bad = copy.deepcopy(final)
        bad["data"][key] = value
        with pytest.raises(ValueError, match="Final"):
            validate(bad)


def test_export_split_guard():
    dev, final = load_config("M0"), load_config("M0_final")
    assert resolve_split(dev) == "val" and resolve_split(dev, "val") == "val"
    assert resolve_split(final) == "test" and resolve_split(final, "test") == "test"
    with pytest.raises(ValueError, match="reserved for final"):
        resolve_split(dev, "test")
    with pytest.raises(ValueError, match="trained on"):
        resolve_split(final, "val")


def test_old_checkpoint_config_without_train_splits_still_exports_val():
    cfg = load_config("M0")
    cfg["data"].pop("train_splits", None)
    assert train_splits(cfg) == ["train"] and resolve_split(cfg) == "val"


@pytest.fixture
def data(tmp_path):
    root = make_synthetic_sysu(tmp_path / "data", train=range(1, 5), val=range(5, 7),
                               test=range(7, 9), images_per_camera=2, inject_anomalies=False)
    rows, _ = build_manifest(root, workers=1)
    return root, rows


@pytest.mark.parametrize("experiment", ["M0_final", "M0_CA_final"])
def test_final_entrypoint_trains_on_train_plus_val(data, tmp_path, monkeypatch, experiment):
    import src.train as training
    root, rows = data
    cfg = load_config(experiment)
    cfg["model"].update(TINY)
    cfg["training"].update(epochs=2, stage1_epochs=1, amp=False,
                            identities_per_batch=2, images_per_identity_per_modality=2)
    cfg["data"]["train_identities"] = 6  # synthetic: 4 train + 2 val identities
    seen = {}

    def loader(cfg, **kwargs):
        seen.update(kwargs)
        return build_train_loader(rows, root, train_splits=kwargs["train_splits"],
                                  identities_per_batch=2, images_per_identity_per_modality=2,
                                  image_size=(32, 20), num_workers=0, batches_per_epoch=2, pin_memory=False,
                                  channel_exchange=cfg.get("augmentation", {}).get("channel_exchange", 0.0))

    monkeypatch.setattr(training, "PROJECT", tmp_path)
    monkeypatch.setattr(training, "load_config", lambda _: copy.deepcopy(cfg))
    monkeypatch.setattr(training, "get_train_loader", loader)
    monkeypatch.setattr(training, "provenance", lambda *args: {"manifest_sha256": "synthetic", "pretrained_sha256": "synthetic"})
    initial = tmp_path / "initial.pt"
    torch.save(PMT(6, TINY).base.state_dict(), initial)
    args = SimpleNamespace(resume=None, experiment=experiment, run_id="final", seed=17, device="cpu",
                           pretrained=str(initial), root=str(root), num_workers=0, max_steps=None,
                           log_every=20, save_every=1)
    training.run(args)
    assert seen["train_splits"] == ("train", "val")
    ckpt = load_checkpoint(tmp_path / "checkpoints/final/last.pt")
    assert ckpt["num_classes"] == 6 and sorted(ckpt["label_map"]) == list(range(1, 7))
    assert ckpt["config"]["data"]["evaluation_split"] == "test"
    assert resolve_split(ckpt["config"]) == "test"


def test_final_identity_count_mismatch_is_rejected(data, tmp_path, monkeypatch):
    import src.train as training
    root, rows = data
    cfg = load_config("M0_final")
    cfg["model"].update(TINY)
    cfg["training"].update(amp=False, identities_per_batch=2, images_per_identity_per_modality=2)
    monkeypatch.setattr(training, "PROJECT", tmp_path)
    monkeypatch.setattr(training, "load_config", lambda _: copy.deepcopy(cfg))
    # Loader that ignores train_splits and returns only the 4 train identities.
    monkeypatch.setattr(training, "get_train_loader", lambda cfg, **kwargs: build_train_loader(rows, root,
        identities_per_batch=2, images_per_identity_per_modality=2, image_size=(32, 20),
        num_workers=0, batches_per_epoch=2, pin_memory=False))
    initial = tmp_path / "initial.pt"
    torch.save(PMT(4, TINY).base.state_dict(), initial)
    args = SimpleNamespace(resume=None, experiment="M0_final", run_id="bad", seed=17, device="cpu",
                           pretrained=str(initial), root=str(root), num_workers=0, max_steps=1,
                           log_every=20, save_every=1)
    with pytest.raises(ValueError, match="Unexpected training identity count"):
        training.run(args)


def test_get_train_loader_forwards_train_splits(monkeypatch):
    import src.data as D
    calls = {}
    monkeypatch.setattr(D, "prepare_data", lambda *a, **k: {"manifest": "m.csv", "root": "r"})
    monkeypatch.setattr(D, "_require_passed_audit", lambda *a, **k: None)
    monkeypatch.setattr(D, "build_train_loader_from_config", lambda cfg, **kw: calls.update(kw))
    get_train_loader(load_config("M0_final"), train_splits=("train", "val"))
    assert calls["train_splits"] == ("train", "val")


def test_dev_experiment_still_rejects_unknown_final_method(monkeypatch):
    import src.train as training
    cfg = load_config("M0_final")
    cfg["experiment"] = "L3_B050_final"
    monkeypatch.setattr(training, "load_config", lambda _: cfg)
    with pytest.raises(ValueError, match="final-stage"):
        training.run(SimpleNamespace(resume=None, experiment="L3_B050_final"))
