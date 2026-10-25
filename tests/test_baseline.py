"""B baseline checks. Tiny models/synthetic images do not count as training results."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from src.config import load_config
from src.model import PMT, VisionTransformer
from src.losses import PMTLoss, hard_triplet, msel, dcl
from src.training import (seed_all, rng_state, restore_rng, make_optimizer, EpochSchedule,
                          save_checkpoint, load_checkpoint, restore_training, isolated_batches)
from src.train import stage_at
from src.export_features import export_dataset
from src.data import build_manifest, build_train_loader, SYSUImageList, select_rows
from synthetic_sysu import make_synthetic_sysu

TINY = dict(image_size=[32, 20], patch_size=8, patch_stride=6, global_feature_dim=24,
            depth=2, heads=3, dropout=.1, drop_path=.1)


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(1)
    seed_all(17)


def test_real_grid_and_gradient():
    model = PMT(3, TINY)
    output = model(torch.randn(8, 3, 32, 20))
    assert output["patch_grid"] == (5, 3)
    assert output["patch_features"].shape == (8, 15, 24)
    assert output["global_logits"].shape == (8, 3)
    assert output["local_logits"] is None
    output["global_logits"].square().sum().backward()
    assert torch.isfinite(model.base.patch_embed.proj.weight.grad).all()
    assert not model.bottleneck.bias.requires_grad
    model.eval()
    torch.testing.assert_close(model(torch.randn(2, 3, 32, 20))["retrieval_feature"].norm(dim=1), torch.ones(2))


def test_default_overlap_grid():
    from src.model import PatchEmbed
    layer = PatchEmbed((256, 128), 16, 12, 24)
    patches, grid = layer(torch.randn(1, 3, 256, 128))
    assert grid == (21, 10) and patches.shape[1] == 210


def test_pretrained_load_and_missing_key(tmp_path):
    src = VisionTransformer(image_size=(32, 32), patch_size=8, patch_stride=8, dim=24, depth=2, heads=3)
    dst = PMT(3, TINY).base
    state = src.state_dict()
    path = tmp_path / "pretrained.pt"
    torch.save(state, path)
    dst.load_pretrained(path)
    torch.testing.assert_close(dst.blocks[0].attn.qkv.weight, src.blocks[0].attn.qkv.weight)
    del state["cls_token"]
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        dst.load_pretrained(path)


def test_hand_calculated_losses():
    # RGB: ID0 [0,2], ID1 [8,10]; IR: ID0 [1,3], ID1 [9,11].
    x = torch.tensor([[0.], [2.], [8.], [10.], [1.], [3.], [9.], [11.]])
    labels = torch.tensor([0, 0, 1, 1] * 2)
    torch.testing.assert_close(msel(x, labels, 2), torch.tensor(.5))
    # Centers 1.5,9.5; mean positive=1; selected negative means=7 each.
    torch.testing.assert_close(dcl(x, labels, 2), torch.tensor(1 / 7))
    assert hard_triplet(x, labels) == 0


@pytest.mark.parametrize("stage", [1, 2, "V0"])
def test_stage_loss_gradients(stage):
    x = torch.randn(12, 16, requires_grad=True)
    scores = torch.randn(12, 3, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2] * 2)
    terms = PMTLoss(k=2)({"global_feature": x, "global_logits": scores}, labels, stage)
    terms["total"].backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(scores.grad).all()
    if stage != 2:
        assert terms["msel"] == terms["dcl"] == 0


def test_degenerate_dcl_finite():
    labels = torch.tensor([0, 0, 1, 1] * 2)
    for features in (torch.zeros(8, 4), torch.tensor([[0.], [0.], [4.], [4.]] * 2)):
        features.requires_grad_()
        loss = dcl(features, labels, 2)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(features.grad).all()


def test_bad_layout_rejected():
    labels = torch.tensor([0, 1, 0, 1] * 2)
    with pytest.raises(ValueError):
        msel(torch.randn(8, 4), labels, 2)


def test_stage_boundary_and_schedule():
    cfg = load_config("M0")
    assert [stage_at(cfg, e) for e in (0, 5, 6, 23)] == [1, 1, 2, 2]
    assert stage_at(load_config("V0"), 0) == "V0"
    model = PMT(3, TINY)
    optimizer = make_optimizer(model, cfg["training"])
    schedule = EpochSchedule(optimizer, cfg["training"])
    schedule.step(1)
    assert max(g["lr"] for g in optimizer.param_groups) == pytest.approx(.000102)
    assert min(g["lr"] for g in optimizer.param_groups) == pytest.approx(.000052)
    schedule.step(24)
    assert all(g["lr"] == pytest.approx(.000003) for g in optimizer.param_groups)


@pytest.fixture
def data(tmp_path):
    root = make_synthetic_sysu(tmp_path / "data", train=range(1, 5), val=range(5, 7),
                               test=range(7, 9), images_per_camera=2, inject_anomalies=False)
    rows, _ = build_manifest(root, workers=1)
    return root, rows


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_matches_uninterrupted_and_export(data, tmp_path, workers):
    root, rows = data
    def components():
        loader, sampler, dataset, labels = build_train_loader(rows, root, identities_per_batch=2,
            images_per_identity_per_modality=2, image_size=(32, 20), num_workers=workers,
            batches_per_epoch=3, pin_memory=False)
        model = PMT(len(labels), TINY)
        optimizer = make_optimizer(model, load_config("M0")["training"])
        scheduler = EpochSchedule(optimizer, load_config("M0")["training"])
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return loader, sampler, model, optimizer, scheduler, scaler
    def step(model, optimizer, batch):
        optimizer.zero_grad()
        loss = PMTLoss(k=2)(model(batch["images"]), batch["labels"], 2)["total"]
        loss.backward()
        optimizer.step()
        return loss.detach()
    loader, sampler, model, optimizer, scheduler, scaler = components()
    scheduler.step(7)
    iterator = isolated_batches(loader)
    step(model, optimizer, next(iterator))
    sampler.set_epoch(0, start_batch=1)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, optimizer, scheduler, scaler, sampler)
    expected_loss = step(model, optimizer, next(iterator))
    expected = copy.deepcopy(model.state_dict())
    loader2, sampler2, model2, optimizer2, scheduler2, scaler2 = components()
    restore_training(load_checkpoint(path), model2, optimizer2, scheduler2, scaler2, sampler2)
    actual_loss = step(model2, optimizer2, next(isolated_batches(loader2)))
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for key, value in model2.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    dataset = SYSUImageList(rows, root, select_rows(rows, ["val"], modality="ir"), image_size=(32, 20))
    output = tmp_path / "features.npz"
    metadata = export_dataset(model2, dataset, output, {"synthetic": True}, batch_size=3)
    arrays = np.load(output, allow_pickle=False)
    assert len(arrays["paths"]) == len(dataset) == metadata["count"]
    assert set(arrays["identities"]) == {5, 6}
    assert set(arrays["modalities"]) == {1}
    np.testing.assert_allclose(np.linalg.norm(arrays["features"], axis=1), 1., atol=1e-6)
    with pytest.raises(FileExistsError):
        export_dataset(model2, dataset, output, {}, batch_size=3)


def test_rng_roundtrip():
    state = rng_state()
    values = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(state)
    assert random.random() == values[0]
    assert np.random.rand() == values[1]
    torch.testing.assert_close(torch.rand(3), values[2], rtol=0, atol=0)


def test_training_entrypoint_crosses_stage_and_resumes(data, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.train as training
    root, rows = data
    cfg = load_config("M0")
    cfg["model"].update(TINY)
    cfg["training"].update(epochs=2, stage1_epochs=1, amp=False,
                            identities_per_batch=2, images_per_identity_per_modality=2)
    cfg["data"]["train_identities"] = 4
    monkeypatch.setattr(training, "PROJECT", tmp_path)
    monkeypatch.setattr(training, "load_config", lambda _: copy.deepcopy(cfg))
    monkeypatch.setattr(training, "get_train_loader", lambda cfg, **kwargs: build_train_loader(rows, root,
        identities_per_batch=2, images_per_identity_per_modality=2, image_size=(32, 20),
        num_workers=0, batches_per_epoch=2, pin_memory=False))
    monkeypatch.setattr(training, "provenance", lambda *args: {"manifest_sha256": "synthetic", "pretrained_sha256": "synthetic"})
    initial = tmp_path / "initial.pt"
    torch.save(PMT(4, TINY).base.state_dict(), initial)
    args = SimpleNamespace(resume=None, experiment="M0", run_id="resumed", seed=17, device="cpu",
        pretrained=str(initial), root=str(root), num_workers=0, max_steps=1, log_every=20, save_every=1)
    training.run(args)
    path = tmp_path / "checkpoints/resumed/last.pt"
    args.resume = str(path)
    args.max_steps = None
    training.run(args)
    resumed = load_checkpoint(path)
    args.resume, args.run_id = None, "continuous"
    training.run(args)
    continuous = load_checkpoint(tmp_path / "checkpoints/continuous/last.pt")
    for key, value in continuous["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    records = [json.loads(line) for line in (tmp_path / "artifacts/results/resumed/train.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in records] == [1, 1, 2, 2]
    assert [r["gray_count"] for r in records] == [4, 4, 0, 0]
    assert resumed["sampler"]["epoch"] == 2 and resumed["sampler"]["start_batch"] == 0
    assert json.loads((tmp_path / "artifacts/results/resumed/summary.json").read_text())["status"] == "completed"


def test_small_batch_can_fit():
    cfg = dict(TINY, dropout=0., drop_path=0.)
    model = PMT(2, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    images = torch.randn(8, 3, 32, 20)
    labels = torch.tensor([0, 0, 1, 1] * 2)
    criterion = PMTLoss(k=2)
    values = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(images), labels, 2)["total"]
        loss.backward()
        optimizer.step()
        values.append(float(loss.detach()))
    assert np.isfinite(values).all()
    assert values[-1] < .6 * values[0]


def test_activation_checkpointing_preserves_gradients_and_rng():
    model = PMT(2, TINY)
    recomputed = PMT(2, dict(TINY, activation_checkpointing=True))
    recomputed.load_state_dict(model.state_dict())
    images = torch.randn(8, 3, 32, 20)
    labels = torch.tensor([0, 0, 1, 1] * 2)
    initial = rng_state()
    criterion = PMTLoss(k=2)
    loss = criterion(model(images), labels, 2)["total"]
    loss.backward()
    expected_rng = torch.get_rng_state()
    restore_rng(initial)
    repeated = criterion(recomputed(images), labels, 2)["total"]
    repeated.backward()
    torch.testing.assert_close(loss, repeated, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for p, q in zip(model.parameters(), recomputed.parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_amp_overflow_skips_update_and_recovers():
    from src.training import scaled_optimizer_step
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=4.)
    original = model.weight.detach().clone()
    # Finite loss, deliberately overflowing backward gradient.
    handle = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    updated, before, after = scaled_optimizer_step(model(torch.ones(1, 1)).sum(), model, optimizer, scaler)
    assert not updated and before == 4. and after == 2.
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    handle.remove()
    optimizer.zero_grad(set_to_none=True)
    updated, before, after = scaled_optimizer_step(model(torch.ones(1, 1)).sum(), model, optimizer, scaler)
    assert updated and before == after == 2.
    torch.testing.assert_close(model.weight, original - .1)


def test_non_amp_nonfinite_gradient_still_fails():
    from src.training import scaled_optimizer_step
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    with pytest.raises(FloatingPointError, match="without AMP"):
        scaled_optimizer_step(model(torch.ones(1, 1)).sum(), model, optimizer,
                              torch.amp.GradScaler("cpu", enabled=False))
