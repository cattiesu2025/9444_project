"""AGW/CAJ baseline checks. Synthetic images and few steps do not count as training results."""
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.config import load_config
from src.agw import (AGW, AGWLoss, wrt, adp, weighted_extremes, make_sgd, StepWarmupSchedule,
                     make_transforms, ChannelExchange, ChannelAdapGray, collate_caj, check_caj_layout)
from src.data import build_manifest, build_train_loader, SYSUImageList, select_rows, MODALITY_CODE
from src.export_features import export_dataset
from src.training import seed_all, load_checkpoint
from synthetic_sysu import make_synthetic_sysu

SIZE = (64, 32)
PRETRAINED = Path(__file__).resolve().parents[1] / "checkpoints/resnet50-19c8e357.pth"


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(2)
    seed_all(17)


def small_cfg(experiment):
    cfg = load_config(experiment)
    cfg["model"]["image_size"] = list(SIZE)
    return cfg


def test_configs_follow_upstream():
    agw, caj = load_config("AGW"), load_config("CAJ")
    assert agw["training"]["epochs"] == 81 and caj["training"]["epochs"] == 100
    assert agw["loss"]["method"] == "ID+WRT" and caj["loss"]["method"] == "ID+ADP"
    for cfg in (agw, caj):
        assert cfg["model"]["image_size"] == [288, 144]
        assert cfg["data"]["train_identities"] == 296 and cfg["data"]["evaluation_split"] == "val"
        assert cfg["training"]["identities_per_batch"] == 8
        assert cfg["training"]["images_per_identity_per_modality"] == 4


def test_modality_routing_and_shapes():
    model = AGW(3, small_cfg("AGW")["model"]).eval()
    images = torch.randn(6, 3, *SIZE)
    modalities = torch.tensor([0, 1, 0, 1, 1, 0])
    with torch.no_grad():
        mixed = model(images, modalities)
        rgb = model(images[modalities == 0], modalities[modalities == 0])
        ir = model(images[modalities == 1], modalities[modalities == 1])
    assert mixed["pool_feature"].shape == (6, 2048) and mixed["logits"].shape == (6, 3)
    torch.testing.assert_close(mixed["retrieval_feature"][modalities == 0], rgb["retrieval_feature"])
    torch.testing.assert_close(mixed["retrieval_feature"][modalities == 1], ir["retrieval_feature"])
    torch.testing.assert_close(mixed["retrieval_feature"].norm(dim=1), torch.ones(6))
    with torch.no_grad():
        model.thermal[0].weight.add_(1.)
        changed = model(images, modalities)["retrieval_feature"]
    torch.testing.assert_close(changed[modalities == 0], mixed["retrieval_feature"][modalities == 0])
    assert not torch.allclose(changed[modalities == 1], mixed["retrieval_feature"][modalities == 1])


def test_architecture_matches_upstream():
    model = AGW(3, small_cfg("AGW")["model"])
    assert sorted(model.non_local) == ["1_2", "1_3", "2_3", "2_4", "2_5"]
    assert model.layers[3][0].conv2.stride == (1, 1) and model.layers[3][0].downsample[0].stride == (1, 1)
    assert not model.bottleneck.bias.requires_grad and model.classifier.bias is None
    assert all(block.W[1].weight.abs().sum() == 0 for block in model.non_local.values())
    model.train()
    output = model(torch.randn(4, 3, *SIZE), torch.tensor([0, 0, 1, 1]))
    output["logits"].square().sum().backward()
    assert torch.isfinite(model.visible[0].weight.grad).all() and torch.isfinite(model.thermal[0].weight.grad).all()


@pytest.mark.skipif(not PRETRAINED.is_file(), reason="run scripts/prepare_pretrained.py --model resnet50")
def test_pretrained_strict_load_into_both_stems():
    from src.agw import load_resnet50_state
    state = load_resnet50_state(PRETRAINED)
    model = AGW(3, small_cfg("AGW")["model"], pretrained=PRETRAINED)
    torch.testing.assert_close(model.visible[0].weight, state["conv1.weight"])
    torch.testing.assert_close(model.thermal[0].weight, state["conv1.weight"])
    torch.testing.assert_close(model.layers[2][5].conv3.weight, state["layer3.5.conv3.weight"])
    assert model.visible[0].weight.data_ptr() != model.thermal[0].weight.data_ptr()


def reference_extremes(x, labels):
    """Loop version of upstream softmax-weighted mining, self-match included."""
    d = torch.cdist(x, x).clamp(min=1e-6)
    fp, cn = [], []
    for i in range(len(x)):
        pos, neg = d[i][labels == labels[i]], d[i][labels != labels[i]]
        fp.append((pos * torch.softmax(pos, 0)).sum())
        cn.append((neg * torch.softmax(-neg, 0)).sum())
    return torch.stack(fp), torch.stack(cn)


def test_weighted_mining_matches_loop_reference():
    x = torch.randn(12, 5, dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2] * 2)
    positive, negative = weighted_extremes(x, labels)
    expected_p, expected_n = reference_extremes(x, labels)
    torch.testing.assert_close(positive, expected_p, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(negative, expected_n, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(wrt(x, labels), torch.nn.functional.softplus(expected_p - expected_n).mean(),
                               rtol=1e-5, atol=1e-5)


def test_adp_squared_difference_direction():
    labels = torch.tensor([0, 0, 1, 1])
    near = torch.tensor([[0.], [1.], [3.], [4.]])
    far = torch.tensor([[0.], [1.], [9.], [10.]])
    assert adp(far, labels) < adp(near, labels)        # wider margin, lower loss
    violated = torch.tensor([[0.], [5.], [1.], [6.]])   # positives further than negatives
    p, n = weighted_extremes(violated, labels)
    assert (p > n).all()
    expected = torch.nn.functional.softplus((p - n).pow(2)).mean()
    torch.testing.assert_close(adp(violated, labels), expected)
    terms = AGWLoss(load_config("CAJ")["loss"])({"pool_feature": violated.requires_grad_(),
                                                 "logits": torch.zeros(4, 2)}, labels)
    terms["total"].backward()
    assert set(terms) == {"total", "id", "triplet"}


def test_channel_augmentations():
    img = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    for seed in range(20):
        torch.manual_seed(seed)
        random.seed(seed)
        out = ChannelExchange(gray=2)(img.clone())
        assert any(torch.equal(out, img[c].expand(3, -1, -1)) for c in range(3))
        out = ChannelAdapGray(0.5)(img.clone())
        assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2]) or torch.equal(out, img)


def test_step_warmup_schedule():
    cfg = load_config("AGW")["training"]
    model = AGW(3, small_cfg("AGW")["model"])
    optimizer = make_sgd(model, cfg)
    schedule = StepWarmupSchedule(optimizer, cfg)
    expected = {0: .01, 9: .1, 10: .1, 19: .1, 20: .01, 49: .01, 50: .001, 80: .001}
    for epoch, lr in expected.items():
        schedule.step(epoch)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(lr * .1)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(lr)
        assert optimizer.param_groups[2]["lr"] == pytest.approx(lr)
    assert optimizer.defaults["nesterov"] and optimizer.defaults["weight_decay"] == 5e-4
    head = {id(p) for p in optimizer.param_groups[1]["params"] + optimizer.param_groups[2]["params"]}
    assert id(model.bottleneck.weight) in head and id(model.classifier.weight) in head


@pytest.fixture
def data(tmp_path):
    root = make_synthetic_sysu(tmp_path / "data", train=range(1, 5), val=range(5, 7),
                               test=range(7, 9), images_per_camera=2, inject_anomalies=False)
    rows, _ = build_manifest(root, workers=1)
    return root, rows


def loader_for(rows, root, experiment):
    loader, sampler, dataset, labels = build_train_loader(rows, root, identities_per_batch=2,
        images_per_identity_per_modality=2, image_size=SIZE, num_workers=0, batches_per_epoch=2,
        pin_memory=False, gray_probability=0.0)
    dataset.transforms = make_transforms(small_cfg(experiment))
    return loader, sampler, dataset, labels


def test_caj_batch_layout(data):
    from torch.utils.data import DataLoader
    root, rows = data
    _, sampler, dataset, _ = loader_for(rows, root, "CAJ")
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_caj)
    batch = next(iter(loader))
    n = sampler.P * sampler.K
    assert batch["images"].shape == (3 * n, 3, *SIZE)
    assert batch["modalities"].tolist() == [0] * 2 * n + [1] * n
    check_caj_layout(batch, sampler.P, sampler.K)
    view1 = batch["images"][n:2 * n]
    assert all(torch.equal(v[0], v[1]) and torch.equal(v[1], v[2]) for v in view1)  # one channel replicated
    assert not torch.equal(batch["images"][:n], view1)


@pytest.mark.parametrize("experiment", ["AGW", "CAJ"])
def test_entrypoint_resume_and_export(data, tmp_path, monkeypatch, experiment):
    import src.train_agw as training
    root, rows = data
    cfg = small_cfg(experiment)
    # Random-init ResNet + upstream lr 0.1 diverges on 4-image batches; this checks wiring, not convergence.
    cfg["training"].update(epochs=2, warmup_epochs=1, lr_milestones=[1], identities_per_batch=2,
                           images_per_identity_per_modality=2, learning_rate=1e-3)
    cfg["data"]["train_identities"] = 4
    monkeypatch.setattr(training, "PROJECT", tmp_path)
    monkeypatch.setattr(training, "load_config", lambda _: copy.deepcopy(cfg))
    monkeypatch.setattr(training, "get_train_loader", lambda cfg, **kwargs: build_train_loader(rows, root,
        identities_per_batch=2, images_per_identity_per_modality=2, image_size=SIZE, num_workers=0,
        batches_per_epoch=2, pin_memory=False))
    monkeypatch.setattr(training, "provenance", lambda *args: {"manifest_sha256": "synthetic", "pretrained_sha256": "synthetic"})
    from torchvision.models import resnet50
    initial = tmp_path / "initial.pt"
    torch.save(resnet50(weights=None).state_dict(), initial)
    args = SimpleNamespace(resume=None, experiment=experiment, run_id="resumed", seed=17, device="cpu",
                           pretrained=str(initial), root=str(root), num_workers=0, preload=False,
                           max_steps=1, log_every=20, save_every=1)
    training.run(args)
    path = tmp_path / "checkpoints/resumed/last.pt"
    args.resume, args.max_steps = str(path), None
    training.run(args)
    resumed = load_checkpoint(path)
    args.resume, args.run_id = None, "continuous"
    training.run(args)
    continuous = load_checkpoint(tmp_path / "checkpoints/continuous/last.pt")
    for key, value in continuous["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    records = [json.loads(l) for l in (tmp_path / "artifacts/results/resumed/train.jsonl").read_text().splitlines()]
    assert [r["global_step"] for r in records] == [1, 2, 3, 4]
    assert {r["batch_rows"] for r in records} == {12 if experiment == "CAJ" else 8}
    assert records[0]["lr"] == pytest.approx([1e-4, 1e-3]) and records[-1]["lr"] == pytest.approx([1e-5, 1e-4])
    assert json.loads((tmp_path / "artifacts/results/resumed/summary.json").read_text())["status"] == "completed"

    model = AGW(resumed["num_classes"], resumed["config"]["model"])
    model.load_state_dict(resumed["model"], strict=True)
    dataset = SYSUImageList(rows, root, select_rows(rows, ["val"], modality="ir"), image_size=SIZE)
    metadata = export_dataset(model, dataset, tmp_path / "query.npz", {"synthetic": True}, batch_size=3)
    arrays = np.load(tmp_path / "query.npz", allow_pickle=False)
    assert metadata["feature_dim"] == 2048 and set(arrays["modalities"]) == {MODALITY_CODE["ir"]}
    np.testing.assert_allclose(np.linalg.norm(arrays["features"], axis=1), 1., atol=1e-5)
