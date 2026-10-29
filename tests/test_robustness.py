"""E：鲁棒性扰动与流程自检。小模型/合成图片只验证流程，不代表真实鲁棒性结果。"""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import build_manifest, write_manifest, SYSUImageList, select_rows
from src.evaluate import build_gallery_trials, evaluate, gallery_rows, load_features, sha256_file
from src.export_features import export_dataset
from src.model import PMT
from src.robustness import CONDITIONS, FILL, Perturbation, PerturbedImageList, extract, run_robustness
from synthetic_sysu import make_synthetic_sysu

TINY = dict(image_size=[32, 20], patch_size=8, patch_stride=6, global_feature_dim=24,
            depth=2, heads=3, dropout=.1, drop_path=.1)


def random_image(height=256, width=128, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize("region,ratio,expected", [("top", .1, (0, 26)), ("top", .2, (0, 51)),
                                                   ("middle", .1, (115, 141)), ("middle", .2, (102, 153)),
                                                   ("bottom", .1, (230, 256)), ("bottom", .2, (205, 256))])
def test_occlusion_band_geometry_and_fill(region, ratio, expected):
    # 256 行：10% → round(25.6)=26 行，20% → round(51.2)=51 行；middle 居中
    image = random_image()
    out = np.array(Perturbation("occlusion", region, ratio)(image))
    start, stop = expected
    assert (out[start:stop] == FILL).all()
    before = np.array(image)
    keep = np.r_[0:start, stop:256]
    assert (out[keep] == before[keep]).all()                   # 带外像素不变
    assert abs((stop - start) / 256 - ratio) <= 0.5 / 256       # 面积比例误差 ≤ 半行


def test_fill_is_imagenet_mean_pixel():
    assert FILL == (124, 116, 104)


def test_blur_is_deterministic_and_size_preserving():
    image = random_image(seed=1)
    for sigma, kernel in ((1., 7), (2., 13)):
        p = Perturbation("blur", sigma=sigma)
        assert p.kernel == kernel
        a, b = np.array(p(image)), np.array(p(image))
        assert a.shape == np.array(image).shape and (a == b).all()
        assert not (a == np.array(image)).all()
    weak, strong = (np.array(Perturbation("blur", sigma=s)(image)).astype(float) for s in (1., 2.))
    assert strong.std() < weak.std() < np.array(image).astype(float).std()   # 越模糊对比越低


def test_clean_is_identity_and_conditions_fixed():
    image = random_image(seed=2)
    assert Perturbation()(image) is image
    assert list(CONDITIONS) == ["clean", "occlusion_top_10", "occlusion_top_20", "occlusion_middle_10",
                                "occlusion_middle_20", "occlusion_bottom_10", "occlusion_bottom_20",
                                "blur_sigma1", "blur_sigma2"]


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(17)
    root = make_synthetic_sysu(tmp_path / "data", train=range(1, 5), val=range(5, 9),
                               test=range(9, 11), images_per_camera=2, inject_anomalies=False)
    rows, _ = build_manifest(root, workers=1)
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest)
    model = PMT(4, TINY).eval()
    checkpoint = tmp_path / "last.pt"
    torch.save({"model": model.state_dict(), "config": {"model": TINY}, "num_classes": 4, "run_id": "synthetic",
                "seed": 17, "manifest_sha256": sha256_file(manifest)}, checkpoint)
    meta = {"checkpoint_sha256": sha256_file(checkpoint), "split": "val", "run_id": "synthetic"}
    for name, modality in (("query", "ir"), ("pool", "rgb")):
        dataset = SYSUImageList(rows, root, select_rows(rows, ["val"], modality=modality), image_size=(32, 20))
        export_dataset(model, dataset, tmp_path / f"{name}.npz", meta, batch_size=4)
    gallery = tmp_path / "gallery.json"
    gallery.write_text(json.dumps(build_gallery_trials(load_features(tmp_path / "pool.npz"), [0, 1, 2], split="val")))
    return dict(root=root, rows=rows, manifest=manifest, model=model, checkpoint=checkpoint, gallery=gallery, tmp=tmp_path)


def run(s, out="robust", **kwargs):
    return run_robustness(s["checkpoint"], s["tmp"] / "pool.npz", s["manifest"], s["root"], s["tmp"] / out,
                          s["gallery"], s["tmp"] / "query.npz", **kwargs)


def test_end_to_end_clean_matches_plain_evaluation(setup):
    s = setup
    pool_sha = sha256_file(s["tmp"] / "pool.npz")
    report = run(s)
    assert list(report["conditions"]) == list(CONDITIONS)
    clean = report["conditions"]["clean"]
    assert clean["export_consistency"]["status"] == "PASS" and clean["export_consistency"]["max_abs_diff"] < 1e-5
    query, pool = load_features(s["tmp"] / "query.npz"), load_features(s["tmp"] / "pool.npz")
    gallery = json.loads(s["gallery"].read_text())
    direct = evaluate(query, pool, gallery_rows(pool, gallery))["mean_over_gallery"]
    assert clean["mean_over_gallery"]["mAP"] == pytest.approx(direct["mAP"], abs=1e-6)
    assert clean["drop_vs_clean_points"]["mAP"] == pytest.approx(0)
    assert sha256_file(s["tmp"] / "pool.npz") == pool_sha            # gallery 未被改动
    assert (s["tmp"] / "robust" / "perturbation_examples.png").exists()
    with pytest.raises(FileExistsError):
        run(s)


def test_perturbation_changes_query_features_only(setup):
    s = setup
    indices = select_rows(s["rows"], ["val"], modality="ir")
    feats = {name: extract(s["model"], PerturbedImageList(s["rows"], s["root"], indices, image_size=(32, 20),
                                                          perturbation=CONDITIONS[name]), batch_size=4)["features"]
             for name in ("clean", "occlusion_middle_20", "blur_sigma2")}
    assert not np.allclose(feats["clean"], feats["occlusion_middle_20"])
    assert not np.allclose(feats["clean"], feats["blur_sigma2"])
    np.testing.assert_allclose(np.linalg.norm(feats["blur_sigma2"], axis=1), 1, atol=1e-5)


def test_rejects_mismatched_checkpoint_or_manifest(setup):
    s = setup
    other = s["tmp"] / "other.pt"
    state = torch.load(s["checkpoint"], weights_only=False)
    torch.save(dict(state, run_id="other"), other)
    with pytest.raises(ValueError, match="different checkpoint"):
        run_robustness(other, s["tmp"] / "pool.npz", s["manifest"], s["root"], s["tmp"] / "x", s["gallery"])
    s["manifest"].write_text(s["manifest"].read_text() + "\n")
    with pytest.raises(ValueError, match="manifest"):
        run(s, out="y")
