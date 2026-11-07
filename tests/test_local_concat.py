"""E：测试时局部特征拼接脚本的自检。小模型/合成图片只验证流程和数学，不代表真实结果。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_concat_eval as lce  # noqa: E402
from test_robustness import build_setup  # noqa: E402


def unit(rng, *shape):
    return lce.l2(rng.normal(size=shape))


@pytest.mark.parametrize("q,weight", [(2, 1 / 3), (3, 1 / 4), (3, .5)])
def test_combined_cosine_is_weighted_sum_of_part_cosines(q, weight):
    rng = np.random.default_rng(q)
    ga, gb = unit(rng, 5, 8), unit(rng, 5, 8)
    pa, pb = rng.normal(size=(5, q, 8)), rng.normal(size=(5, q, 8))
    a, b = lce.combine(ga, pa, weight), lce.combine(gb, pb, weight)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1, atol=1e-12)
    expected = weight * (ga * gb).sum(1) + (1 - weight) / q * (lce.l2(pa) * lce.l2(pb)).sum(2).sum(1)
    np.testing.assert_allclose((a * b).sum(1), expected, atol=1e-12)


def test_equal_weight_two_regions_matches_earlier_formula():
    # 2 区域等权与 2026-11-04 那次运行的公式（拼接后除以 sqrt(3)）完全一致
    rng = np.random.default_rng(0)
    g, parts = unit(rng, 4, 6), rng.normal(size=(4, 2, 6))
    old = np.concatenate([g, lce.l2(parts[:, 0]), lce.l2(parts[:, 1])], 1) / np.sqrt(3)
    np.testing.assert_allclose(lce.combine(g, parts, 1 / 3), old, atol=1e-12)


def run_main(s, out, *extra):
    tmp = s["tmp"]
    argv = ["x", "--checkpoint", str(s["checkpoint"]), "--gallery-pool", str(tmp / "pool.npz"),
            "--query", str(tmp / "query.npz"), "--manifest", str(s["manifest"]), "--root", str(s["root"]),
            "--output", str(out), "--gallery-file", str(s["gallery"]), "--device", "cpu", "--num-workers", "0",
            "--batch-size", "4", "--conditions", "clean", "occlusion_bottom_20", *extra]
    old, sys.argv = sys.argv, argv
    try:
        lce.main()
    finally:
        sys.argv = old
    return json.loads(Path(out).read_text())


def test_three_region_local_head_end_to_end(tmp_path):
    s = build_setup(tmp_path, local=True, regions=3)
    report = run_main(s, tmp_path / "concat.json")
    assert report["regions"] == 3                               # 取自局部头，不需要 --regions
    assert {k: v["dim"] for k, v in report["variants"].items()} == {
        "global": 24, "global+parts": 96, "global+local": 96, "global+local_g50": 96}
    assert all(c["status"] == "PASS" for c in report["export_consistency"].values())
    assert set(report["results"]) == {"clean", "occlusion_bottom_20"}


def test_without_local_head_uses_regions_argument(tmp_path):
    s = build_setup(tmp_path)
    report = run_main(s, tmp_path / "concat.json", "--regions", "3")
    assert report["regions"] == 3
    assert {k: v["dim"] for k, v in report["variants"].items()} == {"global": 24, "global+parts": 96}
    with pytest.raises(FileExistsError):
        run_main(s, tmp_path / "concat.json")
