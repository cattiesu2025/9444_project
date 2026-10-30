"""Cross-check E's evaluator against the pinned PMT checkout (eval_sysu + process_gallery_sysu).

Usage: python scripts/check_eval_reference.py /path/to/PMT
No upstream files are copied into this repository. Requires only numpy.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import RGB_CAMERAS, sample_gallery, sysu_metrics

COMMIT = "0b31da97b06428473bc4a1134b322ea226340818"


def module(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def random_case(rng, ids=40, queries=300):
    """模拟 SYSU 结构：每个 (身份, RGB 摄像头) 至多一张，部分摄像头缺失，部分 query 身份不在 gallery。"""
    g = [(i, c) for i in range(1, ids + 1) for c in (1, 2, 4, 5) if rng.random() > 0.2]
    g_ids, g_cams = map(np.asarray, zip(*g))
    q_ids = rng.integers(1, ids + 4, size=queries)
    q_cams = rng.choice([3, 6], size=queries)
    dist = rng.normal(size=(queries, len(g_ids)))
    dist -= 1.5 * (q_ids[:, None] == g_ids[None])          # 让正样本更靠前，接近真实排序
    return dist, q_ids, g_ids, q_cams, g_cams


def metric_cases(eval_sysu, cases=200):
    worst = {"cmc": 0., "mAP": 0., "mINP": 0.}
    for seed in range(cases):
        dist, q_ids, g_ids, q_cams, g_cams = random_case(np.random.default_rng(seed))
        ref_cmc, ref_map, ref_minp = eval_sysu(dist, q_ids, g_ids, q_cams, g_cams, max_rank=20)
        ours = sysu_metrics(dist, q_ids, g_ids, q_cams, g_cams, max_rank=20)
        diffs = {"cmc": np.abs(np.asarray(ours["cmc"]) - ref_cmc).max(),
                 "mAP": abs(ours["mAP"] - ref_map), "mINP": abs(ours["mINP"] - ref_minp)}
        for key, value in diffs.items():
            if value > 1e-6:
                raise AssertionError(f"case {seed}: {key} differs from reference by {value}")
            worst[key] = max(worst[key], float(value))
    return worst


def gallery_cases(process_gallery_sysu, seeds=range(10)):
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ids = sorted(rng.choice(np.arange(1, 534), size=30, replace=False).tolist())
        (root / "exp").mkdir()
        (root / "exp" / "test_id.txt").write_text(",".join(map(str, ids)) + "\n")
        rows = []
        for pid in ids:
            for cam in (1, 2, 4, 5):
                if rng.random() < 0.2:
                    continue
                d = root / f"cam{cam}" / f"{pid:04d}"
                d.mkdir(parents=True)
                for k in sorted(rng.choice(np.arange(1, 40), size=int(rng.integers(1, 12)), replace=False)):
                    (d / f"{k:04d}.jpg").touch()
                    rows.append((f"cam{cam}/{pid:04d}/{k:04d}.jpg", pid, cam))
        order = rng.permutation(len(rows))                     # 行顺序与目录顺序无关
        paths, identities, cameras = (np.asarray(v)[order] for v in zip(*rows))
        checked = 0
        for search in RGB_CAMERAS:
            for seed in seeds:
                ref_paths, ref_ids, ref_cams = process_gallery_sysu(str(root), mode=search, trial=seed, gall_mode="single")
                ref = [Path(p).relative_to(root).as_posix() for p in ref_paths]
                ours = sample_gallery(identities, cameras, paths, seed, search)
                if paths[ours].tolist() != ref:
                    raise AssertionError(f"gallery differs from reference: search={search} seed={seed}")
                if identities[ours].tolist() != list(ref_ids) or cameras[ours].tolist() != list(ref_cams):
                    raise AssertionError(f"gallery labels differ: search={search} seed={seed}")
                checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/eval_checks/reference.json"))
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError(f"Expected upstream commit {COMMIT}, got {actual}")
    root = args.checkout / "Pytorch-PMT-VI-ReID"
    eval_sysu = module(root / "eval_metrics.py").eval_sysu
    process_gallery_sysu = module(root / "datamanager.py").process_gallery_sysu
    report = {"upstream_commit": actual, "metric_cases": 200,
              "max_abs_diff": metric_cases(eval_sysu), "gallery_trials_identical": gallery_cases(process_gallery_sysu),
              "result": "PASS"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
