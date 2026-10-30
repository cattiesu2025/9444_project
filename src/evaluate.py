"""E：SYSU-MM01 IR→RGB single-shot 检索评价（固定 gallery、cam3→cam2 排除、身份级 CMC、mAP/mINP）。

输入为 B 导出的 IR query 与 RGB 候选池（`src/export_features.py` 的 npz），不使用训练分类标签。
协议与 PMT 固定 commit 的 `datamanager.process_gallery_sysu` 和 `eval_metrics.eval_sysu` 一致：
- gallery：按身份升序、摄像头顺序，每个 (身份, RGB 摄像头) 用 `random.Random(seed)` 依次从按路径排序的
  候选中抽一张；摄像头无图则跳过且不消耗随机数。All-search 用 cam1/2/4/5，Indoor-search 用 cam1/2。
- 距离为 -(query @ gallery.T)；cam3 的 query 丢弃 cam2 的 gallery（同一位置）。
- CMC 为身份级：排序中同一身份只保留首次出现；AP/mINP 在丢弃后的实例级排序上计算。
- gallery 中没有正样本的 query 不计入任何指标，单独计数并记录。
指标单位 0–1。用法：

    python -m src.evaluate run --query Q.npz --gallery-pool G.npz --output-dir artifacts/results/<run_id>/eval
    python -m src.evaluate summarize --metrics a/metrics.json b/metrics.json --output summary.json
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
IR_CAMERAS = (3, 6)
RGB_CAMERAS = {"all": (1, 2, 4, 5), "indoor": (1, 2)}
RGB, IR = 0, 1
REPORT_RANKS = (1, 5, 10, 20)
METRIC_KEYS = ("mAP", "mINP") + tuple(f"rank{r}" for r in REPORT_RANKS)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_features(path):
    """读取导出 npz，返回数组字典及 `metadata`（原 metadata_json 解析后）。"""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(str(arrays.pop("metadata_json"))) if "metadata_json" in arrays else {}
    arrays["paths"] = arrays["paths"].astype(str)
    arrays["metadata"] = metadata
    return arrays


def check_inputs(query, pool, search="all"):
    """拒绝模态/摄像头错位、维度不一致、非有限值或来自不同权重的特征。"""
    for name, part, modality, cameras in (("query", query, IR, IR_CAMERAS), ("gallery pool", pool, RGB, RGB_CAMERAS["all"])):
        n = len(part["features"])
        if part["features"].ndim != 2 or n == 0:
            raise ValueError(f"{name}: features must be a non-empty [N, D] array")
        for key in ("identities", "cameras", "modalities", "paths"):
            if len(part[key]) != n:
                raise ValueError(f"{name}: {key} has {len(part[key])} rows, expected {n}")
        if not np.isfinite(part["features"]).all():
            raise ValueError(f"{name}: nonfinite features")
        if not (part["modalities"] == modality).all():
            raise ValueError(f"{name}: expected modality code {modality} on every row")
        if not np.isin(part["cameras"], cameras).all():
            raise ValueError(f"{name}: cameras outside {cameras}")
        if len(set(part["paths"].tolist())) != n:
            raise ValueError(f"{name}: duplicate paths")
    if query["features"].shape[1] != pool["features"].shape[1]:
        raise ValueError("query and gallery feature dimensions differ")
    if search not in RGB_CAMERAS:
        raise ValueError(f"search must be one of {sorted(RGB_CAMERAS)}")
    qm, gm = query.get("metadata", {}), pool.get("metadata", {})
    for key in ("checkpoint_sha256", "split", "run_id"):
        if key in qm and key in gm and qm[key] != gm[key]:
            raise ValueError(f"query/gallery metadata mismatch on {key}: {qm[key]} vs {gm[key]}")


def pool_fingerprint(pool):
    """候选池内容指纹（与行顺序无关），用于确认不同模型共用同一 gallery。"""
    lines = sorted(f"{p}\t{i}\t{c}" for p, i, c in zip(pool["paths"], pool["identities"], pool["cameras"]))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def sample_gallery(identities, cameras, paths, seed, search="all"):
    """返回一次 single-shot gallery 抽样的候选池行号（顺序同 PMT：身份升序 × 摄像头顺序）。"""
    identities, cameras, paths = np.asarray(identities), np.asarray(cameras), np.asarray(paths).astype(str)
    rng = random.Random(seed)
    chosen = []
    for pid in sorted(set(identities.tolist())):
        for cam in RGB_CAMERAS[search]:
            rows = np.flatnonzero((identities == pid) & (cameras == cam))
            if len(rows):
                rows = sorted(rows.tolist(), key=lambda r: paths[r])
                chosen.append(rng.choice(rows))
    return np.asarray(chosen, dtype=np.int64)


def build_gallery_trials(pool, seeds, search="all", split=None):
    """生成固定 gallery 记录；以路径保存，便于在不同模型的导出（行顺序可能不同）间复用。"""
    if not np.isin(pool["cameras"], RGB_CAMERAS[search]).any():
        raise ValueError(f"gallery pool has no images on cameras {RGB_CAMERAS[search]} ({search}-search impossible on this split)")
    return {"protocol": {"direction": "IR-to-RGB", "search": search, "shot": "single",
                         "rgb_cameras": list(RGB_CAMERAS[search]), "sampler": "PMT process_gallery_sysu (random.Random(seed).choice)"},
            "split": split, "pool_fingerprint": pool_fingerprint(pool), "pool_count": int(len(pool["paths"])),
            "trials": {str(seed): pool["paths"][sample_gallery(pool["identities"], pool["cameras"], pool["paths"], seed, search)].tolist()
                       for seed in seeds}}


def gallery_rows(pool, gallery):
    """把 gallery 记录的路径映射回当前候选池行号；候选池内容不同则拒绝。"""
    if gallery["pool_fingerprint"] != pool_fingerprint(pool):
        raise ValueError("gallery pool differs from the one the fixed gallery file was built from")
    index = {p: i for i, p in enumerate(pool["paths"].tolist())}
    return {int(seed): np.asarray([index[p] for p in paths], dtype=np.int64) for seed, paths in gallery["trials"].items()}


def sysu_metrics(distmat, q_ids, g_ids, q_cams, g_cams, max_rank=20):
    """SYSU 评价。distmat 越小越相似；返回身份级 CMC（长度 max_rank）、mAP、mINP 与有效 query 统计。"""
    distmat = np.asarray(distmat, dtype=np.float64)
    q_ids, g_ids, q_cams, g_cams = map(np.asarray, (q_ids, g_ids, q_cams, g_cams))
    order = np.argsort(distmat, axis=1, kind="stable")
    cmc = np.zeros(max_rank)
    aps, inps, invalid = [], [], []
    for q in range(len(q_ids)):
        ranked = order[q]
        if q_cams[q] == 3:
            ranked = ranked[g_cams[ranked] != 2]
        ranked_ids = g_ids[ranked]
        hits = np.flatnonzero(ranked_ids == q_ids[q])
        if len(hits) == 0:
            invalid.append(q)
            continue
        # 身份级名次 = 首个正确匹配之前出现过的不同身份数
        identity_rank = len(np.unique(ranked_ids[:hits[0]]))
        if identity_rank < max_rank:
            cmc[identity_rank:] += 1
        aps.append(np.mean(np.arange(1, len(hits) + 1) / (hits + 1)))
        inps.append(len(hits) / (hits[-1] + 1))
    valid = len(aps)
    if valid == 0:
        raise ValueError("no query has a positive match in the gallery")
    cmc /= valid
    result = {"cmc": cmc.tolist(), "mAP": float(np.mean(aps)), "mINP": float(np.mean(inps)),
              "num_queries": int(len(q_ids)), "num_valid_queries": valid, "num_invalid_queries": len(invalid),
              "invalid_query_rows": invalid}
    result.update({f"rank{r}": float(cmc[r - 1]) for r in REPORT_RANKS if r <= max_rank})
    return result


def evaluate(query, pool, trial_rows, max_rank=20):
    """对每组固定 gallery 评价，并给出该训练种子在各 gallery 上的平均。"""
    trials = []
    for seed, rows in sorted(trial_rows.items()):
        distmat = -(query["features"].astype(np.float64) @ pool["features"][rows].astype(np.float64).T)
        metrics = sysu_metrics(distmat, query["identities"], pool["identities"][rows],
                               query["cameras"], pool["cameras"][rows], max_rank)
        trials.append(dict(gallery_seed=seed, gallery_size=int(len(rows)), **metrics))
    mean = {key: float(np.mean([t[key] for t in trials])) for key in METRIC_KEYS if key in trials[0]}
    mean["cmc"] = np.mean([t["cmc"] for t in trials], axis=0).tolist()
    mean["gallery_sd"] = {key: float(np.std([t[key] for t in trials], ddof=1)) if len(trials) > 1 else None
                          for key in ("mAP", "rank1")}
    return {"trials": trials, "mean_over_gallery": mean}


def summarize(runs):
    """跨训练种子汇总：每个 run 先取 gallery 平均，再算 mean 与样本 SD（ddof=1）。"""
    if not runs:
        raise ValueError("no runs to summarize")
    for key in ("protocol", "gallery_fingerprint", "split"):
        if len({json.dumps(r[key], sort_keys=True) for r in runs}) != 1:
            raise ValueError(f"runs differ on {key}; they are not comparable")
    seeds = [r.get("training_seed") for r in runs]
    if None not in seeds and len(set(seeds)) != len(seeds):
        raise ValueError("duplicate training seeds: gallery repeats are not independent training runs")
    out = {"protocol": runs[0]["protocol"], "split": runs[0]["split"], "gallery_fingerprint": runs[0]["gallery_fingerprint"],
           "runs": [{"run_id": r.get("run_id"), "training_seed": r.get("training_seed"),
                     **{k: r["mean_over_gallery"][k] for k in METRIC_KEYS}} for r in runs],
           "num_training_seeds": len(runs)}
    for key in METRIC_KEYS:
        values = [r["mean_over_gallery"][key] for r in runs]
        out[key] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return out


def _write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def run(args):
    query, pool = load_features(args.query), load_features(args.gallery_pool)
    check_inputs(query, pool, args.search)
    split = query["metadata"].get("split", "val")
    if args.seeds is None:
        from .config import load_config
        args.seeds = load_config("M0")["evaluation"]["gallery_seeds"]
    gallery_file = Path(args.gallery_file or PROJECT / f"artifacts/manifests/gallery_{split}_{args.search}_single.json")
    if gallery_file.exists():
        gallery = json.loads(gallery_file.read_text(encoding="utf-8"))
        missing = sorted(set(map(str, args.seeds)) - set(gallery["trials"]))
        if missing:
            raise ValueError(f"{gallery_file} has no trials for gallery seeds {missing}")
        gallery["trials"] = {s: gallery["trials"][str(s)] for s in args.seeds}
    else:
        gallery = build_gallery_trials(pool, args.seeds, args.search, split)
        _write_new(gallery_file, gallery)
        print(f"fixed gallery written to {gallery_file}")
    result = evaluate(query, pool, gallery_rows(pool, gallery))
    qm = query["metadata"]
    report = {"run_id": qm.get("run_id"), "split": split, "training_seed": qm.get("seed"),
              "epoch_completed": qm.get("next_epoch_zero_based"), "global_step": qm.get("global_step"),
              "checkpoint_sha256": qm.get("checkpoint_sha256"),
              "protocol": gallery["protocol"] | {"camera_exclusion": "cam3 query drops cam2 gallery", "cmc": "identity-level (SYSU)",
                                                  "distance": "-(query @ gallery.T)", "invalid_query_policy": "excluded and counted"},
              "units": "fraction 0-1", "gallery_file": str(gallery_file), "gallery_file_sha256": sha256_file(gallery_file),
              "gallery_fingerprint": gallery["pool_fingerprint"],
              "inputs": {"query": str(args.query), "query_sha256": sha256_file(args.query),
                         "gallery_pool": str(args.gallery_pool), "gallery_pool_sha256": sha256_file(args.gallery_pool)},
              "feature_l2_norm_range": {name: [float(n.min()), float(n.max())] for name, n in
                                        (("query", np.linalg.norm(query["features"], axis=1)),
                                         ("gallery_pool", np.linalg.norm(pool["features"], axis=1)))},
              **result}
    _write_new(Path(args.output_dir) / "metrics.json", report)
    m = result["mean_over_gallery"]
    print(f"{report['run_id']} | {len(result['trials'])} galleries | mAP {m['mAP']:.4f} | mINP {m['mINP']:.4f} | "
          f"R1 {m['rank1']:.4f} | R5 {m['rank5']:.4f} | R10 {m['rank10']:.4f} | R20 {m['rank20']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="评价一个训练 run 的导出特征")
    p.add_argument("--query", required=True, help="IR query npz")
    p.add_argument("--gallery-pool", required=True, help="RGB 候选池 npz")
    p.add_argument("--output-dir", required=True, help="写入 metrics.json（已存在则拒绝）")
    p.add_argument("--search", choices=sorted(RGB_CAMERAS), default="all")
    p.add_argument("--seeds", type=int, nargs="+", help="gallery 种子，默认取配置 evaluation.gallery_seeds")
    p.add_argument("--gallery-file", help="固定 gallery 记录；不存在则生成，存在则复用并校验")
    s = sub.add_parser("summarize", help="跨训练种子汇总多个 metrics.json")
    s.add_argument("--metrics", nargs="+", required=True)
    s.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        summary = summarize([json.loads(Path(m).read_text(encoding="utf-8")) for m in args.metrics])
        _write_new(args.output, summary)
        print(json.dumps({k: summary[k] for k in METRIC_KEYS}, indent=2))


if __name__ == "__main__":
    main()
