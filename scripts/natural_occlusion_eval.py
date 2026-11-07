"""在自然遮挡子集上比较多个模型（服务 L：M0 vs M2 等），复用 E 的评价协议与固定 gallery。

子集定义在测试集 IR query 上（验证集逐张检查后没有符合标准的样例，见 annotations/README.md）。
测试集规则：标注必须在看任何模型测试结果之前完成并提交到 Git；只在配置冻结后运行一次，做描述性报告，
不得据此修改方法或挑选模型。脚本会拒绝使用有未提交修改的标注文件，并在结果中记录标注文件的提交号。

不重新提取特征：直接从 B 导出的 query_val.npz 中按路径取出子集行，RGB gallery 与 E 的固定 gallery 文件完全相同。
报告以下分组的 10 组 gallery 平均指标：
    all                该划分的全部 IR query
    natural_occlusion  自然遮挡子集
    not_occluded       其余未标注遮挡的 query
    half_upper / half_lower / half_both            按被挡区域（对应 L 的 q=2 上下两区）
    occluder_object / occluder_person / occluder_frame_edge   按遮挡来源
    cam3_* / cam6_*    按摄像头拆分的全部 / 遮挡 / 未遮挡（遮挡集中在某摄像头时用同摄像头未遮挡 query 对照）

用法（项目根目录，每个模型一个 --run 名称 query gallery_pool）：
    python scripts/natural_occlusion_eval.py \
        --run M0 artifacts/results/<M0_final>/query_test.npz artifacts/results/<M0_final>/gallery_pool_test.npz \
        --run M2 artifacts/results/<M2_final>/query_test.npz artifacts/results/<M2_final>/gallery_pool_test.npz \
        --output artifacts/results/natural_occlusion/M0_vs_M2.json

注意：子集样本量小、且是按遮挡挑出的，只能做描述性对比；与人工遮挡结果分开报告，不推断总体遮挡分布。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D  # noqa: E402
from src.evaluate import PROJECT, check_inputs, evaluate, gallery_rows, load_features, sha256_file  # noqa: E402

SHOW = ("mAP", "rank1", "rank5", "mINP")


def groups_from_subset(records, all_paths, cameras=None):
    occluded = {r["relative_path"] for r in records}
    groups = {"all": list(all_paths), "natural_occlusion": [p for p in all_paths if p in occluded],
              "not_occluded": [p for p in all_paths if p not in occluded]}
    if cameras is not None:   # 按摄像头拆开：遮挡可能集中在某个摄像头，需同摄像头的未遮挡 query 作对照
        for cam in sorted(set(int(c) for c in cameras)):
            on_cam = [p for p, c in zip(all_paths, cameras) if int(c) == cam]
            groups[f"cam{cam}_all"] = on_cam
            groups[f"cam{cam}_natural_occlusion"] = [p for p in on_cam if p in occluded]
            groups[f"cam{cam}_not_occluded"] = [p for p in on_cam if p not in occluded]
    for key, values in (("half", D.HALVES), ("occluder", D.OCCLUDERS)):
        for value in values:
            chosen = {r["relative_path"] for r in records if r[key] == value}
            groups[f"{key}_{value}"] = [p for p in all_paths if p in chosen]
    return groups


def subset_git_status(path):
    """返回 (提交号, 是否有未提交修改)；不在 Git 仓库中时提交号为 None。"""
    path = Path(path).resolve()
    try:
        commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", path.name], cwd=path.parent,
                                         text=True, stderr=subprocess.DEVNULL).strip() or None
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", path.name], cwd=path.parent,
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return None, True


def evaluate_run(query_path, gallery_path, records, gallery, split):
    query, pool = load_features(query_path), load_features(gallery_path)
    check_inputs(query, pool)
    if query["metadata"].get("split", "val") != split:
        raise ValueError(f"{query_path} 的 split 为 {query['metadata'].get('split')}，与子集划分 {split} 不一致")
    trial_rows = gallery_rows(pool, gallery)
    groups = groups_from_subset(records, query["paths"].tolist(), query["cameras"].tolist())
    missing = {r["relative_path"] for r in records} - set(query["paths"].tolist())
    if missing:
        raise ValueError(f"{len(missing)} 条子集路径不在 {query_path} 中，例如 {sorted(missing)[:3]}")
    result = {}
    for name, paths in groups.items():
        if not paths:
            result[name] = {"num_queries": 0}
            continue
        part = D.subset_by_paths(query, paths)
        mean = evaluate(part, pool, trial_rows)["mean_over_gallery"]
        result[name] = {"num_queries": len(paths), "num_identities": int(len(set(part["identities"].tolist()))),
                        **{k: mean[k] for k in ("mAP", "mINP", "rank1", "rank5", "rank10", "rank20")}}
    meta = {"run_id": query["metadata"].get("run_id"), "checkpoint_sha256": query["metadata"].get("checkpoint_sha256"),
            "query": str(query_path), "query_sha256": sha256_file(query_path),
            "gallery_pool": str(gallery_path), "gallery_pool_sha256": sha256_file(gallery_path)}
    return meta, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs=3, action="append", metavar=("NAME", "QUERY_NPZ", "GALLERY_POOL_NPZ"),
                        required=True, help="可重复；第一个作为差值的参照（通常是 M0）")
    parser.add_argument("--subset", default=str(D.NATURAL_OCCLUSION_FILE))
    parser.add_argument("--split", default=D.NATURAL_OCCLUSION_SPLIT, help="子集所在划分，默认 test")
    parser.add_argument("--allow-uncommitted-subset", action="store_true",
                        help="仅用于调试：允许使用未提交到 Git 的标注文件（结果会标记 subset_committed=false）")
    parser.add_argument("--manifest", default=str(D.DEFAULT_MANIFEST_DIR / "manifest.csv"))
    parser.add_argument("--gallery-file", default=None,
                        help="E 的固定 gallery 记录，默认 artifacts/manifests/gallery_<split>_all_single.json；必须已存在")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    commit, dirty = subset_git_status(args.subset)
    if dirty and not args.allow_uncommitted_subset:
        raise SystemExit(f"{args.subset} 有未提交修改或不在 Git 中：请先提交标注，保证标注在看结果之前固定")
    gallery_file = Path(args.gallery_file or PROJECT / f"artifacts/manifests/gallery_{args.split}_all_single.json")
    if not gallery_file.exists():
        raise SystemExit(f"{gallery_file} 不存在：先用 src.evaluate run 生成固定 gallery，所有评价共用同一份")
    gallery = json.loads(gallery_file.read_text(encoding="utf-8"))
    rows = D.load_manifest(args.manifest) if Path(args.manifest).exists() else None
    records = D.load_natural_occlusion(args.subset, rows, args.split)

    report = {"split": args.split, "subset": str(args.subset), "subset_sha256": sha256_file(args.subset),
              "subset_git_commit": commit, "subset_committed": not dirty, "subset_size": len(records),
              "gallery_file": str(gallery_file), "gallery_file_sha256": sha256_file(gallery_file),
              "note": "descriptive comparison on a small, purposively selected subset; report separately from synthetic occlusion",
              "runs": {}}
    for name, query_path, gallery_path in args.run:
        meta, result = evaluate_run(query_path, gallery_path, records, gallery, args.split)
        report["runs"][name] = {"inputs": meta, "groups": result}

    names = [r[0] for r in args.run]
    ref = names[0]
    if len(names) > 1:
        report["difference_vs_" + ref] = {
            name: {g: {k: report["runs"][name]["groups"][g][k] - report["runs"][ref]["groups"][g][k] for k in SHOW}
                   for g, v in report["runs"][ref]["groups"].items() if v["num_queries"]}
            for name in names[1:]}

    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有结果 {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    header = f"{'group':22s} {'n':>5s} " + " ".join(f"{n + ' ' + k:>14s}" for n in names for k in ("mAP", "R1"))
    print(header)
    for g, v in report["runs"][ref]["groups"].items():
        if not v["num_queries"]:
            continue
        cells = " ".join(f"{report['runs'][n]['groups'][g][k] * 100:14.2f}" for n in names for k in ("mAP", "rank1"))
        print(f"{g:22s} {v['num_queries']:5d} {cells}")
    print(f"[occlusion] written to {out}")


if __name__ == "__main__":
    main()
