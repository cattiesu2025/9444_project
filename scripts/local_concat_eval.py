"""E：探索性对照——测试时把局部特征拼到全局特征上，是否改善检索（含遮挡）。

不是主协议：主表只用 768 维全局特征。这里改变了检索特征的维度，结果单独报告。
特征方案事先固定，不按扰动结果挑选：
- global：BN 后 L2 全局特征（与导出一致）。
- global+parts：再拼上 q 个水平区域的 patch 均值（不经 BN，各自 L2），q+1 段等权。
  不依赖局部头，M0 和 M2 都能算，用来区分"测试时用局部信息"与"训练时的局部监督"。
- global+local：带局部头的权重专用，区域特征先过局部头训练好的 BN neck 再 L2，q+1 段等权。
- global+local_g50：同上，但全局固定占一半、q 个区域平分另一半，便于比较不同 q（等权时全局占比随 q 变小）。
拼接向量为 [sqrt(w)·g, sqrt((1-w)/q)·l_1, …]，余弦相似度 = w·全局相似度 + (1-w)/q·Σ区域相似度。
区域数 q 取自局部头（即权重里的配置）；没有局部头时用 --regions（默认 2）。
区域边界与 LocalHead 相同（第 j 段为 floor(j*H/q) 到 floor((j+1)*H/q) 行）。
gallery 用同一权重重新提取（需要局部特征），其全局部分与导出特征做一致性检查。

python scripts/local_concat_eval.py --checkpoint <last.pt> --gallery-pool <run>/gallery_pool_val.npz \
    --query <run>/query_val.npz --manifest <run>/training_manifest.csv --root data/archive \
    --output artifacts/results/<exp>_seed17/local_concat.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import SYSUImageList, load_manifest, select_rows  # noqa: E402
from src.evaluate import PROJECT, _write_new, evaluate, gallery_rows, load_features, sha256_file  # noqa: E402
from src.robustness import CONDITIONS, PerturbedImageList, compare_with_export, load_model  # noqa: E402

DEFAULT_REGIONS = 2


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def extract_parts(model, dataset, regions, batch_size=128, num_workers=0, device="cpu"):
    """一次前向同时取全局特征、区域 patch 均值，以及（若有局部头）BN 后区域特征。"""
    import torch
    from torch.utils.data import DataLoader
    from src.data import collate_eval
    head = getattr(model, "local_head", None)
    out = {k: [] for k in ("global", "parts", "local", "identities", "cameras", "modalities", "paths")}
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_eval)
    with torch.inference_mode():
        for batch in loader:
            result = model(batch["images"].to(device))
            h, w = result["patch_grid"]
            spatial = result["patch_features"].reshape(-1, h, w, result["patch_features"].shape[-1])
            parts = torch.stack([spatial[:, j * h // regions:(j + 1) * h // regions].mean(dim=(1, 2))
                                 for j in range(regions)], dim=1)
            out["global"].append(result["retrieval_feature"].float().cpu().numpy())
            out["parts"].append(parts.float().cpu().numpy())
            if head is not None:
                if head.regions != regions:
                    raise ValueError("local head region count differs from the requested regions")
                out["local"].append(torch.stack([neck(parts[:, j]) for j, neck in enumerate(head.necks)], 1)
                                    .float().cpu().numpy())
            for key in ("identities", "cameras", "modalities"):
                out[key].append(batch[key].numpy())
            out["paths"].extend(batch["paths"])
    arrays = {k: (np.asarray(v) if k == "paths" else np.concatenate(v)) for k, v in out.items() if len(v)}
    for k in ("global", "parts", "local"):
        if k in arrays and not np.isfinite(arrays[k]).all():
            raise FloatingPointError(f"nonfinite {k} features")
    return arrays


def combine(g, parts, global_weight):
    q = parts.shape[1]
    return np.concatenate([np.sqrt(global_weight) * g]
                          + [np.sqrt((1 - global_weight) / q) * l2(parts[:, j]) for j in range(q)], 1)


def variants(arrays):
    g, q = arrays["global"], arrays["parts"].shape[1]
    out = {"global": g, "global+parts": combine(g, arrays["parts"], 1 / (q + 1))}
    if "local" in arrays:
        out["global+local"] = combine(g, arrays["local"], 1 / (q + 1))
        out["global+local_g50"] = combine(g, arrays["local"], .5)
    return out


def as_features(arrays, features, metadata):
    return {"features": features.astype(np.float32), "identities": arrays["identities"], "cameras": arrays["cameras"],
            "modalities": arrays["modalities"], "paths": arrays["paths"], "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gallery-pool", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gallery-file")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--regions", type=int, default=DEFAULT_REGIONS,
                        help="only for checkpoints without a local head; otherwise taken from the head")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    pool = load_features(args.gallery_pool)
    checkpoint_sha = sha256_file(args.checkpoint)
    if pool["metadata"]["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("gallery pool was exported from a different checkpoint")
    model, checkpoint = load_model(args.checkpoint, args.device)
    if sha256_file(args.manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("manifest differs from the one used for training")
    rows = load_manifest(args.manifest)
    split = pool["metadata"].get("split", "val")
    size = tuple(checkpoint["config"]["model"]["image_size"])
    gallery_file = Path(args.gallery_file or PROJECT / f"artifacts/manifests/gallery_{split}_all_single.json")
    gallery = json.loads(gallery_file.read_text(encoding="utf-8"))
    meta = {"checkpoint_sha256": checkpoint_sha, "split": split}
    head = getattr(model, "local_head", None)
    regions = head.regions if head is not None else args.regions

    # gallery：重新提取后按导出顺序排列，保证固定 gallery 的行号一致
    g = extract_parts(model, SYSUImageList(rows, args.root, select_rows(rows, [split], modality="rgb"), image_size=size),
                      regions, args.batch_size, args.num_workers, args.device)
    order = {p: i for i, p in enumerate(g["paths"].tolist())}
    if set(order) != set(pool["paths"].tolist()):
        raise ValueError("re-extracted gallery differs from the exported pool")
    idx = [order[p] for p in pool["paths"].tolist()]
    g = {k: v[idx] for k, v in g.items()}
    consistency = {"gallery": compare_with_export({"features": g["global"], "paths": g["paths"]}, pool)}
    trial_rows = gallery_rows(pool, gallery)
    gallery_variants = variants(g)

    report = {"run_id": checkpoint.get("run_id"), "checkpoint_sha256": checkpoint_sha, "split": split,
              "gallery_file_sha256": sha256_file(gallery_file), "regions": regions,
              "note": "exploratory; changes retrieval dimension; not comparable to the main table",
              "variants": {name: {"dim": int(f.shape[1])} for name, f in gallery_variants.items()},
              "export_consistency": consistency, "results": {}}
    query_indices = select_rows(rows, [split], modality="ir")
    for name in args.conditions:
        q = extract_parts(model, PerturbedImageList(rows, args.root, query_indices, image_size=size,
                                                    perturbation=CONDITIONS[name]),
                          regions, args.batch_size, args.num_workers, args.device)
        if name == "clean":
            consistency["query"] = compare_with_export({"features": q["global"], "paths": q["paths"]},
                                                       load_features(args.query))
        report["results"][name] = {}
        for variant, features in variants(q).items():
            result = evaluate(as_features(q, features, meta), as_features(g, gallery_variants[variant], meta),
                              trial_rows)
            report["results"][name][variant] = {
                "mean_over_gallery": result["mean_over_gallery"],
                "trials": [{k: t[k] for k in ("gallery_seed", "mAP", "rank1")} for t in result["trials"]]}
        print(name.ljust(22), "  ".join(f"{v}: mAP {100 * r['mean_over_gallery']['mAP']:.1f} "
                                        f"R1 {100 * r['mean_over_gallery']['rank1']:.1f}"
                                        for v, r in report["results"][name].items()), flush=True)
    print("consistency", consistency)
    _write_new(output, report)


if __name__ == "__main__":
    main()
