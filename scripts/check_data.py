"""A：从真实数据取出训练 batch，检查接口契约、可复现性与采样统计，并保存证据。

先运行 scripts/build_manifest.py，再在项目根目录运行：
    python scripts/check_data.py --root data/SYSU-MM01 --experiment M0

输出（默认 artifacts/results/data_check/，不入 Git）：
    batch_example.json   第一步 batch 的形状、dtype、逐行身份/标签/摄像头/模态/灰度/路径
    batch_grid.png       同一 batch 的图像：每行一个身份，左 K 张为 RGB（第一阶段为灰度），右 K 张为 IR
    sampler_stats.json   配置 P/K 与 Q 对照 P/K 在一个 epoch 的身份/图片覆盖、放回抽取次数（不训练）
    check_report.md      每项检查的结果
退出码 0 表示全部检查通过。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src.config import EXPERIMENTS, load_config  # noqa: E402


def denormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(D.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(D.IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()


def save_grid(batch: dict, P: int, K: int, path: Path) -> None:
    h, w = batch["images"].shape[2:]
    pad, label_h = 4, 14
    grid = Image.new("RGB", (2 * K * (w + pad) + pad * 3, P * (h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(grid)
    n = P * K
    for i in range(P):
        for j in range(2 * K):
            row = i * K + j if j < K else n + i * K + (j - K)
            x = pad + j * (w + pad) + (pad * 2 if j >= K else 0)
            y = pad + i * (h + label_h + pad)
            grid.paste(Image.fromarray(denormalize(batch["images"][row])), (x, y + label_h))
            tag = f"id{batch['identities'][row].item()} c{batch['cameras'][row].item()}"
            tag += " g" if batch["gray"][row] else ""
            draw.text((x + 1, y), tag, fill="black")
    grid.save(path)


def batch_record(batch: dict, P: int, K: int) -> dict:
    n = P * K
    return {
        "layout": f"rows 0..{n - 1}: RGB-source (P={P} identities x K={K}); rows {n}..{2 * n - 1}: IR, same identity order",
        "images": {"shape": list(batch["images"].shape), "dtype": str(batch["images"].dtype),
                   "min": round(batch["images"].min().item(), 4), "max": round(batch["images"].max().item(), 4),
                   "mean": round(batch["images"].mean().item(), 4)},
        "fields": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in batch.items()
                   if isinstance(v, torch.Tensor) and k != "images"},
        "gray_rows": int(batch["gray"].sum()),
        "rows": [
            {"row": i, "label": batch["labels"][i].item(), "identity": batch["identities"][i].item(),
             "camera": batch["cameras"][i].item(),
             "modality": "rgb" if batch["modalities"][i].item() == 0 else "ir",
             "gray": bool(batch["gray"][i]), "path": batch["paths"][i]}
            for i in range(len(batch["paths"]))
        ],
    }


def run_checks(root=None, manifest=None, experiment: str = "M0", out_dir=None,
               num_workers: int = 2, timing_batches: int = 10) -> bool:
    """在真实数据上检查训练接口并写出证据，全部通过时返回 True。"""
    manifest = Path(manifest) if manifest else D.DEFAULT_MANIFEST_DIR / "manifest.csv"
    out_dir = out_dir or ROOT / "artifacts" / "results" / "data_check"
    cfg = load_config(experiment)
    P = cfg["training"]["identities_per_batch"]
    K = cfg["training"]["images_per_identity_per_modality"]
    seed = cfg["training"]["seeds"][0]
    size = tuple(cfg["model"]["image_size"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = D.load_manifest(manifest)
    root = D.ensure_dataset(root, download=False)
    checks: dict[str, tuple[bool, str]] = {}

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = (bool(ok), detail)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")

    print(f"[check] experiment={experiment} P={P} K={K} seed={seed} image_size={size}")
    common = dict(manifest=rows, root=root, identities_per_batch=P, images_per_identity_per_modality=K,
                  seed=seed, image_size=size, pin_memory=False)

    # 1. 身份隔离与标签映射
    splits = D.load_official_splits(root)
    loader, sampler, _, label_map = D.build_train_loader(num_workers=num_workers, gray_probability=1.0, **common)
    train_ids, other = set(splits["train"]), set(splits["val"]) | set(splits["test"])
    check("label_map_only_train_identities", set(label_map) <= train_ids and not set(label_map) & other,
          f"{len(label_map)} train identities")
    check("labels_contiguous_0_to_C-1", sorted(label_map.values()) == list(range(len(label_map))))
    manifest_split_ok = all(r["split"] == ("train" if r["identity"] in train_ids else
                                           "val" if r["identity"] in set(splits["val"]) else
                                           "test" if r["identity"] in set(splits["test"]) else "unassigned")
                            for r in rows)
    check("manifest_split_matches_official_files", manifest_split_ok)

    # 2. 第一阶段（全部 RGB 转灰度）的真实 batch
    sampler.set_epoch(0)
    t0 = time.time()
    it = iter(loader)
    batch = next(it)
    first_batch_s = time.time() - t0
    try:
        D.check_pmt_layout(batch, P, K)
        check("pmt_batch_layout", True, f"2PK={2 * P * K}")
    except AssertionError as e:
        check("pmt_batch_layout", False, str(e))
    check("image_shape", list(batch["images"].shape) == [2 * P * K, 3, *size], str(list(batch["images"].shape)))
    check("images_finite", bool(torch.isfinite(batch["images"]).all()))
    check("labels_match_identities",
          all(label_map[i] == l for i, l in zip(batch["identities"].tolist(), batch["labels"].tolist())))
    check("batch_identities_in_train_split", set(batch["identities"].tolist()) <= train_ids)
    check("stage1_all_rgb_gray", int(batch["gray"][:P * K].sum()) == P * K and not batch["gray"][P * K:].any())
    rec = batch_record(batch, P, K)
    rec.update(experiment=experiment, epoch=0, batch=0, gray_probability=1.0, seed=seed)
    (out / "batch_example.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    save_grid(batch, P, K, out / "batch_grid.png")

    # 3. 第二阶段（M0：不转灰度）同一步抽到同样的图片，只是灰度标记不同
    sampler_stage2 = D.PMTBatchSampler(rows, label_map, P, K, seed=seed, gray_probability=0.0)
    sampler.set_epoch(0)
    tickets_stage1 = next(iter(sampler))
    tickets_stage2 = next(iter(sampler_stage2))
    check("gray_probability_does_not_change_sampling",
          [t[0] for t in tickets_stage1] == [t[0] for t in tickets_stage2]
          and [t[2] for t in tickets_stage1] == [t[2] for t in tickets_stage2])
    check("stage2_no_gray", not any(t[1] for t in tickets_stage2))

    # 4. 可复现：同种子重建得到完全相同的张量；num_workers=0 与 >0 一致
    other_workers = 2 if num_workers == 0 else 0
    loader0, sampler0, _, _ = D.build_train_loader(num_workers=other_workers, gray_probability=1.0, **common)
    sampler0.set_epoch(0)
    batch0 = next(iter(loader0))
    check("deterministic_across_num_workers", torch.equal(batch0["images"], batch["images"])
          and batch0["paths"] == batch["paths"], f"workers {num_workers} vs {other_workers}")
    check("epochs_differ", sampler0.epoch_plan(1)[0]["rows"] != sampler0.epoch_plan(0)[0]["rows"])

    # 5. 断点恢复：从第 3 步继续与完整迭代的第 3 步一致
    sampler0.set_epoch(0, start_batch=3)
    resumed = next(iter(sampler0))
    sampler0.set_epoch(0)
    full = list(sampler0)
    check("resume_mid_epoch", resumed == full[3] and len(full) == sampler0.batches_per_epoch)

    # 6. 采样统计：配置的 P/K 与 Q 对照（P=16, K=2）；仅统计，不训练
    stats = {"configured": sampler.epoch_stats(0)}
    q_cfg = load_config("Q")["training"]
    qP, qK = q_cfg["identities_per_batch"], q_cfg["images_per_identity_per_modality"]
    q_sampler = D.PMTBatchSampler(rows, label_map, qP, qK, seed=seed)
    stats["Q_alternative"] = q_sampler.epoch_stats(0)
    check("Q_same_batch_size_and_steps", q_sampler.batch_size == sampler.batch_size
          and q_sampler.batches_per_epoch == sampler.batches_per_epoch,
          f"P{qP}K{qK}: {q_sampler.batch_size} imgs x {q_sampler.batches_per_epoch} steps")
    (out / "sampler_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    # 7. 读取速度（供 B 判断数据加载是否成为瓶颈）
    n_timing = min(timing_batches, sampler.batches_per_epoch - 1)
    t0 = time.time()
    for _ in range(n_timing):
        next(it)
    dt_s = time.time() - t0
    speed = n_timing * sampler.batch_size / dt_s if dt_s > 0 else float("nan")
    print(f"[check] loading speed: {speed:.1f} images/s with {num_workers} workers")

    passed = all(ok for ok, _ in checks.values())
    s = stats["configured"]
    lines = [
        "# 数据接口检查报告", "",
        f"- 时间：{dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- manifest：`{manifest}`（sha256 `{D.file_sha256(manifest)}`）",
        f"- 数据根目录：`{root}`",
        f"- 配置：{experiment}，P={P}，K={K}，batch=2PK={2 * P * K}，seed={seed}，输入 {size[0]}×{size[1]}",
        f"- 每 epoch 步数：{sampler.batches_per_epoch}（= max(#RGB, #IR) // (P·K)，与作者实现一致）",
        f"- 读取速度：{speed:.1f} 张/秒（{num_workers} 个 worker，首个 batch {first_batch_s:.1f}s）",
        "", f"## 结论：{'全部通过' if passed else '存在未通过项'}", "",
        "| 检查 | 结果 | 说明 |", "| --- | --- | --- |",
    ]
    lines += [f"| {k} | {'通过' if ok else '**未通过**'} | {d} |" for k, (ok, d) in checks.items()]
    lines += ["", "## 一个 epoch 的采样统计（只统计抽样计划，不训练）", "",
              "| 配置 | 步数 | 身份覆盖 | 每身份被抽次数 min/max | RGB 图覆盖 | IR 图覆盖 | 放回抽取次数 |",
              "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for name, st in stats.items():
        lines.append(f"| {name} P={st['P']} K={st['K']} | {st['batches_per_epoch']} | {st['identity_coverage']:.1%} | "
                     f"{st['identity_visits_min_max']} | {st['rgb_image_coverage']:.1%} | "
                     f"{st['ir_image_coverage']:.1%} | {st['with_replacement_draws']} |")
    lines += ["", f"某模态不足 K={K} 张的身份：RGB {s['identities_below_K']['rgb'] or '无'}；"
              f"IR {s['identities_below_K']['ir'] or '无'}。", "",
              "`batch_grid.png`：每行一个身份，左侧 K 张为可见光来源（第一阶段已转灰度，标记 g），右侧 K 张为同一身份的 IR。"
              "人工确认同一行是同一个人。", ""]
    (out / "check_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[check] {'ALL PASSED' if passed else 'SOME CHECKS FAILED'}; outputs in {out}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="数据目录；默认自动查找（不会下载）")
    parser.add_argument("--manifest", default=str(D.DEFAULT_MANIFEST_DIR / "manifest.csv"))
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="M0")
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "results" / "data_check"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timing-batches", type=int, default=10)
    args = parser.parse_args()
    passed = run_checks(args.root, args.manifest, args.experiment, args.out_dir,
                        args.num_workers, args.timing_batches)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
