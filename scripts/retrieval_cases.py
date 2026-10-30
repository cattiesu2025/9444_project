"""E：基于真实排序生成 Top-k 检索案例图（IR query → RGB gallery，绿框正确 / 红框错误）。

样本按固定规则和随机种子抽取，并把选择方式写入 cases.json，避免只挑好看的例子：
  hit      Rank-1 正确
  near     Rank-1 错误，但正确身份在第 2–3 位
  miss     Top-k 内没有正确身份
也可用 --query-paths 指定具体 query（应在报告中注明为目的性选择）。

python scripts/retrieval_cases.py --query Q.npz --gallery-pool G.npz --data-root data/SYSU-MM01 \
    --select near --count 4 --output-dir artifacts/results/<run>/retrieval_cases
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import PROJECT, gallery_rows, load_features

TILE_W, TILE_H, BORDER, GAP, HEADER = 128, 256, 6, 14, 40
GREEN, RED, IR_COLOUR, INK = (34, 139, 76), (200, 40, 40), (181, 80, 106), (27, 42, 58)


def font(size, bold=False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def symbol_font(size):
    """✓/✗ 需要符号字体；常规 UI 字体缺这两个字形。"""
    for name in ("seguisym.ttf", "DejaVuSans.ttf", "Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return font(size, True)


def select(order, q_ids, g_ids, mode, k):
    top = g_ids[order[:, :k]] == q_ids[:, None]
    if mode == "hit":
        return np.flatnonzero(top[:, 0])
    if mode == "near":
        return np.flatnonzero(~top[:, 0] & top[:, 1:3].any(1))
    if mode == "miss":
        return np.flatnonzero(~top.any(1))
    raise ValueError(mode)


def tile(path, colour):
    image = Image.open(path).convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS)
    framed = Image.new("RGB", (TILE_W + 2 * BORDER, TILE_H + 2 * BORDER), colour)
    framed.paste(image, (BORDER, BORDER))
    return framed


def render(root, query_path, ranked, output):
    """ranked: [(path, correct, similarity)]，按名次排列。"""
    w = TILE_W + 2 * BORDER
    width = w * (len(ranked) + 1) + GAP * len(ranked) + 3 * GAP
    height = HEADER + TILE_H + 2 * BORDER + 34
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(tile(root / query_path, IR_COLOUR), (0, HEADER))
    draw.text((w / 2, HEADER / 2), "IR query", fill=IR_COLOUR, font=font(22, True), anchor="mm")
    x0 = w + 3 * GAP
    draw.text((x0 - 1.5 * GAP, HEADER + TILE_H / 2 + BORDER), "→", fill=INK, font=font(30, True), anchor="mm")
    for k, (path, correct, sim) in enumerate(ranked):
        x = x0 + k * (w + GAP)
        canvas.paste(tile(root / path, GREEN if correct else RED), (x, HEADER))
        colour = GREEN if correct else RED
        draw.text((x + w / 2 - 4, HEADER / 2), f"#{k + 1}", fill=colour, font=font(22, True), anchor="rm")
        draw.text((x + w / 2 + 4, HEADER / 2), "✓" if correct else "✗", fill=colour, font=symbol_font(22), anchor="lm")
        draw.text((x + w / 2, HEADER + TILE_H + 2 * BORDER + 17), f"sim {sim:.2f}", fill=INK, font=font(16), anchor="mm")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", required=True)
    parser.add_argument("--gallery-pool", required=True)
    parser.add_argument("--data-root", required=True, help="含 cam1–cam6 的 SYSU-MM01 目录")
    parser.add_argument("--gallery-file", default=str(PROJECT / "artifacts/manifests/gallery_val_all_single.json"))
    parser.add_argument("--gallery-seed", type=int, default=0)
    parser.add_argument("--select", choices=("hit", "near", "miss"), default="near")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--query-paths", nargs="+", help="指定 query 路径（目的性选择）")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    query, pool = load_features(args.query), load_features(args.gallery_pool)
    gallery = json.loads(Path(args.gallery_file).read_text(encoding="utf-8"))
    rows = gallery_rows(pool, gallery)[args.gallery_seed]
    sim = query["features"].astype(np.float64) @ pool["features"][rows].astype(np.float64).T
    if (query["cameras"] == 3).any():
        sim[np.ix_(query["cameras"] == 3, pool["cameras"][rows] == 2)] = -np.inf   # cam3→cam2 排除
    order = np.argsort(-sim, axis=1, kind="stable")
    g_ids, g_paths = pool["identities"][rows], pool["paths"][rows]

    if args.query_paths:
        index = {p: i for i, p in enumerate(query["paths"].tolist())}
        chosen, selection = [index[p] for p in args.query_paths], "purposive (--query-paths)"
    else:
        eligible = select(order, query["identities"], g_ids, args.select, args.top)
        rng = np.random.default_rng(args.sample_seed)
        chosen = sorted(rng.choice(eligible, size=min(args.count, len(eligible)), replace=False).tolist())
        selection = f"random among '{args.select}' queries (n={len(eligible)} of {len(query['paths'])}), sample seed {args.sample_seed}"

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)
    cases = []
    for n, q in enumerate(chosen):
        top = order[q, :args.top]
        ranked = [(g_paths[j], bool(g_ids[j] == query["identities"][q]), float(sim[q, j])) for j in top]
        name = f"case{n + 1}_{Path(query['paths'][q]).parent.name}_{Path(query['paths'][q]).stem}.png"
        render(root, query["paths"][q], ranked, out / name)
        cases.append({"figure": name, "query": query["paths"][q], "identity": int(query["identities"][q]),
                      "top": [{"rank": k + 1, "path": p, "correct": c, "similarity": s} for k, (p, c, s) in enumerate(ranked)]})
    record = {"run_id": query["metadata"].get("run_id"), "gallery_seed": args.gallery_seed,
              "gallery_fingerprint": gallery["pool_fingerprint"], "selection": selection, "cases": cases}
    (out / "cases.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(json.dumps({"selection": selection, "figures": [c["figure"] for c in cases]}, indent=2))


if __name__ == "__main__":
    main()
