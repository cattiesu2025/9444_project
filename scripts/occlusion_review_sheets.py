"""A：生成自然遮挡标注/复核用的拼图（只在本地查看，含数据集图像，不入 Git）。

两种模式（在项目根目录运行）：
    # 标注：测试集（默认，--split 可改）每个身份的全部 IR query 拼成一张图，图下标出图片序号，按 annotations/README.md 的标准逐张判断
    python scripts/occlusion_review_sheets.py --mode all

    # 复核：把 annotations/natural_occlusion_test_ir.csv 中已标注的样例按遮挡来源分组拼图，图下标出标签
    python scripts/occlusion_review_sheets.py --mode subset

输出默认在 artifacts/results/natural_occlusion/sheets/。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D  # noqa: E402

CELL_W, CELL_H, LABEL_H, PAD = 96, 192, 30, 6


def draw_sheet(items, root, title, path, columns=10):
    """items: [(relative_path, caption)]，按行排列，每格上方是图、下方是说明。"""
    rows = (len(items) + columns - 1) // columns
    width = columns * (CELL_W + PAD) + PAD
    height = 28 + rows * (CELL_H + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 8), title, fill="black")
    for k, (rel, caption) in enumerate(items):
        r, c = divmod(k, columns)
        x, y = PAD + c * (CELL_W + PAD), 28 + r * (CELL_H + LABEL_H + PAD)
        sheet.paste(D.load_image(root / rel, CELL_H, CELL_W), (x, y))
        for i, line in enumerate(caption.split("\n")[:2]):
            draw.text((x + 2, y + CELL_H + 2 + 13 * i), line, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("all", "subset"), default="subset")
    parser.add_argument("--root", default=None, help="数据目录；默认自动查找（不会下载）")
    parser.add_argument("--manifest", default=str(D.DEFAULT_MANIFEST_DIR / "manifest.csv"))
    parser.add_argument("--subset", default=str(D.NATURAL_OCCLUSION_FILE))
    parser.add_argument("--split", default=D.NATURAL_OCCLUSION_SPLIT)
    parser.add_argument("--out-dir", default=str(D.PROJECT_ROOT / "artifacts" / "results" / "natural_occlusion" / "sheets"))
    args = parser.parse_args()

    root = D.ensure_dataset(args.root, download=False)
    rows = D.load_manifest(args.manifest)
    out = Path(args.out_dir)
    if args.mode == "all":
        by_id = defaultdict(list)
        for i in D.select_rows(rows, [args.split], modality="ir"):
            by_id[rows[i]["identity"]].append(rows[i]["relative_path"])
        for pid, paths in sorted(by_id.items()):
            items = [(p, Path(p).stem) for p in sorted(paths)]
            draw_sheet(items, root, f"{args.split} IR query  identity {pid:04d}  ({len(items)} images)",
                       out / f"all_{args.split}_{pid:04d}.jpg")
        print(f"[occlusion] {len(by_id)} identity sheets written to {out}")
    else:
        groups = defaultdict(list)
        for rec in D.load_natural_occlusion(args.subset, rows, args.split):
            caption = f"c{rec['camera']} {rec['identity']:04d}/{Path(rec['relative_path']).stem}\n{rec['region']}-{rec['half']}-{rec['severity']}"
            groups[rec["occluder"]].append((rec["relative_path"], caption))
        for occluder, items in sorted(groups.items()):
            for start in range(0, len(items), 40):
                part = items[start:start + 40]
                draw_sheet(part, root, f"natural occlusion: {occluder}  ({start + 1}-{start + len(part)} of {len(items)})",
                           out / f"subset_{occluder}_{start // 40 + 1:02d}.jpg")
        print(f"[occlusion] review sheets for {sum(len(v) for v in groups.values())} images written to {out}")


if __name__ == "__main__":
    main()
