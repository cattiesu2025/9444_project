"""A：扫描 SYSU-MM01，生成 manifest、官方划分、标签映射与数据审计报告。

通常直接用 scripts/prepare_data.py（含自动下载与接口检查）；本脚本只做 manifest 与审计：
    python scripts/build_manifest.py --root data/SYSU-MM01 --source <下载来源> --downloaded-on 2026-10-24

输出（默认 artifacts/manifests/，不入 Git）：
    manifest.csv                      每张图一行，字段见 src/data.py MANIFEST_FIELDS
    splits/{train,val,test}_ids.txt   官方身份列表（每行一个）
    label_map_dev.csv                 开发阶段 train(296) 身份 → 连续标签
    label_map_final.csv               最终阶段 train+val(395) 身份 → 连续标签
    data_audit.json / data_audit.md   审计结果；status=FAIL 时不要开始正式长训练
    dataset_info.json                 来源、日期、路径、各文件 sha256
只需要 numpy 与 Pillow，不需要 torch。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="数据目录；默认自动查找（不会下载）")
    parser.add_argument("--out-dir", default=str(D.DEFAULT_MANIFEST_DIR))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--K", type=int, default=4, help="每身份每模态图片数，用于标记不足 K 张的身份")
    parser.add_argument("--source", default="", help="下载来源 URL")
    parser.add_argument("--downloaded-on", default="", help="下载日期 YYYY-MM-DD")
    args = parser.parse_args()

    root = D.ensure_dataset(args.root, download=False)
    audit = D.build_data_artifacts(root, args.out_dir, source=args.source, downloaded_on=args.downloaded_on,
                                   K=args.K, workers=args.workers)
    sys.exit(0 if audit["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
