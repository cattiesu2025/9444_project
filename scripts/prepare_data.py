"""A：一条命令准备数据——有数据就直接用，没有就从 Kaggle 下载；生成 manifest 与审计；检查训练接口。

在项目根目录运行：
    python scripts/prepare_data.py

查找数据的顺序：--root → 环境变量 SYSU_MM01_ROOT → data/SYSU-MM01；
都没有时用 kagglehub 把课程镜像直接下载到 data/SYSU-MM01（需要 pip install kagglehub；若提示需要登录，按屏幕说明配置 Kaggle API
或手动下载 zip）。已有 manifest 时不会重复扫描，--rebuild 强制重新生成。

结果：
    artifacts/manifests/          manifest、划分、标签映射、data_audit.md
    artifacts/results/data_check/ 真实 batch、batch_grid.png、check_report.md
全部通过时退出码为 0。之后成员 B 用 `from src.data import get_train_loader` 即可取数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from src import data as D  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="已有数据目录；不填则自动查找或下载")
    parser.add_argument("--no-download", action="store_true", help="找不到数据时报错而不是下载")
    parser.add_argument("--rebuild", action="store_true", help="重新扫描数据并生成 manifest 与审计")
    parser.add_argument("--source", default=None, help="手动下载时填写来源 URL，写入审计报告")
    parser.add_argument("--downloaded-on", default=None, help="手动下载时填写日期 YYYY-MM-DD")
    parser.add_argument("--experiment", default="M0", help="用哪个实验配置的 P/K 做接口检查")
    parser.add_argument("--num-workers", type=int, default=0, help="检查时的读取进程数；Windows 建议 0")
    parser.add_argument("--skip-checks", action="store_true", help="只准备数据，不做接口检查（不需要 torch）")
    args = parser.parse_args()

    print("== 1/2 准备数据与审计 ==")
    try:
        prep = D.prepare_data(args.root, download=not args.no_download, rebuild=args.rebuild,
                              source=args.source, downloaded_on=args.downloaded_on)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n[data] {exc}")
        sys.exit(2)
    print(f"[data] dataset: {prep['root']}")
    print(f"[data] audit: {prep['audit_status']} -> {prep['manifest_dir'] / 'data_audit.md'}")
    if prep["audit_status"] != "PASS":
        print("[data] 审计未通过，请先查看报告。")
        sys.exit(1)
    if args.skip_checks:
        sys.exit(0)

    print("\n== 2/2 检查训练接口（真实 batch） ==")
    from check_data import run_checks
    passed = run_checks(prep["root"], prep["manifest"], args.experiment, num_workers=args.num_workers)

    print("\n== 结果 ==")
    print("全部通过。" if passed else "存在未通过项，见 artifacts/results/data_check/check_report.md")
    if passed:
        print("成员 B 取训练数据：\n"
              "    from src.data import get_train_loader\n"
              "    loader, sampler, dataset, label_map = get_train_loader(\"M0\")")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
