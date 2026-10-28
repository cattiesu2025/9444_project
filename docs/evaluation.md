# 检索评价（E）

实现：`src/evaluate.py`（仅依赖 numpy，不需要 torch/GPU）。输入是 B 用 `src/export_features.py` 导出的 IR query 与 RGB 候选池 npz，不接收训练分类标签。

## 协议

| 项目 | 设置 |
| --- | --- |
| 方向 | IR query（cam3/6）→ RGB gallery |
| 搜索 | All-search：cam1/2/4/5（主）；Indoor-search：cam1/2（辅，`--search indoor`） |
| Single-shot | 每个（身份, RGB 摄像头）抽一张；摄像头无图则跳过 |
| 固定 gallery | 种子 0–9（`configs/base.json` 的 `evaluation.gallery_seeds`）；抽样步骤与 PMT `process_gallery_sysu` 相同：`random.Random(seed)`，身份升序 × 摄像头顺序，候选按路径排序后 `choice` |
| 距离 | `-(query @ gallery.T)`，特征为 BN 后 L2 归一化；排序用稳定排序，平局保持 gallery 顺序 |
| 摄像头排除 | cam3 query 丢弃全部 cam2 gallery（同一位置），与身份无关 |
| CMC | SYSU 身份级：排序中同一身份只保留首次出现；Rank-k = 首个正确身份位于前 k 个不同身份内 |
| mAP / mINP | 在丢弃后的实例级排序上计算 |
| 无正样本的 query | 不计入任何指标；在 `num_invalid_queries` / `invalid_query_rows` 中记录 |
| 汇总 | 每个训练种子先平均 10 组 gallery，再跨训练种子算 mean 与样本 SD（ddof=1）；gallery SD 另列，不当作训练重复 |

指标单位 0–1，展示时再转百分比。

## 运行

```bash
python -m src.evaluate run \
  --query artifacts/results/dev_M0_seed17_katana_20261025_retry1/query_val.npz \
  --gallery-pool artifacts/results/dev_M0_seed17_katana_20261025_retry1/gallery_pool_val.npz \
  --output-dir artifacts/results/dev_M0_seed17_katana_20261025_retry1/eval_all
```

- 首次运行生成固定 gallery 记录 `artifacts/manifests/gallery_val_all_single.json`（以图片路径保存，并含候选池指纹）；之后所有模型复用该文件，候选池内容不同会被拒绝，保证各方法共用同一 gallery。
- 写出 `metrics.json`：run_id、split、训练种子、已完成 epoch、checkpoint/输入/gallery 文件 SHA256、协议、每组 gallery 的 CMC/mAP/mINP/Rank-1/5/10/20 与有效 query 数、10 组平均及 gallery SD、特征范数范围。已存在则拒绝覆盖。
- 输入检查：query 必须全为 IR 且摄像头为 3/6，候选池全为 RGB；维度一致、无非有限值、路径不重复；两份导出的 checkpoint/split/run_id 必须一致。

跨训练种子汇总：

```bash
python -m src.evaluate summarize --metrics run17/eval_all/metrics.json run42/eval_all/metrics.json run2026/eval_all/metrics.json --output artifacts/results/summary_M0_all.json
```

协议、split 或 gallery 指纹不同的 run 拒绝汇总；重复训练种子拒绝汇总。

## 正确性检查

- `python -m pytest tests/test_evaluate.py -q`：手算 AP/mINP/CMC、身份级 CMC 与实例级的区别、cam3→cam2 排除（含错误身份）、无正样本 query 的排除与计数、多 query 平均、平局确定性、gallery 抽样与 PMT 步骤一致且与行顺序无关、候选池变更拒绝、输入错位拒绝、样本 SD、命令行生成并复用固定 gallery、拒绝覆盖。
- `python scripts/check_eval_reference.py /path/to/PMT`（固定 commit `0b31da9`）：200 组随机 SYSU 结构用例与上游 `eval_sysu` 比较 CMC/mAP/mINP；在临时 SYSU 目录上与 `process_gallery_sysu` 比较 All/Indoor × 10 个种子的 gallery 是否逐张相同。报告写入 `artifacts/results/eval_checks/reference.json`。

与上游的已知差异：上游用 `np.argsort` 默认排序，平局顺序未定义，本实现用稳定排序；上游在不同身份数少于 max_rank 时 CMC 长度不齐会报错，本实现补齐到 max_rank。真实浮点特征几乎不出现平局，两者在正常输入下数值一致。

## 状态

评价器与自检已完成；尚未在真实导出特征上运行，目前没有 baseline 检索指标。
