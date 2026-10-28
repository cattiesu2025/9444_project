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

与上游的已知差异：上游用 `np.argsort` 默认排序，平局顺序未定义，本实现用稳定排序；上游在不同身份数少于 max_rank 时 CMC 长度不齐会报错，本实现补齐到 max_rank；上游用 float32 计算距离，本实现用 float64。

**对照结果（2026-10-28，PASS）：**

- 合成用例：200 组 CMC 最大差 3.0e-8、mAP 1.1e-16、mINP 0；All/Indoor × 10 种子共 20 组 gallery 与 `process_gallery_sysu` 逐张相同。报告 `artifacts/results/eval_checks/reference.json`。
- 真实特征（M0/M1/F seed17 × 10 组 gallery）：直接调用上游 `eval_sysu`。M0 全部一致（最大差 2.7e-7）。M1/F 的 seed 8 gallery 中有 4 个 query 存在距离差约 1e-7 的近似平局，float32/float64 舍入使其顺序互换，导致 1 个 query 在某一 CMC 名次上跨界：该点差 1/1980 = 0.0005，mAP 差 5e-7。若改用与上游相同的 float32 距离，结果与上游一致（CMC 差 3e-8，mAP 0）。该差异来自浮点舍入而非协议，对报告的 10 组平均指标影响 ≤ 0.00005。

## 内部验证集的摄像头覆盖（重要限制）

内部 val（99 身份）的导出中 IR query 只有 cam3（1,980 张），RGB 候选只有 cam4（990）/cam5（984）；没有 cam1/2/6。因此：

- val 上的 All-search 实际是 cam3 → cam4/5，每组 gallery 195 张；cam3→cam2 排除规则在 val 上不起作用（仍在测试中覆盖）。
- val 上无法做 Indoor-search（cam1/2 无图），评价器会直接报错。
- 这与 test 集的摄像头组合不同，val 指标只用于开发比较，不能与论文 SYSU 测试结果直接对比。

## 首批结果（内部 val，All-search，单训练种子 17，10 组固定 gallery 平均）

| 实验 | run_id | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | dev_M0_seed17_katana_20261025_retry1 | 0.3438 | 0.2424 | 0.3207 | 0.6027 | 0.7264 | 0.8388 |
| M1 | dev_M1_seed17_katana_20261027_l40s | 0.3409 | 0.2401 | 0.3135 | 0.6056 | 0.7330 | 0.8401 |
| F | dev_F_seed17_katana_20261027_retry1 | 0.3439 | 0.2419 | 0.3167 | 0.6121 | 0.7396 | 0.8453 |

三者共用同一 gallery 文件（指纹 `d799087d813f…`），1,980 个 query 均有效。gallery 间 SD：mAP 约 0.015–0.017、Rank-1 约 0.019–0.024。逐 gallery 配对差（相对 M0）：M1 mAP −0.003（−0.010～+0.005）、Rank-1 −0.007；F mAP 0.000（−0.009～+0.009）、Rank-1 −0.004。差异小于 gallery 波动且只有一个训练种子，不能据此判断 S 有效或无效。原始结果见各 run 的 `eval_all/metrics.json`（本地，不入 Git）。

## 状态

评价器与自检已完成，并通过 PMT 固定版本的参考实现对照；M0/M1/F seed17 已在内部 val 复评。
