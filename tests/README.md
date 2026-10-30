# 后续检查计划

各负责人实现模块时加入有意义的检查：A 身份隔离及 P/K batch；B 损失、小样本下降、阶段与恢复；C 窗口首尾/期望暴露量/恢复；D 非正方形网格/梯度；E 手算 AP/CMC、摄像头排除与参考实现交叉验证。框架阶段不将配置检查当作模型正确性测试。


已实现：`test_data.py`（A，当前参数化共 32 项，使用 `synthetic_sysu.py` 生成的合成目录，不需要真实数据）。运行 `python -m pytest tests/test_data.py -q`。


## B baseline（2026-10-25）

`python -m pytest tests/test_baseline.py -q`：18 项通过；与 A 联合运行共 50 项通过（含 AMP 溢出恢复和 FP32 非有限梯度保护）。
覆盖激活检查点与普通前反向的逐位一致性、小批拟合下降、手算 MSEL/DCL、两阶段/V0 梯度、退化 DCL、布局拒绝、非正方形 patch 网格、严格预训练加载、阶段边界、学习率、完整 RNG/优化状态恢复和导出。用模拟 SYSU 图片+小模型运行完整训练入口，比较中途恢复与连续训练跨阶段后的最终参数（逐位相同）。这些自检不是本组真实训练结果。

`python scripts/check_pmt_reference.py /path/to/PMT`：验证固定上游 commit；比较 backbone 前向、3 种 P/K×3 seeds×3 losses 的值和梯度，以及 24 epochs 学习率。报告写入 `artifacts/results/baseline_checks/reference.json`。


## E 评价器（2026-10-28）

`python -m pytest tests/test_evaluate.py -q`：14 项通过。期望值为手算：AP/mINP/身份级 CMC、cam3→cam2 排除、无正样本 query 计数、gallery 抽样与 PMT 步骤一致、固定 gallery 复用与拒绝、跨种子样本 SD、命令行端到端。不涉及真实特征。

`python scripts/check_eval_reference.py /path/to/PMT`：与固定 commit 的 `eval_sysu`、`process_gallery_sysu` 数值对照，报告写入 `artifacts/results/eval_checks/reference.json`。详见 [docs/evaluation.md](../docs/evaluation.md)。

`python -m pytest tests/test_robustness.py -q`：12 项通过（需 PyTorch）。遮挡带位置/面积/填充值、带外像素不变、模糊确定性、条件列表固定、合成数据小模型端到端（干净条件与普通评价一致、gallery 不变、拒绝覆盖）、扰动只改变 query 特征、拒绝不匹配的权重或 manifest。不代表真实鲁棒性结果。
