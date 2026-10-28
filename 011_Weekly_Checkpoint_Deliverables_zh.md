# 本周进度与课堂 Checkpoint 汇报（PMT Transformer 版）

**更新日期：** 2026-10-28，Australia/Sydney  
**English version:** [011_Weekly_Checkpoint_Deliverables.md](011_Weekly_Checkpoint_Deliverables.md)（两版内容一致）  
**对应设计：** [Project Design](010_Project_Design.md)  
**人员：** A–E 为待替换姓名的成员代号。未勾选的条目均为计划，不表示已完成。  
**日期：** 正式 checkpoint 时间以课程公告为准。10/25 数据接口、10/28 结果汇总仅为内部目标，不能替代或延后课程截止日。

**课堂形式：** tutor 口头询问项目进度，不要求现场检查代码。以下实现与自检清单用于团队推进完整项目，不是课堂必须提交的代码包。课堂准备说明本人已完成什么、当前结果、遇到的问题和下一步即可；代码始终按功能放在项目目录中。

## 1. 本周目标与名称说明

课程要求：确定 baseline，完成实现、用本组数据训练、评价并记录讨论初步结果。

本组 baseline 是 **PMT：共享 Transformer＋原论文两阶段训练及损失**。本周先完成一个训练种子的初步结果，后续改进不作为 baseline 完成的前置条件。

- **平滑灰度过渡（简称 S，Smooth transition）：** 第二阶段仍以一定概率使用灰度图，再逐步降低该概率到 0，代替立即全部切换到 RGB。这是本组计划验证的改进，由 C 负责。
- **局部特征监督（简称 L，Local supervision）：** 将 Transformer 的图像 patch 特征按水平区域聚合，增加局部身份监督。这是本组计划验证的改进，由 D 负责。
- S/L 只是设计中的因素代号，不是成员姓名，也不是 PMT 原论文的两个组件名称。

**分工原则：一人负责一份独立产物，自己实现、检查并撰写说明。** B 独立负责完整 baseline；C/D 分别交独立改进模块，不承担 B 的 baseline 核验。E 负责评价与汇总，不代写其他人的技术说明。各自维护自己的模块和配置，不要求多人同时修改同一个训练文件。

## 2. 每个人本周交付什么

| 成员 | 独立负责内容 | 本周交付物 | 交付顺序 |
|---|---|---|---|
| A | 数据准备、划分、采样与 EDA | 数据入口、划分/标签说明、采样器、审计和 EDA 初稿 | 数据接口最先交；EDA 可稍后 |
| B | 完整 PMT baseline 的实现与训练 | 环境、训练/特征导出入口、配置、训练日志、模型权重、方法说明 | A 数据接口后完成真实训练，再交 E |
| C | 平滑灰度过渡改进 | 独立调度模块、概率示例、自检结果和方法说明 | 可与 A/B 同时做；不阻塞 baseline |
| D | 局部特征监督改进 | 独立局部头模块、形状/梯度自检和方法说明 | 可与 A/B 同时做；不阻塞 baseline |
| E | 检索评价与初步结果汇总 | 评价代码、指标自检、固定 gallery、baseline 指标、检索案例和汇总页 | 先独立准备评价；拿到 A 数据及 B 权重后产出指标 |

## 3. A：数据准备与 EDA

**项目位置：** `src/data.py`、`data/`、`artifacts/manifests/`；EDA 结果放入 `artifacts/results/`。

必须交付：

1. 数据路径及来源说明，RGB/IR、身份、摄像头的映射规则。
2. `manifest.csv`、训练/验证/测试划分文件及检查记录：身份不重叠，图片可读取，缺失/异常有记录。
3. 可执行数据读取与身份/模态平衡采样入口；提供一条命令及真实 batch 示例。
4. batch 包含图片、连续训练标签、原始身份、模态、摄像头；默认 `2PK=64`，P=8、K=4，具体配置写明。
5. 初步 EDA：摄像头/模态数量、每身份样本分布、同身份 RGB/IR 样例。每类图写清观察及对训练的影响。

默认开发划分为 296 个训练身份、99 个验证身份，96 个测试身份保留。不得按图片随机拆分而混入相同身份。

**完成标准：** 数据接口可直接读取；检查记录完整；不必等全部 EDA 画完才交接口。本周不要求完成采样对照训练。

## 4. B：完整 PMT baseline

**项目位置：** `src/model.py`、`src/losses.py`、`src/train.py`、`configs/`；权重放入 `checkpoints/`，日志放入 `artifacts/results/`。

必须交付：

1. 环境说明：作者代码 URL/commit、依赖、设备、预训练来源和实际修改。记录论文 MindSpore 与公开 PyTorch 实现的差异。
2. 完整 PMT 模型与训练入口，包括共享 ViT、两阶段输入、PHT/MSEL/DCL、保存恢复；B 自己核验原方法及阶段边界。
3. 配置文件：输入、patch/stride、采样、增强、损失、优化器、种子和训练长度。原论文 SYSU 配置 24 epochs、前 6 epochs 第一阶段作为核对起点，不能未经核对就声称一致。
4. 本组真实数据训练日志及 checkpoint 路径/checksum。日志包括各项 loss、stage、epoch/global step、学习率、耗时与峰值显存。
5. 模型加载与特征导出入口：输出特征及对应原始身份、摄像头、模态和图片路径，供 E 评价；说明特征维度、归一化和距离。
6. 简短方法与运行说明：一条训练命令、一条特征导出命令、已完成训练进度和已知问题。

**完成标准：** 实际训练并保存可加载权重，输入/损失/恢复自检通过。baseline 不启用平滑过渡或局部监督；B 不等待 C/D 改进模块。

完整方法与完整训练分开表述：部分 epochs 的结果写“阶段性结果”；作者预训练模型仅用于校验不能代替本组训练。若实际只完成基础 ViT，必须标为基础 ViT，不得称完整 PMT。

## 5. C：平滑灰度过渡模块

**项目位置：** `src/schedules.py`、`configs/experiments/M1.json`、`configs/experiments/F.json`。

必须交付：

1. `src/schedules.py`：独立概率调度函数，输入第二阶段步数、过渡窗口和初始概率，输出当前灰度概率；给出关闭开关及恢复所需状态。
2. 独立运行示例：打印或绘制直接切换、平滑过渡、固定概率控制的概率序列；解释固定控制怎样匹配期望灰度暴露量。
3. 自检记录：概率范围、窗口首尾、窗口后归零、断点恢复连续性、关闭时不改变原路径。短示例可以用模拟步数，不冒充训练结果。
4. 方法说明：问题假设、公式、默认参数、预期输入输出、如何接入第二阶段，以及后续对照实验。

**完成标准：** 不依赖 B 正在运行的训练任务也能执行和验证模块。灰度转换针对可见光来源，IR 不变；不能同时更改 PMT 损失切换规则。

**本周不要求：** 完成改进模型训练或证明提升。B baseline 交付后，C 在其固定版本基础上独立完成自己的改进配置与训练入口；不要求 B 替 C 集成。

## 6. D：局部特征监督模块

**项目位置：** `src/local_head.py`、`configs/experiments/M2.json`、`configs/experiments/M3.json`。

必须交付：

1. `local_head.py`：输入去除特殊 token 的 patch 特征及显式 H_p/W_p，按水平区域聚合；初版 q=2，每区有局部身份分类头。
2. 独立调用示例：输入模拟 patch 特征，输出局部特征、分类 logits 和局部交叉熵；记录张量形状与梯度检查。
3. 自检记录：token 数与 H_p×W_p 匹配、特殊 token 排除约定、非正方形网格、梯度有限、关闭时不影响全局输出。
4. 方法说明：全局/局部结构、损失与 beta、错位风险、接入位置和后续实验。真实 token 网格必须从 patch embedding 输出读取，不能开平方猜测。

**完成标准：** 模块可独立运行，不需要等待 baseline 完整训练。模拟特征只验证模块接口与反向传播，不证明真实图像上有效。初版测试仍只用原全局特征，局部头仅作训练辅助监督。

**本周不要求：** 完成改进模型训练或证明遮挡性能提升。B baseline 交付后，D 独立完成真实 token 接入和自己的改进训练配置，不要求 B 替 D 集成。

## 7. E：评价与初步结果

**项目位置：** `src/evaluate.py`、`src/robustness.py`、`artifacts/manifests/`、`artifacts/results/`。

必须交付：

1. 评价代码与协议说明：内部验证身份、IR→RGB、All-search/Single-shot、摄像头排除、固定 gallery、距离和有效 query 处理。
2. 指标自检：人工距离矩阵手算 Rank-1/AP，与参考实现核对；检查 SYSU 身份级 CMC 和 cam3 query 排除 cam2 gallery 的规则。
3. 固定 10 组 gallery 索引，种子 0–9；使用 A 的身份/摄像头信息和 B 的特征导出入口进行评价。
4. `metrics.json`/`metrics.csv`：各次 Rank-1/mAP、均值、run_id、split、epoch、有效 query 数及 checkpoint checksum。
5. 基于真实排序生成 Top-5 检索案例，注明样本选择方式；输出初步分析和限制。
6. 在本文末尾的进度记录中汇总 baseline、训练进度、指标及证据路径，记录全组讨论、问题与下一步，供课堂口头汇报使用。

**完成标准：** 指标来自本组实际权重，原始记录可追溯。默认报告内部 validation，不用测试集调参。10 次 gallery 重复不等于 10 次独立训练；若只有一次临时评价，如实标注 trial 数。

E 自己检查评价正确性，不设其他成员的审批环节；上游数据或权重有问题时反馈给其负责人修复。每个人对自己的交付物负责。

## 8. 只需遵守的交付顺序

**Baseline 主线：A 数据接口 → B 训练权重与特征导出入口 → E 指标、案例和汇总。**

**改进模块：C 平滑过渡、D 局部监督同时独立准备 → B baseline 版本交付后，各自接入和训练自己的改进。** 本周仅要求模块及自检，改进训练留到后续。

| 顺序 | 交付 | 内部目标 |
|---|---|---|
| 1 | A 先交数据接口/划分；B 准备原模型；C/D 准备独立模块；E 准备评价器 | 10/24 起同时开始，数据接口目标 10/25 |
| 2 | B 使用 A 的数据实际训练，交可评价权重、配置与导出命令；A 补 EDA，C/D 完成自检 | 数据接口可用后立即开始 |
| 3 | E 使用权重完成指标和检索案例；五人各自交说明，E 汇总讨论记录 | 内部目标 10/28，且不晚于正式 checkpoint |

没有 C/D 先核验才允许 B 训练的依赖，也没有必须等 A 全部 EDA 完成才开工的要求。B 可用已核验小批数据先试跑，正式长训练前完成数据与输入检查。

后续联合方案由 D 在 B 的固定 baseline 版本上接入 C 已交付的调度模块，D 负责联合配置和训练；C 不需与 D 同时改同一个文件。E 统一评价各版本，B 继续负责原 baseline。

## 9. 最终交付检查

- [ ] A：数据/采样接口、划分检查和 EDA 初稿。
- [x] B：实际训练的 baseline、配置、日志、权重和特征导出入口。
- [x] C：可独立执行的平滑过渡模块、自检与说明。
- [ ] D：可独立执行的局部监督模块、自检与说明。
- [x] E：评价代码、自检、真实指标、案例与汇总。
- [ ] 全组：讨论初步结果，写清问题、限制与后续任务。
- [ ] 汇总页区分已完成训练、阶段性结果、模块原型和未完成项。

课堂 checkpoint 不单建目录或代码包。最终项目仍整合为一个 Notebook：[submission/project.ipynb](submission/project.ipynb)；`checkpoints/` 仅保存模型权重及断点恢复状态，内容不上传 Git。


## 10. 本周进度记录（课堂汇报用）

**当前进度：** 数据接口已就绪，M0 baseline 已完成 24 epochs 训练及验证集特征导出；评价器已通过手算自检与 PMT 参考实现对照，M0（以及已训练的 M1、F）seed17 已得到内部验证集检索指标。

| 成员 | 已完成 | 待完成 | 下一步 |
| :---: | --- | --- | --- |
| A | 数据接口、采样器与真实数据检查 | EDA | 补充数据分析与样例 |
| B | M0 seed17 训练、恢复与特征导出 | 本周任务已完成 | 向 E 交付 baseline 权重与特征（已完成） |
| C | 平滑灰度调度 `src/schedules.py`（linear / 固定期望暴露量 / off）并接入训练入口；M1、F 配置；调度自检（边界、等期望暴露量、按配置与 step 重建）；方法说明；M1、F seed17 已在 Katana 完成 24 epochs 并由 E 复评 | 断点恢复端到端验证；多种子 | 种子 42/2026；切换期训练曲线分析 |
| D | 尚无提交：`src/local_head.py` 仍为占位，仅有框架阶段的 M2/M3 配置 | 局部头、非正方形网格与梯度自检 | 按设计 10/29 前完成形状/梯度/小样本检查；10/31 前仍不成立则保留失败证据并降为探索工作 |
| E | 评价器（固定 10 组 gallery、cam3→cam2 排除、身份级 CMC、mAP/mINP、跨种子汇总）；14 项手算自检；PMT `eval_sysu`/`process_gallery_sysu` 对照通过；M0/M1/F seed17 复评；M0 Top-5 检索案例 | `metrics.csv` 导出；鲁棒性评价（`src/robustness.py`）；多种子汇总 | 种子 42/2026 到位后汇总 mean±SD；失败案例分析 |

### 证据与说明

- **A：** [数据接口说明](docs/data_interface.md)、[数据准备入口](scripts/prepare_data.py)。真实数据审计及 batch 检查均通过；本地记录见 `artifacts/manifests/data_audit.md` 和 `artifacts/results/data_check/check_report.md`。
- **B：** [baseline 运行说明](docs/baseline.md)、[整合 Notebook](submission/project.ipynb)。Katana L40S 完成 M0 seed17 的 24 epochs；A+B 共 50 项测试及交付核验通过，已导出 IR 1,980 条、RGB 1,974 条验证特征。
- **B 的本地交付：** 日志、配置与特征位于 `artifacts/results/dev_M0_seed17_katana_20261025_retry1/`；权重位于同名 `checkpoints/` 目录。AMP 问题已修复，正式作业成功结束；检索指标见下方 E 的初步结果。

- **C：** [平滑过渡方法与检查](docs/smooth_transition.md)、[调度实现](src/schedules.py)、[调度自检](tests/test_schedule.py)（3 项通过）；已合并至 main（PR #1、#2）。M1/F 训练记录：`dev_M1_seed17_katana_20261027_l40s`、`dev_F_seed17_katana_20261027_retry1`，交付核验 PASS（24 epochs、导出校验和与逐行对齐）。
- **D：** 截至 2026-10-28 GitHub 上没有 D 的分支或提交。
- **E：** [评价协议与运行说明](docs/evaluation.md)、[评价器](src/evaluate.py)、[自检](tests/test_evaluate.py)、[参考对照脚本](scripts/check_eval_reference.py)、[检索案例脚本](scripts/retrieval_cases.py)。B 交付的 6 个特征 npz 的 SHA256 与导出记录一致；验证集 3,954 张原图与训练 manifest 的 SHA256 逐张一致。对照报告 `artifacts/results/eval_checks/reference.json`；固定 gallery `artifacts/manifests/gallery_val_all_single.json`（本地，不入 Git）。

### 初步结果（E 汇总，2026-10-28）

协议：内部验证集（99 身份，未用于训练）、IR→RGB、All-search、Single-shot；单训练种子 17，10 组固定 gallery（种子 0–9）取平均；指标单位 %。三个模型共用同一组 gallery，1,980 个 query 均有效。

| 实验 | run_id | 训练 | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **M0（PMT baseline）** | dev_M0_seed17_katana_20261025_retry1 | 24 epochs | **34.4** | 24.2 | **32.1** | 60.3 | 72.6 | 83.9 |
| M1（PMT + 平滑过渡） | dev_M1_seed17_katana_20261027_l40s | 24 epochs | 34.1 | 24.0 | 31.3 | 60.6 | 73.3 | 84.0 |
| F（固定概率对照） | dev_F_seed17_katana_20261027_retry1 | 24 epochs | 34.4 | 24.2 | 31.7 | 61.2 | 74.0 | 84.5 |

- gallery 间波动（SD）：mAP 约 1.5–1.7，Rank-1 约 1.9–2.4。
- 逐 gallery 配对差（相对 M0）：M1 mAP −0.3（−1.0～+0.5）、Rank-1 −0.7；F mAP 0.0（−0.9～+0.9）、Rank-1 −0.4。
- 原始记录：各 run 的 `eval_all/metrics.json`（E 本地，解压自 B 交付的结果包，含 checkpoint 与输入 SHA256）。

**解读与限制：**

1. 本组 baseline 已建立：PMT M0 在内部验证集上 mAP 34.4%、Rank-1 32.1%，作为后续比较基准。
2. M1、F 与 M0 的差异小于 gallery 间波动，且只有一个训练种子，**目前不能判断平滑过渡有效或无效**；需种子 42/2026 后按"先平均 gallery、再跨种子 mean±SD"汇总。
3. 内部验证集的 IR 只有 cam3、RGB 只有 cam4/5（无 cam1/2/6），All-search 实际为 cam3→cam4/5，无法做 Indoor-search；验证指标只用于组内比较，**不能与论文 SYSU 测试集结果直接对比**（论文为 395 训练身份、测试集、完整摄像头组合）。
4. Top-5 检索案例（从"Rank-1 错、正确身份在第 2–3 位"的 371 个 query 中随机抽取 4 个，`artifacts/results/M0_seed17/retrieval_cases/near/`）显示：红外中深色衣物可呈亮色，颜色/明暗线索失效，模型易混淆发型、体型相近的人。

**待全组讨论（讨论后在此记录结论）：**

- 种子 42/2026 的训练排期（M0、M1，及 F 是否需要多种子）。
- 内部验证集摄像头覆盖有限是否需要在报告中单独说明，或调整验证协议。
- M2/M3（局部监督）接入后共用同一 gallery 文件与评价命令。

**全组讨论结论：** 待实际讨论后填写。
