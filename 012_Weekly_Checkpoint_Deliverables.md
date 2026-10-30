# 新一周计划：评价器上线与四个 baseline 对比

**更新日期：** 2026-10-30，Australia/Sydney  
**对应设计：** [Project Design](010_Project_Design.md)；上周计划见 [011](011_Weekly_Checkpoint_Deliverables.md)  
**周期：** 10/30–11/05（内部目标）。正式 checkpoint 时间以课程公告为准。  
**人员：** A–E 为成员代号；标"待定"的负责人需在本周组会确定。

## 1. 本周目标

1. **评价器可用（最高优先级）。** E 完成并核验 `src/evaluate.py`，先给出 M0 的第一组 Rank-1/mAP。
2. **补齐上周未完成项：** A 的 EDA、D 的局部监督模块、E 的评价与汇总，以及 M1/F 的评价。
3. **四个 baseline 在同一协议下训练和对比：** V0、M0、AGW、CAJ。

老师要求增加 2 个 baseline。本组选 PMT 论文 SYSU-MM01 对比表中的 AGW 与 CAJ（CAJ 也是课程第 4 篇论文 P4）。V0 是去掉 PMT 组件的 ViT 消融下界，是否计入老师要求的 baseline 需在 checkpoint 上向老师确认。

## 2. 当前状态（2026-10-30）

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 数据接口、审计、采样器（A） | 已完成 | [数据接口说明](docs/data_interface.md) |
| EDA（A） | 未完成 | — |
| M0 seed17 训练与验证集特征（B） | 已完成，24 epochs | `artifacts/results/dev_M0_seed17_katana_20261025_retry1/` |
| V0 代码与配置（B） | 已实现，未训练 | [configs/experiments/V0.json](configs/experiments/V0.json) |
| S 调度模块（C） | 已完成；M1、F seed17 已训练并导出特征 | [docs/smooth_transition.md](docs/smooth_transition.md)、`artifacts/results/dev_M1_*`、`dev_F_*` |
| 局部监督模块（D） | 已实现并有自检，在分支 `feat/local-supervision`，未合并；M2 未训练 | `src/local_head.py`、`tests/test_local_head.py`、`docs/local_supervision.md`（该分支） |
| 评价器（E） | 已完成：14 项手算自检；与 PMT `eval_sysu`/`process_gallery_sysu`（commit `0b31da9`）对照通过；M0/M1/F seed17 已评价 | [评价说明](docs/evaluation.md)、`src/evaluate.py`、PR #3 |
| 鲁棒性扰动（E） | 已完成：9 种固定扰动（上/中/下遮挡 10%/20%、模糊 σ=1/2），只扰动 IR query；12 项自检；M0/M1/F seed17 已运行 | [评价说明 · 鲁棒性](docs/evaluation.md)、`src/robustness.py` |
| AGW / CAJ 代码 | 已实现并通过 14 项自检，未训练 | 分支 `feat/agw-caj-baselines`：`src/agw.py`、`src/train_agw.py` |

**M0/M1/F seed17 已有检索指标（内部 val，All-search，10 组固定 gallery 均值）：**

| 实验 | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |
| --- | --- | --- | --- | --- | --- | --- |
| M0 | 34.4 | 24.2 | 32.1 | 60.3 | 72.6 | 83.9 |
| M1 | 34.1 | 24.0 | 31.3 | 60.6 | 73.3 | 84.0 |
| F | 34.4 | 24.2 | 31.7 | 61.2 | 74.0 | 84.5 |

三者差别小于 gallery 间波动（mAP SD 约 1.5–1.7，Rank-1 SD 约 1.9–2.4），且只有一个训练种子，暂不能判断 S 的作用。内部 val 只有 cam3 query 和 cam4/5 gallery，无法做 Indoor-search，数值不能与论文测试集结果直接比较。

## 3. 上周未完成工作

| 成员 | 任务 | 交付物 | 完成标准 |
| :---: | --- | --- | --- |
| E | 评价器 | `src/evaluate.py`、指标自检记录 | 手算小矩阵的 Rank-1/AP 一致；与 mangye16 `eval_metrics.py` 交叉核对；cam3 query 排除 cam2 gallery；固定 gallery 种子 0–9 |
| E | M0/M1/F 首轮评价 | `metrics.json`/`metrics.csv` | 每个 run 记录 10 次 gallery 的 Rank-1/5/10、mAP、均值和标准差、run_id、checkpoint checksum |
| E | Top-5 检索案例 | 案例图及选择方式说明 | 来自真实排序，注明样本怎样选出 |
| A | EDA | 设计文档 §3.4 中 E1–E5 的图表 | 每张图写清"观察—影响—决策" |
| D | 局部监督模块 | `src/local_head.py`、形状/梯度自检、方法说明 | 网格取自 patch embedding 的实际输出（21×10）；L 关闭时全局输出不变 |
| D | M2 训练 | M2 seed17 权重与特征 | 模块自检通过后提交 |
| C | M0/M1/F 对比分析 | 评价器出结果后写对比结论 | 区分调度顺序的效果与额外灰度暴露量的效果 |
| E | 鲁棒性扰动 | `src/robustness.py` | 可降级：主表完成后再做 |

## 4. 四个 baseline 训练与评价对比

### 4.1 对比对象

| ID | 方法 | 主干 | 训练入口 | 负责人 | 状态 |
| --- | --- | --- | --- | --- | --- |
| V0 | ViT + ID + 全局 hard-triplet（无灰度阶段、MSEL、DCL） | ViT-B/16 | `src.train --experiment V0` | B | 待训练 |
| M0 | 完整 PMT | ViT-B/16 | `src.train --experiment M0` | B | 已训练 |
| AGW | 双流 ResNet-50 + Non-local + GeM，ID + WRT | ResNet-50 | `src.train_agw --experiment AGW` | 待定 | 待训练 |
| CAJ | AGW 基础上加通道交换/灰度增强，ID + ADP 平方差损失 | ResNet-50 | `src.train_agw --experiment CAJ` | 待定 | 待训练 |

### 4.2 统一的部分（四个 baseline 必须一致）

- 数据：同一个 manifest、296 个训练身份；评价用 99 个验证身份，测试身份不碰。
- 采样：A 的采样器，P=8、K=4，每步先放 RGB 再放 IR。
- 评价：E 的同一个评价器，IR→RGB、All-search、Single-shot，固定 gallery 种子 0–9。
- 特征：BN 之后做 L2 归一化，余弦距离；不做翻转测试增强，不做 re-ranking。
- 记录：配置、日志、权重 checksum、环境，与 M0 的格式一致。

### 4.3 保留作者设置、在表注中说明的部分

| 项目 | V0 / M0 | AGW / CAJ |
| --- | --- | --- |
| 输入尺寸 | 256×128 | 288×144 |
| 优化器 | AdamW，cosine | SGD 0.1，10 epoch warmup，第 20、50 epoch 各降 10 倍 |
| 训练长度 | 24 epochs | AGW 81 epochs，CAJ 100 epochs |
| 精度 | CUDA AMP | FP32（与作者一致） |
| 每步图片数 | 64 | AGW 64；CAJ 96（RGB 两个视图 + IR） |

AGW/CAJ 作者原本用 train+val 395 个身份训练，本组改为 296 个，以避免验证身份泄漏。因此作者论文中的数字只能作为参考栏，不能和本组结果直接比。

### 4.4 执行顺序

1. **E 完成评价器** → 评价已有的 M0、M1、F。
2. **同时进行：** B 提交 V0；AGW/CAJ 负责人完成下面的准备工作。
3. **AGW/CAJ 准备：**
   - 在 Katana 上运行 `python scripts/prepare_pretrained.py --model resnet50`（计算节点不能下载）。
   - 编写 AGW/CAJ 的 PBS 脚本。现有 `train.pbs` 限时 2 小时，不够用；可以分多次运行，用 `--resume` 接上。
   - 先跑短训练（约 1–2 epoch），检查 loss 是否下降、训练 ID 准确率是否上升，导出特征并用评价器算一次 Rank-1。如果接近随机水平（约 1%），先排查问题，再提交完整训练。
   - 根据短训练的每步耗时估算完整训练时长，再设置 walltime。
4. **完整训练** AGW、CAJ → 导出 `query_val.npz` / `gallery_pool_val.npz` → E 统一评价。
5. **E 汇总主表**，由全组讨论结论。

四个 baseline 各跑 seed 17 一个种子；三个种子留给 M0–M3 的消融。

### 4.5 主表格式

| 方法 | 主干 | 训练身份 | 输入 | Rank-1 | Rank-10 | mAP | 参数量 | FLOPs | 训练时长 | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | ViT-B/16 | 296 | 256×128 | | | | | | | L40S |
| M0 | ViT-B/16 | 296 | 256×128 | | | | | | | L40S |
| AGW | ResNet-50 | 296 | 288×144 | | | | | | | L40S |
| CAJ | ResNet-50 | 296 | 288×144 | | | | | | | L40S |
| 论文报告值（参考，不可直接比） | — | 395 | — | | | | — | — | — | — |

指标为 10 次 gallery 评价的均值 ± 标准差，均在内部验证集上。

### 4.6 要回答的问题

- **V0 和 M0 比：** PMT 的灰度阶段、MSEL 和 DCL 比普通 ViT 好多少？
- **AGW 和 M0 比：** 在本组设置下，PMT 与已发表的 CNN 方法（AGW）相比表现如何？两者的主干和训练方法都不同，差距不能只归因于主干（Transformer 还是 CNN）。
  - PMT 仓库 README 中的 `AGW*` 是"PMT 方法 + AGW 主干 + 随机擦除"，不是 AGW 原方法，代码也未公开。它的 67.09% / 64.25% 不能作为本组 AGW 行的参考值。本周不做 `AGW*`；如果主表完成后还有 GPU 时间，可以把"AGW 网络 + PMT 训练"作为选做实验，用来分开主干和训练方法两个因素，并标为本组实现。
- **CAJ 和 AGW 比：** 只加通道/灰度增强能带来多少提升？这为 S（平滑灰度过渡）提供对照：如果简单增强已有同等收益，S 的价值就要打折扣。

## 5. 分工与时间

| 成员 | 本周任务 | 内部目标 |
| :---: | --- | --- |
| A | EDA 图表与结论 | 11/02 |
| B | 提交 V0；把 M0 交给 E 评价 | V0 于 11/01 提交 |
| C | 评价器可用后写 M0/M1/F 对比 | 评价器可用后 2 天 |
| D | 局部模块与自检；M2 首轮训练 | 模块 11/01，M2 11/03 提交 |
| E | 评价器与核验 → 评价 M0/M1/F → 评价 V0/AGW/CAJ → 主表 | 评价器 10/31，主表 11/05 |
| 待定 | AGW/CAJ 的 PBS 脚本、短训练检查、完整训练、特征导出 | 短训练 11/01，完整训练 11/04 |

## 6. 风险与降级

| 风险 | 处理 |
| --- | --- |
| 评价器推迟 | 所有结果都会被推迟；E 优先做评价器，其他人不等待，继续训练并导出特征 |
| AGW/CAJ 训练时间过长或 GPU 排队 | 分段运行并用 `--resume` 接续；实在不够时先完成 AGW，CAJ 标为"阶段性结果"，注明实际 epoch 数 |
| AGW/CAJ 短训练指标接近随机水平 | 先检查身份映射、模态路由和标签，不提交完整训练 |
| 老师认为 V0 不算 baseline | AGW、CAJ 已满足"再加 2 个"；V0 保留为消融下界 |
| 本地磁盘不足 | AGW/CAJ 的测试会写入多个 ResNet-50 checkpoint；运行测试前确认至少有几 GB 空闲 |

## 7. 下次 checkpoint 要问老师的问题

1. V0（去掉 PMT 组件的 ViT）是否算作一个 baseline？
2. AGW/CAJ 用作者代码移植到本组划分上重新训练是否可以，还是需要独立实现？
3. 外部 baseline 只跑一个种子是否足够？

## 8. 进度记录（课堂汇报用）

**当前进度：** E 的评价器、M0/M1/F 首轮评价、检索案例与鲁棒性已完成（2026-10-30）；其余成员待填写。

| 成员 | 已完成 | 待完成 | 下一步 |
| :---: | --- | --- | --- |
| A | | | |
| B | | | |
| C | | | |
| D | | | |
| E | 评价器及核验；M0/M1/F seed17 评价；Top-k 检索案例；鲁棒性工具及 M0/M1/F 运行（下方遮挡 20% 时 mAP 约降 8 点，上方 20% 约降 5.6 点，中间遮挡和模糊不超过约 2 点；三个模型下降几乎相同） | V0、AGW、CAJ、M2 的评价；主表（含参数量、FLOPs）；M2 鲁棒性对比 | 等 B 交 V0 特征、AGW/CAJ 短训练特征、D 交 M2 特征和权重；拿到即评价 |
| 待定（AGW/CAJ） | | | |

**全组讨论：** 待实际讨论后填写；引用指标时注明 run_id、评价 split 和原始记录位置。
