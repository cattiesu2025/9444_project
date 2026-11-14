# D：独立注册的局部监督实验

本轮暂停平滑扩展，保留 PMT 原本的两阶段直接切换。两个变体使用独立实验名，不覆盖 M2 配置，不需要复制项目目录。

| 实验 ID | 配置路径 | 区域数 | beta | 首轮种子 |
|---|---|---:|---:|---:|
| L3_B050 | configs/experiments/L3_B050.json | 3 | 0.5 | 17 |
| L2_B025 | configs/experiments/L2_B025.json | 2 | 0.25 | 17 |

原 M2 的 q=2、beta=0.5 及训练种子计划保持原样。既有 L3_B025 保持原样。新增配置沿用它的 seeds=[17] 约定。

三个区域是图像的三个水平条带，默认21×10 token网格分为三个7×10条带，没有关键点定位，不保证严格对应人体头/上身/下身。

## 实现变更

新增两份实验配置。在 src/config.py 的 EXPERIMENTS，以及 src/train.py 的训练实验白名单、局部监督启用白名单和命令行 choices 中登记两个名称。模型、损失实现和 PBS 不变。

训练保存的完整配置包含各自 experiment 名，便于评价元数据和后续汇总区分。独立 run_id 仍然用于识别某次运行。E 的外部表格/Drive若另有固定白名单，需要由 E 同步两个名称。

## 在同一个 Katana 项目根目录运行

先使用组内环境检查配置：

```bash
python scripts/inspect_config.py --experiment L3_B050
python scripts/inspect_config.py --experiment L2_B025
python -m src.train --help
```

应分别显示 local.regions=3/2、local.weight=0.5/0.25，smooth.mode=off，retrieval_feature=global_only。

沿用组内已有数据、manifest、预训练权重和环境路径。资源/环境变量按 Katana 实际配置传入。以下名称用于尚不存在同名目录的新实验。

### 20步预检

```bash
qsub -v EXPERIMENT=L3_B050,SEED=17,RUN_ID=dev_L3_B050_seed17_20261107,MAX_STEPS=20 scripts/katana/train.pbs
qsub -v EXPERIMENT=L2_B025,SEED=17,RUN_ID=dev_L2_B025_seed17_20261107,MAX_STEPS=20 scripts/katana/train.pbs
```

检查每个运行的 config.json、total/local/local_weighted、优化器更新、AMP overflow及last.pt。预检通过后分别恢复：

```bash
qsub -v EXPERIMENT=L3_B050,SEED=17,RUN_ID=dev_L3_B050_seed17_20261107,RESUME=1,MAX_STEPS=0 scripts/katana/train.pbs
qsub -v EXPERIMENT=L2_B025,SEED=17,RUN_ID=dev_L2_B025_seed17_20261107,RESUME=1,MAX_STEPS=0 scripts/katana/train.pbs
```

只恢复各实验自己的checkpoint。不要将旧的 experiment=M2 探索checkpoint改名后当成新实验；如已开始旧实验，保留原记录并与B/E商定迁移方式。

### 评价

完整训练后现有PBS默认导出全局特征，然后运行：

```bash
python -m src.evaluate run --query artifacts/results/dev_L3_B050_seed17_20261107/query_val.npz --gallery-pool artifacts/results/dev_L3_B050_seed17_20261107/gallery_pool_val.npz --output-dir artifacts/results/dev_L3_B050_seed17_20261107/eval_all
python -m src.evaluate run --query artifacts/results/dev_L2_B025_seed17_20261107/query_val.npz --gallery-pool artifacts/results/dev_L2_B025_seed17_20261107/gallery_pool_val.npz --output-dir artifacts/results/dev_L2_B025_seed17_20261107/eval_all
```

与M0、M2共用固定gallery记录 artifacts/manifests/gallery_val_all_single.json，并由B/E核对训练预算、划分、种子和初始化一致。不启用局部特征拼接作为此次核心比较。使用验证集选q/beta，不用测试集调参。单训练种子结果仅为探索。

## 当前状态

配置与入口注册已实现。合成输入/自动化检查不代表完成真实训练和检索评价；mAP、Rank-1等只能在真实评价完成后填写。详细本地检查见随包交付的验证记录。

## 横条遮挡增强：L3_B050_BE 与 M0_BE（E，2026-11-14）

**动机：** 三种子下 L3_B050 与 M0 基本持平（验证集 mAP 34.67 ± 0.34 vs 34.57 ± 0.46），遮挡下也没有一致优势，只在中间遮挡有约 +0.3 的趋势（见 [evaluation.md](evaluation.md)）。原有训练已带作者的随机擦除（`PMTRandomErasing`，p=0.5，面积 2%–40%，位置随机），所以"再加 random erasing"等于原配置。这里改为与局部区域对齐的横条擦除：某一区域整条被挡住时，模型只能依靠其他区域和全局特征识别。

**做法：** `src/data.py` 的 `BandErasing`。以概率 p 选一条区域（高度等分为 q 条，第 j 条为行 `[floor(j*H/q), floor((j+1)*H/q))`，与局部头的划分方式一致），整行宽度擦除，填充值与作者随机擦除相同。作用于 RGB 与 IR、两个训练阶段，排在原随机擦除之后。p=0 时不消耗随机数，已有实验的增强序列逐像素不变（有测试覆盖）。

| 实验 | 与谁对照 | 唯一差别 |
| --- | --- | --- |
| L3_B050_BE | L3_B050（已有三种子） | `augmentation.band_erasing = 0.5` |
| M0_BE（可选对照） | M0（已有三种子） | 同上；`local.regions = 3` 只决定擦除带位置，局部监督仍关闭 |

**训练（B）：** 每个实验跑 seed 17/42/2026，其余与已有运行完全相同：

```bash
qsub -v EXPERIMENT=L3_B050_BE,SEED=17,RUN_ID=dev_L3_B050_BE_seed17_20261114 scripts/katana/train.pbs
qsub -v EXPERIMENT=L3_B050_BE,SEED=42,RUN_ID=dev_L3_B050_BE_seed42_20261114 scripts/katana/train.pbs
qsub -v EXPERIMENT=L3_B050_BE,SEED=2026,RUN_ID=dev_L3_B050_BE_seed2026_20261114 scripts/katana/train.pbs
# 可选对照：把 EXPERIMENT 换成 M0_BE，RUN_ID 相应改名
```

**评价注意（E）：** 验证集的人工遮挡测试用的也是横条，训练时见过同类遮挡，结果会偏乐观；干净图片 mAP 与测试集的自然遮挡子集（多为侧面柱子，形状不同）才是公平检验。结论按三种子配对比较给出，单种子只作探索。
