# D 的两个局部监督变体

## 本次完成的内容

新增两份局部监督配置快照，不修改模型、训练源码、测试、PBS 或共享 `configs/base.json`。

两个变体均沿用现有 M2 实验标识，关闭平滑扩展，保留 PMT 原有的两阶段灰度直接切换。

| 变体 | 区域数 q | 局部损失权重 beta | 建议 run_id |
|---|---:|---:|---|
| L_q3_beta050 | 3 | 0.5 | dev_M2_q3_beta050_seed17_20261107 |
| L_q2_beta025 | 2 | 0.25 | dev_M2_q2_beta025_seed17_20261107 |

原 M2 默认配置为 q=2、beta=0.5。第一项只改变区域数，第二项只改变局部损失权重。

现有 `L3_B025` 配置为 q=3、beta=0.25，同时改变两个因素，因此不作为本次两项实验的替代。

三个区域是等高水平条带：默认 21×10 token 网格分成三个 7×10 区域。它们对应图像上、中、下部分，没有人体关键点定位，不保证严格对应头部、上身和下身。

## 配置文件位置

两份配置快照分别保存在：

```text
configs/variants/L_q3_beta050/M2.json
configs/variants/L_q2_beta025/M2.json
```

现有训练入口不会自动读取 `configs/variants/`。运行前，需要将所选配置复制到：

```text
configs/experiments/M2.json
```

`--experiment` 仍使用 `M2`，通过不同的 `run_id` 和保存的完整配置区分实验。

## 准备两个独立运行目录

建议为两个变体分别准备项目副本，例如：

```text
9444_L_q3
9444_L_beta025
```

在各自项目根目录执行以下命令。

### 三区域、beta=0.5

```bash
cp configs/variants/L_q3_beta050/M2.json configs/experiments/M2.json
python scripts/inspect_config.py --experiment M2
```

确认：

```text
local.enabled = true
local.regions = 3
local.weight = 0.5
smooth.mode = off
model.retrieval_feature = global_only
```

### 两区域、beta=0.25

```bash
cp configs/variants/L_q2_beta025/M2.json configs/experiments/M2.json
python scripts/inspect_config.py --experiment M2
```

确认：

```text
local.enabled = true
local.regions = 2
local.weight = 0.25
smooth.mode = off
model.retrieval_feature = global_only
```

数据集和预训练权重沿用组内已有路径，不必重复复制。各项目副本还需能够访问相同的 manifest、划分记录和 Python 环境。

不要在两个排队作业之间修改它们共用的 `M2.json`，否则作业启动时可能读取错误配置。使用独立项目目录可以避免这一问题。

## Katana 预检与训练

以下命令在对应项目根目录执行，沿用组内已验证的 Katana 环境和 PBS 资源设置。

如果环境需要 `SYSU_MM01_ROOT`、`PMT_PRETRAINED`、`PMT_ENV_SETUP` 或 `PMT_PYTHON`，按组内实际路径设置并传入作业。

两个实验均使用 seed17，保持相同训练轮数、学习率、数据划分和初始化来源。提交长训练前，与 B 核对这些设置是否与对照实验一致。

### 1. 各运行 20 步预检

三区域配置：

```bash
qsub -v EXPERIMENT=M2,SEED=17,RUN_ID=dev_M2_q3_beta050_seed17_20261107,MAX_STEPS=20 scripts/katana/train.pbs
```

两区域、beta=0.25 配置：

```bash
qsub -v EXPERIMENT=M2,SEED=17,RUN_ID=dev_M2_q2_beta025_seed17_20261107,MAX_STEPS=20 scripts/katana/train.pbs
```

检查以下证据：

- 保存的 `config.json` 包含正确的区域数、权重和关闭平滑的设置。
- 日志中的 `total`、`local`、`local_weighted` 为有限值。
- 优化器成功更新，无连续 AMP overflow。
- 正确保存 `checkpoints/<run_id>/last.pt`。

20 步预检只能检查训练初期，不能替代阶段切换、完整训练及最终评价检查。

### 2. 从各自预检 checkpoint 继续训练

三区域配置：

```bash
qsub -v EXPERIMENT=M2,SEED=17,RUN_ID=dev_M2_q3_beta050_seed17_20261107,RESUME=1,MAX_STEPS=0 scripts/katana/train.pbs
```

两区域、beta=0.25 配置：

```bash
qsub -v EXPERIMENT=M2,SEED=17,RUN_ID=dev_M2_q2_beta025_seed17_20261107,RESUME=1,MAX_STEPS=0 scripts/katana/train.pbs
```

恢复入口读取 checkpoint 内保存的配置。每个实验必须从自己的 checkpoint 恢复，不可用其他区域数或其他 beta 的 checkpoint 代替独立训练。

以上 `run_id` 仅适用于尚不存在同名实验的情况；已有运行应正确恢复，或为新实验选择新的唯一名称。

## 特征导出与评价

现有 PBS 默认在完整训练后导出：

```text
artifacts/results/<run_id>/query_val.npz
artifacts/results/<run_id>/gallery_pool_val.npz
```

核心比较仍使用全局特征，不启用测试时局部特征拼接。

分别执行以下评价命令。

三区域配置：

```bash
python -m src.evaluate run \
  --query artifacts/results/dev_M2_q3_beta050_seed17_20261107/query_val.npz \
  --gallery-pool artifacts/results/dev_M2_q3_beta050_seed17_20261107/gallery_pool_val.npz \
  --output-dir artifacts/results/dev_M2_q3_beta050_seed17_20261107/eval_all
```

两区域、beta=0.25 配置：

```bash
python -m src.evaluate run \
  --query artifacts/results/dev_M2_q2_beta025_seed17_20261107/query_val.npz \
  --gallery-pool artifacts/results/dev_M2_q2_beta025_seed17_20261107/gallery_pool_val.npz \
  --output-dir artifacts/results/dev_M2_q2_beta025_seed17_20261107/eval_all
```

与 M0 和原 M2 比较时，由 E 核对各项目使用同一份固定 gallery 记录：

```text
artifacts/manifests/gallery_val_all_single.json
```

保持相同的验证数据、预处理、训练种子、训练预算和评价协议。指标以保存的原始评价结果为准。

## 验证与当前状态

在 CPU PyTorch 2.14.1+cpu 环境中完成了以下检查：

- 两份配置通过校验。
- 相对于原 M2，共用训练、主干和评价配置保持一致。
- 两个变体通过微型 PMT 的前向、局部输出形状、加权损失、主干与局部头梯度及优化器单步检查。
- 灰度调度保持原两阶段直接切换，没有启用平滑扩展。
- 新版原有的 7 项局部监督自检通过。

详细验证记录保存在本地交付包的 `validation.json`，未随本次三文件配置提交上传。

上述检查使用合成输入，不代表已完成完整 ViT 的真实数据训练、Katana 运行或检索评价。目前两次真实实验仍待执行，尚无可报告的 mAP 或 Rank-1 结果。

## 结果记录

训练和评价完成后填写，未完成前保留为空，不填写估计值。

| 设置 | seed | run_id | 完成 epoch | mAP | Rank-1 | Rank-5 | Rank-10 | 耗时/显存 |
|---|---|---|---|---|---|---|---|---|
| M0 | 17 | 与 B/E 核对 | | | | | | |
| 原 M2：q2、beta0.5 | 17 | 与 B/E 核对 | | | | | | |
| q3、beta0.5 | 17 | dev_M2_q3_beta050_seed17_20261107 | | | | | | |
| q2、beta0.25 | 17 | dev_M2_q2_beta025_seed17_20261107 | | | | | | |

结果汇总时明确指标单位，将每个训练种子的 10 次固定 gallery 评价先取平均。单个训练种子的结果仅作为探索，不据此声称统计显著提升。区域数和 beta 的选择使用验证集，不使用测试集。

## 组内进度说明

已完成两个局部监督配置快照、运行说明和模块兼容性检查。本轮暂停平滑扩展，下一步在 Katana 运行两项 seed17 实验，导出全局特征并共用固定 gallery 评价，与原 M2 和 M0 比较。

配置准备完成不等于训练或评价完成。
