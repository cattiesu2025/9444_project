# PMT + CAJ 通道增强：M0_CA 与 L3_B050_CA

## 动机

在我们的验证集结果里，唯一明显的提升来自 CAJ 的通道增强：CAJ_fix 比 AGW 高 4.1 mAP，Rank-1 高 6.2，10 个 gallery 全部更高（两者都是 ResNet-50）。PMT 只有灰度阶段，没有用通道增强。这里把 CAJ 的通道增强加到 PMT 的训练里，检验它能否改进 PMT。差距不能全部归因于通道增强（CAJ 同时还有 ADP 损失、随机擦除等差别），所以需要在 PMT 上单独受控检验。

## 做法

`src/data.py` 的 `PMTTrainTransform` 新增 `channel_exchange`（配置 `augmentation.channel_exchange`），复用 `src/agw.py` 中 CAJ 的实现，排在作者随机擦除之后：

- **RGB**：以概率 p 做 `ChannelExchange(gray=2)`，随机复制一个颜色通道，得到类似 IR 的单通道图。
- **IR**：做 `ChannelAdapGray(0.5)`（随机复制单通道或转灰度）。

与 CAJ 原版的差别：CAJ 给每张 RGB 额外生成一个交换视图，RGB 数量翻倍；这里改为以 p=0.5 替换，以保持 PMT 的 P×K 批次结构和 MSEL/DCL 的配对方式。期望上与原版一样，一半 RGB 样本是交换后的。p=0 时不消耗随机数，已有实验不受影响。

| 实验 | 对照 | 唯一差别 |
| --- | --- | --- |
| **M0_CA**（主实验） | M0（已有三种子） | `augmentation.channel_exchange = 0.5` |
| L3_B050_CA（可选） | L3_B050（已有三种子） | 同上，检验能否与局部监督叠加 |

## 训练命令（B）

```bash
qsub -v EXPERIMENT=M0_CA,SEED=17,MAX_STEPS=0,RUN_ID=dev_M0_CA_seed17_20261115 scripts/katana/train.pbs
qsub -v EXPERIMENT=M0_CA,SEED=42,MAX_STEPS=0,RUN_ID=dev_M0_CA_seed42_20261115 scripts/katana/train.pbs
qsub -v EXPERIMENT=M0_CA,SEED=2026,MAX_STEPS=0,RUN_ID=dev_M0_CA_seed2026_20261115 scripts/katana/train.pbs
# 可选：EXPERIMENT=L3_B050_CA，RUN_ID 相应改名
```

开发阶段，在验证集上评价，交付方式与之前相同（checkpoint + `artifacts/results/<RUN_ID>/` + `SHA256SUMS`）。

## 评价与决策规则（E，结果出来前写定）

- 主指标：验证集 All-search，只用全局特征，与 M0 三种子在同一组 10 个 gallery 上配对比较；同时报告 9 种扰动的鲁棒性。
- 只有当 M0_CA 的三种子均值高于 M0、且三个种子方向一致时，才考虑把它加入最终阶段（需要另外 3 次 395 身份重训）；否则作为负结果报告。最终阶段的 M0 和 L3_B050 不等这组实验，照常进行。

## 状态

代码、配置与自检（`tests/test_channel_aug.py`）已写好，**自检尚未运行**；建议 B 在训练前先运行一次：

```bash
python -m pytest tests/test_channel_aug.py -q
```
