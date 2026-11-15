# 最终阶段：M0_final 与 M0_CA_final（B，2026-11-15）

## 决定

最终阶段只比较两个模型：**M0**（PMT 基线）与 **M0_CA**（PMT + CAJ 通道增强）。各跑 3 个种子（17/42/2026），共 6 次训练，测试集只评价一次。

依据（只用验证集，测试集未参与选择）：

- M0_CA 是唯一明显超过 M0 的配置。三种子验证集 mAP 为 36.02 ± 0.37，M0 为 34.57 ± 0.46。配对差 +1.45，三个种子方向一致，30 对种子×gallery 中有 28 对更高。这满足 [channel_augmentation.md](channel_augmentation.md) 事先写定的规则。
- L3_B050 与 M0 持平（34.67 ± 0.34），L3_B050_CA 低于 M0_CA。所以 L3_B050 不进入最终阶段，它的持平结论用验证集三种子结果报告。
- 这与设计文档"最终测试只报告两个模型（M0 + 验证集事先选出的候选）"一致。

上述数字由 B 用 E 的评价器（`src.evaluate`，固定 gallery `d799087d813f…`）在本地算出，正式表格以 E 写入 [evaluation.md](evaluation.md) 的为准。

**时间顺序须在报告中说明：** 11/08 冻结时的候选是 L3_B050。通道增强实验在冻结后加入（PR #16），决策规则在结果出来前写定。候选因此由 L3_B050 改为 M0_CA。通道增强来自 CAJ，不是本组提出的方法。

## 配置

| 实验 | 对应开发配置 | 训练身份 | 评价划分 |
| --- | --- | --- | --- |
| `M0_final` | M0 | train+val（395） | test（96） |
| `M0_CA_final` | M0_CA（`augmentation.channel_exchange = 0.5`） | train+val（395） | test（96） |

除 `stage`、`data.train_splits`、`data.train_identities`、`data.evaluation_split` 外，模型、损失、优化器、学习率、24 个 epoch 均与开发阶段相同（`tests/test_final_stage.py` 逐项核对）。两者都从 ImageNet 初始化重新训练，不接着开发阶段的模型续训。395 身份每个 epoch 的批次更多，总步数相应增加。epoch 数沿用开发阶段的 24，不按测试集选择 checkpoint，只用最后一个。

## 防泄漏规则（代码强制）

- `src/config.py`：开发配置必须是 train → val；最终配置必须是 train+val → test，且声明 395 身份。
- `src/train.py`：按 `data.train_splits` 建立 label map，实际身份数与配置不符就报错。只接受 `M0_final` / `M0_CA_final` 两个最终实验。
- `src/export_features.py`：只导出 checkpoint 自己的评价划分。开发 checkpoint 不能导出 test，最终 checkpoint 不能导出 val（已用于训练）。

## 训练命令（B，Katana）

```bash
E=PMT_ENV_SETUP=scripts/katana/environment.sh,PMT_PYTHON=/srv/scratch/z5535967/venvs/pmt-py311/bin/python
for s in 17 42 2026; do
  qsub -v $E,EXPERIMENT=M0_final,SEED=$s,MAX_STEPS=0,RUN_ID=final_M0_seed${s}_20261115 scripts/katana/train.pbs
  qsub -v $E,EXPERIMENT=M0_CA_final,SEED=$s,MAX_STEPS=0,RUN_ID=final_M0_CA_seed${s}_20261115 scripts/katana/train.pbs
done
```

训练结束后 `train.pbs` 按 `config.json` 的评价划分自动导出 `query_test.npz` 与 `gallery_pool_test.npz`（开发运行仍是 `*_val.npz`）。交付方式同前：checkpoint + `artifacts/results/<RUN_ID>/` + `SHA256SUMS`。

## 评价（E）

```bash
python -m src.evaluate run --query artifacts/results/<RUN_ID>/query_test.npz \
  --gallery-pool artifacts/results/<RUN_ID>/gallery_pool_test.npz --output-dir artifacts/results/<RUN_ID>/eval_test
```

评价器从特征元数据读取 `split=test`，使用 `artifacts/manifests/gallery_test_all_single.json`（首次运行时用固定种子 0–9 生成）。测试集有完整的摄像头组合，可以另外报告 Indoor-search。自然遮挡子集（测试集 IR query）只对这 6 个最终模型做描述性评价。

## 状态

配置、代码与自检已完成：`tests/test_final_stage.py` 11 项，全部测试本地 149 项通过；Katana 上 final_stage/baseline/channel_aug 共 40 项通过。Katana 的 manifest 在 train+val 上同样得到 395 个身份。

2026-11-15 已在 Katana 提交，做法与 11/08 冻结运行相同：先跑 20 步 preflight，完整运行依赖 preflight 成功（`afterok`），从 preflight 的 checkpoint 续训。完整运行结束后自动导出测试集特征。只有 `final_M0_seed17` 的 preflight 另外做了一次 pilot 导出（`pilot/*_test.npz`），用来提前检查测试集导出路径。这份 20 步模型的特征不参与评价。

| Run ID | Preflight | 完整运行 |
| --- | --- | --- |
| final_M0_seed17_20261115 | 9796618 | 9796619 |
| final_M0_seed42_20261115 | 9796620 | 9796621 |
| final_M0_seed2026_20261115 | 9796622 | 9796623 |
| final_M0_CA_seed17_20261115 | 9796624 | 9796625 |
| final_M0_CA_seed42_20261115 | 9796626 | 9796627 |
| final_M0_CA_seed2026_20261115 | 9796628 | 9796629 |

提交不等于训练完成；完成状态以 `summary.json` 与作业退出码为准。
