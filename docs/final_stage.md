# 最终阶段：395 身份重训与测试集评价

配置已冻结后，用开发阶段完全相同的配置，把 train（296 身份）和 val（99 身份）合并训练，然后在官方测试集（96 身份）上只评价一次。

## 跑哪些模型

| 模型 | 配置 | 种子 |
| --- | --- | --- |
| M0（PMT baseline） | `M0` | 17、42、2026 |
| L（冻结的局部监督） | `L3_B050` | 17、42、2026 |

S（M1）、横条遮挡（L3_B050_BE、M0_BE）和其他 L 变体在验证集上都是负结果或探索，不进入最终阶段。

## 代码改动

- `src/train.py`：新增 `--final`。`finalize()` 只改 `data` 的四项：`stage = "final"`、`train_splits = ["train", "val"]`、`evaluation_split = "test"`、`train_identities = 395`；模型、损失、优化器、训练轮数等全部不变。最终阶段的 `RUN_ID` 必须以 `final_` 开头，开发阶段不能以它开头；恢复训练时 `--final` 必须与 checkpoint 一致。
- `src/export_features.py`：`--split` 新增 `test`。最终模型训练时见过验证集身份，只允许导出测试集；开发模型导出测试集需加 `--allow-dev-checkpoint`（仅作降级方案，报告中注明）。特征元数据记录 `training_stage`。
- `scripts/katana/train.pbs`：`FINAL=1` 时传 `--final`，训练结束后导出 `query_test.npz` 和 `gallery_pool_test.npz`（pilot 导出同理）。
- 自检：`tests/test_final_stage.py`。真实数据上，最终阶段的标签映射为 395 个身份，其中包含全部 99 个验证身份、不含任何测试身份；每个 epoch 695 个 batch（开发阶段 633 个），训练时间约增加 10%。

## 训练命令（B）

```bash
qsub -v EXPERIMENT=M0,SEED=17,FINAL=1,MAX_STEPS=0,RUN_ID=final_M0_seed17 scripts/katana/train.pbs
qsub -v EXPERIMENT=M0,SEED=42,FINAL=1,MAX_STEPS=0,RUN_ID=final_M0_seed42 scripts/katana/train.pbs
qsub -v EXPERIMENT=M0,SEED=2026,FINAL=1,MAX_STEPS=0,RUN_ID=final_M0_seed2026 scripts/katana/train.pbs
qsub -v EXPERIMENT=L3_B050,SEED=17,FINAL=1,MAX_STEPS=0,RUN_ID=final_L3_B050_seed17 scripts/katana/train.pbs
qsub -v EXPERIMENT=L3_B050,SEED=42,FINAL=1,MAX_STEPS=0,RUN_ID=final_L3_B050_seed42 scripts/katana/train.pbs
qsub -v EXPERIMENT=L3_B050,SEED=2026,FINAL=1,MAX_STEPS=0,RUN_ID=final_L3_B050_seed2026 scripts/katana/train.pbs
```

需要先做 20 步预检时，去掉 `MAX_STEPS=0` 提交一次，再加 `RESUME=1,MAX_STEPS=0` 续跑（与开发阶段相同）。交付时与之前一样打包 checkpoint 和 `artifacts/results/<RUN_ID>/`，附 `SHA256SUMS`。

## 测试集评价规则（E，测试前写定）

- **主结果**：只用全局特征（与开发阶段主协议一致），All-search 与 Indoor-search，10 个固定 gallery（种子 0–9），按三种子均值 ± 标准差报告。
- **事先声明的附加分析**：L 的测试时局部特征融合（3 区域、全局 50%）、9 种人工扰动、A 标注的自然遮挡子集（224 张）。附加分析不作为"改进有效"的依据。
- 测试集只评价一次，结果不再用于选择模型或参数。
