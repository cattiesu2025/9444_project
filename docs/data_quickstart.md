# 数据快速上手（给成员 B）

A 的数据接口已在真实 SYSU-MM01 上运行并通过检查（2026-10-24）：审计 PASS；开发阶段 296 个训练身份；batch = 64（P=8 身份 × K=4 张 × RGB/IR 两个模态）；每 epoch 633 步。完整说明见 [data_interface.md](data_interface.md)。

## 1. 在你的机器上准备一次

```bash
pip install numpy pillow torch torchvision kagglehub
python scripts/prepare_data.py
```

没有数据时会自动从课程 Kaggle 镜像下载到 `data/SYSU-MM01`（约 2 GB，需要 Kaggle 登录时按屏幕提示操作），然后生成 manifest、审计并检查接口。看到"全部通过"即可。已有数据可用 `--root <目录>` 或环境变量 `SYSU_MM01_ROOT` 指定。Katana 计算节点可能不能联网，请先在登录节点运行一次。

## 2. 训练时取数据

```python
from src.data import get_train_loader

loader, sampler, dataset, label_map = get_train_loader("M0", num_workers=4)
num_classes = len(label_map)          # 开发阶段 296；get_train_loader("M0", final=True) 为 395

for epoch in range(24):               # epoch 从 0 开始
    sampler.set_gray_probability(1.0 if epoch < 6 else 0.0)   # 第一阶段 RGB 全转灰度，第二阶段不转
    sampler.set_epoch(epoch)
    for batch in loader:
        images = batch["images"]      # [64, 3, 256, 128]，已做 PMT 增强与 ImageNet 归一化
        labels = batch["labels"]      # [64]，0..num_classes-1
        feats = ...                   # model(images)
        feat_rgb, feat_ir = feats.chunk(2, 0)   # 前 32 行 RGB，后 32 行 IR
```

batch 还包含 `identities`（原始身份）、`cameras`、`modalities`（0=RGB，1=IR）、`gray`（是否已转灰度）、`paths`。

## 3. 必须知道的三条约定

1. **排列固定**：前一半 RGB、后一半 IR，两半身份顺序相同，每个身份连续 4 行。作者的 MSEL / DCL / Triplet 依赖这个排列，可直接使用；`check_pmt_layout(batch, 8, 4)` 可校验。若想用作者 `train.py` 的写法，`as_pmt_tuple(batch)` 返回 `(input1, input2, label1, label2)`。
2. **可复现**：同一 seed 下每个 epoch 的 batch 完全相同，与 `num_workers` 无关。断点恢复保存 `sampler.state_dict()`，恢复时 `sampler.load_state_dict(...)`。
3. **只用验证集调参**：训练只含 train 身份；验证用 `get_eval_dataset("val", ...)`，测试集留到配置冻结后。

## 4. 导出特征给 E

```python
from src.data import get_eval_dataset, collate_eval
from torch.utils.data import DataLoader

query = DataLoader(get_eval_dataset("val", modality="ir"), batch_size=128, collate_fn=collate_eval)
gallery = DataLoader(get_eval_dataset("val", modality="rgb"), batch_size=128, collate_fn=collate_eval)
# 每个 batch：images, identities, cameras, modalities, paths（不含训练标签）
```

## 5. 遇到问题

- 审计报告：`artifacts/manifests/data_audit.md`；接口检查：`artifacts/results/data_check/check_report.md`。
- 数据或接口问题请直接找 A，不要在本地修改 `src/data.py` 后自行训练，否则各实验的数据不一致。
