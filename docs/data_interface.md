# 数据接口说明（A → B / C / E）

负责人：A。代码：`src/data.py`；脚本：`scripts/prepare_data.py`（一条命令）、`scripts/build_manifest.py`、`scripts/check_data.py`；自检：`tests/test_data.py`（30 项，合成数据）。

## 1. 快速开始

### 准备数据（每台机器一次）

```bash
pip install numpy pillow torch torchvision kagglehub
python scripts/prepare_data.py
```

这条命令依次完成：

1. **找数据**：`--root` 参数 → 环境变量 `SYSU_MM01_ROOT` → `data/SYSU-MM01` → 上次自动下载的位置。
2. **找不到就下载**：用 kagglehub 下载课程镜像 `coconutjean/sysumm01`（约 2 GB）到 kagglehub 缓存目录，并把位置记在 `data/dataset_location.json`，下次自动找到。若 Kaggle 要求登录，屏幕会给出两种办法：配置 Kaggle API token，或手动下载 zip 并解压到 `data/SYSU-MM01`。
3. **生成 manifest 与审计**：写入 `artifacts/manifests/`。已有且数据位置未变时直接复用，`--rebuild` 强制重建。
4. **检查训练接口**：取真实 batch，检查排列、标签、复现、断点恢复，写入 `artifacts/results/data_check/`。

全部通过时退出码为 0，并打印 B 的调用方式。常用参数：`--root <已有数据目录>`、`--no-download`、`--skip-checks`（只准备数据，不需要 torch）、`--num-workers 0`（默认值，Windows 上更稳）。

### B：取训练数据（两行）
1. 有用的
```python
from src.data import get_train_loader
loader, sampler, dataset, label_map = get_train_loader("M0")   # 开发阶段：train 296 身份
num_classes = len(label_map)
```

2. 解释
(test)在 (.venv) 窗口、9444_project 目录下输入 python，进入 Python 后逐行粘贴：

```python
from src.data import get_train_loader
loader, sampler, dataset, label_map = get_train_loader("M0", num_workers=0)
print("类别数:", len(label_map), " 每轮步数:", len(loader))

sampler.set_epoch(0)
batch = next(iter(loader))        # 从发牌机取一手牌
print(batch.keys())
print(batch["images"].shape)      # 64 张图，每张 3×256×128
print(batch["identities"][:8])    # 前 8 张是谁（原始编号）
print(batch["modalities"])        # 0=RGB，1=IR：前 32 个是 0，后 32 个是 1
```

你会看到：

类别数是 296，每轮 633 步。
images 的形状是 [64, 3, 256, 128]。
modalities 前一半全是 0，后一半全是 1。

这就是 B 每一步训练拿到的东西。

3. B 的训练代码大概长这样

```python
loader, sampler, dataset, label_map = get_train_loader("M0")
model = 建模型(类别数=len(label_map))

for epoch in range(24):
    sampler.set_epoch(epoch)                              # 告诉发牌机第几轮
    sampler.set_gray_probability(1.0 if epoch < 6 else 0.0)  # 前 6 轮 RGB 转灰度
    for batch in loader:
        输出 = model(batch["images"])
        损失 = 计算损失(输出, batch["labels"])
        ...

```

第一次调用时如果还没准备数据，会自动完成第 1 步中的 1–3（不做接口检查）；审计未通过会报错，不给出训练数据。最终阶段用 `get_train_loader("M0", final=True)`（train+val 395 身份）。其余参数原样传给 `build_train_loader`，例如 `num_workers=8`、`seed=42`。

### B / E：特征导出与评价

```python
from src.data import get_eval_dataset, collate_eval
from torch.utils.data import DataLoader

query = get_eval_dataset("val", modality="ir")           # IR query 候选
gallery_pool = get_eval_dataset("val", modality="rgb")   # RGB gallery 候选（Single-shot 抽样由 E 负责）
loader = DataLoader(query, batch_size=128, collate_fn=collate_eval)
```

评价 batch 含 `images, identities, cameras, modalities, row_indices, paths`，**不含训练标签**。`cameras=[1, 2]` 可筛摄像头（Indoor-search）。

## 2. 训练循环怎么接

```python
from src.config import load_config
cfg = load_config("M0")
loader, sampler, dataset, label_map = get_train_loader(cfg, num_workers=4)

for epoch in range(cfg["training"]["epochs"]):              # 0-based
    stage1 = epoch < cfg["training"]["stage1_epochs"]
    sampler.set_gray_probability(1.0 if stage1 else 0.0)    # 第一阶段 RGB 全部转灰度；M0 第二阶段为 0
    sampler.set_epoch(epoch)
    for step, batch in enumerate(loader):
        images = batch["images"].cuda(non_blocking=True)   # [2PK, 3, 256, 128]
        labels = batch["labels"].cuda()                     # [2PK] 连续训练标签
        ...
```

batch 字典（各字段逐行对齐）：

| 字段 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `images` | `[2PK,3,256,128]` float32 | ImageNet 归一化后的图像 |
| `labels` | `[2PK]` int64 | 连续训练标签（分类头使用） |
| `identities` | `[2PK]` int64 | 原始 SYSU 身份 |
| `cameras` | `[2PK]` int64 | 摄像头 1–6 |
| `modalities` | `[2PK]` int64 | 0=rgb，1=ir |
| `gray` | `[2PK]` bool | 该行是否已转灰度（只可能出现在 RGB 行） |
| `row_indices` | `[2PK]` int64 | manifest 行号 |
| `paths` | `list[str]` | 相对数据根目录的路径 |

**排列契约**（作者 MSEL/DCL 依赖，`check_pmt_layout(batch, P, K)` 可校验）：前 P·K 行为 RGB 来源，后 P·K 行为 IR；两半身份顺序相同；每个身份在每半连续 K 行。因此 `model(batch["images"])` 后可直接 `chunk(2, 0)`，与作者 `torch.cat([input1, input2])` 等价；`as_pmt_tuple(batch)` 返回作者格式 `(input1, input2, label1, label2)`。我们用作者的 MSEL、DCL、Triplet 在本接口的 batch 上做过前向/反向检查，损失有限、梯度正常。

**每 epoch 步数**：`max(#RGB, #IR) // (P·K)`，与作者 `IdentitySampler` + `drop_last` 一致。Q 对照（P=16, K=2）步数和 batch 大小与 M0 相同。

**增强**：等价于作者 `transforms.py`。RGB：翻转 →（可选）灰度 → 归一化 → RandomErasing；IR：翻转 → ColorJitter 或 GaussianBlur 二选一 → 归一化 → RandomErasing。如需替换，向 `SYSUTrainData` 传 `transforms={"rgb": f, "ir": g}`，`f(img, gray=bool)` 返回张量。

## 3. 复现与断点恢复

- 每个 epoch 的抽样只由 `(seed, epoch)` 决定；同一 seed 在任何机器、任意 `num_workers` 下得到相同 batch 张量。
- 身份/图片抽样与灰度/增强是两条独立随机流：改变灰度概率（M0 ↔ M1 ↔ F）不会改变抽到的图片或其增强种子，只改变灰度标记。
- 数据增强不消耗全局 RNG，不影响模型初始化和 dropout。
- 恢复：保存 `sampler.state_dict()`，恢复时 `sampler.load_state_dict(state)`。保存前把 `start_batch` 设为已完成步数：`sampler.set_epoch(epoch, start_batch=step + 1); state = sampler.state_dict()`。

## 4. 平滑灰度过渡（C 使用）

`sampler.set_gray_probability(fn)`，其中 `fn(global_step) -> [0, 1]`，`global_step = epoch * sampler.batches_per_epoch + 步内序号`（从 0 开始）。C 的调度把"第二阶段起点后的步数 r"换算为 `global_step - stage1_epochs * batches_per_epoch`。实际灰度数量见 `batch["gray"].sum()`，应写入训练日志。IR 行永远不转灰度。

## 5. 约定

| 项目 | 规则 |
| --- | --- |
| 模态 | cam1/2/4/5 → `rgb`（编码 0），cam3/6 → `ir`（编码 1）；RGB 转灰度后仍记为 `rgb` |
| 划分 | 只用官方 `exp/{train,val,test}_id.txt`，按身份划分；开发 296/99/96 |
| 训练标签 | 原始身份升序编号 0..C-1，只含同时有 RGB 与 IR 的身份；开发 `("train",)`，最终 `("train", "val")` |
| 图像 | 读入后转 3 通道，LANCZOS 缩放到 256×128（高×宽），与作者 `process_sysu.py` 一致 |
| 损坏图片 | 记录在审计中，采样器与标签映射自动排除 |
| 作者的 `.npy` | Kaggle 包里的 `train_*_resized_*.npy` 是作者预处理的 train+val 合并数组，不使用 |

## 6. 产物

| 文件 | 内容 |
| --- | --- |
| `artifacts/manifests/manifest.csv` | 每图一行：`relative_path, identity, camera, modality, split, width, height, channels, file_bytes, sha256`，另含解码结果与亮度/通道统计 |
| `artifacts/manifests/splits/*_ids.txt` | 官方 train/val/test 身份 |
| `artifacts/manifests/label_map_dev.csv` / `label_map_final.csv` | train(296) / train+val(395) 身份 → 连续标签 |
| `artifacts/manifests/data_audit.md` / `.json` | 审计结论（PASS/FAIL）、数量差异、重复、尺寸、模态证据、磁盘占用 |
| `artifacts/manifests/dataset_info.json` | 来源、日期、根目录、各文件 sha256、代码版本 |
| `artifacts/results/data_check/` | `batch_example.json`、`batch_grid.png`、`sampler_stats.json`、`check_report.md` |
| `data/dataset_location.json` | 自动下载时记录的数据位置 |

以上都不入 Git。manifest 中的路径相对数据根目录；不同机器的下载副本可比较 `dataset_info.json` 中的 sha256。

## 7. 依赖与限制

- manifest/审计：Python ≥ 3.10、numpy、Pillow。训练/评价接口：另需 torch、torchvision。自动下载：kagglehub。正式版本由 B 锁定。
- 已在合成数据上验证全部代码路径；真实数据的数量、重复、尺寸等以审计报告为准。
- Katana 计算节点可能不能联网：先在登录节点或本机运行 `prepare_data.py` 下载，或用 `--root` 指向共享数据目录。
- 采样器每 epoch 独立抽身份，不保证每个身份每 epoch 恰好出现一次（与作者实现相同）；覆盖率见 `sampler_stats.json`。
- `preload=True` 会把训练图缩放后放入内存（约 3 GB），只在内存充足时使用。
