# B：PMT baseline 实现与运行

负责人：B。实现日期：2026-10-25。代码实现、自检、真实训练分别记录；完成自检不代表完成本周训练交付。

## 当前状态

- 已实现：共享 ViT-B/16、重叠 patch、BN neck、ID/PHT/MSEL/DCL、M0 两阶段、V0 基础对照、AdamW、warmup/cosine、完整恢复、验证集特征导出。
- 本地 Python 3.11.15 / PyTorch 2.14.1 / torchvision 0.29.1，macOS ARM；沙箱内检测不到 CUDA/MPS，沙箱外确认可用 Apple MPS，统一内存 16GB；无 CUDA。完整依赖见 `requirements-lock.txt`，这是已测试的本地环境锁，不是已验证的 Katana CUDA 环境。
- A+B 测试 48 项通过。上游对照：主干前向、27 组损失数值/梯度、24 个 epoch 的各参数组学习率通过。小模型真实数据接口的模拟图像测试覆盖中途恢复及阶段边界，恢复后参数与连续运行逐位一致。
- ImageNet 初始化已下载并校验，完整 ViT-B 严格加载及前反向已通过；本机真实 SYSU 数据审计及接口检查均 PASS（45,863 图片、296 train IDs、633 steps/epoch）；尚无完整 24-epoch 训练或检索指标。
- 本地真实训练联调已完成：`local_M0_mps_smoke_seed17`，先 2 步、恢复后再 1 步，总计 3/633 个第一阶段 batch；FP32 MPS＋激活检查点，P=8/K=4 未变。日志、权重和 summary 已保存，status=`partial`，没有跑到第二阶段。
- 正式训练目标为 Katana。SSH 别名 `katana` / `kdm` 已验证可用。远端项目为 `/srv/scratch/z5535967/9444_project`，独立环境为 `/srv/scratch/z5535967/venvs/pmt-py311`。H200 设备探测作业 `9609590` 成功（k201，驱动 610.43.02）；PyTorch 2.14.1+cu130 已安装，实际训练检查待运行。

## 方法与来源核对

核对 [PMT 论文](https://arxiv.org/abs/2212.00226) 与 [作者公开 PyTorch 实现](https://github.com/hulu88/PMT/tree/0b31da97b06428473bc4a1134b322ea226340818)，固定 commit `0b31da97b06428473bc4a1134b322ea226340818`。本组按算法和接口独立实现；没有把上游模型或损失文件整份放入仓库。作者仓库没有声明统一许可证，不能据此认定可任意再分发；已有 A 增强代码的登记仍保留。

| 项目 | 核对及本组行为 |
| --- | --- |
| 框架 | 论文报告 MindSpore/RTX3090；本组跟随作者公开 PyTorch 路径，不声称数值复现 MindSpore |
| 输入 | A 提供 `[64,3,256,128]`；RGB/IR 各 32，P=8、K=4；前半 RGB，两半身份排列相同 |
| Backbone | ViT-B/16，12 层、12 heads、768 维；patch=16、stride=12、无 padding；实际卷积输出网格 21×10 |
| 位置编码 | 单 CLS；224×224 ImageNet 的 14×14 patch 位置编码用 bilinear、align_corners=False 插值到 21×10 |
| Dropout | token/MLP/projection .03；attention 0；逐层 stochastic depth 0→.1；LayerNorm eps=1e-5 |
| 分类与检索 | CLS 特征→BN neck→无 bias 分类头；BN bias 冻结。度量损失使用 BN 前未归一化特征；导出 BN 后 L2 特征，768 维 |
| ID | 两个模态分别计算 mean CE 后相加，保持作者损失尺度 |
| 第一阶段 | epoch 1–6（内部 0–5）：RGB 全转灰度，IR 不变；ID＋两个模态的各自 hard-triplet 之和 |
| 第二阶段 | epoch 7–24（内部 6–23）：RGB/IR；ID＋全 batch hard-triplet＋0.5 MSEL＋0.5 DCL |
| 论文/代码差异 | arXiv v1 Eq.12 未列 ID；作者 `train.py` 的第二阶段明确包含 ID。本组保留 ID，跟随公开代码 |
| MSEL | 按 Eq.5–7，对每个 anchor 比较 K−1 个非自身同模态正样本均距与 K 个跨模态正样本均距，平方后双向平均 |
| DCL | 两模态同身份中心→图片距离；选小于该中心负样本平均距离的负样本，正均距/选中负均距 |
| 优化器 | AdamW，base lr=3e-4，weight decay=1e-4；patch conv 与 Transformer blocks 的 lr×0.5，其他参数×1 |
| 学习率 | warmup=3 epochs，initial/minimum=base lr×.01；按作者在 epoch 开始调用 one-based step；cosine 原点不减去 warmup |
| 训练范围 | 开发仅 train 296 身份；val 99 用于 E 评价；不使用 test 调参。区别于作者 train+val 395 身份设置 |
| V0 | 同模型、RGB/IR、ID＋全局 hard-triplet，始终无灰度/MSEL/DCL；不称完整 PMT |

数值边界处理：所有度量损失在 FP32 计算，欧氏距离平方下限 1e-12。Triplet/MSEL 显式排除自匹配。DCL 保留作者正距离 >1e-6 的筛选；若某中心没有严格低于负均距的负样本，回退到该中心全部负样本；分母至少 1e-6；全部正距离退化时分子为 0。这些退化处理防止原代码空集合 NaN，已用退化输入测试；正常随机输入的损失/梯度与作者一致。没有把退化保护称为新方法。

## 环境与初始化

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/prepare_pretrained.py
python scripts/prepare_data.py
python -m pytest tests/test_data.py tests/test_baseline.py -q
```

初始化来自作者 README 链接的 `jx_vit_base_p16_224-80ecf9dd.pth`，下载脚本检查发布文件名中的 SHA256 前缀并记录完整 checksum。加载必须严格匹配全部 backbone 参数，不能静默跳过缺失权重。分类头按本组训练身份重新初始化。完整训练入口不允许隐式随机初始化。

Katana 应重新建立 Linux Python 环境，不可复制 macOS `.venv`。选用能在获配 GPU/驱动上工作的 CUDA PyTorch，在计算节点确认 `torch.cuda.is_available()`；PBS 会记录实际设备和 `pip freeze`。本地 lock 中未包含 Linux CUDA 传递依赖；新安装的 Linux/CUDA 环境已另存 `requirements-katana-lock.txt`。

## 训练、恢复与交付

```bash
# 先用真实数据做 2 个 batch，之后从同一状态继续；不改变24 epoch配置。
python -m src.train --experiment M0 --run-id dev_M0_seed17 --device cuda --num-workers 4 --max-steps 2
python -m src.train --run-id dev_M0_seed17 --device cuda --num-workers 4 --resume checkpoints/dev_M0_seed17/last.pt
```

首次训练前必须有预训练文件和 A 的审计通过的数据；训练器不在计算节点自动下载。可用 `--root` 指定已有数据。已有 run-id 不覆盖；恢复读取 checkpoint 内配置、种子与标签映射，并检查当前 manifest checksum。AMP 仅在 CUDA 开启，CPU/MPS 使用 FP32 并明确记录；恢复不允许切换 AMP 模式。可选 `--activation-checkpointing` 重算 Transformer block 以降低激活内存，保持 batch/P/K 和损失不变，开关保存在配置；已验证随机层下损失、梯度与 RNG 一致。每 100 batch、epoch 末和 `--max-steps` 退出时保存 `last.pt`，可用 `--save-every` 调整。

- `artifacts/results/<run_id>/config.json`：实际合并配置。
- `environment.json` / `resume_<step>.json`：环境、设备、代码 commit/diff、源码 hash、数据和预训练 checksum。
- `train.jsonl`：每 batch 的 stage、epoch、global_step、全部 loss 原值/加权值、各组 lr、灰度概率/数量、耗时与峰值 CUDA 显存。
- `summary.json`：`partial/completed`、下一 epoch/batch、实际步数、权重路径/checksum。只有 24 epochs 全部结束才标 completed。
- `checkpoints/<run_id>/last.pt` 与 `.sha256`：模型、optimizer、scheduler、scaler、sampler、全部可用 RNG、阶段及来源状态。只加载自己信任的训练 checkpoint，因为完整 RNG 恢复需要 Python 对象反序列化。

本组增强会设随机种子，训练器隔离数据迭代的 CPU/CUDA/MPS RNG，避免改变模型 dropout 随机流。CPU 单/多进程数据加载下已验证逐位恢复；MPS 已实际加载真实断点并推进下一 batch；CUDA 的跨设备/跨版本逐位一致性仍需实机核验。恢复时日志会截到 checkpoint 的 global_step，避免上次未保存步骤在日志中重复。

## 导出给 E

```bash
python -m src.export_features --checkpoint checkpoints/dev_M0_seed17/last.pt --split val --modality ir --output artifacts/results/dev_M0_seed17/query_val.npz --device cuda
python -m src.export_features --checkpoint checkpoints/dev_M0_seed17/last.pt --split val --modality rgb --output artifacts/results/dev_M0_seed17/gallery_pool_val.npz --device cuda
```

输出 `features, identities, cameras, modalities, paths, row_indices, metadata_json`，可用 `np.load(path, allow_pickle=False)` 读取。特征为 BN 后 L2 归一化、默认 768 维；距离为 `-(query @ gallery.T)`，不做 flip augmentation。metadata/同名 JSON 记录训练进度、split、seed、checkpoint checksum。RGB 输出是完整候选池，E 再生成固定 10 组 single-shot gallery；不把候选池直接当 single-shot gallery。默认只允许内部 val。输出存在则拒绝覆盖。

## Katana

[官方指南](https://docs.restech.unsw.edu.au/using_katana/troubleshooting_navigation/) 区分 KDM 文件传输主机与 Katana 登录/作业主机。当前模板 `scripts/katana/train.pbs` 请求 1 GPU、4 CPU、32GB 系统内存、2 小时，队列按站点默认；GPU 显存不能由 `mem=32gb` 推断。需在账号实际可用资源上验证。

```bash
ssh z5535967@katana.restech.unsw.edu.au
cd /实际项目目录/9444_project
# 数据、权重、CUDA虚拟环境已准备后，先提交2步检查。
qsub -v RUN_ID=dev_M0_seed17,MAX_STEPS=2 scripts/katana/train.pbs
qstat -u z5535967
# 检查日志及summary后继续同一run；按获准时限调整walltime。
qsub -v RUN_ID=dev_M0_seed17,MAX_STEPS=0,RESUME=1 scripts/katana/train.pbs
```

可通过 `PMT_PYTHON` 指定远端 Python 绝对路径，`PMT_ENV_SETUP` 指向加载模块的 shell 文件，`SYSU_MM01_ROOT` 指向共享数据，`PMT_PRETRAINED` 指向初始化文件。本地 MPS 短跑只用于联调，CUDA 正式实验应从同一 ImageNet 初始化新建 run，不能将 FP32 MPS 断点伪装成同一 AMP 运行继续。`ACTIVATION_CHECKPOINTING=1` 可让新建 CUDA run 也启用激活检查点。这些路径需使用实际远端值；模板不猜测模块版本或项目目录。没有在登录节点运行训练。

## 本地验证记录

初始化完整 SHA256：`80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c`。完整 ViT-B（296 类，86,038,272 参数）已用该权重严格加载，对模拟 `[8,3,256,128]` 输入执行第二阶段损失反向，patch 卷积梯度有限。此项是实现检查，不是 SYSU 训练。证据：`artifacts/results/baseline_checks/full_backbone.json`、`reference.json`、`local_validation.json`。

本地真实短跑证据：`artifacts/results/local_M0_mps_smoke_seed17/train.jsonl`、`summary.json`；权重 `checkpoints/local_M0_mps_smoke_seed17/last.pt`，SHA256 `6930aa9f481fa41c6f3f698f25dce2e841ab640f0c52bfd29d7b86732ce9e0e2`。三步 total loss 为 22.5227、21.7838、21.4705（不同 batch，不能据此判定收敛或检索性能）。这份权重仅供联调，不代替正式 baseline。已导出 val IR query 1,980 条和 RGB 候选 1,974 条（各 99 身份），均为 768 维；路径为同一 run 目录下 `query_val.npz` 和 `gallery_pool_val.npz`，逐行原始身份/摄像头/模态/路径与 manifest 对齐检查 PASS，见 `export_validation.json`。尚未计算检索指标。

## 上游核验重现

```bash
git clone https://github.com/hulu88/PMT.git /tmp/pmt-reference
git -C /tmp/pmt-reference checkout 0b31da97b06428473bc4a1134b322ea226340818
python scripts/check_pmt_reference.py /tmp/pmt-reference
```

对照报告写入 `artifacts/results/baseline_checks/reference.json`；仅读取上游 checkout。测试入口与结果解释见 `tests/README.md`。


## Katana 本次提交（2026-10-25）

SSH 已配置，源代码、45,863 张图片/官方划分和已校验初始化已部署至 `/srv/scratch/z5535967/9444_project`。未使用 Kaggle 的 train+val `.npy` 数组。专用环境 `/srv/scratch/z5535967/venvs/pmt-py311`，加载 `python/3.11.3`；Linux/CUDA 依赖见 `requirements-katana-lock.txt`。

- H200 设备探测 `9609590`：成功，节点 k201，驱动 610.43.02。它只证明设备可用，不是训练结果。
- GPU preflight `9609832`：申请 L40S / 4 CPU / 32GB / 30分钟，队列 `csegpu2`。先跑测试与真实数据审计，再进行两步训练；截至本次提交记录仍排队。
- 完整训练 `9609976`：同一 run `dev_M0_seed17_katana_20261025`，L40S / 4 CPU / 32GB / 最长12小时，队列 `csegpu12`；依赖 `afterok:9609832`，因此当前 H 状态表示等待检查成功。检查失败不会启动。
- 完整作业从两步 checkpoint 恢复，目标仍为总计24 epochs（不是额外24）；完成后自动导出 val IR/RGB 特征。AMP 与激活检查点启用；预留时限不是预估训练时长。

```bash
ssh katana 'qstat -u z5535967'
ssh katana 'tail -n 30 /srv/scratch/z5535967/9444_project/pmt-preflight.o9609832'
ssh katana 'tail -n 30 /srv/scratch/z5535967/9444_project/pmt-M0-seed17.o9609976'
```

训练进度证据位于远端 `artifacts/results/dev_M0_seed17_katana_20261025/`；权重在 `checkpoints/dev_M0_seed17_katana_20261025/last.pt`。排队不表示已完成训练，本地3步联调与 Katana 正式 run 分开。提交登记：本地 `artifacts/results/katana_deployment.json`。
