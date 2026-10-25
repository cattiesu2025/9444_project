# COMP9444 · RGB–红外行人重识别

基于 **PMT（Progressive Modality-shared Transformer）**，研究平滑灰度过渡（S）与局部特征监督（L），在 SYSU-MM01 上完成受控消融与检索评价。

> 数据接口（A）与 PMT baseline 实现（B）已具备，A+B 共 50 项自检通过。B 已在 Katana 完成 M0 seed17 的 24 epochs 训练并导出验证集特征，配置、日志和权重均已保存；E 的检索评价待执行，目前没有检索指标。详见 [B 的方法与运行说明](docs/baseline.md)。

## 项目入口

- [完整设计与研究问题](010_Project_Design.md)
- [本周进度与课堂汇报](011_Weekly_Checkpoint_Deliverables.md)
- [工程架构与接口](docs/architecture.md)
- [数据接口说明（A）](docs/data_interface.md)
- [PMT baseline 说明与 Katana 运行（B）](docs/baseline.md)
- [代码来源与复现差异登记](ATTRIBUTION.md)
- [整合 Notebook（baseline 已写入）](submission/project.ipynb)
- [Notebook 结构与评分覆盖](submission/README.md)
- [课程评分标准](COMP9444_Group_Project_Marking_Criteria.pdf)

## 研究路线

输入 IR query，在 RGB gallery 中检索同一身份，以 mAP 和 Rank-1/5/10 评价。训练分类准确率不能替代检索指标。

| 实验 | 方法 | 负责人 | 计划训练种子数 |
| --- | --- | --- | --- |
| V0 | 基础 ViT + ID / global triplet | B | 1 |
| M0 | 完整 PMT，两阶段 + PHT/MSEL/DCL | B | 3 |
| M1 | PMT + S：第二阶段平滑降低灰度概率 | C | 3 |
| M2 | PMT + L：水平区域局部身份监督 | D | 3 |
| M3 | PMT + S + L | D（接入 C 的模块） | 3 |
| F | 同窗口固定灰度概率，匹配期望暴露量 | C | 1 |
| Q | 固定 batch 大小的 P/K 采样对照 | A | 1 |

M0–M3 测试均只用全局特征，保持检索维度一致。S/L 是待验证扩展，尚不宣称性能提升或首次提出。

## 目录

```text
configs/
  base.json                 # 共享开发配置（待核对）
  experiments/*.json         # V0/M0–M3/F/Q 的差异配置
src/
  config.py                 # 配置加载、合并与检查（已实现）
  data.py                   # A：数据查找/下载、manifest、审计、PMT 兼容 P/K 采样（已实现）
  model.py, losses.py        # B：PMT 主干、原损失（已实现）
  schedules.py              # C：灰度调度（接口占位）
  local_head.py             # D：局部监督（接口占位）
  train.py, training.py     # B：训练/完整恢复（已实现）
  export_features.py        # B：BN后归一化特征与元数据导出
  evaluate.py, robustness.py # E：评价/扰动（接口占位）
scripts/inspect_config.py    # 可直接运行的配置检查
scripts/prepare_data.py      # A：一条命令准备数据（查找/下载、审计、接口检查）
scripts/build_manifest.py    # A：manifest、划分、标签映射与数据审计
scripts/check_data.py        # A：真实 batch、接口契约与采样统计
docs/                       # 架构、接口与协作约定
tests/                      # 后续模块正确性检查说明
data/                       # 本地数据，不入 Git
checkpoints/                # 预训练和训练权重，不入 Git
artifacts/manifests/         # 本地数据/划分/gallery 清单，不入 Git
artifacts/results/          # 本地运行证据，不入 Git
notebooks/                  # 开发用 notebook（提交前清空输出）
submission/                 # 最终 Notebook、报告与展示的交付说明
```

## 开始使用

```bash
git clone https://github.com/cattiesu2025/9444_project.git
cd 9444_project
python3 -m venv .venv
source .venv/bin/activate
python scripts/inspect_config.py --experiment M0
python scripts/inspect_config.py --experiment M3
```

以上 clone 命令适用于框架提交并推送后。已有本地项目无需再次 clone。配置检查仅依赖 Python 标准库（Python ≥3.10），输出合并配置，不下载数据、不训练模型。`--experiment` 支持表中所有实验 ID。

本地已验证环境见 `requirements-lock.txt`（Python 3.11.15、PyTorch 2.14.1，macOS ARM）；Katana CUDA 环境需在实际计算节点另行验证和锁定。模型/损失、训练/恢复与导出命令见 [baseline.md](docs/baseline.md)，PBS 模板为 `scripts/katana/train.pbs`。测试通过不表示已完成真实训练。

## 数据与评价约定

数据由 A 的数据接口自动准备：有数据就直接用，没有就从课程 Kaggle 镜像下载，然后生成 manifest、审计并检查训练接口。详见 [数据接口说明](docs/data_interface.md)。

```bash
pip install numpy pillow torch torchvision kagglehub
python scripts/prepare_data.py              # 查找/下载数据 → manifest 与审计 → 真实 batch 检查
python -m pytest tests/test_data.py -q      # 合成数据自检，不需要真实数据
```

B 取训练数据只需：

```python
from src.data import get_train_loader
loader, sampler, dataset, label_map = get_train_loader("M0")   # final=True 用 train+val 395 身份
```

- RGB cameras：1/2/4/5；IR cameras：3/6。
- 开发：296 身份训练、99 身份验证；96 身份测试集保留，不用于调参。
- 最终：配置冻结后合并 395 个训练身份，从同类 ImageNet 初始化重新训练；最终配置尚待开发完成后建立。
- 主协议：IR→RGB、All-search、Single-shot；cam3 query 排除 cam2 gallery，采用 SYSU 身份级 CMC。
- 固定 gallery 种子 0–9，各方法共用索引；主实验训练种子 17/42/2026。
- 每个训练种子先平均 10 次 gallery 评价，再跨训练种子汇总均值与样本标准差。

评价器通过手算矩阵与参考实现核验后才能支撑正式实验。文件数、实际划分与所有指标以执行证据为准。

## 分工与协作

| 成员 | 负责模块 | 首要交付 |
| --- | --- | --- |
| A | 数据、划分、采样、EDA | 核验后的真实 batch 与标签映射 |
| B | 完整 PMT、损失、训练/恢复、特征导出 | 本组实际训练的 baseline 与配置 |
| C | 平滑过渡和固定概率控制 | 独立调度模块及边界/恢复自检 |
| D | 局部监督、M2/M3 集成 | 非正方形 token 网格与梯度自检 |
| E | 评价、gallery、鲁棒性、汇总 | 手算指标核验及真实 baseline 复评 |

课堂 checkpoint 是 tutor 询问进度的口头交流，不要求现场检查代码；进度、问题和下一步统一记录在 [本周进度文档](011_Weekly_Checkpoint_Deliverables.md)。代码按项目功能组织，不按课堂汇报或成员另建目录。`checkpoints/` 指训练权重和恢复状态，与课堂 checkpoint 无关。

主线为 **A 数据接口 → B 训练与导出 → E 评价**。C/D 可独立准备模块，在 B 固定版本上分别接入；M3 由 D 集成。每人维护自己的实现、检查和技术说明。

建议使用 `feat/data`、`feat/pmt`、`feat/smooth-transition`、`feat/local-supervision`、`feat/evaluation` 分支。共享接口变更写入架构文档，每次实验记录代码 commit、完整配置、种子、数据/权重 checksum、设备与原始指标。不要提交完整数据集、权重、凭据或缓存；需要分享的小型结果表经审阅后单独加入 Git。

## 交付与进度

- [x] 项目目录、配置模板与 README。
- [ ] 锁定作者代码版本、许可证与训练依赖。
- [x] 数据接口与采样器（合成数据自检通过）。
- [x] 本机真实数据审计与真实 batch 检查。
- [x] B 的模型、损失、训练恢复和导出实现；真实数据 3 步短跑。
- [ ] 评价器正确性验证（E）。
- [x] 本组实际训练的 M0 baseline（seed17，24 epochs）。
- [ ] S/L 模块、M0–M3 消融与 F/Q 控制。
- [ ] 最终冻结配置训练、鲁棒性与错误分析。
- [x] 整合 Notebook 结构与 baseline 章节（其余实验/评价仍待补齐）。
- [ ] 最终完整 Notebook、正文不超过 4 页的报告、全员展示。

内部目标为 2026-11-15 形成可提交版本，2026-11-18 是内部暂定截止日；正式日期以课程公告为准。

## 来源

研究路线依据本仓库设计文档；主要外部来源为 [PMT 作者仓库](https://github.com/hulu88/PMT)、[PMT 论文](https://arxiv.org/abs/2212.00226) 和 [SYSU-MM01](https://github.com/wuancong/SYSU-MM01)。A 的数据增强已有第三方来源；B 按固定作者版本核对并独立实现。具体 commit、许可证及修改记录见 [ATTRIBUTION.md](ATTRIBUTION.md)。
