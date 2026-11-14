# COMP9444 · RGB–红外行人重识别

基于 **PMT（Progressive Modality-shared Transformer）**，研究平滑灰度过渡（S）与局部特征监督（L），在 SYSU-MM01 上完成受控消融与检索评价。

> 数据接口（A）、PMT baseline（B）和评价器（E）已完成并通过自检。四个 baseline（V0、M0、AGW、CAJ）seed17 均已在 Katana 训练，并在内部验证集上用同一组 10 个固定 gallery 评价：M0 mAP 34.4 / Rank-1 32.1，V0 33.0 / 29.0，AGW 30.2 / 26.9。CAJ 按作者代码训练时发生特征塌缩（7.0% Rank-1），结果无效；修正一处负样本加权后重跑（`CAJ_fix`），达到 mAP 34.4 / Rank-1 33.1，与 M0 持平。原因和修正见[本周计划 §4.7](012_Weekly_Checkpoint_Deliverables.md)。详见[评价说明](docs/evaluation.md)和[整合 Notebook](submission/project.ipynb) §5。
>
> **扩展实验（2026-11-07）：** M1、F、M2、M3 seed17 均已训练并评价。S 未见明确收益（M1 mAP −0.3，F 与 M0 持平）；M2、M3 在 10 个 gallery 上一致略低于 M0（mAP −0.6 / −1.0），均为单种子、差距小于 1 个点。L 的补充实验按区域数 q × 损失权重 β 补齐 4 组（见下表），S 窗口探索不做（11/07 决定）。

## 项目入口

- [完整设计与研究问题](010_Project_Design.md)
- [本周计划：评价器与四个 baseline 对比](012_Weekly_Checkpoint_Deliverables.md)（上周见 [011](011_Weekly_Checkpoint_Deliverables.md)）
- [工程架构与接口](docs/architecture.md)
- [数据接口说明（A）](docs/data_interface.md)
- [PMT baseline 说明与 Katana 运行（B）](docs/baseline.md)
- [检索评价协议与运行（E）](docs/evaluation.md)
- [代码来源与复现差异登记](ATTRIBUTION.md)
- [整合 Notebook（baseline 与四个 baseline 对比已写入）](submission/project.ipynb)
- [Notebook 结构与评分覆盖](submission/README.md)
- [课程评分标准](COMP9444_Group_Project_Marking_Criteria.pdf)

## 研究路线

输入 IR query，在 RGB gallery 中检索同一身份，以 mAP 和 Rank-1/5/10 评价。训练分类准确率不能替代检索指标。

| 实验 | 方法 | 负责人 | 计划种子数 | 当前状态（11/08） |
| --- | --- | --- | --- | --- |
| V0 | 基础 ViT + ID / global triplet | B | 1 | seed17 已训练并评价 |
| M0 | 完整 PMT，两阶段 + PHT/MSEL/DCL | B | 3 | seed17、42、2026 均训练完成并导出特征（11/07），seed42/2026 待 E 评价 |
| M1 | PMT + S：第二阶段平滑降低灰度概率 | C | 3 | seed17、42、2026 均训练完成并导出特征（11/07），seed42/2026 待 E 评价 |
| M2 | PMT + L：水平区域局部身份监督（q=2, β=0.5） | D | 3 | seed17 完成；seed42/2026 未跑 |
| M3 | PMT + S + L | D（接入 C 的模块） | 3 | seed17 完成；seed42/2026 未跑 |
| F | 同窗口固定灰度概率，匹配期望暴露量 | C | 1 | seed17 已训练并评价 |
| Q | 固定 batch 大小的 P/K 采样对照（P=16,K=2 对比 M0 的 P=8,K=4，总图片数和步数相同） | A | 1 | 训练完成并导出验证集特征（11/07），待 A 分析 |
| L3_B050 | L 探索：q=3（等高三条带，大致对应头/上身/下身）, β=0.5 | D | 1 | 训练完成并导出验证集特征（11/07），待 E 评价 |
| L3_B025 | L 探索：q=3, β=0.25（两个变体的组合） | C | 1 | 训练完成并导出验证集特征（11/07），待 E 评价 |
| L2_B025 | L 探索：q=2, β=0.25（原计划的 β 对照） | D | 1 | 训练完成并导出验证集特征（11/07），待 E 评价 |
| L3_B050_BE | L3_B050 + 与局部区域对齐的横条遮挡增强（p=0.5） | E（B 训练） | 3 | 代码与配置就绪（11/14），待训练 |
| M0_BE | M0 + 同样的横条遮挡增强（可选对照） | E（B 训练） | 3 | 代码与配置就绪（11/14），待训练 |
| AGW / CAJ_fix | 外部 CNN baseline | — | 1 | seed17 已训练并评价 |

M2、L2_B025、L3_B050、L3_B025 构成 q（2/3）× β（0.5/0.25）的 2×2，可分别归因区域数和损失权重，并看两者能否叠加。三个新配置只改配置、不改代码，均为 seed17 单种子探索，报告中标为 exploratory，不与主矩阵种子合并求均值。三次训练正好是设计文档允许的辅助探索上限（3 次），S 窗口 1/6 不再占用。

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
  schedules.py              # C：灰度调度（已实现）
  local_head.py             # D：局部监督（已实现并接入训练/导出）
  train.py, training.py     # B：训练/完整恢复（已实现）
  export_features.py        # B：BN后归一化特征与元数据导出
  evaluate.py               # E：固定 gallery、SYSU 指标与跨种子汇总（已实现，待真实特征复评）
  robustness.py             # E：固定遮挡/模糊扰动下的鲁棒性（已实现并自检，M0/M1/F seed17 已运行，结果见 docs/evaluation.md）
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

本地已验证环境见 `requirements-lock.txt`（Python 3.11.15、PyTorch 2.14.1，macOS ARM）；Katana CUDA 环境需在实际计算节点另行验证和锁定。模型/损失、训练/恢复与导出命令见 [baseline.md](docs/baseline.md)。PBS 模板：ViT 系列（V0/M0/M1/F/M2/M3 及 L 探索配置）用 `scripts/katana/train.pbs`，AGW/CAJ/CAJ_fix 用 `scripts/katana/train_agw.pbs`（`EXPERIMENT=` 指定）；加 `PILOT_EXPORT=1` 会在短训练后把特征导出到 `<run>/pilot/`，用于开训前的评价检查。测试通过不表示已完成真实训练。

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

| 成员 | 负责模块 | 已完成 | 未完成（截至 11/08） |
| --- | --- | --- | --- |
| A | 数据、划分、采样、EDA | 数据接口、审计、采样器、真实 batch 检查；自然遮挡子集初标（验证集无符合样例，改在测试集 IR query 上标注 224 张，见 [annotations/](annotations/README.md)） | EDA 要出 5 组图表（设计 §3.4）：E1 数据构成、E2 身份覆盖、E3 图像质量、E4 RGB/IR 配对示例、E5 划分与质量审计；自然遮挡子集：复核 4 张待复核样本，并把 [annotations/README.md](annotations/README.md) 中“配置已于 10/31 冻结”改为实际情况（标注于 11/07 完成，当时无任何测试集结果；配置尚未冻结，子集只在冻结后对最终模型做一次描述性评价）；Q 结果分析（覆盖身份数、重复采样率、与 M0 对比；训练已由 B 跑完，结果随其他运行上传 Drive） |
| B | 完整 PMT、损失、训练/恢复、特征导出；Katana 运行与 Drive 上传 | M0 三个种子、V0 seed17；训练/恢复/导出；Katana 上已跑完 M0/M1 seed42/2026、三个 L 探索变体和 Q（共 8 次，均已导出验证集特征） | 上述 8 次运行结果上传 Drive；冻结后提交 M2/M3 seed42/2026；最终阶段代码（395 身份训练、测试集特征导出）与最终 M0 |
| C | 平滑过渡和固定概率控制 | S 模块；M1 三个种子、F seed17；M1/F 负结果分析；L3_B025 配置与训练 | — |
| D | 局部监督、M2/M3 集成 | 局部监督模块；M2、M3 seed17；L3_B050、L2_B025 配置（PR #9），两者已训练完成 | 冻结后确认 M2/M3 seed42/2026 用哪组 q/β（若改配置需提供对应配置文件） |
| E | 评价、gallery、鲁棒性、汇总 | 评价器核验；9 种扰动；所有 seed17 模型已评价；局部特征拼接探索 | **当前最急**：L3_B050/L2_B025/L3_B025 评价、鲁棒性和局部特征拼接（拼接脚本已支持任意 q），以及 M0/M1 seed42/2026 和 Q 评价；出 L 探索对比表（**只用验证集**），作为冻结配置的依据；自然遮挡子集在测试集上，只能在配置冻结后对最终模型做描述性报告，不能用于选 q/β；三种子汇总表；Top-5 检索案例图；错误分析 |
| 全组 | — | baseline 主表；Notebook 框架 | 配置冻结决定；最终阶段是否降级；**整合 Notebook、4 页报告、PPT 展示与彩排**（结构与分工待全组再对）；写作中 S 须引用 R10（PBSVT）并说明机制差异与曝光对照，L 须引用 R3/R12/R14 等并说明 q=3 依据来自已有文献；ATTRIBUTION 锁定 |

Katana 作业统一由 B 用 rsync 同步代码后提交，结果上传 Google Drive；各模块代码由其负责人提供，B 不代写成员模块。

课堂 checkpoint 是 tutor 询问进度的口头交流，不要求现场检查代码；进度、问题和下一步统一记录在 [本周进度文档](011_Weekly_Checkpoint_Deliverables.md)。代码按项目功能组织，不按课堂汇报或成员另建目录。`checkpoints/` 指训练权重和恢复状态，与课堂 checkpoint 无关。

主线为 **A 数据接口 → B 训练与导出 → E 评价**。C/D 可独立准备模块，在 B 固定版本上分别接入；M3 由 D 集成。每人维护自己的实现、检查和技术说明。

建议使用 `feat/data`、`feat/pmt`、`feat/smooth-transition`、`feat/local-supervision`、`feat/evaluation` 分支。共享接口变更写入架构文档，每次实验记录代码 commit、完整配置、种子、数据/权重 checksum、设备与原始指标。不要提交完整数据集、权重、凭据或缓存；需要分享的小型结果表经审阅后单独加入 Git。

## 交付与进度

### 已完成

- [x] 项目目录、配置模板与 README。
- [x] 数据接口与采样器（合成数据自检通过）；本机真实数据审计与真实 batch 检查。
- [x] B 的模型、损失、训练恢复和导出实现；真实数据 3 步短跑。
- [x] 评价器正确性验证（E）：手算自检 + PMT `eval_sysu`/`process_gallery_sysu` 对照通过；鲁棒性扰动工具及 M0/M1/F 运行。
- [x] 四个 baseline 对比（seed17，内部 val）：V0、M0、AGW 已训练并评价；CAJ 按作者代码训练崩溃，修正版（`CAJ_fix`）已重跑并评价。
- [x] S 模块（C）；M1、F seed17 已训练并评价。
- [x] 整合 Notebook 结构、baseline 章节与 baseline 检索对比。
- [x] L 模块（D）合并并接入训练/导出；M2、M3 seed17 已训练并评价，含鲁棒性。
- [x] M1/F 负结果分析（C）；局部特征测试时拼接的探索评价（E）。
- [x] S/L 文献核查（设计文档 §4）：S 最接近 R10（PBSVT，无公开代码，不复现）；L 与 R3/R12/R14 等重叠，按受控应用定位。

### 剩余工作（按顺序，2026-11-07 更新）

M2/M3 首轮已完成，原步骤 2 已结束。下表日期为按内部目标 11/15 倒排的建议，需全组确认；与 [设计文档](010_Project_Design.md) §12.3 冲突时以本节为准。

| 步骤 | 内容 | 负责人 | 状态 | 建议完成 |
| :---: | --- | --- | --- | --- |
| 1 | 向 tutor 确认：V0 是否算 baseline、CAJ 修正版能否使用、外部 baseline 单种子是否足够 | 全组 | 待确认 | 下次 tutorial |
| 2 | L 补充实验：L3_B050、L2_B025、L3_B025 跑完并评价；A 的自然遮挡子集；E 的探索对比表 | D、B、C、A、E | 三个 L 变体均训练完成（11/07），待 E 评价 | 11/08 |
| 3 | S 窗口 1/6 epochs 探索：**不做**（11/07 决定），报告中说明；Q 对照照常 | A | Q 训练完成，待 A 分析 | 11/09 |
| 4 | 冻结配置：E 出验证集对比表后，全组决定 M0–M3 维持 q=2/β=0.5 还是改用探索结果（改则旧运行标 exploratory）；决定、日期和依据数字提交到 Git | 全组 | 等 E 对比表 | 11/08 晚 |
| 5 | M0–M3 补 seed42、seed2026（8 次训练），E 汇总均值 ± 标准差 | B、C、D、E | 未开始 | 11/12 |
| 6 | 最终阶段（395 身份重训 + 测试集一次评价）：按剩余时间决定执行或降级，降级须在报告中写明 | 全组 | 待决定 | 与冻结一起决定 |
| 7 | 完整 Notebook、正文 ≤4 页报告（含 R10、R3/R12 等文献定位）、全员展示与彩排 | 全组 | Notebook 有框架，§3.2/§5 扩展实验/§6 待补 | 11/15 |

另需在最终提交前锁定作者代码版本、许可证与训练依赖（见 [ATTRIBUTION.md](ATTRIBUTION.md)）。

内部目标为 2026-11-15 形成可提交版本，2026-11-18 是内部暂定截止日；正式日期以课程公告为准。

## 来源

研究路线依据本仓库设计文档；主要外部来源为 [PMT 作者仓库](https://github.com/hulu88/PMT)、[PMT 论文](https://arxiv.org/abs/2212.00226) 和 [SYSU-MM01](https://github.com/wuancong/SYSU-MM01)。A 的数据增强已有第三方来源；B 按固定作者版本核对并独立实现。具体 commit、许可证及修改记录见 [ATTRIBUTION.md](ATTRIBUTION.md)。
