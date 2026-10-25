# Notebook 提交结构

`project.ipynb` 是唯一整合 Notebook，按评分表分成六部分：

1. Introduction, Motivation and Problem Statement
2. Data Sources
3. Exploratory Analysis of Data
4. Models and Methods
5. Results
6. Discussion（末尾附 References）

Writing 是全文的表达与组织质量，不单独设章节。正文呈现研究问题、方法、实验结果和解释，不放评分映射、成员任务清单、PBS作业号、校验和或排障流水账。

## 当前内容

已写入baseline的方法、必要训练参数，以及真实训练第1、6、7、24个epoch的平均损失。完整24轮数据保存在 `evidence/m0_seed17_epochs.csv`；检索指标仍明确标为Pending。完整EDA、训练曲线、检索分析和扩展实验尚需补充。

Notebook保存已执行输出，内嵌重建表格所需的小型数据，默认执行不训练、不下载、不依赖数据集或权重。`project.html` 是阅读预览，不是第二份提交Notebook。运行工具依赖见根目录 `requirements-notebook.txt`。

## 工程记录的位置

- `docs/baseline.md`：训练、恢复、AMP修复和完整复现命令。
- `artifacts/results/`：原始日志、运行环境、特征和校验记录。
- `submission/evidence/`：可版本管理的结果摘要与Notebook执行记录。

这些记录支持查证，不全部搬入正文。完整复训仍需仓库源码、配置、数据和训练依赖；辅助文件允许的提交方式以课程公告为准。开发用Notebook不作为额外最终Notebook提交。

当前是工作稿。正式提交前还需补齐真实评价和实验讨论、成员信息，并完成报告（正文最多4页，不含参考文献）与展示材料。
