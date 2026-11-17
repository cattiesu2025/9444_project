# Notebook 提交结构

`project.ipynb` 是唯一整合 Notebook，按评分表分成六部分：

1. Introduction, Motivation and Problem Statement
2. Data Sources
3. Exploratory Analysis of Data
4. Models and Methods
5. Results
6. Discussion（末尾附 References）

Writing 是全文的表达与组织质量，不单独设章节。正文呈现研究问题、方法、实验结果和解释，不放评分映射、成员任务清单、PBS作业号、校验和或排障流水账。

## 当前内容（2026-11-17）

骨架已按最终结论重排，B 负责的部分已写完：§4.1 PMT、§4.2 对照 baseline、§4.6 训练设置、§4.8 最终阶段（395 身份、防泄漏、候选更换的时间顺序）、§5.1 训练损失（含 24 epoch 曲线）、§5.2 四个 baseline 三种子对比（配对差、CMC）、§5.3 CAJ 塌缩与修正、§5.6 M0 vs M0_CA 与候选选择、§5.7 最终 6 次训练记录、§6.1/§6.3，以及 §4.1/§4.8/§5.3 中 B 的核心代码片段（原样摘自 `src/`）和附录 A 的代码仓库说明（GitHub 链接指向提交前要打的 tag `submission-2026-11-23`）。顶部已加入组名和成员表。

其余部分是醒目的 `[TODO — 负责人]` 框，由各自负责人填写（搜索 `[TODO` 即可找到，共 18 处）：A 写 §2 补充与 §3.2 EDA；C 写 §4.3/§5.4；D 写 §4.4/§5.5；E 写 §4.4 band erasing、§4.5 CA、§4.7、§5.5–§5.8 的评价部分（包括测试集结果）；全组核对成员表，写总结、§6.2/§6.4。C/D/E 各自在方法小节里放 30–60 行核心代码片段（TODO 里已写明）。提交前要删掉所有 TODO 框。

Notebook 保存了执行后的输出，小型结果数据直接写在单元里，默认执行时不训练、不下载、不依赖数据集或权重。完整代码不放进 notebook，通过附录 A 的仓库链接提供。`project.html` 只是阅读预览，不是第二份提交的 Notebook。运行依赖见根目录 `requirements-notebook.txt`（另需 numpy、matplotlib、torch；核心代码单元会 import torch）。

## 工程记录的位置

- `docs/baseline.md`：训练、恢复、AMP修复和完整复现命令。
- `artifacts/results/`：原始日志、运行环境、特征和校验记录。
- `submission/evidence/`：可版本管理的结果摘要与Notebook执行记录。

这些记录支持查证，不全部搬入正文。完整复训仍需仓库源码、配置、数据和训练依赖；辅助文件允许的提交方式以课程公告为准。开发用Notebook不作为额外最终Notebook提交。

当前是工作稿。正式提交前还需补齐真实评价和实验讨论、成员信息，并完成报告（正文最多4页，不含参考文献）与展示材料。
