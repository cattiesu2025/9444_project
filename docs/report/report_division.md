# Report 分工

初稿：[COMP9444_Project_Report_v2.docx](COMP9444_Project_Report_v2.docx)（已有完整内容）。按模块分工：谁负责哪个模块，就核对、修改和补充报告里的哪一部分。

| 成员 | 模块 | 负责的报告部分 | 具体内容 |
| --- | --- | --- | --- |
| Yuanrong Wu (A) | 数据 | Experiments: Dataset and exploration; Splits and protocol（划分部分） | SYSU-MM01 介绍；从 EDA 图表 E1–E5 中选 1–2 张放进报告；296/99 划分；自然遮挡子集的标注说明 |
| Zixin Su (B) | PMT baseline | Methods: Baseline PMT; Experiments: Training details | PMT 结构与损失；训练设置（24 epochs、batch、AdamW）；AGW 与 CAJ_fix 的复现描述（CAJ 特征塌缩及修正） |
| Di Shao (C) | S | Related work: Gap 1; Methods: Extension S; Results: RQ1 | 引用 PBSVT [6] 并说明机制差异；S 与 F（fixed-grey control）的结果和解释 |
| Junru Mao (D) | L | Related work: Gap 2; Methods: Extension L; Results: RQ2 与 M3 | 引用 PCB 等局部方法，说明 q=3 的依据；M2、M2\*、M3 的结果 |
| Hao Lu (E) | 评价 | Experiments: Protocol 与 Robustness; Results: RQ4 与 final test; Discussion; Conclusion | 10 个固定 gallery、配对 seed 比较、9 种扰动；Table I–III；CA 结果；错误分析；结论与 limitations |
| 全组 | — | Abstract、Introduction、ATTRIBUTION、排版 | Hao 统稿；ATTRIBUTION 每人写一两句自己的贡献；最后每人通读一遍 |

## 约定

- 篇幅限 4 页，每人负责的部分约半页到一页；超页时优先删减 Related work 和 Methods 的细节。
- 所有数字以 PPT 第 8 页和 Table I 为准；报告里的 "L3" 统一改为 M2\*，与 PPT 一致。
- 在初稿副本上修改，并标出改动（Word 修订模式或批注），改完交给统稿人合并，不要多人同时改同一个文件。
