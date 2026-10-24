# 来源与复现登记

目前没有复制或引入第三方实现。采用代码前填写下表并保留适用的许可声明；未确认许可不等于可任意再发布。本仓库尚未选择统一开源许可证。

| 组件 | 来源 | 锁定 commit | 许可状态 | 原始路径 / 本组路径 | 修改 / 负责人 |
| --- | --- | --- | --- | --- | --- |
| PMT | https://github.com/hulu88/PMT | 待核对 | 待核对 | 待导入 / src/model.py、src/losses.py | B |
| SYSU 协议 | https://github.com/wuancong/SYSU-MM01 | 待核对 | 待核对 | 待核对 / src/evaluate.py | E |
| 评价交叉参考 | https://github.com/mangye16/Cross-Modal-Re-ID-baseline | 待核对 | 待核对 | eval_metrics.py / 尚未引入 | E |
| S 扩展 | 本地设计 §5.2；相关文献待补查 | 不适用 | 待团队确定 | src/schedules.py | C |
| L 扩展 | 本地设计 §5.3；相关文献待补查 | 不适用 | 待团队确定 | src/local_head.py | D |

## 论文—作者代码—本组设置

| 项目 | 设计文档记录 | 作者代码 | 本组实际设置 / 差异原因 |
| --- | --- | --- | --- |
| 框架 | 论文 MindSpore / 公开实现 PyTorch | 待锁定 | 尚未实现 |
| 输入与主干 | 256×128，ViT-B/16，stride 12 | 待核对 | 配置提案，未验证 |
| 训练阶段 | 24 epochs，前 6 epochs 第一阶段 | 待核对 | 配置提案，未验证 |
| 损失与增强 | PHT/MSEL/DCL，阶段对应的增强 | 待逐项核对 | 尚未实现 |
| 预训练 | ImageNet | URL/checksum 待登记 | 尚未下载 |
| 检索特征 | BN 后 768 维 | 归一化/距离待核对 | 尚未实现 |
