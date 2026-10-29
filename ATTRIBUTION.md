# 来源与复现登记

目前只引入一处第三方实现：作者的 RandomErasing（见下表）。采用代码前填写下表并保留适用的许可声明；未确认许可不等于可任意再发布。本仓库尚未选择统一开源许可证。

| 组件 | 来源 | 锁定 commit | 许可状态 | 原始路径 / 本组路径 | 修改 / 负责人 |
| --- | --- | --- | --- | --- | --- |
| PMT 数据增强 | https://github.com/hulu88/PMT | 0b31da97b06428473bc4a1134b322ea226340818 | 仓库未声明许可证，待核对 | Pytorch-PMT-VI-ReID/transforms.py（RandomErasing、增强顺序与参数）、process_sysu.py（预缩放）/ src/data.py | 灰度转换改为采样器逐样本决定，便于平滑过渡；每样本独立增强种子；A |
| PMT | https://github.com/hulu88/PMT | 0b31da97b06428473bc4a1134b322ea226340818 | 仓库未声明统一许可证 | model/、loss/、train.py、optimizer.py、scheduler.py 作核对参考 / src/model.py、src/losses.py、src/training.py、src/train.py | 按算法独立实现，保留上游参数名便于初始化和数值核验；新增数值退化保护、完整恢复与证据记录；B |
| SYSU 协议 | https://github.com/wuancong/SYSU-MM01 | 待核对 | 待核对 | 待核对 / src/evaluate.py | E |
| 评价交叉参考 | https://github.com/mangye16/Cross-Modal-Re-ID-baseline | 待核对 | 待核对 | eval_metrics.py / 尚未引入 | E |
| AGW / CAJ 外部 baseline | https://github.com/mangye16/Cross-Modal-Re-ID-baseline | e817a2672c1e14e8802617acb980a79520ef9545 | MIT，许可文本见 third_party_licenses/mangye16-Cross-Modal-Re-ID-baseline-LICENSE | model.py、loss.py、resnet.py、train.py；ICCV21_CAJ/ChannelAug.py、data_loader.py、loss.py、train_ext.py / src/agw.py、src/train_agw.py | 按上游行为移植（含 Non_local inter_channels=1、ADP 平方差损失）；改用本组 296 身份划分、A 的 P/K 采样器与统一特征导出；训练入口与恢复复用 B 的实现；负责人待定 |
| S 扩展 | 本地设计 §5.2；相关文献待补查 | 不适用 | 待团队确定 | src/schedules.py | C |
| L 扩展 | 本地设计 §5.3；相关文献待补查 | 不适用 | 待团队确定 | src/local_head.py | D |

## 论文—作者代码—本组设置

| 项目 | 设计文档记录 | 作者代码 | 本组实际设置 / 差异原因 |
| --- | --- | --- | --- |
| 框架 | 论文 MindSpore / 公开实现 PyTorch | 固定 0b31da9 | 本组 PyTorch，已本地验证；Katana CUDA 尚待验证 |
| 输入与主干 | 256×128，ViT-B/16，stride 12 | 已核对，21×10 网格 | 已实现重叠 patch、位置编码插值、BN neck |
| 训练阶段 | 24 epochs，前 6 epochs 第一阶段 | START_EPOCH=1，epoch≤6 为第一阶段 | 内部 0-based：epoch<6；尚无完整真实训练 |
| 损失与增强 | PHT/MSEL/DCL，阶段增强 | 第二阶段仍含 ID；正常输入下损失/梯度对照通过 | 跟随代码保留 ID；显式排除自匹配、DCL 退化保护，见 baseline.md |
| 预训练 | ImageNet | 作者链接 jx_vit_base_p16_224-80ecf9dd.pth | 下载脚本验证发布 hash 前缀并记录完整 SHA256；训练必须严格加载 |
| 检索特征 | BN 后 768 维 | L2 归一化，负内积距离 | 已实现；内部 val 完整候选池供 E 抽 gallery |

详细参数、论文 Eq.12 与代码的差异、学习率/AMP/开发划分差异及自检方法见 [B baseline](docs/baseline.md)。
