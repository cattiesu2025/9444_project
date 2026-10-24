# 工程架构与接口

本文件将设计文档第 9 节落到实际目录；算法模块目前只有职责说明。`src/local_head.py` 独立于 B 的 `src/model.py`，便于 D 独立开发，后续由模型适配器接入。

## 数据流

```text
base.json + experiments/<ID>.json
                ↓ 合并并检查
data → batch → train → model → losses → checkpoint
                 ↑       ↑
             schedules  local_head
                         ↓ 全局特征导出
manifest + 固定 gallery → evaluate → metrics → 图表/Notebook
                              ↑
                         robustness
```

配置采用 JSON 深度合并：字典递归覆盖，标量和列表整体替换；实验文件只记录相对 base 的差异。所有配置都是开发提案，不代表训练器已支持。最终 395 身份配置需在开发配置冻结后另建，不能仅切换 evaluation split。

## 共同接口（待各模块实现）

| 接口 | 必需字段与约束 |
| --- | --- |
| manifest | relative_path, identity, camera, modality, split, width, height, channels, file_bytes, sha256 |
| batch | images `[2PK,3,256,128]`、labels（连续训练标签）、identities（原始 ID）、cameras、modalities、paths；逐行对齐 |
| model output | global_feature `[N,768]`、global_logits `[N,C]`、local_features、local_logits；L 关闭时局部字段为 None |
| patch grid | 从 patch embedding 读取 H_p/W_p；去除特殊 token；不得开平方推断 |
| exported features | features、identities、cameras、modalities、paths；记录 BN 位置、归一化、距离及权重 checksum |
| evaluator | 特征和原始身份/摄像头，加固定 gallery 索引；不接收训练分类标签作为检索身份 |

模态约定：`rgb` 表示可见光来源（转灰度后仍为 rgb），`ir` 表示红外。公共模型 forward 和训练调用签名由 B 在集成前固定；此处只定义数据契约。

## 运行证据

推荐目录 `artifacts/results/<run_id>/`，例如 `dev_M1_train296_seed17_transition3`。每次保存合并配置、代码 commit/dirty diff、环境、数据 checksum、预训练来源、日志、指标和权重位置及 checksum。

日志必须包含 stage、epoch、global_step、各损失原值/加权值、学习率、p_gray、实际灰度数量、耗时和峰值显存。恢复 checkpoint 包含模型、optimizer、scheduler、scaler、epoch/global_step、阶段、调度状态以及全部 RNG；独立增强和局部头初始化 RNG 不干扰公共采样和主干。

`metrics.json` 至少记录 run_id、split、epoch、协议、训练种子、gallery trial/种子、有效 query 数、Rank-1/5/10、mAP、checkpoint checksum。记录指标单位（推荐统一 0–1），展示时再转换百分比。没有实际执行的数据不得填占位分数。

## 开发顺序

1. A 固定数据与 batch；B 核对上游版本；E 用人工矩阵检查评价协议。
2. B 实现模型、原损失、两阶段训练与恢复，完成真实小批检查。
3. C/D 独立验证调度/局部头；B 导出 baseline；E 复评。
4. C/D 在固定 baseline 接入扩展；D 负责 M3；共用 gallery 与全局测试特征。
5. E 从原始证据生成汇总，所有人提供自己的技术章节。

不得将空模块、配置检查、模拟张量或作者权重验证写成已训练结果。
