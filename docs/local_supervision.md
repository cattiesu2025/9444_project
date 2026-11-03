# D：局部身份监督

按设计文档 v2.0 第 5.3 节实现。`src/local_head.py` 提供独立局部头、PMT 适配器和损失组合函数。默认两个区域、权重 0.5，两个训练阶段使用相同的局部监督。

## 方法与接口

输入为不含 CLS 的 `patch_features [N,H*W,D]` 和显式 `patch_grid=(H,W)`。区域 j 使用行 `[floor(j*H/q), floor((j+1)*H/q))`，对区域内的全部 patch 平均。默认实际网格 21×10 分成 10 行和 11 行，既不重叠也不遗漏。每个区域有独立 BatchNorm1d 和无偏置身份分类头。BN 偏置冻结，与全局头保持一致。训练 batch 必须多于一张图，以满足 BatchNorm 要求。

局部特征为 `[N,q,D]`，logits 为 `[N,q,C]`。局部交叉熵在区域间取平均，总损失为原阶段 PMT 损失加 0.5 倍局部交叉熵。分类损失以 FP32 计算。局部初始化保存并恢复 CPU 随机状态，不改变公共主干初始化及 dropout 随机序列。

## 后续训练器接入方式

```python
from src.model import PMT
from src.local_head import PMTWithLocal, add_local_loss

model = PMT(num_classes, cfg['model'], pretrained=pretrained)
model = PMTWithLocal(model, regions=cfg['local']['regions'],
                     enabled=cfg['local']['enabled'], seed=seed).to(device)
# 在以上步骤之后创建 optimizer，使其包含局部头参数。
output = model(images)
# terms = 原有 PMTLoss 调用的返回字典
terms = add_local_loss(terms, output, labels, cfg['local']['weight'])
terms['total'].backward()
```

适配器保留原主干、全局 BN 和分类头的 state_dict 名称。局部头及其 BN 状态也进入 state_dict，恢复时必须构建相同区域数和开关设置的模型。测试仍取 `retrieval_feature`，维度及计算与原 PMT 一致；无需拼接局部特征。关闭开关时不创建局部头；权重为 0 时保留原损失 total。

## 自检

在项目根目录，使用已有 PyTorch 环境运行：

```bash
python -m pytest tests/test_local_head.py -q
```

覆盖实际非正方形网格、奇数高度手算池化、错误 token 数、空区域拒绝、随机状态隔离、CE 手算对照、关闭/零权重回退、局部头及主干梯度、全局检索特征一致性和模型状态重载。

## 完成范围

本次为模块与合成输入自检交付。原 `src/model.py` 和 `src/train.py` 未修改。现有训练入口仍拒绝 M2/M3；需要按上述接口接入训练器、记录 local/local_weighted，并核对特征导出器与恢复入口后才能启动实验。未执行真实数据训练、完整训练恢复或检索评价，不能报告性能提升。

课堂汇报：已实现水平局部池化、独立 BN/身份分类头和辅助交叉熵，并提供非正方形网格与梯度检查；下一步将其接入 M2，再结合 C 的调度接入 M3。真实训练及检索效果尚待验证。自检通过与否以实际测试输出为准。
