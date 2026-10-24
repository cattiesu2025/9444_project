# 后续检查计划

各负责人实现模块时加入有意义的检查：A 身份隔离及 P/K batch；B 损失、小样本下降、阶段与恢复；C 窗口首尾/期望暴露量/恢复；D 非正方形网格/梯度；E 手算 AP/CMC、摄像头排除与参考实现交叉验证。框架阶段不将配置检查当作模型正确性测试。


已实现：`test_data.py`（A，30 项，使用 `synthetic_sysu.py` 生成的合成目录，不需要真实数据）。运行 `python -m pytest tests/test_data.py -q`。
