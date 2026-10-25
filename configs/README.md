# 配置规则

base.json 与 experiments/<ID>.json 递归合并；数组整体替换。运行 `python scripts/inspect_config.py --experiment M0` 查看最终配置。M0/V0 的模型、损失、优化器和学习率已按作者固定版本核验并由 B 的训练器支持；扩展配置仍待 C/D 接入。运行时的权重 checksum 保存到 checkpoint/环境记录，不写固定占位值。

S 的 linear 模式在第二阶段窗口 W≥2 步内使用 p0×(1-r/(W-1))，窗口外为 0；fixed_expected_exposure 同窗口使用 p0/2。window_epochs 由训练器根据实际每 epoch 优化步数转换成 W，恢复不得重置。off 恢复原 PMT 数据路径。V0 没有 PMT 第一阶段和 MSEL/DCL。

当前只有开发配置。BN 后 L2 特征、负 cosine 距离、warmup 与作者学习率已固定；本地依赖锁已提供。最终 395 身份训练配置和 Katana CUDA 环境仍待实际核验。阶段编号、来源差异及命令见 docs/baseline.md。
