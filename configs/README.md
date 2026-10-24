# 配置规则

base.json 与 experiments/<ID>.json 递归合并；数组整体替换。运行 `python scripts/inspect_config.py --experiment M0` 查看最终配置。数值来自本地设计，尚非上游核验结果；null 字段须由负责人补齐。

S 的 linear 模式在第二阶段窗口 W≥2 步内使用 p0×(1-r/(W-1))，窗口外为 0；fixed_expected_exposure 同窗口使用 p0/2。window_epochs 由训练器根据实际每 epoch 优化步数转换成 W，恢复不得重置。off 恢复原 PMT 数据路径。V0 没有 PMT 第一阶段和 MSEL/DCL。

当前只有开发配置。最终 395 身份训练配置、数据 checksum、评价距离/归一化、增强顺序、warm-up 和依赖均需核验并冻结后补齐。
