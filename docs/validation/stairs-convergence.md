# 楼梯三档离散收敛实测报告

模型版本为 `helical-box-cable-v3`，MuJoCo 版本为 `3.15.0`。本轮输入来源为 `246f5f3f89a3d8b76da5bb2f6581b0020d230416`，工作流为 [37580189844](https://github.com/FanYu-Nijika/slinky-lab/actions/runs/37580189844)。当前 checker 是未提交工作树中的 `scripts/validate_stairs.py`，SHA256 为 `61f68fddbf3df73df72fc7ac2146423d03c5660341b7ef7b24ef9b58e5222944`；记录中的 recheck 使用 SHA256 `378049475896353d28a56b8463d1e51a3af345bed5cb47b772fbe5397aa2ac1e`。原始记录的 33 个 artifact hash 均通过复核。

| case | 配置 | 墙钟时间 | 模拟时间 | 确认翻转 | 端圈序列 | 警告/非有限 | 结果 |
|---|---|---:|---:|---:|---|---|---|
| baseline | 8 段/圈，50 µs | 576.176 s | 2.400000 s | 3 | last → first → last | 否 / 否 | completed |
| half_dt | 8 段/圈，25 µs | 941.017 s | 2.400025 s | 4 | last → first → last → first | 否 / 否 | completed |
| mesh | 16 段/圈，12.5 µs | 3600.373 s | 0.8025875 s | 1 | last | 否 / 否 | timeout |

baseline 与 half_dt 的前三阶端圈序列一致，但第二阶支撑时刻从 `1.058200 s` 变为 `0.989475 s`，相对变化 `6.495%`，超过既定 `5%` 标准。mesh 在 3600 s 墙钟预算内未完成 2.4 s，只观察到第一阶支撑。

本轮收敛结论为 **false**。mesh 的 `timeout` 是计算时间预算结果；该记录没有 MuJoCo warning，也没有非有限状态，因此不能把超时等同于积分器不稳定，也不能将未完成的 mesh 结果当作通过证据。完整原始配置、模型、轨迹、摘要、支撑诊断和 warnings 保存在 `reports/stairs-validation/`。

这是宏观演示几何的数值证据，尚无独立实验支持。本报告不包含参考视频或派生图。
