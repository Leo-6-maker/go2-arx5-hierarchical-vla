# SEA-GRIP 工具与物理开度回读

Go2 的 **42 mm 是每指滑块行程，对应 85.048 mm 的内侧指垫间距**。冻结场景的几何关系为 1.048 mm 加两指行程之和。实测观测状态使用两滑块行程平均值除以 44 mm，历史 H40 分数使用单指 `44 × fraction`，已有官方动作桥接提案使用单指 `2 + 40 × fraction` mm。这三套语义分别记录。 [已封存的场景与测量](../../SLA_GO2_V19/SEA0/TOOL_AND_SOURCE_CLOCK.md) · [Go2 代码映射](../../../scripts/sla_go2_v18/world_intent_clock.py)

ARX X5 原厂家明确区分单轨和双轨夹爪，双臂 URDF 的末端只定义到 `link6`。另一份公开 ARX SDK 模型定义了沿 X 轴 145 mm 的固定工具关节。两份公开资产都缺少与本次两段录像对应的设备和指垫安装证明。源文件的一维归一化读数仍无法转换成实测内垫间距。 [厂家型号与 URDF](https://github.com/ARXroboticsX/X5/tree/9a255e38156e3f5a263f341df209ab028d49dad9) · [公开 SDK 固定工具关节](https://github.com/real-stanford/arx5-sdk/blob/36eabcb618c2668dfdff228b18b30507fde71784/models/X5.urdf)

源工具适配待补同一录制台的夹爪型号、从 `link6` 到共同工具点的变换、归一化读数与内垫间距的实测表，以及录制时钟对应。接触仍待对应场景的允许接触对、STOP 和 H0 资格。 [机器可读边界](ACTUAL.json)
