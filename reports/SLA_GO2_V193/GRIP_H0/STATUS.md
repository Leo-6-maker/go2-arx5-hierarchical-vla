# V1.9.3 接触前置证据边界

真实源开度仍为归一化单标量。复核了官方 RoboDojo 源码固定版本的完整目录和 `arx_x5` 环境配置。该配置引用仿真参数，未提供两个已核验实机 HDF5 对应的夹爪、指垫与共同工具测量。现有 ARX 公开模型工具尺寸仍仅作几何参考。 [官方版本](https://github.com/robodojo-benchmark/RoboDojo/tree/c3af1ac6ad7302c0c09d17f4b050fd3a5c68c921) · [源文件与工具回读](../../SLA_GO2_V192/SEA_GRIP/ACTUAL.md)

后继 H0 可以独立使用已知目标物理参考。旧 G00/G01 站立监测器明确拒绝任务物体接触，并要求 15 mm 净空，属于无接触入口。现有 `holdout_effort_preflight.py` 已提供核对原生接触角色与 Python 接触谓词的方法。需在两个新的水果任务开发布局绑定实际物体、站立目标、允许接触对和终态 STOP 后复用该方法，并按新分配执行教师。旧移动教师及无接触 H0 成绩均保留原适用范围。此次接触分支新增启动为 0。 [站立监测器](../../../scripts/sla_go2_v18/standing_reach.py) · [原接触角色核验](../../../scripts/sla_go2_v18/holdout_effort_preflight.py)
