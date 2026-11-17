# V6 活跃阶段回读

观察时间 2026-11-18T06:24:51.478381+08:00。T 两个固定端点已封存，见 [ACTUAL.md](ACTUAL.md)。S1000 已持久提交 150 次更新及 4,800 次访问。S0 已完成 TRAIN96 与 SOURCE_DEV96 的 192 次生成，四窗反传诊断另计且增加 0 次更新。SOURCE_DEV 的 48 个动态锚点，S0 归一化参考误差为 0.125056，当前状态保持基线为 0.173941。S1000 尚未封存，迁移门未作决定。

| Core 受管单元 | MainPID | 当前状态 | 截止 SGT |
|---|---:|---|---|
| wm02-sla-v6-source1000-20261118-02 | 1505424 | REGISTERED_SOURCE_CHILD | 16:10:54 |
| wm02-sla-v6-target-followup-20261118-02 | 1601452 | WAIT_EXISTING_T_AND_S_CAPACITY_COLD_BOUNDARY | 16:28:08 |
| wm02-sla-v6-source-test96-20261118-01 | 1608316 | WAIT_FIXED_S1000_AND_INDEPENDENT_RESTORES | 18:13:56 |
| wm02-sla-v6-multitask-interfaces-20261118-02 | 1615664 | WAIT_EXISTING_T_NATIVE_OWNER_COLD_BOUNDARY | 18:18:57 |

四个服务均从实际进程 cgroup 和 lease 回读。SOURCE_TEST96 只在两个独立 RGB16 恢复与固定 S1000 后准备，GPU 评价等待现役 T 后继关闭。T 后继的 615 次离线请求、新24及条件确认尚未派发。SOURCE_TEST 标签和生成均为0。

F2 已在同一个保存的起始姿态上比较六组纯视觉区域。候选03的 body RGB 可见红392、蓝294、左区77、右区96像素，腕图两区仍为0。选择按 body 四个可见计数中的最小值排序。物体、桌面、碰撞和相机配方一致。新 MJB SHA `e54f34ef6c24ec2fcf6ddca56285249a682fa794bf9cfa7f3d67f88ddee58fdf`，候选任务记录器 `1d40b6c8c1060df9010f20cf6fe03bad62ac6ec9e0867e993a95a871108f375e`。19.10 s wall／13.46 s Python CPU，编译子进程另未计时。

36 个新场景簇均已实际编译，获同一个受保护机器人指纹。布局 population SHA `97a586c492811e212558f8e1c3ff978e2ec205cb9f8d7b90b8ba12c8622e3ec5`。簇覆盖预定48个 Teacher（含先行6）、每臂24个筛选、24个确认、DEV60及TEST60。每个簇共享六条指令，独立 DEV／TEST 各10场景簇。实际2,680,941,636字节，37.59 s wall／14.25 s CPU。接口检查待实际执行，Teacher 还未 dispatch。

M 接口首个单元的原生 Python 缺 SciPy，导入退出前未启动任何子用例。保留失败，父子三个 lease 均已关闭且 MainPID0。去掉等待函数的学习器依赖后，通过同一原生解释器的实际导入，按同一面板与期限接续两个未启动接口。接续已经独立确认等待 T 世界锁释放。新的41／38／20状态由完整 native journal核验，原 trace 每条仍只容纳96个值，其末尾三个 ctrl 分量不可用。诊断记录对此单列，完整后步数据仍依赖原生状态日志。

本阶段 T 和 M 新原生请求、积分和任务成功均为0。未动原截止、WBC、安全门、旧负结果和历史 TEST。下一交付是 S1000 与独立恢复终态、T 固定24逐条件结果及两组新的非接触原生接口。
