# 仓库执行审阅：V5.2与RoboDojo-Real接入

审阅日期2026-11-18（SGT）。只读固定PR405 head `01b4a11c80f95956742379f73404be8fbdf909e1`，并核对上游官方源码。**这是GitHub已提交证据审阅，不是SSH/GPU/进程现场检查。** 最新实际记录时间为2026-11-18 00:38:43.459925 SGT；不可把这个历史快照当成接手时仍然相同的实时状态。

## 1. 已完成与尚未完成

| 项目 | 已提交事实 | 不应作出的推论 |
|---|---|---|
| D0 |320输入、319动作、1376 C20标签、8专家550计划、2501保存状态；零新积分|不是新闭环试验|
| 专家端点余量 |全部C20首尾加速度最大约0.15，距原0.25门约0.10|不保证学生状态分布下合法|
| B1/B2训练 |各500更新/16000访问；共同87初态；500步访问/LR与4000微批次索引等匹配|不是一个模型累计新增1000步|
| 恢复 |各87数组GPU重载逐位相同；独立普通RGB16逐位相同|不是任务成功|
| 新离线生成 |792；诊断反传2次/8窗单列|不记为0数据读取|
| 旧TRAIN96/CN38 |三臂各96/96、180/180；CLOSE20+36无遗漏/迟到|不是未见布局泛化|
| ROLL96 |B0 93/96，B1 93/96，B2 96/96；旧三拒绝修复0/3、1/3、3/3|不能认为B2已解决所有轨迹拒绝|
| E2前缀 |9/24启动、8终态、1活跃、15未启动；5成功、1接触STOP、2H40拒绝|不能提前作8条件/模型晋级判断|
| 确认/E3/DEV/TEST |均0请求；条件评价代码已通过61项CPU检查但未部署|没有整条后续DAG已挂载的证据|

E2终态：F1/T0 B0碰撞，B1/B2成功；F1/T1 B0首端拒绝，B1/B2成功；F1/T2 B0成功、B1未终态、B2内部拒绝。快照尚未覆盖F2。5/8是已终态前缀，5/9包含1个pending，不报告成完整成功率。

闭合请求计量为`4653*25+13+8*5703=161962` tick；活跃条件至少15453 tick。worker完成508生成=468闭合条件＋40活跃条件。原结果字段误扫`queries/`曾显示0；派生审计核对`shift7-queries/`的request/return与SHA，原结果字节保留。后继只读计数器应复用正确路径，不热改当前胶囊。

来源：[ACTUAL](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/reports/WM02/sla-mobile-f2-v52-20261117/ACTUAL.md)、[E1](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/reports/WM02/sla-mobile-f2-v52-20261117/E1_PAIR_ACTUAL.md)、[离线](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/reports/WM02/sla-mobile-f2-v52-20261117/E1_OFFLINE_ACTUAL.md)、[E2](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/reports/WM02/sla-mobile-f2-v52-20261117/E2_PREFIX_ACTUAL.md)。

## 2. 两个不能回避的新事实

**失败机制并未消失。** B2/F1-T2在query54、任务27秒出现内部k34加速度0.261575m/s²，首尾分别0.117765/0.019200；B0/F1-T1首端0.277729。前者不属于当前端点loss直接约束的位置。它支持继续研究全轨迹/新状态行为，不支持中途改loss、删后缀或放宽0.25门。[新拒绝归因](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/reports/WM02/sla-mobile-f2-v52-20261117/E2_REJECTION_PREFIX_ACTUAL.md)。

**名义配对不是完整动态状态配对。** F1/T0的B0/B1初始化qpos最大差0.02831055，是混合平移/角度分量的数组统计，不可一概叫28mm。B1/B2也不是完全相同。不得把结果差全部归因于唯一loss。后继按物理量分开报告TCP位置/旋转、base姿态、关节、qvel/ctrl；完整WBC/selected-reference身份未知部分保持未知，不为追求配对重置或篡改控制历史。

D0已在保存状态做FK：腕部–方块小穿透与原pre-step接触时刻不是同一个状态；最近专家仍相距7.712mm，不是同状态专家纠正标签。[D0](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/reports/WM02/sla-mobile-f2-v52-20261117/D0_BOUNDARY_CONTACT.md)。

## 3. 实际可复用与不可直接复用的代码

- `mobile_successor_v52.py`已经修复真实SHIFT7父加载、fresh Adam和lineage问题；不要重做此前C20 parent1000缺陷。但它仍硬绑定500/16000、原C20 manifest和B1/B2，不能把真机目录塞入旧spec。
- `mixed_visits_v4.py`已经有任务→episode→阶段→anchor和固定配额；只需窄扩展500步、10-update精确配比及嵌套替换槽。旧类限定3K/4K、域名T/S_cmd，不能假装已有Real10。
- `MobileDataset`限制20 EP、五阶段、5703+250n、P10已消费、统计fit人口与全部manifest一致。真实来源必须独立reader，不伪造native ticks/消费证书。
- `gripper_time_v1.label_parameters`把运动事件总时长硬设为2秒，要求单调单事件，context要求上一发出width与query相差50ms。真实遥操作数据不能直接按它编码。
- `mobile_selection_v52.py`与`mobile_evaluation_v52.py`可复用评价模式；新研究仍需新arm和人口schema，不能改原V5.2选择器后重算历史胜者。

来源：[后继训练](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/src/go2arx5_vla/sla_adapter/mobile_successor_v52.py)、[配额规划](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/src/go2arx5_vla/sla_adapter/mixed_visits_v4.py)、[夹爪标签](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/src/go2arx5_vla/sla_adapter/gripper_time_v1.py)、[选择规则](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/src/go2arx5_vla/sla_adapter/mobile_selection_v52.py)、[未部署后继](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/reports/WM02/sla-mobile-f2-v52-20261117/CONDITIONAL_EVALUATION_READY.md)。

## 4. 上游配方核对

官方主仓库最新核对提交`18896e757647ed9c2a68816b35a7100afda6e287`，官方README链接的Memory实现是PR237，仍open/unmerged，固定head`c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454`。不能假定XPolicyLab main已包含它。

Sim配置仅仿真；Real配置只混合三种真实机器人。Memory加载器对不等source权重直接报错，等权使用ConcatDataset；主仓库持续预训练才使用`N_i*w_i`加权采样。**未发现官方下游sim:real固定最佳比例。** 0/10/25%是本项目设计，不是官方结论。

官方转换器分别读取HDF5 state/action，频率来自文件或preview元信息，真实pose使用臂局部frame，夹爪在后续处理归一化到0–1。字段名和格式不能单独证明命令实际发出或消费、硬件时间戳、TCP偏移与未来动作计划语义。

来源：[Real配置](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/external_config/train_robodojo_real.py)、[Sim配置](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/external_config/train_robodojo_sim.py)、[Memory加载器](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/training/data_loader.py)、[转换器](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/scripts/convert_robodojo_eef.py)、[坐标处理](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/policies/robodojo_policy.py)、[主仓库预训练](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/external_config/pretrain.py)、[主仓库加载器](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/training/data_loader.py)。

## 5. 身份、期限与未知项

训练source=`8ebdcbb7ab39c0f707c5b2ba94b0b097abc937e5`；运行inference/native source=`2797d0c2c458667e241aa0d24d6d0b005f3caf1a`；报告head不同。快照中的followup02只自动继续原E2余项。服务截止为**2026-11-18 10:12:18.289509 SGT**；历史allocation截止为**2026-11-19 02:51:23.855488 SGT**，不是同一期限。后继须查实际有效许可，不延长旧服务。

HF只核到公开数据仓库元信息；本轮未下载或逐文件审计任何RoboDojo HDF5，未取得可绑定的dataset revision与样本SHA。它们在执行前解析并冻结，不能填入猜测值。本轮没有现场owner、GPU空闲、服务器新终态或真机资格。

本审阅不推翻三次旧成功、旧F2失败、RP8/JOIN10/R3，也不替换任何有效负结果。结论是：保留现役比较，先解决真实来源与现有表示是否兼容，再检验数据占比，而不是直接扩采或宣布混合必然有益。
