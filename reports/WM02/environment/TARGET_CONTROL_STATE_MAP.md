# 目标控制状态与动作时间映射：E2离线绑定

2026-11-04。实际目标为冻结MuJoCo＋ROS2／OCS2 Go2–ARX5，2ms积分／50ms原子步／400ms Actor决策。**完整live target snapshot／restore尚未实现或资格通过。** 新代码只实现数值容器和离线完整性合同；不是控制器导出器，也不会启动服务。

| 状态块 | 实际可见接口／证据 | 角色及恢复缺口 |
|---|---|---|
| MuJoCo完整积分 | 已有`task_state.parse_state`读GRTSTATE，ABI nq34/nv32/nu20；只有qpos/qvel/ctrl、时钟和接触。MuJoCo软件具备get/setState API | 物理量PREDICT；完整integration为RESTORE_ONLY。现有进程未导出完整mjSTATE_INTEGRATION，不能用qpos/qvel或`mj_forward`代替；activation、warmstart、外力等仍需原进程支持 |
| WBC／腿策略历史 | 实际`rl_sim_mujoco.cpp:961–965`使用history_obs_buf及history_obs；ObservationBuffer实现环形历史；桥接header还有yaw unwrap／command feedback历史 | RESTORE_ONLY。需历史顺序、游标、previous action、controller memory和时钟；reset/清空历史不等于恢复。没有修改冻结WBC或腿策略 |
| MPC solver／policy | 实际Go2节点与WbcBridgeCore读取MRT getPolicy；节点可resetMpcNode | solver、warmstart和完整policy/cache为RESTORE_ONLY。getPolicy或reset不能证明solver内部状态已完整导出／恢复；跨进程一致切面未绑定 |
| ROS receiver | 原`native_receiver`含trajectory_source／trajectory、bound_start、pending message、sequence/ref id/deadline；原冻结parser已确认41节点 | installed reference与相对时钟ANALYTIC_UPDATE；队列／接收器记忆RESTORE_ONLY。安装与真正消费ACK、空间进度和MPC/仿真时钟映射仍需原生实测 |
| 夹爪 | 原GRTGRIP1携带sequence/source/opening/expiry，实际finger q及速度来自原生状态 | q、速度PREDICT，命令解析更新／剩余执行状态必须恢复。保持q/.044机械fraction和V2工作行程[.002/.044,.042/.044]；不隐式缩放或裁剪 |
| RNG、monitor | 原任务时钟／接触／终态、grasp/hold和安全记忆需要跨分叉一致 | RESTORE_ONLY。保存各实际存在的RNG和monitor状态；不得把Teacher phase／成功真值送入Actor，不能让reset继承产生新success-onset奖励 |
| goal／geometry | 原layout/task card绑定的固定目标和几何 | CONTEXT。实际tensor shape／frame须在真实数值replay入口核验，不用S域缺失BASE字段补零冒充目标监督 |

本轮只读调查了选定native源码树中118个包含相关标识的文件。代表性源码SHA：

- Go2 OCS2 node：`1caf2f24fc18a6210d0ff7c81c46e10e9bf53623248e56d8e0a9ee4a50aa5e68`。
- WbcBridgeCore：`bc21fd6d32754d3713479205a035e5377e5557c061d009770a28c7b5cf2b1f0e`。
- WBC bridge header：`5ddf926e4105e973cc1c9e4c68585244f9bdc172bd2a420e14a90331da83d79e`。
- 原rl_sim_mujoco源码：`e05de9ee87d3030871743445a50b2e69e4565e3a786429b8566f24b012263d29`。
- ObservationBuffer header：`c0a4e1ee8f39b90ac7878f23dcdd6d2eae450a9fee9352661b3d8fa5cbe51466`。

这些是接口调查身份，不是新的控制器版本或原二进制可重建承诺。未绑定完整导出器，不等于证明所有外部OCS2库都没有序列化能力。

`ACTION_TIME_MAP`固定要保留：原Actor56参数、完整H40、wire sha/ref id、source MPC time、receiver MPC time、提交／发布安装日志、实际native ACK tick、实际消费进度、achieved和终态前状态。41节点的0…2.0秒语义已由原parser CPU验证；`installed`发布日志不是WBC真实消费证明，不能按时间推断“必然消费前8行”。H40后32节点仍属于完整preview条件，不作已执行expert标签。

恢复方案：优先由原进程在一致暂停切面导出上述完整状态，再由原受控入口恢复；如果原冻结二进制没有该API，候选替代为新实例＋相同原生初始化和完整因果prefix重放。后者须在有效动态许可下，比较至少两个状态的控制记忆、实际TCP/BASE、接触、参考进度和终止；多线程MPC确定性不能预先保证。现在不重编译／热改旧二进制，不使用用户许可字段绕过隔离。

[control_snapshot.py](../../../src/go2arx5_vla/world_model/control_snapshot.py)要求七类完整块及必要字段、dtype/shape/hash、原native tick/time和冻结身份；拒绝qpos-only、对象pickle、缺块、错误身份和篡改。数据全部标RESTORE_ONLY，`native_exporter_bound=false`、`live_restore_performed=false`。离线四项测试含完整数值容器往返与评分反例，普通及`-O`通过；未调用任何live restore。

```bash
PYTHONPATH=src "$WM02_CPU_PYTHON" -m unittest discover -s tests/world_model -p test_target_interfaces.py -v
```
