# WM02目标控制器接口：本轮只读证据与替代路线

本轮没有目标积分／ROS节点／控制器修改。P3仍未取得live完整snapshot/restore资格；此文明确可复用API、缺失信息和可行路径，不把离线容器当原生实现。[本轮5份源文件SHA再次MATCH及实际行号](TARGET_API_ACTUAL.json)，上轮[完整状态映射](../environment/TARGET_CONTROL_STATE_MAP.md)保留。

| 组件 | 当前实际接口与时钟 | 不能恢复的信息／下一步 |
|---|---|---|
| MuJoCo | GRTSTATE读取qpos34/qvel32/ctrl20、step/time及contact；`task_state.parse_state`标contact_time=t−.002 | 没有外部完整mjSTATE_INTEGRATION导出／反向写入入口。qacc warmstart、activation、applied force等不能由forward重建为原状态。未来若原受控API不可提供，不为此改冻结WBC |
| WBC bridge | 源header第202–211行有previous_input/time、previous native velocity command/time、有无上一命令、yaw_offset/prev_raw_yaw等 | 这些是进程内记忆，不在GRTSTATE；reset/清零历史不等于恢复，缺一致切面导出和回填 |
| 腿观测历史 | rl_sim第961–965行insert/get_obs_vec/forward；ObservationBuffer第53行get_obs_vec，第62行obs_buf | 能读到历史向量不等于完整buffer/顺序/游标/控制状态已可跨进程导出恢复；不能只复制当前qpos |
| MPC | 节点getPolicyResponseModel（151）、resetMpcNode（269），bridge读取getPolicy（222/243/261） | policy输出和reset服务不提供完整solver/warmstart恢复；MPC observation时钟与ROS wall/steady时钟需保留映射。未声称所有第三方OCS2库都无任何序列化能力 |
| receiver | Python对象中trajectory/pending/sequence/bound_start/deadline/ref_id；原publish的installed trace、MPC observation time | `installed`只是发布边界。原`target_mapped_ack`需要policy bytes、真实MPC observation、before/after native tick和独立prefix safety verdict；其旧CORR/R0科学条件不能直接冒作WM02条件，更不能拿policy12维当全身安全 |
| gripper/monitor | 独立GRTGRIP1的seq/source/fraction/expiry，原高率状态记录与机械范围monitor | 要保留执行状态和监控历史；工作行程不变。物体碰撞安全布尔不能替代倾倒、腿／全关节、控制器及原prefix安全链 |

源文件SHA：node `1caf2f24fc18a6210d0ff7c81c46e10e9bf53623248e56d8e0a9ee4a50aa5e68`；bridge cpp `bc21fd6d32754d3713479205a035e5377e5557c061d009770a28c7b5cf2b1f0e`；bridge header `5ddf926e4105e973cc1c9e4c68585244f9bdc172bd2a420e14a90331da83d79e`；rl_sim `e05de9ee87d3030871743445a50b2e69e4565e3a786429b8566f24b012263d29`；buffer `c0a4e1ee8f39b90ac7878f23dcdd6d2eae450a9fee9352661b3d8fa5cbe51466`。这些源码不是已证明可重建对应二进制的build provenance。

站立T1传感绑定要沿原`ActualState.update(packet)`提取物体／TCP、左右接触、opposing／relative motion、机械夹爪状态及原32秒监控；它目前的`safety_valid`只来自对象接触允许表。还必须绑定真实free-base姿态、原全身安全verdict、高率全关节范围、时钟／contact的2ms偏移与一致采样。原`JournalRangeGuard`检查每个native高率包的全部limited joints，但也不单独证明全身安全。**不得用ActualState对象碰撞safe或SDK joint-cost代替`native_fullbody_safety_valid`。** 旧任务／安全阈值和评分保持，新WM02_STANDING_T1_V1尚未native qualified。

优先路线是原进程已有受控完整导出／恢复API；本轮选定冻结app入口未暴露一个可直接调用并已资格的完整接口。新增一致checkpoint API若需要重编译／改冻结控制器，当前不执行。

替代候选为新进程完整原生初始化＋逐tick完整动作前缀重放：须记录相同全部冻结输入、seed、controller start与reference clocks、实际安装与消费，并在未来有效原生准入下，用两个独立实例、至少两个切点比较TCP/BASE、接触、终止、reference进度及能读取到的MPC/WBC历史；内部历史读不到时，不能把外部轨迹一致升级为严格完整状态同一性。多线程MPC求解／DDS时序可能不确定，不能预先声称等价。

若该重建不能严格资格，保留“同状态五动作反事实／真实梯度诊断”未完成；**在独立有效动态许可和原安全链满足时，正常顺序采集仍可先进行，不以缺exact restore永久阻止WM学习。** 当前原S1/T2隔离未解除，所以这两种目标动态路线均未运行；本轮没有通过新metadata或GPU空闲绕过它。
