# WM02 当前唯一执行入口：AUTO1

2026-11-04，Asia/Singapore。继续未合并PR #355的实际代码与证据，接入PR #354固定提交 `930bb2ef0ff178771e48fb5f03ea3e37fa717582` 的AUTO1委托。不假设main已合并，不为#353/#354启动两套owner，不切换现役源码。

1. 先读根[AGENTS.md](AGENTS.md)的AUTO1补充、[执行交接](docs/handover/2026-11-04/WM02_AUTO1_CODEX_EXECUTE.md)、[G0–G6协议](docs/protocols/WM02_AUTO1_WBC_TO_WM_V1.md)及[唯一campaign配置](experiments/wm02/autonomous_wbc_to_wm_v1.json)。
2. 新WM02_AUTO1_SIM_ONLY已于19:54:21 SGT通过原owner/profile/registry/Journal/lease实际登记；成功候选的两次0.1s、两次2s及两次5s保持均通过，共7100tick。原生参考消费、目标WM、Actor仍未完成。见[当前实际命令、全部SHA与负结果](reports/WM02/auto1-target-entry/EXECUTION_REPORT.md)。
3. G1成功候选构建／运行源码 `959337e68034fdb29f5cfa0819249cbd574e1e97`，二进制SHA256 `e9aab70b6c1d2af76c6e178fed1fd8e86299dffb9624d15a85e58ba4d3f426ff`；实际全过程7350tick含旧失败250。原参考install/cancel/HOLD入口随后已通过（源码 `7939f7caf6a443a65faa21b60e0b7eef5d51c0cb`）；累计目标12353tick。下一优先G2实际动作消费、八项诊断和顺序记录，保持唯一物理owner及原安全；真实安全STOP未触发，NOT_TESTED，不能用暂停关闭代替。
4. 目标保持与消费合格后推进顺序采集→真实目标Replay→200次WM主更新及恢复→第二批4次采集、评价与100次后继更新。不等待S模型胜过规则、任意中途restore或完整任务成功；接触、预测与控制价值分别验收。
5. 源域FIT单参数响应模型已拟合并严格JSON重载，既有8父释放诊断开度H16 RMSE由.06947降至.000380、正regret3/8→0/8；这是事后因果预测诊断，没有新在线执行或M4-B参数变化。见[续轮证据](reports/WM02/auto1-target-entry/G2_AND_OPENING_REPORT.md)。并行源域仅针对已定位的夹爪动作/时间响应做最小修复；冻结M4-B，不扩泛化M5/布局/GPU来替代目标主线。正式Actor、主动行走、实机、控制数学与安全阈值修改不在AUTO1范围内。

#353已完成工作与负结果保留在[执行报告](reports/WM02/autonomous-recovery/EXECUTION_REPORT.md)：完整PC mesh录播、M4-A/B共400主更新+2恢复探针、6752条S转移；M4-B联合DEV门通过，尚无动作选择增益，目标新候选无动态通过证据。[释放排序事后诊断](reports/WM02/autonomous-recovery/POSTHOC_RELEASE_RANK_REPORT.md)不是部署验证。旧初始化、STOP、隔离和数据/模型原件均保留。

AUTO1在合法冷边界接续#353已结束的有限worker，不热改旧scope、不重开旧lease、不另建平台。拟实现CLI先编码和测试；历史qualify --domain T固定exit78不是commissioning入口。现场注册、实际命令和G0–G6状态以新执行回执为准。
