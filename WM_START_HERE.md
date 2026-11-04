# WM02 当前唯一执行入口：AUTO1

2026-11-04，Asia/Singapore。继续未合并PR #355的实际代码与证据，接入PR #354固定提交 `930bb2ef0ff178771e48fb5f03ea3e37fa717582` 的AUTO1委托。不假设main已合并，不为#353/#354启动两套owner，不切换现役源码。

1. 先读根[AGENTS.md](AGENTS.md)的AUTO1补充、[执行交接](docs/handover/2026-11-04/WM02_AUTO1_CODEX_EXECUTE.md)、[G0–G6协议](docs/protocols/WM02_AUTO1_WBC_TO_WM_V1.md)及[唯一campaign配置](experiments/wm02/autonomous_wbc_to_wm_v1.json)。
2. 当前新项目仿真范围为USER_AUTHORIZED；须通过原EXISTING_LOCAL_PC/profile/registry/Journal/lease实际登记WM02_AUTO1_SIM_ONLY。现场登记及原生资格分开报告，旧CLOSED lease不表示缺少用户GO，新文档不表示已运行。
3. 最高优先：G0实际登记与受控目标入口，随后G1两次0.1s→两次2s→两次5s，按实际scope逐级通过。使用最新trace候选：构建源码 `82439355dd5edabba9dc8999a577f435f4a877a4`，二进制SHA256 `5e2c7d290f6323eb1322757a217fd08755b9c50058a7fd2ee01fcdf53c2c5cf7`。同一物理owner、单实例、原安全；首个异常留证后有依据修复，不盲重跑。
4. 目标保持与消费合格后推进顺序采集→真实目标Replay→200次WM主更新及恢复→第二批4次采集、评价与100次后继更新。不等待S模型胜过规则、任意中途restore或完整任务成功；接触、预测与控制价值分别验收。
5. 并行源域仅针对已定位的夹爪动作/时间响应做最小修复；冻结M4-B，不扩泛化M5/布局/GPU来替代目标主线。正式Actor、主动行走、实机、控制数学与安全阈值修改不在AUTO1范围内。

#353已完成工作与负结果保留在[执行报告](reports/WM02/autonomous-recovery/EXECUTION_REPORT.md)：完整PC mesh录播、M4-A/B共400主更新+2恢复探针、6752条S转移；M4-B联合DEV门通过，尚无动作选择增益，目标新候选无动态通过证据。[释放排序事后诊断](reports/WM02/autonomous-recovery/POSTHOC_RELEASE_RANK_REPORT.md)不是部署验证。旧初始化、STOP、隔离和数据/模型原件均保留。

AUTO1在合法冷边界接续#353已结束的有限worker，不热改旧scope、不重开旧lease、不另建平台。拟实现CLI先编码和测试；历史qualify --domain T固定exit78不是commissioning入口。现场注册、实际命令和G0–G6状态以新执行回执为准。
