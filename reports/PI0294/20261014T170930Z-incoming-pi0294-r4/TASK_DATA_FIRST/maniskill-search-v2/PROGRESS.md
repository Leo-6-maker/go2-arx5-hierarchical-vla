# PI0294 MP-S2 实际进度与独立复算

状态：**PARTIAL_NO_SUCCESSFUL_TARGET_BODY_SEED**。这是实际物理结果，不是CPU_READY，也不是完整任务PASS。MP-S2尚未封存终局；真人辅助路径尚未启动。旧pilot、native安全失败、V1/V2与旧ManiSkill组件终局全部保留。

## 1. 已执行与未执行

采用固定c2ac452bf873f1c8187d0340f2da9c4d696e08c5，采用收据固定cb3389f18fffa604f3f33ca53c141e0af5fb7005。CPU初始来源eefd43c619e2929cece77bfa5fff95004d9da98c；T2真实运行b08580ddce029cb6cf977394d35abd7c46432c0b；独立CPU审计与RGB恢复08afa0189aaf34939297fe9b15443808b69b09b6。小收据见[ACTUAL_PROGRESS_RECEIPT.json](ACTUAL_PROGRESS_RECEIPT.json)。

| 条件 | 实际结果 | 后续资格 |
|---|---|---|
| 16项新增CPU契约 | 16/16通过，原24项证据复用 | 不冒充物理成功 |
| 原首帧T1有界规划 | NO_LEGAL_REGISTERED_PAIR | T1 AUTO短任务未准入，无物理重跑 |
| 原首帧T2有界规划 | 候选4（−60°）获得合法H40；1001密集点检查通过 | 实际启动自身短任务 |
| T2 AUTO短任务 | 完整12秒，独立完整性通过；有效短里程碑FAIL | 不准入AUTO完整任务，不按成绩重试 |
| T1/T2完整任务 | 都未启动 | 如走ASSISTED，仍需真实人类与实际验证的交互接口 |

T2真实完整Go2＋ARX5执行：5秒warmup＋12秒任务，8500实际积分（8503保守入账）、241观察点、30条提交并安装的H40事务；安全停止0。安装150条native消息的最大XYZ差2.980e−8m、四元数分量差3.374e−8，低于原1e−6门。新观察/源时钟、prefix8、标签与保存native状态的独立复算通过；不由安装PASS推导实际MPC采用或任务成功，MPC selected-policy消费仍UNKNOWN。

## 2. 真实任务行为

距离选定预接近点：初始0.252917m、最小0.220274m、最终0.339089m。达到原“至少2cm接近”一项，但未达到safe-precontact、实际物体接触或推送。500Hz审计接触ticks=0，物体最大位移/抬升=0。末端实际起点[0.354463,0.014099,0.717493]m、终点[0.163128,0.031449,0.491927]m。

这说明本条件的合法参考生成与安装尚未带来真实接触，不证明全局IK无解、不证明WBC不能完成任务，也不是GR00T策略FAIL。本系列没有新模型查询/训练。保留负结果，不进入新的WBC归因或参数调整。

## 3. 记录器故障与非物理恢复

首次原RESULT保持DATA_RECORDER_INCOMPLETE：父记录进程没有有效OpenGL上下文。**未因此重跑物理。** 用同一保存native packet、MJB、相机与时间重建241组RGB；逐文件hash和来源绑定通过，再完成独立审计。图像必须标作RGB_FROM_SAVED_ACTUAL_STATE_RECONSTRUCTION，不宣称首次记录器成功或实时录像。原RESULT和故障hash均保留；恢复RGB已补计父账。

初始CPU证书formatter曾错误填IK_calls/RRT_calls=0，不能解释成零计算；原件保留。以持久SEARCH_LEDGER为准，本轮32个bundle累计94目标IK调用、1504已预留CLIK初值、62次RRT、31031预留密集点；预留量不冒称全部实际迭代完成。补充收据省略错误字段，不覆盖历史证书。

## 4. 现场账本与边界

截至本收据，MP-S2新增1启动/8503入账积分/241双RGB，所有入账CPU343.020566秒。父账38启动/353508积分/12248双RGB/15613.158760 Core CPU秒。无活动CPU reservation，现场只读PID核查无本run活跃进程。原T0与截止未重置；新GPU/PBS/GR00T/训练0。该快照不是最终归档收据，未宣称MP-S2整包私有readback完成。

外部示范建议已单独作为[来源与适配审计](../../../../../docs/reviews/2026-10-15_SUCCESSFUL_DEMO_SOURCE_AND_RETARGET_AUDIT.md)准备，不自动替换本系列教师/候选或重开有效失败。当前卡external_demo_downloads=0；本轮只读取公开元数据与源码，没有下载轨迹payload或启动新来源实验。WBC/安全/原任务评分不变，selected/deployment=NONE。
