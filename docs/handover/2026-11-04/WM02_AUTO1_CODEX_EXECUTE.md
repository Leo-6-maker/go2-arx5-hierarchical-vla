# Codex：WM02 AUTO1 长程接续

从 PR #352 固定 `76bb8115c3b588b70fd55bdde13f10926c60bce7` 的后继分支接管；不要假设main已合并。先读根AGENTS的AUTO1补充，再读 `docs/protocols/WM02_AUTO1_WBC_TO_WM_V1.md` 和 `experiments/wm02/autonomous_wbc_to_wm_v1.json`。

## 并发计划协同：只启动一个campaign

本交接编写期间出现同基线的草稿 PR #353（`a5619af01a8e508f170ecaec4aa994936098a985`），其内容保持不变，未由本轮关闭或覆盖。**不要为#353和AUTO1分别启动两个派发owner或重复任务。** 按本交接执行的新campaign使用AUTO1；若其他计划已经实际注册/启动，先尊重其现役scope和终态核清，不热改、更不抢锁，只能在合法冷边界登记后继。现有真正的外部站点/owner权限要求仍优先。本文件提供项目委托与分级范围，不冒充现场注册或资格。

## 最新直接用户要求

“给codex更多授权，制定一个长程自主推进计划，跑通WBC采集到WM的训练入口”。

将该请求作为本轮新项目仿真scope的授权来源，在既有owner/profile/registry/Journal/lease中实际登记 `WM02_AUTO1_SIM_ONLY`。不再为每个小修、每个有限批次或阶段通过询问GO。用户授权不等于已经注册或已通过安全资格；真正外部站点权限缺失必须报告，不可伪造。

旧S1/T2失败、STOP和CLOSED lease完整保留；此次不是复活旧运行，而是新候选分级commissioning。先核已构建R1＋锁存＋trace候选，随后实际执行两次最多0.1秒冷启动，原生通过再两次2秒、两次5秒保持。安全失败立即停当前候选；有证据的包装/同步/ready/命令配对修复及回归后，可在新身份继续，不改腿策略/WBC控制律、参数或安全阈值。

## 不要再绕回源域模型门槛

M3差异损失可通过但绝对偏差/事件失败，不适合Actor。它**不阻挡目标初始化、顺序采样和真实WM入口**。不要求先赢规则、完成视觉、任意中途MPC恢复或已有完整任务成功。

本轮主交付：目标保持 → EEF实际消费 → 首批8次真实顺序采集 → 目标schema/Replay → 200次真实WM更新＋恢复 → 第二批4次采集/评价＋100次后继更新。接触资格和预测有效性单独报告。Actor、主动行走、实机、A40/H200正式长队列不在本轮默认范围。

## 首轮应做的实际工作

1. 固定独立worktree，核查运行中owner而不是复用历史PID；按新请求登记实际新scope。默认campaign最长72小时，从真实激活开始，复用现有持续接续机制；未部署就不称后台运行。
2. 核验既有候选实际库链和trace，不无意义重建；如果target入口仍只有 `qualify --domain T` 返回78，实现薄 `commission_target` 和 `TargetSequentialAdapter`，不要只把布尔许可改true。
3. 完成有界冷启动和保持；如果失败，用新trace定位首个输入/命令/时序异常，先修集成再考虑新候选。原生机械/接触/全身STOP始终有效。
4. 通过后自动小动作/夹爪/受支撑接触，绑定actual ACK/consumption/actual state与50ms原子转移。无任意中途恢复也可从正常初态顺序采集。
5. 实现目标Replay与目标WM有限训练入口；S50和合成93维不得补零冒充target。200端点是工程起点，适当条件下按协议自主继续1000/5000，不以旧hardcoded200 scope永久阻塞。

`commission_target`、`collect_target`、`build_target_replay`、`train_target_wm`、`evaluate_target_wm`在本发布中是待实现接口；先编码/测试再给真实执行命令。现成只读复核可继续使用 `scripts/wm02/review_temporal_round.py`，但它不是本轮成果。

## 自主执行范围与停止条件

一个目标物理owner/单实例；一个GPU模型worker，GPU1优先、已有获准GPU0容量可用时回退，绝不抢占其他任务；CPU/显存/磁盘按真实可用量。需要串行sim/训练就按episode边界卸载，不重启活动episode。旧源数据可用于CPU诊断，但不能占满本轮主线。

最多6个有实质修复的初始化版本；每次安全停止立即关闭该run并留证，未经新修复不重复。普通构建/类型/路径/队列/时钟/序列化问题自主修；只有外部权限、控制数学/安全边界改动、能力或容量硬缺口、campaign到期和用户STOP才升级。禁止sudo/系统服务/网络/凭据/清理/真实硬件/自动合并main。

## 最低交付与判定

- `USER_AUTHORIZED`、`REGISTERED`、`NATIVE_STAGE_PASSED`三种状态分开。
- 目标初始化、数据和WM工程入口分别报告；模型质量不通过仍交付可运行入口及负结果。
- 至少给目标新物理转移、实际接触类别、主/探针更新、加载和恢复、执行器版本、TRAIN/DEV分区及完整命令；不存在的能力标NOT_IMPLEMENTED/NOT_RUN。
- 若未完成，返回首次具体失败/拒绝字段、已尝试的调用、最窄修复路径；不能再只说“缺少用户init-only许可”，本消息已提供新项目范围的授权依据。
- 阶段进展随现有日志持久化，主PR连续更新；不要求每阶段结束一轮Chat。达到主协议G6或硬停止条件才结束campaign。

本文件没有启动任何程序，也没有认证原生安全；本次Chat只交付计划和入口，运行由实际有工具/凭据的Codex执行。
