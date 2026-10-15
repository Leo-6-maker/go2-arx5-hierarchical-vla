# TF-NIGHT R1单项后继草案：固定夹爪终点，改变入口请求闭合曲线

**NOT_AUTHORIZED / authorizes_execution=false / launch_allowed=false。** 本文件是原active goal下的有界接续草案，不恢复任何旧失败source，不启动新物理。原6个有效科学条件已耗尽，需要明确采用下述**新增1个科学开发条件**；不能仅用工程debug豁免把真实安全FAIL去掉。原累计总资源上限与原T0/09:00/10:00窗口不增加、不重置。若采用时已经超过2026-10-16 09:00SGT，不得据此开新任务。

## 1. 已保存实际证据与反证

最后固定T2/PC2/full Go2＋ARX5运行在task6.990s首次右指q<0，6.996s达−0.229538mm；当时尚无物体接触。18安装/140RGB/5998物理包独立复算，夹爪原PD逐步误差0N，目标22mm→2mm，无丢包或二次映射证据。原安全FAIL、版本隔离、所有raw成本保留。

新增13个保存状态只读mj_forward分解：右指bias约0.376—0.451N；停止时ctrl为向外+0.976345N、被动阻尼+0.085487N、关节约束项+0.189941N，而右指速度仍向内−0.028496m/s。进入越界时仍有动量，随后在减速。bias/kp400的静态量级最多约1.13mm，小于2mm工作裕度；**不能把它简化成“静态重力偏置超过裕度”**。这些是post-step clone量，不是原求解器warmstart/完整力因果干预；不能宣称唯一根因，也不调整重力/增益/物理。

目前Teacher闭合请求在20Hz从中点直接跳到closed；原执行器再以0.04m/s限速逼近，末端目标速率突然停止。唯一待检验假设：入口请求序列在闭合末端提前降速，可以在相同2mm终点与原执行器下减轻动态穿越。它不是已证明的修复。

实际12条development/失败记录均被现有数据准入拒绝，3条无任务audit、9条有独立audit；成功export、statistics拟合、实际consumer仍0。这只是实际负例准入检查，不能写成数据闭环PASS。

## 2. 唯一变化

仅T2入口的**20Hz请求夹爪setpoint序列**由step改变为五次进度曲线：

`q(u)=q_mid−(q_mid−q_closed)(10u³−15u⁴+6u⁵)`，u从0到1。

q_mid=22mm，q_closed=2mm、机械范围0—44mm、物理actuator slew40mm/s、原PD400/3、force±8N、被动阻尼3全部不变。逻辑量仍经原映射一次，标签记录真正提交的值，不裁剪实际状态或学生输出。

曲线时长只由现有速度上限解析给出：`ceil_50ms((15/8)×20mm/(40mm/s))=0.95s`。不扫描其他profile/时长/终点。保持原实际entry计算得到的闭合名义到达时刻与50ms ACK余量：把开始提前0.45s；保存E4输入对应从6.00s开始、6.95s目标到达、7.00s ACK截止。真实新startup必须从自身实际entry/原neutral接触预览重算该起点；不能复制E4 qpos或原生控制/history进入轨迹。

原EEF路径、姿态、时参、2cm入口、源skill夹爪值、源事件、相机、实际观测、原H40以**当前真实夹爪**做碰撞认证、SafetyGate和最终物体评分均不改。不改task_actuation的PD或slew，不新增command ABI。继续原ROS/policy/clock修复版本；不假装coverage证明最新ref消费。

这是采样setpoint设计，**不声称原2ms slew输出或实际手指获得C2加速度**。20Hz采样与原slew仍可能产生速度折角；CPU只证明目标范围、单调性、最大采样增量、原映射/时刻一致。动态安全和任务成功必须用新实际结果判断。

## 3. 采用后的固定顺序与停止条件

1. 窄实现新的版本化bridge；复用已存数组，核对原EEF/source值逐样本不变、0.95s唯一时长、阶段/标签/时间、正负例。旧CPU/原件/旧stat不重建。
2. 在原样真实startup后，重新绑定当前模型/输入/2cm入口/RNG；原native当前夹爪几何、完整H40、history、速度/加速度、原gate均须通过。静态FAIL是有效失败，不能调起点或换距离；CPU不能代替动态准入。
3. 同一锁定来源执行唯一96秒完整development条件。0.4秒安装前缀、20Hz真实观测、原安全、完整记录不变。任何真实安全FAIL立刻结束、隔离新source；不再尝试另一profile/终点/seed。
4. 若完整原物体任务PASS，按原R1既定repeatability/生产/导出/消费者链自动继续，不再等例行GO、不停在首seed。原重复性门和固定scene-group/split全部保留；DEV不调Teacher。原总资源/搜索余额不足时停对应阶段，不能重置预算。
5. 若唯一新条件有效FAIL，交完整负结果，停止这一假设。工程同语义缺口可按已采用debug规则修复，原始attempt/曝光/资源全累计；不能将真实安全FAIL改成工程FAIL。

## 4. 精确预算

只请求将原有效科学development上限6扩展到7，新增1条件；不增加原52总启动、3458156积分、133172RGB、32coreh及搜索总上限。新条件最多1次有效物理启动，5秒startup＋96秒task＋3保守步=50503积分，1921双RGB；原生worker≤4核/7200 CPU秒，相关CPU资格/审计≤1800秒，总计≤9000 CPU秒，均花原余额；保留原6coreh收口余量。

入口1＋240个固定H40最多241bundle，每bundle仍受原12goal/192内部初值/8RRT与60s上限；总预算取原剩余与本卡较小值。该条件连同入口、H40、closure forecast及只读审计的static/dense预留≤350000点；内部静态/局部IK/审计也照原账。不得重新跑全历史或载入神经模型。工程恢复仍花原总52启动/积分/CPU/RGB，不新增独立无限资源池。

重复性与成功后的生产只用原R1已经授权的剩余槽和预算，不在本草案另发资源。全流程无GPU/PBS/GR00T训练/真机/清理/monitor/mail/merge，selected/deployment保持NONE。

## 5. 当前交付身份

[机器卡](../../reports/PI0294/design/closure-reference-draft-20261016/TASK_CARD_NOT_AUTHORIZED.json) · [纯CPU请求曲线收据](../../reports/PI0294/design/closure-reference-draft-20261016/CPU_PROFILE_RECEIPT.json) · [已保存证据摘要](../../reports/PI0294/design/closure-reference-draft-20261016/SAVED_EVIDENCE_SUMMARY.json)。运行实现NOT_IMPLEMENTED；新geometry/physics/成功seed=0。旧科学停止报告不改，本草案发布不等于采用。
