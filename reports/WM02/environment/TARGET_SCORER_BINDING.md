# WM02_STANDING_T1_V1：独立版本的站立物体任务评分

2026-11-04。实现为 [standing_t1.py](../../../src/go2arx5_vla/world_model/standing_t1.py)。状态为 **OFFLINE_DEFINITION_FROZEN_NATIVE_SENSOR_BINDING_PENDING**；不表示目标任务完成或已获动态许可。

完整复用原`GO2_PLACE_V1::score_episode`，源码SHA `8da98606b9301c789ec1619d15ddae50702f94205e7d78025dd2d4a47fa32720`。原`task_fullbody_audit.py` SHA `4447f2620af4a8c243aae39c9383fb4334d1d2bcae74a1a637e915bcdac58b6c`不变，仍保留其card的额外BASE前移目标；新任务不输出旧全身任务分数。

| 定义 | V1冻结值／要求 |
|---|---|
| 几何与目标 | 原目标T1的4cm立方体、桌高.44m、目标中心高.46m；goal XY来自实际card，初始物体与t=0传感状态必须一致，源目标XY距离≥.15m |
| 时钟 | 0…32秒完整641个20Hz观测；无缺帧或插值冒充；28…30秒41个持稳样本全部满足 |
| 抓持／抬升／运输 | 原calibrated grasp和有效双侧物体接触；持续抓持与抬升各≥.30s，抬升≥.02m，抓持时XY搬运≥.10m |
| 放置／释放／撤离 | 物体所有角点位于目标±.07m区域，中心高度误差≤.005m；线速度≤.02m/s，角速度≤.10rad/s；native grasp false，TCP离物体≥.08m |
| 原生安全 | 原object/contact/gripper validity、safety、abort、illegal contact、joint limit、reset、takeover判据全部不变；另要求完整free-body pose流与原全身安全monitor逐帧有效 |
| BASE | 自由底座仍需真实记录，不能焊死或移除腿碰撞。本站立任务不设额外BASE前移成功目标；最大实际位移仍报告。原全身BASE目标和历史分数不变 |

原`ActualState`可重建物体与工具、finger q及接触，但它本身不证明完整全身controller safety。新增`native_fullbody_safety_valid`必须来自原安全链；不能以物体contact-safe替代腿部／机身／MPC abort检查。该传感绑定、实际完整目标card/原始高频数据审计仍待E3；wrapper自身始终报告`native_qualified=false`，不颁发资格。

CPU合成正例通过，失抓、未释放、碰撞、未撤离、全身安全无效五类反例均失败；源32秒真实成功或目标真实成功均没有由这些fixtures新增。相关原有scorer／SO3／SafetyGate回归70项通过，旧阈值没有修改。本轮没有执行新的S域释放撤离或目标原生任务。

E1 Actor采用明确标记的合成工程目标（物体到目标平方距离与动作代价），不是本文尚待native绑定的训练奖励；没有将它称为P4正式WM-Actor或通过原生奖励排序。正式reward五类真实记录核验、reset bank、真实TRAIN数值replay和完整任务学习闭环仍未完成。
