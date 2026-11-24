# 原生相机／ARX5 迁移：进度与采集来源审查 · 2026-11-12

## QUESTION

目标侧现在应推进到哪里？公开 Table30 的 `joint_positions`、`end_effector_pose` 和视频／状态时间对应，哪些已经有证据，哪些仍不能确认？

**结论：相机与目标观测版本已经通过；下一主线是官方父模型的真实 GPU 训练资格，随后完成 A_TARGET500。源采集来源缺口只阻塞 B_SOURCE。** 现有证据尚不能把 Table30 的字段来源和原始采集同步认定为已闭合；没有源准入不能解释为公开迁移负结果。

本审查只读分析已发布证据，没有启动训练、仿真、远程连接修复或对外联系。后续执行顺序见 [目标侧优先计划](../handover/2026-11-12/NATIVE_ARX5_TARGET_FIRST_NEXT_V1.md)。

## 证据截止与身份

- 执行 PR：[#376](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/376)，本次固定读到 `41111540d33398e916224dd07cd2638db2a5675f`。
- 计划基线：[V1 协议](../protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md)，初始计划提交 `ade5ab354af0622741aea92995e5643b4d8106d0`。
- PR 的公开更新为 2026-11-12 02:10:02 SGT；实际 TRAIN4 组最后读回为 **2026-11-12 01:11:30.844 SGT**。此后本地 Core relay SSH 超时，当前远端状态未获确认。
- 下列“0 更新”是该固定回执的实际计数，不是对稍后机器活动的实时判断。其余三个 TRAIN 条件未读回，不能填成失败、成功或“仍在运行”。
- 执行分支的旧根状态页仍包含较早实验叙述；本审查以本轮具体 ACTUAL 回执为准。

## CHANGED / ACTUAL：已经完成到哪一步

| 项目 | 实际证据 | 现在的含义 |
|---|---|---|
| 新 body 相机及目标数据 | 100 episodes、1,500 captures、1,450 监督窗口、50 干预 captures 完整保留；动作／状态／mask／标签／统计与原版一致 | 目标观测版本已固定，不能为方便删窗口或重新拟合统计 |
| 全量视野 | body 的 8 px amodal 边距 1,450/1,450；PREGRASP、DESCEND、CLOSE、LIFT、HOLD 五阶段 raw 和真实最终 Processor 后可见性均通过 | P1 通过，无需重复预览或换相机 |
| 同卡 SFT2000 回放 | 13 GiB allocator 上 16/16 普通生成／解码完成，8 个唯一窗口各重复两次，结果 bit-exact | 新观测的实际推理链可运行；这不是新官方起点的训练资格 |
| raw H40 | 同一回放的 raw H40 包络 0/8；8 个窗口均不满足平移离散加速度条件，1 个还不满足夹爪速度条件 | basis 仍承担轨迹整形，不能宣称原始 H40 已可直接执行 |
| 完整整机 TRAIN4 | 登记 4；已读回终态 1；该终态成功 1；另外 3 未读回 | 只有 TRAIN01 是已证实的完整终态，不形成最终 1/4 成功率 |
| 官方父模型字节身份 | 7 个文件全部流式 SHA256 核验，总计 6,914,922,850 bytes | 下载身份通过，不等于实际模型加载／训练通过 |
| 新训练软件 | 44 项软件检查；loader、cache、batch32 对照、恢复与日程代码已准备 | 实际官方加载、backbone calls、资格更新仍为 0 |
| 候选训练 | A_TARGET 0/500；B_SOURCE 0/500；B_TARGET 0/500 | 本轮还没有新 A/B 检查点或迁移效果结果 |
| 源数据 | 30 项 task metadata 已读，11 项为 ARX5；8 episodes／5,837 状态行已做有限 FK 审计；1 episode 两路视频已查 PTS | 几何／容器时间结构得到有限支持；准入 H40 窗口仍为 0，source training 未准入 |

原始回执：

- [P1 全量相机与实际 Processor](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/41111540d33398e916224dd07cd2638db2a5675f/reports/WM02/native-arx5-transfer-20261112/P1_CAMERA_TRAIN100_AND_FINAL_PROCESSOR_ACTUAL.json)
- [SFT2000 同卡回放及失败保留](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/41111540d33398e916224dd07cd2638db2a5675f/reports/WM02/native-arx5-transfer-20261112/P3_SFT2000_SAME_CARD_REPLAY_ACTUAL.json)
- [首个完整 TRAIN 终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/41111540d33398e916224dd07cd2638db2a5675f/reports/WM02/native-arx5-transfer-20261112/P3_TRAIN4_FIRST_COMPLETED_CONDITION_ACTUAL.json)
- [官方七文件身份](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/41111540d33398e916224dd07cd2638db2a5675f/reports/WM02/native-arx5-transfer-20261112/P3_OFFICIAL_PARENT_BYTE_IDENTITY_ACTUAL.json)
- [训练软件准备／实际零计数](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/41111540d33398e916224dd07cd2638db2a5675f/reports/WM02/native-arx5-transfer-20261112/P3_TRAINING_SOFTWARE_PREPARATION_STATUS.json)
- [全部任务 metadata](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/41111540d33398e916224dd07cd2638db2a5675f/reports/WM02/native-arx5-transfer-20261112/P2_ALL30_TASK_METADATA_ACTUAL.json)
- [FK 与视频时钟有限审计](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/41111540d33398e916224dd07cd2638db2a5675f/reports/WM02/native-arx5-transfer-20261112/P2_FK_AND_VIDEO_CLOCK_BOUNDED_ACTUAL.json)

### TRAIN01 成功的准确解释

方法是 **SFT2000＋固定 endpoint basis＋原消费／控制链＋新 body/wrist 输入**。该条件实际完成 15/15 查询，21,203 个 2 ms 物理步；最大抬升 26.1951 mm，连续真实抓持／抬升／保持 5.95 s，unsafe samples 为 0，技术有效且正常退出。

该成功按既有 `TASK_LEARNING_GRASP_STANDING_SCORE_V3` 成立：base 位移上限 20 mm、旋转上限 5°、足平面位移上限 5 mm、至少 3 足支持，支持力阈值 0.05 N。实测 base 最大位移 6.6936 mm、旋转约 0.8508°、足平面位移 3.9193 mm、最少 4 足支持。旧短任务 2 mm／0.5° pose 诊断仍为 false；它必须保留，不能把 V3 成功写成旧诊断也通过。

这是新观测与整机链路存在可行案例的证据。它没有检验新官方父模型微调、公开数据增益、独立 DEV 泛化、零样本能力或 WM；也不能从一次成功倒推出全部相机位置或任意基座运动均可靠。

## Table30 官方链接：能证明到哪一层

本轮核查了官方 dataset 文档／转换代码／讨论、RoboChallenge 论文与公开推理仓库，以及厂商 X5 SDK。**未找到与固定 Table30 revision 的实际 ARX5 采集 writer 相连、同时闭合反馈来源和原始跨流时间对应的直接证据。** 这是当前检索结果，不是断言该代码或记录不存在。

| 一手来源 | 确认的内容 | 不能据此确认的内容 |
|---|---|---|
| [ARXroboticsX/X5 README，固定提交](https://github.com/ARXroboticsX/X5/blob/9a255e38156e3f5a263f341df209ab028d49dad9/readme.md) | 厂商把 `get_joint_positions`、`get_ee_pose_xyzrpy` 等列作反馈读取接口，和 set 接口区分 | Table30 采集器实际调用了哪个函数、哪版 SDK、主臂还是从臂 |
| [X5 Python single_arm.py，固定提交](https://github.com/ARXroboticsX/X5/blob/9a255e38156e3f5a263f341df209ab028d49dad9/py/arx_x5_python/bimanual/script/single_arm.py) | joint getter 直接调用底层 getter；XYZRPY getter 取底层 pose 后做四元数到 RPY 转换 | Table30 JSON writer 的赋值调用链；底层闭源实现如何获得该 pose |
| [RoboChallenge 官方 mock_rc_robot.py，固定提交](https://github.com/RoboChallenge/RoboChallengeInference/blob/4f8f9304c87a932fa2a05dddb352231c77cf8ddb/mock_server/mock_rc_robot.py) | ARX5 回放按序取一行 state，并对各视频顺序取一帧；`joint_positions` 和 `end_effector_pose` 直接填入回放状态 | 历史采集器是否这样同步、丢帧／重复帧如何处理、曝光时刻与行时间差 |
| [官方 mock_robot_server.py，固定提交](https://github.com/RoboChallenge/RoboChallengeInference/blob/4f8f9304c87a932fa2a05dddb352231c77cf8ddb/mock_server/mock_robot_server.py) | 回放服务生成当前服务时间戳；mock 图像接口同样使用 `time.time()` | 回放时戳不是历史曝光／关节采样时戳，不能反推原采集同步 |
| [RoboChallenge 论文 §2.1–2.2](https://arxiv.org/html/2510.17950v1) | 系统使用机器人驱动、带时间的相机流，并另有示教采集软件 | 没有公开逐字段 writer 赋值或 Table30 frame-to-state 对应细则 |
| [固定 Table30 README](https://huggingface.co/datasets/RoboChallenge/Table30/blob/1eda76b8935b5c7a59cd635ca9e3fab882943e11/README.md)、[转换示例](https://huggingface.co/datasets/RoboChallenge/Table30/blob/1eda76b8935b5c7a59cd635ca9e3fab882943e11/convert_to_lerobot.py) | 提供 schema 和索引转换惯例 | 字段说明及转换器都不足以证明实际反馈生产者和曝光同步；转换器没有实现本轮 timestamp／PTS 合同 |

官方回放代码是比“字段名字像反馈”更强的使用约定证据，但仍是 **consumer 证据**，不是 **collector 证据**。代码中的近似 30 Hz 回放循环、相同帧号和重写的服务时间都不能取代原采集时间。

### 已读数值支持什么

1. `open_the_drawer` 与 `put_cup_on_coaster` 各 4 个 episode，共 5,837 行，与已登记的 X5 FK、145 mm TCP、`Rz(yaw) @ Ry(pitch) @ Rx(roll)` 弧度约定相容。最大 XYZ 差约 0.646790 mm，其中一个 535 行 episode 达到浮点精度一致。它支持该子集的数值约定；`q_actual → FK(q_actual)` 和 `q_command → FK(q_command)` 仍可产生同样的一致性，不能仅由误差判断来源。
2. 已读 8 个 episode 的 state timestamp 量级约 1.757e9、median dt 约 0.033，且匹配 episode Unix 起止时间，支持单位为秒。README 的毫秒文字不能覆盖实际值。新 reader 应对其实际纳入范围逐一验证；不得把全部 timestamp 无条件再除以 1,000，也不得自动外推到未查任务。
3. `put_cup_on_coaster/episode_000000` 的 wrist/global 均为 437 帧、640×480、30 fps，time base 1/15360；两路容器 PTS 相同，范围 0–14.533333 s。`state_timestamp - episode_start - PTS` 为约 5.232–16.811 ms。这里测到的是容器时间关系，不是最大 image age 或两路物理快门 skew。
4. 任务应按 `robot_id`／metadata 选择。全 30 task 中实际识别到 11 项 ARX5；`stack_color_blocks` 是 UR5_2，不能凭任务名作为 ARX5 数据。

因此本轮可记录 `NUMERICAL_CONVENTION_VALIDATED_ON_AUDITED_SUBSET` 与 `OFFICIAL_REPLAY_INDEX_CONVENTION_FOUND`，同时保留 `CAPTURE_PROVENANCE_UNCONFIRMED` 和 `SOURCE_NOT_ADMITTED`。

### 关闭来源缺口需要的最小证据

不要求全部采集系统开源，也不要求每个 MP4 自带绝对 UTC。可信的维护者说明、采集 writer 片段或原始记录说明，只要能与固定数据版本和实际 ARX5 范围关联，并回答以下问题即可继续验证：

- `joint_positions` 来自从臂实际反馈、主臂位置还是发给从臂的命令；`end_effector_pose` 由哪个 getter 或哪组关节 FK 得到；使用哪个 TCP／frame。
- state timestamp 的单位、时钟域，以及采样／接收／写文件中哪个时点被标记。
- 每路视频的 frame ID／PTS 与 state 行如何对应；缓冲、offset、丢帧和重复帧如何处理。
- 有足够的时间界限或原始映射，能够检验本轮已固定的 image age≤50 ms、双路 skew≤25 ms；只有“都是 30 Hz”不够。

不因缺相机内外参就额外阻塞普通双 RGB VLA；本轮需要的是正确的 EEF／TCP 语义与图像状态对应。若未来开展几何投影或 WM 才另行定义所需相机标定。

## NEGATIVE：必须一起保留的结果与成本

- 首次 12 GiB allocator 回放尝试加载了模型、发起 1 次生成、完成 0 次，在 LM head CUDA OOM；不能从成功版报告中删除。其后独立 13 GiB 版本才完成 16 次生成。
- 13 GiB 成功回放 peak allocated 为 13,301,736,448 bytes（约 12.39 GiB），wall 226.5671 s；这不能证明新训练的 backward、Adam 与 cache 在同卡可容纳。
- 目标重渲染 wall 325.8069 s，最终 Processor 检查 20.4208 s；官方七文件核验 4.6811 s。TRAIN01 wall 859.4601 s、CPU 1,200.7111 s、仿真物理时间 42.406 s。这些口径不同，不相加成未经核实的总 GPU 成本，也不能直接当新 A500 ETA。
- 原 raw H40 0/8、旧短任务 pose false、原跨硬件 BF16 REJECT、旧 DEV／TEST、CORR2048／SFT2000／C500 与用户给出的 Teacher24 最终 19/24 负结果继续保留。Teacher24 完整终态 receipt 仍待补，不用旧部分回执覆盖用户给出的最终结果，也不为补证据自动重跑。

## DECISION / NEXT

1. 只读收齐原 TRAIN4 组状态，先消除“远端仍占用还是已结束”的不确定性；不重派未知状态的条件。
2. 目标普通 RGB endpoint 重建与 source reader 分开。前者属于 A 的交付依赖；后者只属于 B。
3. 完成新官方父模型、真实 cache／梯度／恢复的有限资格，随后 A_TARGET500；A500 后固定四 TRAIN 交付诊断，不打开锁定新 DEV，也不按诊断分数追加训练。
4. 源分支只补能改变来源判断的证据，并行做 native TCP／frame 映射审计与未开结果的新 DEV 身份检查；不把扩大 FK／PTS 抽样当作来源证明。
5. A500 交付时若源仍未准入，报告目标侧结果与 `SOURCE_NOT_ADMITTED`，本轮公开迁移问题暂未回答。若 B 完成，才进入原固定 A/B DEV 首筛与有条件扩样。
6. 当前没有新证据要求引入 WM、解冻视觉骨干、改变 WBC 或换大卡。先取得新父模型在当前目标数据上的真实训练与闭环失败谱系，再决定下一轮研究问题。
