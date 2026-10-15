# Codex：TF-NIGHT R1启动—参考—成功数据闭环

用户在夜间真实启动安全FAIL后要求更新goal。你仍是现有local-pc唯一owner，不建立第二调度者。

从本文件固定release读取：
1. `docs/protocols/PI0294_TEACHER_FACTORY_NIGHT_R1_STARTUP_AND_REFERENCE_CONTINUITY.md`
2. `reports/PI0294/design/TEACHER_FACTORY_NIGHT_R1_TASK_CARD.json`
3. AGENTS/START_HERE、#11/#47最新事实、私有runmap和原native状态。

登记一次R1_ADOPTION，引用本次更新请求、固定协议、旧night终局、实际余额及继任run。旧GOAL_CLOSED/安全标记/原件仍拒绝旧run；不能直接删除标记或修改旧deadline重启。09:00停止新工作，2026-10-16 10:00SGT硬收尾；旧T0仍00:23:27。读取时已过期就交接，不顺延下一夜。

## 三个实际问题，不再混为“Teacher失败”

- TF-NIGHT真正启动的是T2/H47/closed。Teacher尚未发动作，就在22步/.044秒发生FL_calf_joint原限位越界。初始机器人状态/固定二进制相同，但新旧命令首次生效step2/10、值和指初态不同；这不是已证明的唯一原因，不能加sleep碰运气。
- V3已观察单侧接触，但其独立原件还有raw右指越界，不能当安全成功种子。query14未来reference的连接附近加速度4.012846783大于原1.0，属于预执行参考拒绝，不是实际WBC关节加速度。
- `task_actuation.cpp`仍硬限制quota<=20000，64秒以上任务在旧adapter下不能完整执行。须把资源配额接上真实card/预留，不能改变控制和安全公式。

夜间摘要与V3原件的任务秒数若不同，按run/native_step/task_t0/ref_id对齐，不能用摘要凑出根因；只读原件即可，不为修报告重跑物理。

## 连续执行顺序

A与B先CPU并行：
A. 对照原有startup-v4及night_worker/coordinator，核查进程epoch、mode/ENTER、pause、sensor/command生成接收消费、ROS与旧history。45分钟内给具体差异/假设。只有自有wrapper的可证实生命周期错误或已有受支持原生流程可以正确重绑，才做最小修复及最多4个有界startup验证。原WBC/腿策略/采样率/增益/时钟算法不改；不要复制旧腿命令、跳过启动安全或从站立状态teleport。
B. 并行复用失败轨迹，修任务侧连续连接、技能及跨chunk时参，完整保留实际参考历史与actual状态差别，量化后再次检查原速度/加速度门。实现新的card-bound长episode quota（最高66503，旧adapter原件保留），证明短路径控制等价和拒绝负例。检查gripper工作范围只映射一次。未有成功数据时可以完成这些CPU和导出实现，不伪造seed。

A取得相同域两次安全5秒启动、B相关契约合格后，进入任务。T1/H47/open优先；另task合法则独立继续。源仍用六条自有TRAIN，不下载新库、不重建通用环境、不调WBC。

新Teacher最多3版本、6开发启动，且受原全夜版本/开发上限。source在episode前锁定；真实接触/抓握/推动/释放判断，不让源事件冒充目标成功。允许已声明少量重规划/事件等待，原32秒不够就在运行前封存64/96/128慢域，运行后不延长，慢seed不改判旧32秒结果。

首条完整成功立即存包，最多3个固定独立开发条件中2次成功后，该task冻结进入生产；不要等另一task或例行签字。一条数据一审、一条完成可进入export。目标先8TRAIN+2DEV的单技能闭环，再在余额内争取两任务各12TRAIN+4DEV。共同初态不成立就paired=false，零BASE意图不叫移动技能。

开始时1worker。原生生命周期、ROS/MPC/history/时钟/quota/输出真正隔离且同版本资格成立才2worker，本修订不直接上4。总8CPU核包含采集/规划/腿RL/渲染/审计。串行能产数据就继续，提速不是新的永久前置。

冻结TRAIN-only统计/EEF13模态和真实数据；独立进程实读全部成员hash、全部numeric窗口及阶段真实RGB，走官方CPU processor/collator。不要加载GR00T/PBS。私有远端容量/认证不足只阻断远端交付，先完成本地bank+独立消费者，不转public、不删旧资产。

## 原安全终局与原余额继续有效

原night真实安全FAIL已1；R1若再出现任何真实native或高频机械范围违规，立即停止所有新物理（全夜上限2不能清零），转只读/CPU/已有成功bank交付。预执行gate拒绝计任务/规划失败但不伪称物理事故。未知共享启动问题且无范围内可执行修复时不原样重试，给出一个具体阻断，不开WBC调参项目。

R1新增最多32coreh/8核（与原64coreh余额取小，至少6coreh留收尾）、52启动/3458156步/133172双RGB；各scope已用/预留全部扣。旧全窗口27.763coreh保守上界不能当0或只用638.9秒metered值得到虚假余额。完整搜索/磁盘上限见机器卡。新GPU/PBS/训练/高层推理/真机/大下载/删除0。

修配额、IPC、序列化、身份、时钟表示和参考时参等明确范围工程问题，最小patch→受影响测试→冻结新source→正常边界继续；不每项再问GO，也不无限改名复试。每根因90分钟主动处理后收敛，独立工作照常推进。

## 最后交付

首先报告：startup是否恢复、完整成功seed几条、实际contact/grasp/push、TRAIN/DEV/配对数、真用几个worker、独立消费者是否读过，随后说明工程。原全部失败与实际成本保留。

新增R1协议/运行收据、startup lifecycle差异、task quota等价、reference continuity、Teacher/version、真实视频私有索引、bank/modality/stats、consumer、资源和恢复文件直接提交GitHub；#11科学/#47工程并回读。没有bank不写空集PASS，没有执行不写完成；selected/deployment始终NONE。10:00前自有worker终态、预留结清，不能以新protocol重新开始旧夜间时钟。

本文件发布仅完成任务交接，本次对话没有实际启动CoreLab。你登记真实采用并核实许可/余额后，按上述有界范围推进，不停在CPU_READY或首成功。
