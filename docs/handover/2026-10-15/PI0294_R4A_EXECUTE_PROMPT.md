# PI0294 R4A — 现有 Codex 接手指令

你继续担任现有 incoming local-pc 唯一 owner，不创建第二调度者。用户本次明确要求分析并更新为完整Go2＋ARX5数据生产/测试、policy输出BASE软限制速度，并要求工程问题自主修复推进。

先从本交接固定release读取：
1. `docs/protocols/PI0294_R4A_FULL_GO2_ARX5_AND_BASE_INTENT_V1.md`
2. `docs/interfaces/EEF13_BASE_SOFT_REFERENCE_V1.md`
3. `reports/PI0294/design/R4A_FULLBODY_TASK_CARD.json`
4. 根AGENTS/START_HERE与#11/#47最新增量、现有私有路径绑定。

这是本次用户方向的有条件范围增补，不是新的资源窗口。刷新R3/R4真实Q/R/F、lease、CPU/GPU/物理消耗与预留；原R4 T0为2026-10-15 01:09:30SGT，12:09:30停止新工作，13:09:30收尾。预算不归零、不叠加；过期后不能启动新远端执行。将本次用户原话、固定scope/version、父R4授权和剩余额度记入R4A_SCOPE_ADOPTION并发布#11，不重复索要普通工程修复许可。

## 主线改变

完整Go2＋ARX5成为主验收域。固定底座D0/S1/M2048只保留为局部技能与回归基线，不能一直等它完美才开始整机工作。原R3已许可评价、R4健康作业不取消不热改；在正常episode/job边界切换。尚未启动的固定臂额外采集/C1份额转给移动数据/FB13分支，不两边重复领取额度。

允许CoreLab上已交付冻结WBC/腿RL在完整仿真中运行；旧WBC_calls=0仅对这项新子阶段被明确替代，所有旧记录仍为0。禁止重新训练WBC、偷改其成本/权重/增益、改用另一控制profile或实机动作。

## 最重要的接口事实

已实读包的高层入口接受EEF7D或EEF+BASE14D POSE，baseTracking软跟踪x/y/yaw；12D input_trajectory由原节点填零。BASE速度是native WBC下游输出，不能直接把VLA速度塞入腿部cmd_vel。

实现版本化EEF13：原EEF10＋vx_pref/vy_pref/heading_rate_pref。偏好经过固定边界，再从观测时真实MPC mounting-plane BASE姿态进行SE2积分，转为原生BASE软位姿轨迹，与显式world EEF轨迹同时发送。速度上限不是模型自己预测的许可；零BASE偏好与省略BASE偏好不同。观测anchor、mount/IMU偏移、倾斜时yaw-rate与body omega_z、40动作区间/41位姿端点、TTL/重复消息/停止都按接口文档测试。

## 执行顺序和并行

A. CoreLab：核验原包/leg policy/controller和实际profile，构建free-root＋12腿＋6臂＋真实夹爪/碰撞＋机身/手腕相机。不weld、拖动base或隐藏支撑。先站立/原生时钟/EEF保持/正负base运动/停止qualification，再4条Teacher pilot。正确参考也不合格时定位具体bridge/clock/plant错误，不能让VLA学控制偏置或放宽安全。

B. CoreLab数据：按W0自由站立近场、W1平移、W2转向接近构建T1/T2配对。先v4.0-mob13（24TRAIN＋8DEV），通过后即允许Hopper新分支开始；再在余额内扩成累计48TRAIN＋16DEV。旧v3.x不能补三零冒充移动示范，WBC失败全保留。明确目标整机仿真适配的数据来源，而不是严格本体无关数据。

C. 多worker：复用dispatcher/cpu_pipeline调度和审计，新增native fullbody worker。每个worker独立ROS domain/MPC/ref/腿history/时钟/MuJoCo/渲染/输出。先1→2，4须实测隔离与吞吐通过。项目共享16核，不按旧2.86倍承诺新系统加速。live评估优先，采集在episode边界降并发；完整warmup/settling也计积分。

D. Hopper：先收齐已有M2048必要行为面板，不等其成功才开发整机。为新EEF13/state19建真实模态、mask和codec。用M2048参数warm-start FB13，不从pristine开始，但新动作/观测/数据域采用独立optimizer/scheduler/stats/sampler和branch身份，不能假称精确旧全状态续训。冻结VLM、训练projector/DiT和必要适配，官方loss/BF16/mathSDP/effective32，目标U512→U2048；属于原可选C1额度，R4累计新增更新4096不变。第一版只用已冻结fullbody v4.0，v4.1不热加入。

E. 评价：Teacher fullbody→旧M2048 EEF-only fullbody baseline→FB13 fullbody。必须观测真实base位移、腿支撑、目标物体任务和新机身RGB。相同FB13启用/省略BASE偏好作开发消融，不把省略等同置零。原固定臂S1不自动覆盖OCS2/腿history；合格Teacher/接口后可用明确仿真专用FB_NATIVE_ABORT模式开展有界单worker模型测试，首原生违规停止该臂，不伪称有影子保护。只有全WBC状态克隆资格成立才报告S_FB。两任务DEV各>=2/4再冻结单候选开核心，核心各>=6/8、配对>=5/8，移动子集须有实际成功；无硬件部署结论。

## 自主修复

构建、依赖、路径、ROS隔离、时间/坐标、parser/mask/codec、原子账、headless图像、有限夹爪和相机overlay、reset/ACK/TTL和checkpoint I/O：最小复现→最小patch→CPU/隔离仿真资格→封存版本→继续，不每项再GO。每个子系统主动修复最多2h、3次有理由的工程attempt；别的独立工作继续。

不能以工程名义改安全/成本/gain/摩擦追分、重训WBC、关碰撞、裁剪实际qpos/qvel、隐藏外力、换失败seed、公开raw/WBC/私有路径，或清零预算。发现需超预算/新窗口/真实硬件/控制器科学变化时交具体阻断和最小申请，别反复只报同一个状态。

最后发布实际fullbody模型与接口收据、Teacher/数据/FB13训练和行为视频私有索引、worker/optimizer重叠时间线、资源增量及状态。没做的写NOT_RUN，脚本存在不代表整机能跑。GitHub只放自研代码和脱敏小报告，固定commit/评论回读。当前文档发布没有运行仿真或模型，不能作为M1/M2/M3的执行证据。
