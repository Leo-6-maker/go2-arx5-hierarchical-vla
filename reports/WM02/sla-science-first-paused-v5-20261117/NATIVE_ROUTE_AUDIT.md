# V5 原生接入的源码边界核查

本轮是**只读源码及限定现场检查**，新增模型生成、训练访问、原生请求和积分均0。没有重跑缓存面板、安装hook或改动运行胶囊。核对源码为`980933e0896c247edeb7d9faf66c11fb987ce4b0`；逐文件SHA和真实行号见[NATIVE_ROUTE_AUDIT.json](NATIVE_ROUTE_AUDIT.json)。以下是静态事实，不能替代动态资格。

## 暂停合同已有原生接缝，不能把mock token直接当原生许可

旧`task_clock.require_pause`已要求新ACK序号，同时冻结`step/time_s/motiontime/leg_history_steps/gait_indices`。`fusion_paused_observation._pause/_after_pause/verify_still_paused`还核对原始状态字节、quota、当前进程及场景身份。未来原生接入应复用这些实际反馈，不能只传一个`native_step`整数或`paused=True`。

当前V5 `PausedMockSession`刻意只校验调用方给出的mock clock，发出的token始终`native_resume_authorized=False`，不存在通向`TargetSequentialAdapter.advance_to`的连接。故2/2缓存接缝通过尚未实现或资格化“实际相机→SLA→WBC”闭环。旧原生代码虽有query暂停能力，也不因该函数存在就获得V5新profile许可。

## STOP和参考身份仍存在两个独立的动态缺口

`TargetSequentialAdapter.cancel_reference`收到的是CANCEL生命周期及新HOLD ACK，返回值明确`physics_stop=NOT_IMPLIED_BY_MODE_ACK`。原子推进经`advance_to→resume_clock_once`，ENTER写入不允许在消费未知时重放。不能把取消参考、臂进入HOLD或Python替身的STOP返回当成原生2ms积分停止证明。

候选close fence在自己的`mj_step`入口前用receiver witness判断几何并闩锁SIGINT；其ABI与代码明确`controller_selected_reference_known=false`。现役源的`native_display_command`只列既有trace/mobile safety/outer/observer库，不会因为新profile字符串自动加入该候选库。**这不是完整已加载链的动态观测**；它说明新hook不能靠复用旧启动命令被视为已安装或已资格化。旧`mobile_safety_v1`保留自己的原始规则，并在`next(m,d)`后作其状态检查；不得据此推断整个冻结原生链的全部碰撞机制或停止延迟。

`accepted_boundary`只将确切receiver轨迹与随后到达的solver target配对，仍标记`DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED`。因此最小所需动态证据是：实际控制消费点的selected-policy/target身份与年龄，以及明确装载次序下的检测→不再积分/可靠停止的最坏响应；暂停query期间还要验证wall STOP。没有这些证据，仍不能声称12/20/34ms的持续告警余量足够。该缺口不能由又一轮SLA训练消除。

## 新发现：旧60秒任务段不能直接套用V5的60秒整episode上限

源码`mobile_wbc_reference_v1.START=5703`。旧专家先完成初始化及128个50ms稳定保持原子，随后`mobile_expert_episode_v3`再允许`MAX_QUERIES=120`，即任务段最多30,000 tick。**若把旧程序完整预算直接复用，总原生上界是35,703 tick / 71.406s，而非V5每episode30,000 tick / 60s。** 这是预算上界计算，不是声称旧41次都执行到该上界，也不修改旧结果。

按V5固定“专家8≤240,000原生tick”的整段上界，保持原初始化不动后每episode剩余24,297 tick / 48.594s，可完整容纳97个P10，余47 tick。该算术**不授权自动将查询数改成97或缩短初始化**；未来可执行spec必须显式包含全部初始化、保持、任务及停止尾部tick，并在已签范围内绑定最终max_ticks。若要求保留整60秒任务段，必须在签发前单列额外初始化预算，不能悄悄挤出既定上限。新scope尚未启动，故现在修正计量边界不会重写科学失败。

## 当前签发核查范围

GitHub PR399、PR401均无新评论，PR401无review。Core对7个相关scope的根目录/control共14个明确位置做了限定文件名检查，未发现readmission/safety-decision文件；R1/R3主unit仍inactive、PID0/cgroup空，三lease CLOSED，F1/F2仍隔离。完整范围见[READMISSION_OWNER_SEARCH.json](READMISSION_OWNER_SEARCH.json)。**未提供规范签发路径，不能把这次限定搜索说成全站不存在任何许可。** 需要真实独立负责人提供其签署决定及对应绑定；现有待答问题继续有效，不重复申请GPU或原生任务。
