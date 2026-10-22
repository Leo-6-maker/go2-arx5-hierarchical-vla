# A2运行期间：两个并行agent推进B训练与解耦评价

日期：2026-10-22，Asia/Singapore。任务ID：VAP1_A2_PARALLEL_AGENTS_V1。
状态：USER_AUTHORIZED_HELPER_DISPATCH_PENDING_NATIVE_ADOPTION。
授权依据：用户报告A2已实训到2053，并明确要求「允许agent同步推进，加速试验主线，发布任务到github」。
审核基线：9422d1690f053302a34e42196d560dd45f754c28。既有主卡：reports/PI0294/design/parallel-science-20261022/GOAL_SUPPLEMENT_AND_PARALLEL_TASK_CARD.json。

**本文件只把已经存在的并行准备分成两个可派发agent任务；不是新实验、新GPU队列、新数据计划或第二个native owner。** 用户这次明确允许并行辅助agent，替换旧交接中「不启动子agent」这一组织限制；单一原生调度者、单一集成发布者、原科学配方和资源上限全部保留。无需再次询问同范围的agent派发许可。先复用现有两个worker槽位；若它们仍在工作，不另外再开两个。

## 1. 起点：从最新证据继续，不重开已完成工作

截至已发布的2026-10-22 20:45:26 SGT现场观察：
- A1已经成功终态、1024及2048全状态审计和结算关闭；不能再次结算或重跑。
- 当前A2为1475719.stdct-mgmt-02，R；运行source 3a1420e76c492196fa83905f4bde92924cc12c28，精确CI39777328633/job119819488833通过。
- 原A_CONTROL_SAVE_RECOVERY1从2048继续；已commit到2053/cursor65696，新增5步、17次兼容查询、2次模型加载，无unknown。最新完整可恢复点仍2048/cursor65536；不能把live2053当成完整checkpoint。
- A2原request 9d1582d5-1c87-41bf-ad53-4e049cc21fe9、nonce3104eb01-637c-4afa-b33e-9aca24bd8204、request SHA298d68cf547e1540e2a259c0e5318e5bb5c7a07bccb3afea01dae43878d67200、manifest ae5acd27be71fb05c46c8a801c7b080877e5887ec9fbc481171aa5cea5d066e1保持。
- 已结算GPU秒40813（旧账字段名H200_s沿用，不能把A40消耗误写成H200独有）；PBS17、更新3072=旧丢失1024+恢复A2048，queries1117。A2 live新增更新在终态再结算，不能提前重复加账。
- 当前预约22560 GPU秒/2048更新/29queries/54000 CPU秒/70,000,000,000输出字节。不释放、重复扣除或挪给agent。
- B1 V3元数据和31项检查已完成，A2终态adapter及8项CPU检查已完成；尚无B1提交。数据索引、曝光表、DEV剩余键、DEC1结果模板已经存在，禁止再做一套。

现场可能已前进，owner刷新后覆盖快照；计划发布不是原生状态查询。原1735?之类无关历史标识不使用；只以以上确切job及最新真实runmap为准。

## 2. 并行拓扑与权限

| 执行者 | 唯一职责 | 允许的输出 | 禁止事项 |
|---|---|---|---|
| 原owner | A2观察、认证/heartbeat/release、实际终态审计/结算；随后B1/B2唯一派发；集成发布 | 原native收据、冻结源、原预算内合法调度 | 不能把辅助任务的未完成当作健康A2停机理由 |
| Agent T：TRAIN_NEXT | B1/B2已实现接续链的独立检查与必要最小修复 | 隔离开发区小型patch、targeted fixtures、一个readiness结果 | 无SSH凭据、qsub/release/lease/ledger写入、模型加载或训练 |
| Agent S：SCIENCE_NEXT | DEC1源侧实际接线缺口、目标ABI依赖、结果聚合工具 | 隔离开发区小型patch、纯CPU验证和接口缺口清单 | 无目标仿真、任务执行、WBC修改或研究方法调参 |

最多两个辅助agent，复用现有槽位；不递归生成子agent、不创建永久watcher。原owner继续现有原生心跳；交互观察沿现有低频规则（不增加独立远端poller）。辅助agent读取owner已批准的脱敏快照或只读源码，不能各自连接服务器取一套不一致状态。

每个agent使用已登记开发根中的隔离worktree/输出目录，固定base SHA与互斥文件归属；不能同时修改共享NFS checkout、活跃capsule、RUN_MANIFEST或源输入。agent只交patch/测试/结果，owner是唯一merge与GitHub集成发布者。已有同名任务或产物成立则增量复用，不重置或复制。

## 3. Agent T任务：A4096结束后尽快接上B，而不是继续开发调度框架

任务ID：T_B_CHAIN_READY。优先级P0。

### 输入直接复用
- 当前主卡的cold_training_preparation.parallel_continuation.B1与A2_terminal；B1元数据SHA 3b4e33f3995c3a778a96096f341766fda069bd4c8a0396c76e987fe1900250d4。
- 原成熟source_pair_inputs、segment intake、typed accounting和保存恢复实现；原1024 TRAIN探针合同。
- DATA_ASSET_AND_PURPOSE_INDEX.json、EXPECTED_TRAINING_EXPOSURE.tsv与已CLOSED的原139、Original126/C1消费收据。

### 只补剩余缺口
1. 对现有V3输入做独立差分：B必须从原CORR@2048建立新optimizer与B sampler，start0、resume=None；不得残留A恢复路径、A lineage、A optimizer或A composite identity。R0、effective32、原B数据配比、seed/loss/精度均固定，不混CR2/VAP1的R1数据。
2. 用小型fixture走通「A4096真实终态/无unknown/独立全状态验收/一次结算关闭 → B1完整future预算与身份 → B0→2048 → B2完整恢复2048→4096」。只有既有qualified入口可派发；fixture不写真实grant/lease、不调用PBS，不伪造A4096来源。
3. 覆盖失败路径：A仍R、4096不完整、错误arm/recipe、缺1024探针、已结算重复、partial copy、stale源、未核清effects、存储/CPU不足必须拒绝。复用原测试，不再复制整个source runner。
4. 若发现已证实的工程缺口，提交一个最小同语义patch及针对性测试。没有缺口就回READY，不为显示进展添加helper。A2活跃source保持不变，补丁只供后续冷态准入。
5. 明确B2已有何种代码与CI、还缺哪份原生输入；不得把CPU假件PASS称CUDA恢复成功。

### 交付与完成条件
在既有parallel-science目录增加一个小型B_CHAIN_AGENT_REVIEW.json：base/输入身份、复用证据、逐项PASS或具体阻断、patch/test、expected-next-owner-action。所有真实准入字段仍为PENDING_UNTIL_A4096_CLOSED。
本任务不要求重下数据、重hash整个模型或重新跑全仓测试。既有实现已经满足时，检查结论即交付；不得因为agent尚未交报告而阻塞原来就合法的B1。

## 4. Agent S任务：把DEC1从模板推进到可交付的接线补丁

任务ID：S_DEC1_WIRING_AND_RESULTS。优先级P0；不阻塞源域训练。

### 必须复用的既有内容
- EVALUATION_AND_CLAIM_PREPARATION.json、DEC1_64_CELL_RESULT_TEMPLATE.tsv、SAVED_DEV48_REMAINING_CONDITIONS.json。
- reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_SOURCE_ONLY_INTERFACE_FREEZE.json。
- 已有execution_progress.py及FRAME_CORRECTED_REFERENCE_V2。I_TIME/I_PROGRESS参数已经source-only固定，不以当前模型或目标结果重调。

### 当前已证缺口与工作
1. 源侧接线审查：原raw完整H40合法性/S1先于映射；mapped完整参考合法性；mapped prefix shadow与tick；实际TCP/夹爪反馈和明确ACK；latched exception后的真实stop。沿既有executor接口写最小连接补丁或证明现有代码已覆盖，不另建物理执行器、安全判定器、preview规则或第二套评分。
2. 纯CPU fake-transport/保存的合格小数组测试，明确mock不是原生物理。至少覆盖raw非法不执行、映射非法不执行、反馈过期/乱序拒绝、chunk reset与gripper顺序、stop锁存、相同进度时两模式共享语义，以及每条执行证据唯一归属。禁止CPU测试启动MuJoCo积分或神经模型。
3. 核对目标团队冻结EXECUTOR_MANIFEST是否真正交付。交付须包括机器人/控制器权重配置版本、frame/TCP/夹爪语义、状态时间、native安全/reset/stop与可调用入口。缺什么就列什么，WBC归档存在不等于ABI合格。不虚构目标、不焊base、不读target真值调高层、不修改WBC；无target不挡训练。
4. 用已有结果schema实现或验证最小聚合入口：区分SUCCESS、有效任务/参考/S1拒绝、原生安全错误、ENGINEERING_MISSING与NOT_RUN；配对分析按8个几何块而不是64独立样本。测试缺失键、重复case、不同模型/噪声混入应被拒绝；不把未执行格填0成功率。只处理已允许的脱敏记录，不读取未开放LOCKBOX。
5. 分开两张表：A/B源域DEV48研究覆盖收益；DEC1首轮固定原CORR/R0的8几何×2任务×2执行域×2接口研究迁移。不得换成A4096或未来B赢家、重新抽容易场景、提前用target调参数。

### 交付与完成条件
DEC1_WIRING_AGENT_REVIEW.json＋如有必要的最小代码/test diff＋owner可冷态调用的已有入口/确切缺口。必须列NATIVE_COMMISSIONING_NOT_RUN、TARGET_NOT_DELIVERED（如果仍缺）及实际source qualification前提。只多写一份空模板不算完成；若外部ABI缺失，完成源侧可做补丁后停止等待，不绕回新数据/WorldModel/Teacher项目。

## 5. 两个agent的共同资源与工作量边界

使用现有本地/管理CPU可用余额，**不新增或重置任何总预算**。首批两agent合计本地CPU上限7200秒、最多并行2个单核进程、墙钟最多6小时且不越过原截止；小输出总计≤16MiB、从已发布小收据获取的传输≤32MiB。不下载模型/图像库，不复制多GB checkpoint，不在Vanda计算节点抢A2的4CPU或磁盘。

这些是派发内界而非额外额度：owner先用原全剩余预算确认有真实余额并沿原机制计量一次；不足则缩至一个agent或只读代码。该限制只约束本地CPU执行，不假装已经知道代理服务/模型token额度，沿既有Codex账号及其限制使用，不新购服务。

原最新存储授权1abb20973d84836b29baebf1d73466377c98759c已采用：DEC1 352GiB、global419GiB，现役与科学checkpoint全部保留、退役credit=0。当前完整预测DEC1存储只剩约83.36MiB余量；辅助输出必须小且计量，不能另生成大型“报告缓存”。旧176GiB/250GiB和清理预期不能覆盖此新授权。全局96GPUh/24PBS、DEC1 52GPUh/11PBS/9216更新、final20h/40GiB仍保留。

两个agent各跑针对性fixture；owner仅在实际代码修改需要集成时执行一次合并后的精确CI，不让每个agent重复全仓suite。若已有CI覆盖且代码未变，复用身份与收据。大文件重hash、全runs树scan、清理失败policy都不属于此次并行工作。尤其他人vla_wbc/PI05完全排除。

## 6. owner的原生主线不变

1. 保持1475719及其固定source、lease、真实更新和观察链，不热merge、不重投、不改变checkpoint频率或训练数值。
2. A2真实到4096且完整保存后，核对原生terminal/effects、完整恢复状态和已保存1024 probes；使用既有terminal adapter一次结算并关闭GPU与CPU-control。进程在R或只是step4096日志时不能提前关闭。预期2048新更新/17查询/2load只是既有成功路径条件，最终按实际原生效果核定，不补造缺失记录。
3. 关闭后重新验证quota/项目/完整自身队列/输出空间/remaining work；将已准备的B1正式冷绑定，从原CORR开始0→2048，再完整关账后B2→4096。不重跑A、不用A权重初始化B；必须处理真实阶段效果，不能靠agent的READY预授权qsub。
4. helper补丁已完成则按原冷态流程纳入；未完成但旧路径已合格就继续，不创造新的B1前置。target委托接口缺失同样不挡B。
5. A/B终点之后，执行器/安全/资源完整准入成立才做已有授权评价。DEC1与源域DEV/LOCKBOX预算分开核算；当前训练+matrix预测不自动支付额外96DEV及80LOCKBOX，不静默缩分母。LOCKBOX仍按原独立选模门，selected/deployment=NONE直至真正通过。

## 7. 派发、交付、终止

现有owner读取本固定版后，在#11/#47合并登记一次PARALLEL_HELPERS_ADOPTED_AND_DISPATCHED，分别记T/S实际agent或会话身份、固定base、互斥写入路径、复用产物、CPU/输出上限与状态。没有实际启动写ASSIGNED_NOT_STARTED，不能把本文件发布称agent正在执行。

每个worker仅一次完成回执：DONE_REUSED、DONE_PATCH_TESTED或BLOCKED_EXACT_INPUT；附真实patch/测试和尚未原生验证部分。owner审阅后统一合并普通push/固定SHA回读，冲突保留并发提交，无force/reset。若Goal面板API不支持改标题/恢复，只更新已有持久卡和现场状态；不要重建Goal、重置usage或声称UI已恢复。

遇活跃训练/心跳/磁盘压力，先暂停辅助任务，不暂停健康A2。出现不明原生副作用、安全/认证权限、缺真实恢复状态或资源时钟不足，由owner报告精确阻断；helper无权解除STOP、清理checkpoint、创建第二actor。数据生产与Teacher恢复继续按已有关闭/NOT_TESTABLE状态，不因“并行”而重新授权。

原时间不变：2026-10-25 02:29:56 SGT停新job，14:29:56停新训练块，18:29:56停新物理；2026-10-26 02:29:56 SGT硬收尾。10月23日20:00报告真实神经/接口结果或准确缺口，不保证正结论。

**本次发布只有GitHub文档与任务派发授权，没有启动agent、远端SSH、PBS、模型、训练、物理或删除。新增权限仅为原scope内两个受限辅助agent的并行开发/审查；原生效果只由现有owner按原规则执行。**
