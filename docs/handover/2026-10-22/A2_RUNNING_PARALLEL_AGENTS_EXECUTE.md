# A2运行期间：两个并行agent推进B训练与解耦评价

日期：2026-10-22，Asia/Singapore。任务ID：VAP1_A2_PARALLEL_AGENTS_V1。
状态：USER_AUTHORIZED_HELPER_DISPATCH_PENDING_NATIVE_ADOPTION。
授权依据：用户报告A2已实训到2053，并明确要求「允许agent同步推进，加速试验主线，发布任务到github」。
审核基线：9422d1690f053302a34e42196d560dd45f754c28。既有主卡：reports/PI0294/design/parallel-science-20261022/GOAL_SUPPLEMENT_AND_PARALLEL_TASK_CARD.json。
派发卡：reports/PI0294/design/parallel-science-20261022/A2_PARALLEL_AGENT_DISPATCH.json。

**本文件把已有并行准备拆成两个可派发agent任务，不新建科学Goal、GPU队列或第二个native owner。** 用户此次许可替换旧交接中「不启动子agent」这一组织限制；原生调度者和集成发布者仍各只有一个，科学配方及资源上限不变。无需再次询问同范围派发许可；复用既有两个worker槽位，不能在它们仍运行时再另开两个。

## 1. 最新已证状态

截至已发布的2026-10-22 20:45:26 SGT现场观察：
- A1已成功终态，1024与2048全状态审计、结算关闭完成，不再结算或重跑。
- 当前A2为1475719.stdct-mgmt-02，R，运行source 3a1420e76c492196fa83905f4bde92924cc12c28，精确CI39777328633/job119819488833通过。
- A_CONTROL_SAVE_RECOVERY1已到2053/cursor65696，A2新增5更新、17兼容查询、2模型加载、unknown空。最新完整恢复点仍2048/cursor65536，live2053不等于新checkpoint。
- request9d1582d5-1c87-41bf-ad53-4e049cc21fe9、nonce3104eb01-637c-4afa-b33e-9aca24bd8204、request SHA298d68cf547e1540e2a259c0e5318e5bb5c7a07bccb3afea01dae43878d67200、manifest ae5acd27be71fb05c46c8a801c7b080877e5887ec9fbc481171aa5cea5d066e1保持。
- 已结算GPU40813秒、PBS17、更新3072=旧丢失1024+恢复A2048、queries1117；旧账H200_s字段不代表A40消耗变成H200独有。A2新增效果在真实终态结算一次，不重复合账。
- A2预约22560 GPU秒/2048更新/29queries/54000 CPU秒/70,000,000,000输出字节。不提前释放或挪给helper。
- B1 V3元数据及31项检查、A2终态adapter及8项CPU检查已完成，尚无B1提交；数据索引、曝光表、DEV剩余键、DEC1结果模板均存在，必须复用。

此处是证据快照，不是发布时的SSH读数。owner先刷新当前真实进度；若已前进，不回滚到本文step。

## 2. 并行拓扑与权限

| 执行者 | 工作 | 允许产出 | 禁止 |
|---|---|---|---|
| 原owner | A2观察/认证/heartbeat/release、终态审计和结算，随后B1/B2唯一派发及集成发布 | 原native收据、冻结源、原预算内合法调度 | 不能因helper未完成中断健康A2或凭空新增训练前置 |
| Agent T：TRAIN_NEXT | B1/B2接续链独立检查、必要的最小修复 | 隔离开发区patch、targeted fixture、readiness结果 | 无原生调度/lease/ledger写入、模型或训练 |
| Agent S：SCIENCE_NEXT | DEC1源侧接线、目标ABI缺口、结果聚合工具 | 隔离开发区patch、CPU验证、缺口清单 | 无目标仿真、WBC修改或方法调参 |

最多两个辅助agent，不递归开子agent，不新增永久watcher。原owner必要心跳不改；交互观察沿既有低频规则，不增加独立远端poller。helper只读取owner提供的脱敏快照和只读源码，不持有SSH、PBS或维护凭据，不各自远端轮询。

每个agent在已登记开发根内用独立worktree/输出目录，固定base SHA与互斥文件归属；不得同时写共享NFS checkout、活跃capsule、RUN_MANIFEST、源输入或中央账本。helper交patch/测试/结果，owner是唯一merge与GitHub集成发布者。已有同名任务优先复用；不新建顶层克隆或调度系统。

## 3. Agent T：A4096后尽快接上B

任务ID：T_B_CHAIN_READY，P0。输入直接复用主卡的B1 V3和A2_terminal，B1元数据SHA3b4e33f3995c3a778a96096f341766fda069bd4c8a0396c76e987fe1900250d4；复用成熟source_pair_inputs、segment intake、typed accounting及保存恢复。DATA_ASSET_AND_PURPOSE_INDEX.json、EXPECTED_TRAINING_EXPOSURE.tsv和原139/Original126+C1的CLOSED回执不重新生成。

必须完成：
1. 独立差分B1输入：B从原CORR@2048建立独立optimizer和B sampler，start0、resume=None，不继承A恢复路径、A lineage、A optimizer或A composite identity。R0、effective32、原B数据配比、seed/loss/精度不变，R1不热混入。
2. 小型fixture走通「真实A4096终态/完整恢复审计/无unknown/一次结算并关闭 → B1未来预算与身份 → B0至2048 → B2完整恢复至4096」。fixture不写真实grant/lease，不调用PBS，不伪造A4096结果。B1不得先于真实A4096关闭提交。
3. 检查A仍R、4096缺成员、错误arm/recipe、缺1024探针、重复settle、partial copy、stale源、未决effects及存储/CPU不足均被拒绝。复用原测试，只加未覆盖失败模式，不复制整个runner。
4. 如有确证工程缺口，提交最小同语义patch及针对性测试；没有缺口则直接回READY，不为显示进展添加新helper。A2运行source不变，新patch只用于后续冷态准入。
5. 明确B2已有代码/CI及仍缺的原生输入。CPU假件PASS不写成CUDA恢复成功。

交付B_CHAIN_AGENT_REVIEW.json：base、已复用证据、各检查PASS/具体缺口、patch/test、下一owner动作。真实准入字段保持PENDING_UNTIL_A4096_CLOSED。不重下数据、不全树hash、不重复全仓测试；既有实现已满足时审查结论即交付。若B1原路径已经合格，不能因本审查未结束而额外阻塞提交。

## 4. Agent S：把DEC1从模板推进到可交付接线

任务ID：S_DEC1_WIRING_AND_RESULTS，P0，不阻塞源域训练。复用EVALUATION_AND_CLAIM_PREPARATION.json、DEC1_64_CELL_RESULT_TEMPLATE.tsv、SAVED_DEV48_REMAINING_CONDITIONS.json，以及reports/PI0294/DEC1/20261018T182956Z-localpc-pv1/DEC1_SOURCE_ONLY_INTERFACE_FREEZE.json。已固定execution_progress.py/FRAME_CORRECTED_REFERENCE_V2参数不重调。

必须完成：
1. 源侧接线：原raw完整H40合法性/S1在映射之前；mapped完整参考合法性、mapped prefix shadow及tick；实际TCP/夹爪反馈与明确ACK；异常锁存后的stop。沿既有executor补最小接线或证明已覆盖，不另写物理执行器、安全判据、preview算法或评分器。
2. 纯CPU fake-transport/合格保存小数组测试，覆盖raw非法和mapped非法不执行、反馈过期/乱序拒绝、chunk reset与gripper顺序、stop锁存、相同进度时两模式共享语义、执行证据唯一归属。明确mock不是物理资格，不启动MuJoCo积分或神经模型。
3. 核对目标团队真正交付的EXECUTOR_MANIFEST：机器人/控制器版本、frame/TCP/夹爪、时间反馈、安全/reset/stop与可调用入口。缺什么列什么；WBC归档存在不等于ABI合格。不得虚构目标、焊base、读target真值调高层或修改WBC。无target不挡训练。
4. 按已有schema实现或验证最小结果聚合入口，区分SUCCESS、有效任务/参考/S1拒绝、原生安全错误、ENGINEERING_MISSING和NOT_RUN；按8个几何块处理配对相关性，而不是64独立样本。测试缺失键、重复case、模型/噪声混入；未运行格不填成0成功率。不打开LOCKBOX。
5. 源域A/B DEV48回答覆盖收益；DEC1首轮固定原CORR/R0的8几何×2任务×2执行域×2接口回答迁移。不得替换为A4096或B赢家、重抽容易场景、提前用target调参。

交付DEC1_WIRING_AGENT_REVIEW.json、必要的小patch/test、owner可冷调用的既有入口及精确依赖。必须标NATIVE_COMMISSIONING_NOT_RUN及TARGET_NOT_DELIVERED（若仍缺），不声称完成64格。不允许只生成又一份空模板；外部ABI缺失时完成源侧可做事项后停止等待，不转向新数据、WorldModel或Teacher项目。

## 5. helper资源与工作量边界

使用现有本地/管理CPU余额，不新增或重置总预算。首批两agent合计本地CPU≤7200秒，最多2个单核进程，墙钟≤6小时且不越原截止；小输出合计≤16MiB、已发布小收据传输≤32MiB。不下模型/图像库、不复制大checkpoint、不占Vanda计算节点A2的CPU或磁盘。

这是派发内界，不是额外额度：owner先确认原全剩余计划下仍有对应CPU/IO余额，按既有机制计量一次；不足则减至一个agent或只读。该界只约束本地计算，代理模型token额度沿既有Codex账户和服务限制，不新购服务。

存储最新授权1abb20973d84836b29baebf1d73466377c98759c已采用：DEC1 352GiB、global419GiB，checkpoint全保留、退役credit0。最新全部未来保留峰值预测DEC1仅余约83.36MiB；helper输出必须小且计量。旧176GiB/250GiB和垃圾清理预期不覆盖新授权。全局96GPUh/24PBS、DEC1 52GPUh/11PBS/9216更新、final20h/40GiB保持。

helper各跑针对性fixture；owner仅在实际代码合并需要时跑一次精确集成CI，不让两个agent重复全仓suite。已通过且源码未变的资格复用。大文件重hash、全runs树scan、清理checkpoint不是本次任务；尤其排除他人vla_wbc/PI05。

## 6. owner的原生关键路径不变

1. 保护1475719及其固定source、lease、实时更新和观察链，不热merge、不重投、不改checkpoint频率或训练数值。
2. A2到4096并真正完整保存后，核验native terminal/effects、完整恢复状态和保留的1024 probes；用既有terminal adapter结算一次，关闭GPU与CPU-control。仅看到step4096日志或作业仍R不能提前关账。2048新增更新/17查询/2load是原成功路径预期，最终依据真实效果，不补造缺失记录。
3. 关账后fresh核对quota/项目/完整本人队列/空间/剩余工作预算，正式冷绑定B1，从原CORR独立0→2048；其终态关闭后B2→4096。不重跑A，不用A权重初始化B；helper READY不是native准入。
4. helper补丁就绪则按冷态流程采用；未就绪但原路径合格就继续，不创建新的B1依赖。目标WBC缺失也不挡B训练。
5. 终点产生后，已有scope和完整资源、安全/执行器资格满足才做配对评价。DEC1与源域DEV/LOCKBOX成本分列；训练+matrix预测不自动支付另96DEV和80LOCKBOX，不静默缩分母。原独立选模门未过前selected/deployment仍NONE。

## 7. 派发与收口

owner读取固定版后，在#11/#47合并登记一次PARALLEL_HELPERS_ADOPTED_AND_DISPATCHED：T/S真实agent或会话身份、base SHA、互斥写入路径、复用产物、CPU/输出界及状态。未启动写ASSIGNED_NOT_STARTED，不能把发布当执行。已有两worker则续用，不双倍开工。

worker各一次完成回执：DONE_REUSED、DONE_PATCH_TESTED或BLOCKED_EXACT_INPUT，附patch/测试及未原生验证部分。owner审查后统一正常push并回读，保留并发提交，不force/reset。Goal面板API不支持编辑/恢复时，只更新现有持久卡和真实现场状态，不重建Goal、重置usage或伪称UI已恢复。

训练心跳、磁盘或控制面有压力时先暂停helper，不暂停健康A2。原生副作用未知、安全/认证权限、缺完整状态或额度/时钟不足由owner上报最小阻断；helper无权解除STOP、删checkpoint或建第二actor。数据/Teacher保持既有关闭或NOT_TESTABLE结论，不因并行许可重开采集。

原时间不变：2026-10-25 02:29:56 SGT停新job，14:29:56停新训练块，18:29:56停新物理；2026-10-26 02:29:56 SGT硬收尾。10月23日20:00报告真实神经/接口结果或精确缺口，不保证正结论。

本次发布只有GitHub文档和任务授权，没有启动agent、SSH、PBS、模型、训练、物理或删除。新增权限仅为原scope内两个受限helper的并行开发与审查；原生效果继续由唯一owner按原规则执行。
