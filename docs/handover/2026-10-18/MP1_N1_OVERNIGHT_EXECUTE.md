# Codex 夜间无人值守接手：MP1-N1

2026-10-18，Asia/Singapore。用户已要求给足工程预算并自主修复。本文件交给现有local-PC Codex owner；不是启动第二个云端agent。不要继续把旧“不得第六次提交/仅12H200小时”当成本次明确扩额后的有效上限，也不要清零旧费用或解除安全门。

## 必须读取的完整固定发布包

1. [完整审核、自治范围与流程](../../protocols/PI0294_MP1_N1_OVERNIGHT_AUTONOMY_20261018.md)
2. [机读授权卡](../../../reports/PI0294/design/MP1_N1_OVERNIGHT_AUTONOMY_20261018.json)
3. 本接手文件。

三者在同一固定commit回读，保存各自hash。然后沿AGENTS、START_HERE及统一工作区入口读取最新#11/#47和真实native状态。此包授权未来有界执行；GitHub发布本身不代表远端运行已经开始。

## 最新审核基线，不要继续引用旧排队/训练快照

科学收据403704527d876bf36b7c122d0f59384a4693c19d：source04/672907 F/1，source累计24119H200秒、父28619秒，预留0；组合2560更新=BC2048+CORR512。BC TRAIN3/4、DEV3/8；CORR尚需1536更新。CORR512 F1T1两次工程partial已耗尽；F1T2为8.4s工程partial，尚有原唯一replacement。32cells中18有效、2工程、12未尝试；21attempts。所有结果和STOP保留。
实际心跳失败年龄为-67.9ms，07b04707已修复先采钟后IO导致的假未来；CPU/CI通过不等于持续native资格。发布前e418bb790d291c6c936ce3878366314f4f5af83e已增加第五次冷态准入代码。接手若第五作业已提交/开始，按它原绑定保持健康执行，计入同一总账，绝不重复同号job。

## 新的有效累计上限

- MP1：H200 80h（原48，新增32）、A40 48h（原32，新增16）；Hopper PBS24、Vanda GPU PBS20/CPU build16。
- source-pair-01：H20036h、PBS12次（第五槽在内）、Core/Hopper CPU180/144core-h、生成12000；新单次allocation最多8h，站点只准4h则服从4h。
- 原32条件/每条件1次工程replacement、4096正式更新/每臂2048、原数据/统计/随机种子/损失、H40/prefix8、原0..300s心跳和20/60/900s约束不变；2700秒allocation收尾不变。
- 父CPU和网络的具体新上限、原物理/存储上限及最后6H200h/4A40h保护见机读卡。所有旧used/reserved计一次；不是新建一个不含旧成本的36h子账。

N1首次实际采用后36h硬关闭，+30h停止新提交/候选、+34h停止新训练实验，并取原MP1对应截止的更早者。原MP1 T0与10月24日03:14:45 SGT硬收尾不变。10月18日10:00 SGT写晨报；晚采用则立即写真实时间的补报。owner每最多4活跃小时更新一次#11，不需要用户守夜逐项批准。

## 直接执行的状态机

**ADOPT -> QUALIFY -> RESUME -> EVALUATE -> DECIDE -> CONFIRM/CLOSE**。

ADOPT：刷新当前账户权限、真实job/process/owner/lease、pending副作用、quota、每次累计费用。记录N1_ADOPTION_AND_EFFECTIVE_LIMITS，保留原birth card/manifests，新增固定N1授权层。旧owner还活着则不抢锁；旧job终态和effects确定后冷态交接。

QUALIFY：先将现有第五次补丁泛化为由固定N1卡限制的连续01..12准入。逐层测试PBS/request/launcher/ColdGuard/session label/矩阵和所有RLIMIT_CPU/预算硬编码；没绑定N1时旧限制仍生效。不要手写06、07、08各一个例外，不删安全或身份assert。心跳必须每次读取后采钟；真实compute至少600s持续读取及80个生产尺寸运输fixture cycle，验证并发replace、受锁统计、元数据失败、迟到答复和退出。故障注入只在隔离fixture，不碰真实heartbeat或动作。原0..300s规则不变，真实过期/未来/错误run仍停止。

RESUME：加载模型前依据已封存面板计算remaining。BC全部结束，跳过重复load/probes/serve；CORR512 F1T1不第三次跑，F1T2仅原一次replacement，终态或机会耗尽即保留缺失并推进，不把中期成功率/全格有效当新训练前置。存在安全STOP或未知action effect例外，先处置。复用最新合格完整CORR状态，恢复optimizer/scheduler/RNG/sampler，不倒退/重复训练；到2048正式停止。已生成probe复用，已知NOT_RUN缺draw可按原manifest补齐，未知effect不重发，目录exists不等于完整。

EVALUATE：完成CORR2048原固定probes/TRAIN4/DEV8，用原评分与Teacher OFF。所有正式有效成功或失败不重跑。原工程replacement用完仍缺失就如实保留；32格全部交代状态，不要求以额外尝试消除不可补缺口。核对两臂各65536实际访问，CORR整体25%原T1/25%纠正T1/50%原T2，逐cursor对照与同M1父权重。视频和大归档不阻塞合格本地评价。

DECIDE：两终点权重与可用对照已固定后，复用8个当前学生锚点与实际noise身份，缺终点输出最多16次、有界独立NI/shadow卡。只有原MP1触发证据满足时自动选择一种后继方法：S-EVENT一次，或最多原总2轮合格Recovery/DAgger，或原4episode/512容量诊断。不混方法、不凭loss/hold比例自动改训练；没有可信T2完整恢复，不生产T2正纠正标签。模型架构、VLM训练范围、控制器、夹爪PD、安全规则全部不改。

CONFIRM/CLOSE：合格共享候选按原开发规则固定，再一次性40条件SOURCE_CONFIRM；不把DEV当隐藏测试、不回流确认数据。Vanda可独立新窗口开展双摄到WBC任务链资格，旧相机NOT_RUN/TF-R3 STOP保留，不重做WBC调参/RMS大工程。真正安全问题阻断目标lane，源研究继续。无候选则交付负结果/缺失和机制判断，不无限加训。

## 自主修复细则

允许有限幂等metadata读取最多3次/120秒重试，心跳服务与大IO隔离；不把不确定query/install/start/qsub自动重发。常规解释器、请求字段、路径、quota、依赖、时间采样、审计和传输问题自行定位/测试/普通commit/冷态部署/续接。两次同机制native故障必须有新诊断与回归证据才再试，每组件最多6个实质版本。原生权限无法非交互恢复时保留一次阻断并做其他合格工作，禁止绕认证。

代码发布、CI、部署、native资格和完整任务结果分别记账。旧工程STOP不删，闭合副作用+新修复资格后以新epoch/fence作用于未来；真实安全STOP不在此列。已有owner可用有限supervisor连续执行，无永久cron/daemon、无第二调度owner。最多两个隔离只读CPU辅助者，正式审计尽量复用同一serial消费者。

当前所需父/BC/CORR/最新resume、唯一失败证据和raw绝不删。可按用户已有授权自主退役无引用的失败临时payload/冗余cache，先核对消费者与身份，保留轻量manifest，不要求逐项再次确认或复制一遍大权重。无可删对象就写0；禁止扫他人目录或follow symlink删目标。

## 必交结果

在现有MP1报告根写：N1_ADOPTION_AND_EFFECTIVE_LIMITS.json、ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json、MATCHED_SOURCE_PAIR_FINAL.json、更新BEHAVIOR_AND_FAILURE_MATRIX.json、N1_MECHANISM_AND_NEXT_BRANCH.md、N1_MORNING_REPORT.md、N1_RESOURCE_FINAL.json、N1_FINAL_SCIENCE_REVIEW.md。使用固定SHA及真实native证据，保留所有工程partial/有效负例/未运行。

若更早完成全部目标，立即规范收口；若全部lane硬阻断或窗口结束，公布精确事实并停止。不要用新的计划、更多测试数量、重复视频上传或再次请求一张作业票替代CORR终点与双任务稳定性证据。
