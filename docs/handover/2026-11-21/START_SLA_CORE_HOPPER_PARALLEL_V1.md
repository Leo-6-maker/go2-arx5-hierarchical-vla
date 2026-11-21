# Codex 启动：Corelab 数据采集与 Hopper H200 长训并行

这是供**具备现场服务器连接的 Codex**执行的完整启动任务。计划本身未提交作业；收到用户此启动指令后应完成实现、资格和真实后台挂载，不停在再写一份计划。

## 固定阅读入口

仓库：`Leo-6-maker/go2-arx5-hierarchical-vla`。先把本计划所在 Git commit 记为 `PLAN_COMMIT`，对文件计算 SHA256。只读获取最新服务器事实，不以文档日期推断现役作业状态。

1. `AGENTS.md` 的最新 Core/Hopper 入口及持续有效的工程约束。
2. `docs/protocols/SLA_CORE_HOPPER_PARALLEL_V1_20261121.md`。
3. 同名 `.json` 和 `SLA_CORE_HOPPER_PARALLEL_V1_POPULATIONS_20261121.json`。
4. `reports/WM02/sla-core-hopper-parallel-20261121/AUDIT_AND_PLAN.md`。
5. JSON 的固定源码版本中与本轮直接相关的训练、数据、transport、site、native producer、G7、评分和PBS入口。无需反复扫描全部历史。

## 任务

启动新的 `SLA-CORE-HOPPER-PARALLEL-V1`，**Corelab 采集与 Hopper H200 训练立即独立推进，不等待 VANDA A40 的 ADAPT5K、Core MIX、旧 Eval、旧 HOLD4、新采集或全部 Teacher 单元成功。** 用户确认 VANDA↔Hopper 网络直连，使用现场已配置、获准的直传路径；Core→VANDA沿既有批准路径。

### A. 先完成两条可以并行的准备

**Hopper链：**核对已有PBS/owner，不扰动健康作业。定位精确ADAPT2K封存父端点，HF身份为 revision `5c460f195f6509f2f4da55dea32c356b45e00d37`、目录 `ADAPT5K/step-02000-2bff547b6f2c`。从封存清单读完整hash。将完整140模型/Adam/RNG数组、优化器和采样状态、冻结SLA底座、tokenizer、统计量、D2/R200/LIBERO直接从VANDA交接至Hopper，校验成员、大小、SHA、实际样本后原子READY。不可用29数组Go2推理包代替，不可偷偷换父权重或等A40跑完。传输用真实CPU/传输队列或现有持久服务。

**Core链：**建立新Goal合法scope与单world owner，修复入口绑定缺陷，编译并封存新TRAIN240/VAL24/DEV24/CONF24的数值几何与身份，核对旧TRAIN/DEV/TEST去重（旧TEST仅身份）。所有科学条件都来自冻结人口；保留旧HOLD4的0请求，不复活旧过期watcher。Core存在资源局部占用时，不将它变成H200依赖，不抢占其他任务。

### B. 实现有限的 H200 新训练入口

旧ADAPT合同写死5K、旧Goal/站点和银行。使用独立worktree，按固定来源选择性整合现成模块与必要依赖，建立新版本20K contract/worker和完整训练状态搬运；复用现有恢复、PBS、Journal/onceIO、owner、日志，不另造大型调度器。运行中capsule不pull/checkout/hotpatch。

正式配方保留每步Go2/RoboDojo/LIBERO 64/32/32、域均值权重2/1/1、4个完整H40、十步生成全反向、原25%状态dropout、原训练范围和精度。恢复Adam/RNG，祖先与新phase计数分开。从父最后实际LR连续接新20K余弦到1e-6，冻结LR序列。

单H200做六个16更新候选（FM mb1/4/8/16、最佳FM配aux2/4），最多一组16更新技术保留，共112工程更新/8GPU小时。每候选4暖态+12完整计时，最多48固定输出+48独立恢复。E1为原串行计算参考，复用首步已有loss/gradient；各域FM按m/B_domain、aux按m/4归约，分别查分项与总梯度。原配方、逐样本随机、一次clip/Adam与预登记数值门保留；不通过缩短辅助反向或放松门提速。选择正确候选中完整更新p50最低者，不按DEV挑后端；大batch不合格则H200 mb1正常长训，不无限工程打磨。工程权重隔离，主臂重新恢复原ADAPT2K。

### C. 正式训练和数据生产持续运行

主臂 `HOPPER-ADAPT20K-V1` 新增20K逻辑步，从D2立即启动。PBS分段最多6小时，按实测p90和15分钟保存退出余量计算本段步数。每100步或30分钟及所有阶段边界封存完整状态。恢复后按真实完整前缀接续；丢失已提交后缀保留账本，最多256次额外Adam重放。未提交更新尝试最多32，独立计读取与生成，不能清零。

Core最多240条TRAIN+24条VAL专家条件，争取120条新完整合格示范；一条通过原G7与实际消费证明即可入库，不等全Teacher资格。前48个专家槽固定TRAIN12→VAL12→TRAIN12→VAL12，之后继续TRAIN；VAL producer在首条VAL前冻结，不用VAL调参。按六任务/F1F2轮转，优先缺少的蓝HOLD、F2红HOLD、红左放和蓝右放。相同TRAIN单元同producer两次连续同因失败后暂停该producer，可做最多两次有证据修复，使用该单元未派发条件，所有实际请求（包括已启动后的技术失败）仍消耗人口和总上限。WBC、腿、动力学、H40/接缝、STOP与原评分冻结，STOP后积分0。

每4条新合格TRAIN或6小时有新增时，封存累计快照；VAL独立。逐episode/快照事件发布，不等TRAIN或VAL全终态。通过VANDA传Hopper，完成真实读取后READY。仅在主臂phase2K/5K/10K/15K边界选已READY的最新合法快照；没有则NO_NEW_DATA_CONTINUE。新配额min(48,4×非原23条的完整episode数)，旧回放至少16；两源完整20K索引/noise银行启动即冻结，目标换库不改变它们，三域总额不变，不热改统计量/decoder。新目标数据不能偷用DEV/CONF或失败轨迹正例。

第一次实际接入本Goal新Core数据边界b，仅触发一次CONTROL2000：共同父状态、相同H200后端/LR/源样本/随机规则；主臂用新数据，控制保留切换前数据。复用主臂b+2000端点。控制可在主臂完成后用同一卡，主臂绝不等它。15K仍无新快照则不触发，不挪作其他试验。

### D. Eval 和原生学生独立排队

主臂0/2K/5K/10K/20K固定端点，每端点424原面板/诊断/恢复，新增固定VAL最多96；条件配对最多再加两端点。VAL在24条终态或T0+48h时冻结可用部分，每条最多4窗按计划中的确定性阶段函数选，源码与清单在模型预测前封存；缺项如实报告，不拿TRAIN补齐。Eval与训练后继同时放行；没有独立GPU则用合法队列或主H200封存段间有界运行，不等评分通过才训练。

每登记模型用新schema从完整seal独立冷态恢复生成Hopper REF16，绑定固定输入/noise/统计/底座；Core原严格RGB16资格后跑同一新DEV24，最终20K新CONF24；条件主/控各DEV24，总学生科学上限192。TECH2另计。部署分支与三域424 Eval并行，不依赖旧FORMAL424包装、不伪造其回执；未完成如实标OFFLINE_EVAL_PENDING。严格复验失败只阻塞该模型部署，采集与未受影响训练继续，不改原门。共同checkpoint/身份/统计完整性故障需冻结受影响下游。Core按最多12采集后6已就绪学生交替，始终一physics owner。

### E. 自治、预算和收尾

首次实际挂载T0写入一次，Goal期限14天，H200本Goal总上限320GPU小时（含持卡编译/恢复/保存）；实际项目grant失效则停相应链。单PBS allocation时限只结束本capsule，保存后继可在新的真实allocation接续，不因第一段6小时到期停止20K链。硬上限以JSON为准：总实际Adam≤22,368，优化访问≤2,863,104，所有模型生成尝试≤125,224；原生含接口≤458请求、20,911,974tick。条件角色未触发不消耗或转移预算；原失败/未派发旧人口/旧CLOCK和旧作业账本保持。

在有权的用户空间自主修复并持续执行，不为常规小修再次索取GO。禁止伪造grant/owner或以改身份绕过scope；UNKNOWN写入先核验，不能双投。账号/站点授权缺失只阻塞受影响角色，其余继续。禁止修改系统网络/SSH/挂载、sudo、删除用户数据、抢占他人GPU、发布私有资产。

真正挂到PBS和现有服务器持久服务后，回读真实job ID、状态、依赖、Core owner、代码/计划/父数据hash、T0/deadline及断开交互后存活证据。不要只报告“脚本已生成”。由服务器负责后继、Eval、传输与收尾，关闭用户电脑后继续。完成或到限时封存实际前缀，退出本Goal子任务、关lease、释放owner并发布实绩。已有监督automation不新增、不重启、不修改。

## 首次回报格式

1. 已实际挂载的角色、PBS/job/service ID、依赖和有效截止；未挂载角色及具体真实原因。
2. H200完整父状态与D2/R/L交接情况；工程候选实测更新/耗时/显存/恢复；正式新phase与祖先分列。
3. Core已实际请求、tick、完整示范、首失败和新入库；0请求的技术拒绝明确为0。
4. 哪些有序后台任务能在用户关机后继续，哪些实际授权/资源缺项仍阻塞；不能把计划上限写成已运行数量。
5. GitHub更新固定代码SHA与脱敏实绩链接。服务器计划文本不替代真正mount回执。
