# SLA RoboDojo-Real → Go2+ARX5 迁移与配比 V1

2026-11-18（SGT）；**PLAN_ONLY / NOT_EXECUTED**。用户本轮请求分析、详细计划与GitHub发布，本次不启动服务器训练、下载大数据或原生世界。用户交给Codex执行配套prompt后，在实际站点权限内按下列有限范围推进；本文件不是GPU配额、真机许可或可直接运行的配置。

[审阅报告](../../reports/WM02/sla-real-sim-plan-audit-20261118/REPOSITORY_AUDIT.md) · [机读规格](SLA_ROBODOJO_REAL_SIM_MIX_V1_20261118.json) · [启动prompt](../handover/2026-11-18/START_SLA_ROBODOJO_REAL_SIM_MIX_V1.md) · [入口](../../SLA_REAL_SIM_MIX_START.md)

## 1. 研究问题与明确不做的事

核心问题：在相同SHIFT7父模型、目标C20、冻结WBC、执行/评价合同与总优化访问数下，用经过语义审计的真实ARX X5源样本替换10%或25%的目标样本，是否改善Go2+ARX5移动抓持的完整闭环、困难F2与新布局表现，同时不损伤F1？

这是**公开真实源域经验的迁移**，不是“真机数据天然优于仿真”的因果检验，也不是零目标数据迁移：共同父模型已有C20目标适配经历，三臂继续使用目标数据。源任务分布、视觉外观和样本多样性也随替换改变，不能把任何收益唯一归因于realism。

首轮不加Memory、Spatial Forcing、BASE head、教师阶段、物体真值、额外解冻、RL、连续实机控制或新夹爪decoder；不同时做从SLA_base重新预适配、顺序训练和多种比例的大网格。保持旧V5.2原门、全部失败和正在运行的24条件。

## 2. 当前基线与优先顺序

审阅head `01b4a11c80f95956742379f73404be8fbdf909e1`。截至已提交快照2026-11-18 00:38:43 SGT，B1/B2各500/16000完成；E2 9/24启动、8终态、1活跃，5成功。B2旧ROLL96过96/96，但新F1/T2内部k34加速度0.261575>0.25拒绝。F2尚无本轮完整结果；不能先宣布B2胜出。详见审阅报告。

P0先由现场执行代理只读核对最新head、服务、owner、未启动条件和真实期限。健康E2继续原24面板。D0、B1/B2训练和旧离线银行不重复。V5.2完成后按原规则进行其确认/纠正/DEV/TEST决定；任何已合法启动后继也优先继续。新数据CPU工作可在独立目录、实际有余量时进行；新GPU/原生阶段排在健康现役消费者的冷态边界后，不能通过杀进程或改owner抢先。

当前后继评价实现只到CPU资格、未部署；E3也不能当作已实现。新研究不自动占用V5.2剩余额度。若现役故障为STOP后继续积分、冻结文件变化、owner未知，则先核清共用执行器问题；普通科学失败不永久阻止其它预登记纯仿真。

## 3. 固定身份

| 对象 | 固定值 |
|---|---|
| 发布基线main |6843163d641eb24a29c3b1557f868765346ba66c|
| V5.2计划 |90d39196bdc345b977ee54ed64e750cf3561aa72|
| 审阅PR405 |01b4a11c80f95956742379f73404be8fbdf909e1|
| 原SHIFT7共同父包 |ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb|
| C20 manifest |4bcc5ffcd8920d874599a54cca73792c50a49ea6013d65ca7826817a9f80d7de|
| C20 statistics |f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e|
| Real0复用B2包 |f95a5bafc2c67cae722c74544fd2c6b6aa1496805f04179dd13c134338081e25|
| 上游SLA源码 |18896e757647ed9c2a68816b35a7100afda6e287|
| 上游Memory源码 |c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454|
| 源数据仓库 |RoboDojo-Benchmark/RoboDojo；只取data/RoboDojo_real/*/arx_x5/|
| 源数据revision/逐文件SHA |**执行前解析并冻结；本次未知，不伪造**|

原0%控制选B2，是固定其“目标FM＋端点loss”算法作为配比基线，不是根据E2前缀把B2选成部署胜者。无论V5.2哪臂最后更好，本配比实验不改共同父或偷偷改成赢家；需要改变时另立科学版本。

## 4. R1：有限真实数据侦察与独立划分

### 4.1 边界与选样

先只读取仓库目录/文件元信息，解析immutable revision、license和可用任务/episode，不跟随浮动main逐文件下载。最多60个ARX X5 episode、最多32GiB新增原始字节；两者先到者生效，首次每个已核实任务2个、最多12个包含在60内。新中间产物含原始数据总计不超过64GiB并服从现场更小容量；不自动删除旧数据腾空间。

先按实际session/operator/scene/episode簇防泄漏划分；每个存在且足够的任务目标10个原始episode，按6 TRAIN、2 SOURCE_DEV、2 SOURCE_TEST分配。候选ID与顺序用固定seed2026111801和排序规则在模型生成前封存。缺文件/不合格保留原因，不无限换样本直到通过；未知session不能宣称跨session泛化，episode近重复需额外检测。若任务数不是六个，按实际人口给出未满额，不虚构任务。

至少检查：`state`与`action`是否不同、各自生产者定义；每臂frame/TCP/units/quat顺序；夹爪的command、measured与归一化语义；实际时间戳/频率/掉帧；RGB颜色与相机映射；任务指令及左右臂依赖。官方converter只是字段复制入口，不是命令消费证明。优先保留原压缩RGB字节，避免再次有损编码；只有需要的anchor图像解码，不在AMD做批量视频处理。

产出`SOURCE_INVENTORY.json`、`SOURCE_SEMANTICS.md/json`与`SPLIT_MANIFEST.json`；包含总候选、读取、每类排除、有效完整episode/片段/窗口分母，不把60个读取episode写成60个可训练成功示范。

### 4.2 两层数据保留

`SOURCE_NATIVE`保留真实原始语义、双臂、原频率、原标签。它不需通过Go2 H40门才有研究价值。

`SOURCE_CANONICAL_COMPAT`才是本次配比实验的单臂全H40样本。只从操作确实可单臂完成、另一臂未承担移动/支撑/开门等必要作用的片段取样；单凭另一臂关节速度小不足以认定独立。动态双手交接、另一臂持物支撑等留在SOURCE_NATIVE，不能删除一半动作后配完整双臂指令。合法子任务文本须可由原操作过程核对，保存原指令与派生理由；没有可验证子任务不伪造标签。

## 5. R2：统一语义，而不是伪装成C20

### 5.1 标签种类必须保留

C20标签是当前query合法专家完整计划，P10消费另有证据。源文件的未来action序列应标为`SOURCE_RECORDED_ACTION_SEQUENCE`或`SOURCE_RESAMPLED_COMMAND_REFERENCE`；它不是在anchor时一次提交的H40，也不是Go2执行数据。

保留`label_kind`、`command_provenance`、原时间戳、重采样映射、原始action/state SHA及消费证据级别。无独立dispatch/ACK时标UNKNOWN，不填成已执行。若生产者表明action实为future achieved state，则转为另一个显式source-only研究方案，本次命令混合不偷偷使用。source的完整未来动作监督可成立，但其运动学/控制语义必须明确，不能因此声称物理消费。

本轮只用所有40点均真实可构造的窗口。episode尾部不足40点就排除并计数；不复制终点补“真标签”，不引入未正确实现的部分loss mask。源侧没有5703初始化或250tick消费，不伪造这些字段。source不填固定底座零作为BASE监督。

### 5.2 Pose、相机与时钟

选择活动臂的主视角＋对应腕视角，使用同一目标图像处理路径和合法缺视图mask；不得把右腕任意改名左腕而不声明映射，不通过无依据镜像改变运动语义。

以真实标定得到的frame与TCP变换统一pose。例如`T_policy_TCP = T_policy_source * T_source_EEF * T_EEF_TCP`。源臂局部frame与Go2 episode固定任务frame各自建绑定；不能把上游双臂0.6m间距的±0.3m偏移复制到Go2。四元数先转SO(3)，再调用项目既有rotvec/rot6d约定；验证逆变换及旋转正交性。不依靠数值相减替代旋转群上的相对姿态。

source anchor优先取真实观测时刻，输入不得使用未来图像/状态。动作按明确的`query+k*0.05`目标网格构造，与目标codec实际索引一致。位置/旋转插值或命令零阶保持由原action语义决定并固定；不能仅因30Hz而每隔一帧抽取当20Hz。不得通过整体放慢、平滑或截断动作，隐瞒真实数据与目标执行包络差异。

若只有nominal frame clock，需核对采集器的同步约定，标`NOMINAL_FRAME_CLOCK`；不声称已测硬件延迟。源侧50ms上下文只是重采样网格，原命令实际发出时刻另存；不得填入目标`previous_actual_emitted`字段冒充真实50ms ACK。训练source reader要明确这个语义分支，目标deployment仍严格使用上一实际发出指令。

### 5.3 夹爪：本轮最可能的兼容性瓶颈

当前SHIFT7 decoder只能表达受约束的单调单事件；旧labeler还假定2秒事件。不能用旧函数给真实曲线强贴2秒标签。先核对源开度/开合方向/目标范围；若使用规范化开度转目标2–42mm，必须称为**retargeted command-equivalent width**，不声称是测得的源毫米宽度。

允许在独立source labeler中对**同一未修改的SHIFT7解码族**做有界逆参数求解，目标、slack、delay仍在原族内；参数从已记录command事件得到，既不输入学生，也不修改原命令。使用全episode事件可作标签分析，不作未来观测输入。每窗存raw、retargeted、reconstructed三份曲线及参数可辨识性；选择规则固定为最小重建误差、同误差取较小delay再较小slack，禁止按模型输出调参数。

本轮预注册可接受误差：目标等效width最大误差≤0.2mm、RMS≤0.1mm；有0.5开度crossing时偏差≤50ms；单调方向与开/闭/保持意图一致；跨query拼接保持上一command上下文一致。它们是本研究的数据近似容限，不是官方参数或安全门。不能修到能过为止。多次开合、超出事件族、延迟不明或未观察到事件边界，不能静默拟合。

数据晋级至少需要12个TRAIN源episode、至少2个源任务；≥256不同全H40窗口，其中至少24个闭合与24个张开转换窗，两个事件类别各来自≥4个不同episode；SOURCE_DEV至少6个episode、≥2任务、至少64不同窗口。SOURCE_TEST需要保留≥64不同窗口供最终一次只读诊断。不能只用静止样本满足这个门，也不把256窗称256独立试验。

### 5.4 统计与执行包络是不同问题

初轮冻结原C20统计、同一SHIFT7相对动作定义和padding语义。所有三臂共用，source不能单独用另一normalizer而不显式声明域路由。检查state/action往返误差、归一化幅值分布、非有限、clip饱和与域间梯度量级，不直接按新混合集重算统计后继续用旧B2当同合同控制。若必须改统计，停止本配比矩阵并提出共同新统计/共同初始化的独立版本；不要回写旧文件。

**源数据正确性不等于目标H40资格。** 目标首尾补零加速度门意味着端点相邻位移≤0.625mm，真实连续遥操作切窗未必满足。对完整源银行报告首尾/内部/夹爪速度违规率及幅度，但不为过门裁剪，不把所有目标门失败自动当坏数据。训练用真实源序列不需要先让Go2执行它；部署每个学生query仍过原完整H40。若源目标明显冲突，报告这是negative-transfer风险并按原研究固定比较，不能以“已在真机执行过”跳过目标安全门。

### 5.5 不兼容时的终态

若语义/夹爪族/最低事件覆盖不成立，输出`BLOCKED_REPRESENTATION`或`BLOCKED_LABEL_SEMANTICS`，保留SOURCE_NATIVE与量化排除原因。首个有效结论是“当前定制表示不能承接哪类真实动作”，不是SLA或真实数据无效。后继可选择source-native空间/动作预适配或新的通用夹爪接口，但那会改变方法，**不自动冒充0/10/25%单变量比较**。不创建新审批框架；对必须改科学接口的原因作一次明确交付即可。

## 6. R3：精确配比，复用已经付出的计算

数据兼容、旧健康消费者已冷态退出、有效资源成立后，才进入本阶段。

| Arm | 共同初始化 | 500步/16000访问 | 本次新增优化 |
|---|---|---|---|
| Real0 |原SHIFT7＋fresh Adam，目标端点loss0.1|16000 C20；**复用V5.2 B2实际端点**|0|
| Real10 |同一原SHIFT7＋匹配fresh Adam|14400 C20＋1600 source|500|
| Real25 |同一原SHIFT7＋匹配fresh Adam|12000 C20＋4000 source|500|

不得从B2继续500再把它与B2比较，也不把B1/B2当两个不同随机seed。28 active数组、52,153,376参数、原frozen fingerprint、batch32/micro4、50 warmup的原学习率数组、state-dropout .25、cached_actual/10步、C20 stats、原SHIFT7 decoder均固定。两个新端点各自累计2000更新/64000访问，与B2相同。原B2训练cost仅引用一次。

objective逐样本定义为`FM32 + 0.1 * is_target * endpoint_loss`，全batch32统一平均。source的端点没有目标H40的静止边界语义，不加目标端点辅助项；这条domain规则在所有比例都相同，Real0恰等于原B2。源与目标每样本同样FM权重，不再额外乘一遍10%/25%。不按source batch单独归一后又改变名义配比。

精确访问：每10个update共320槽；Real10替换32槽，Real25替换80槽，前者集合包含于后者，两个源序列共享这1600个对应样本。至少12000个共同target槽逐项保持原B2的episode/anchor/dropout/noise/time/LR；其余槽从同一固定target银行替换，不因为删去target访问就压缩其cursor。随机张量按原绝对槽生成，不用source episode名称重新取随机数。

按task→episode→已实际存在的event→anchor采source；缺失阶段不能伪造。source metadata及domain只用于reader/loss/计量，不送入学生作为任务答案。实测source读取/优化访问、不同episode、不同窗口、每事件曝光与重复次数分别记录。10%无法由B32每batch固定整数精确表达，因此以10-update superblock约束，不能长期四舍五入成3/32或4/32。

新recipe复用`SuccessorEngine`与`mixed_visits_v4`的实现模式，但不关闭旧guard。新增源reader和RealMixVisits500；旧C20统计的fit人口继续写C20，而不是伪称覆盖整个source混合manifest。

## 7. R4：最小实现验证与离线比较

实现验证包括frame/TCP往返、SO(3)、时间offset、源context与目标context分离、gripper重建、no-future-observation、完整窗口范围、split、嵌套配额和恢复cursor。Real0路径须在相同16输入/噪声下与原B2实现逐位一致；最多32次新完整生成用于双路径比较。允许最多2次诊断反传/8窗口、0优化器更新；实际读取/尝试另计，不伪称没有成本。

两个500端点固定，不按250结果挑选；保存全状态并重载，独立进程普通RGB16复现。新的接口/算子若不能恢复一致，先最小工程修复，不加训练步数寻找偶然成功。

离线银行在新模型生成前固定：原TRAIN96＋CN38 180＋ROLL96=372目标开发条件；SOURCE_DEV64覆盖合法开/闭/保持与至少6episode。Real0目标预测沿用身份正确的B2原结果，新增source64与独立恢复16；Real10/25各372＋64＋16=452。所有source结果是off-policy动作/重建诊断，不是RoboDojo真机任务成绩。最终候选才开启SOURCE_TEST64一次。

target记录完整H40合法率、首/尾/内部margin、旧3拒绝修复、CLOSE时机、空间偏差、进度和新failure indices。source记录原始动作误差、codec近似误差与模型误差分开；在开发源上改善不能代替目标任务收益。

## 8. R5：真正的目标闭环与泛化

### 8.1 三臂开发screen：24请求

F1/F2×T0–T3各一条、三臂各8，使用全新预注册reset/noise身份。与V5.1/V5.2已执行条件独立，不替换旧失败。按条件预先循环交错arm次序；固定一个native world owner。新reader不改变physics/WBC/相机安装/输入定义。既有接口若未变复用资格；变更真实消费接缝才允许最多2个接口回归，每族1，0模型生成。

比较输出逐条件三列和Real10−Real0、Real25−Real0；n=8是开发探索，不写显著提升或最优比例。所有合法模型都进入screen，不因offline科学失败先删控制。每次H40拒绝不安装，计episode失败；碰撞由原STOP终止，保留部分tick，继续其它预登记仿真条件。STOP后积分、未知owner或冻结SHA异常则停受影响campaign。

初始化按frame/TCP/base/关节分别量化实际偏差，记录从reset到query0的时钟与输入SHA。同seed不等于完整WBC动态状态相同；不重置内部状态以凑“逐位配对”，也不择初态重试。

### 8.2 候选选择与确认8

完整24终态后，候选须screen≥6/8、每族≥3/4、0 hardSTOP、0技术失败；TRAIN96/CN38全合法，旧3拒绝输入通过，CLOSE0遗漏且≤100ms。相对Real0还须总成功数不减少、每族成功数不减少。排序依次总成功、较弱family、较少拒绝、成功任务时间中位数，完全同分优先Real0→Real10→Real25。不做逐query mixture或挑模型。

若Real0自己不满足门，Real10/25仍可按上述门晋级；若没有合格者就封存负结果，不打开DEV/TEST、不自动加source比例。只有离线改善而任务未改善，结论必须明确“没有证明目标闭环收益”。

唯一冻结候选做新确认8，同样≥6/8、每族≥3/4、0 hardSTOP/技术失败。确认失败不换候选再试，不以新seed挑通过。

### 8.3 独立DEV32/TEST40

在新训练/选择前封存新的布局/初态簇人口：DEV每族8布局×2seed，TEST每族10另行布局×2seed。与C20/CN38、旧DEV、PR403、V5.2的36个保留布局以及本轮screen/后续数据去重。仅同物体、同相机的局部布局推广；不是新物体、新技能或真机泛化。如果无法建立充分独立且可达的新布局，明确本轮只做开发比较，不把换seed改名TEST。

DEV≥24/32且各族≥12/16、0 hardSTOP/技术失败后才一次TEST；TEST≥32/40且各族≥16/20、0 hardSTOP/技术失败。未过DEV不看TEST输出。以episode和布局簇分别报告，不把同布局两seed视为完全独立。若TEST用于修改数据或方法，该人口退为开发，不能复用刷分。

原评分保持native grasp＋物体抬升≥20mm＋保持≥1s＋任务方向base位移≥80mm，四项及综合分别列。暂停仿真不等于连续350ms实时，更不等于真实硬件成功。

## 9. 预算与执行边界

| 阶段 | 新训练更新/优化访问 | 新模型生成上限 | 新native请求 |
|---|---:|---:|---:|
| 仓库/数据/codec审计 |0/0|0|0|
| 同数值接缝验证 |0/0，另≤2反传/8窗|32|0|
| Real10/25训练＋离线 |1000/32000|904|0|
| Real0源诊断/恢复 |0/0|80|0|
| 最终源TEST64 |0/0|64|0|
| screen24＋确认8＋DEV32＋TEST40 |0/0|≤12480|104|
| 必要接口回归 |0/0|0|≤2|
| **全条件路径上界** |**1000/32000**|**13560=1080离线＋12480原生**|**106**|

科学episode初始化5703tick、任务≤30000tick、总≤35703；接口≤5953tick；总上界`104*35703+2*5953=3725018` tick。技术已发生费用仍计在同一上界，不自动额外跑一条替代。两臂正式训练包含所有有优化器更新的资格步，不在500外暗加smoke updates；诊断仅限前述零更新调用。

新服务单个≤12h、新campaign≤24h且不晚于实际有效allocation。旧E2服务10:12:18截止与历史allocation次日02:51:23不同，不能把后者当成延长前者许可。capacity检查使用实测：V5.2每臂JAX live buffer峰约9.71GB，host峰约14.37/22.12GB；这些不是新输入管线的保证，禁止预先按峰值硬开多模型并行。source处理首阶段至多2CPU线程、4GiB工作内存、1个解码worker，独立目录；容量不足有界等待，不清理他人文件或改站点服务。

应用级路径、类型、IPC、reader、计量、checkpoint和有界source labeler问题由Codex自主修复、定向回归、正常PR发布；不逐小补丁询问GO。每阶段最多2次有新证据的技术接续，先核清UNKNOWN，只续接未启动后缀；不恢复CLOSED或重跑科学失败。AMD不做训练/编码/大扫描。WBC、solver、腿策略、MPC、臂PD、2ms、安全门与评分冻结。

## 10. 实现任务表（拟新增，不冒称已完成）

| 工作包 | 复用实现 | 最小新增及验收 |
|---|---|---|
| R1身份/数据 |upstream converter与现有hash清单|定向下载manifest、source字段/生产者证据、组划分；不执行上游全库下载/删除路径|
| R2 canonical reader |项目codec、原SO(3)、SHIFT7 decoder|source独立schema、单臂过滤、时间/相机/TCP、事件逆编码及误差表；旧C20类不放宽|
| R3比例与训练 |mixed_visits_v4、mobile_successor_v52|500步嵌套替换槽、is_target辅助项、真实source/target计数、固定父包、共同87状态|
| R4包与恢复 |mobile_endpoint_v52、provider_mobile_v52|source语义绑定不送模型、新包声明、旧目标数值回归、全态及独立RGB恢复|
| R5 native比较 |mobile_campaign_v52、mobile_evaluation_v52|新arm/spec/population、正确shift7-queries计数、原停止/TTL/部分P10继承|

候选新文件可命名`robodojo_source_audit_v1.py`、`robodojo_source_dataset_v1.py`、`real_mix_visits_v1.py`、`real_mix_successor_v1.py`；是否沿用文件由最小变更决定。这里的名字不是当前可运行命令。禁止新建重复调度器、账本或多层签发系统。

## 11. 阶段产出、停止点与后续科学问题

每阶段提交紧凑ACTUAL.md/json和必要逐条件表：已做/未做、source/target真实访问、原生请求/tick/partial、失败与缺件、模型/数据/统计/source身份、恢复证据、成本与下一动作。源原始数据、模型大权重、站点路径/凭据不公开。GitHub发布、CPU测试与模型加载不算任务成功。

首个交付是现役E2终态决定＋source兼容性报告，不必等待所有后继预算耗完。通过兼容性后，第二交付是两个真实500端点和24条件配比比较；失败时给出可证伪机制，不无限加步。

本轮即使成功，也只证明在选定真实源子集上改善Go2仿真PICK_HOLD，不能宣称长程规划、完整RoboDojo真机能力或本体零样本。下一单独协议可建立：同场景红/蓝对象×左/右目的地的完整PICK_PLACE（拟40条完整示范起点），语言反事实与未见组合、整体目标vs oracle子目标、必要时匹配Memory消融。新任务须有完整Teacher/执行可行性，但不要求当前抓持刷到100%才允许准备语义数据。**这些是明确后续方向，不计入本轮1000/32000预算，也不自动启动。**

若目标是严格本体无关，另做源任务训练后冻结高层、零Go2任务微调的迁移对照；不能用已有C20适配的SHIFT7替代。真实Go2控制始终不在本计划范围。

## 12. 主要来源

- [PR405实际状态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/reports/WM02/sla-mobile-f2-v52-20261117/ACTUAL.md)、[完整训练证据](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/reports/WM02/sla-mobile-f2-v52-20261117/E1_PAIR_ACTUAL.md)、[新内部拒绝](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/reports/WM02/sla-mobile-f2-v52-20261117/E2_REJECTION_PREFIX_ACTUAL.md)。
- [原V5.2选择](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/src/go2arx5_vla/sla_adapter/mobile_selection_v52.py)、[真实父加载](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/src/go2arx5_vla/sla_adapter/mobile_successor_v52.py)、[现有配额](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/src/go2arx5_vla/sla_adapter/mixed_visits_v4.py)、[夹爪事件假设](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/01b4a11c80f95956742379f73404be8fbdf909e1/src/go2arx5_vla/sla_adapter/gripper_time_v1.py)。
- [上游Real配置](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/external_config/train_robodojo_real.py)、[上游Sim配置](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/external_config/train_robodojo_sim.py)、[非等权限制](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/training/data_loader.py)、[真实字段转换](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/scripts/convert_robodojo_eef.py)、[pose转换](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/policies/robodojo_policy.py)。
- [SLA持续预训练](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/external_config/pretrain.py)、[加权采样](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/training/data_loader.py)。
