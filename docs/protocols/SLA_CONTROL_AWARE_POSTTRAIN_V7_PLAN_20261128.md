# SLA V7 · 长程自主推进 V3：闭环行为保持、控制状态、失败分布与官方空间适配
2026-11-28 SGT；继承 PR #529，历史 V2 固定 67d15fb8fad688a2bbd82e225111bb49ec34559b。本修订是**当前调度与科研决策的接力入口**，覆盖 V2 中过时的“等待 A 资格/训练”与候选优先级，但不改已封存实验、运行配置、原安全或资源授权。已完成实验按原记录结算，新的实际回执优先。用户批准将详细长程计划推送 GitHub，供现役执行者自主推进；不等同于扩张实际集群权限、名额或期限。
[RUNBOOK](SLA_CONTROL_AWARE_POSTTRAIN_V7_RUNBOOK_20261128.md) · [REGISTRY](SLA_CONTROL_AWARE_POSTTRAIN_V7_REGISTRY_20261128.json)。

## 0. 固定最新实绩与不可重复的任务
以下证据固定到执行提交 9ada3206728ffc10e6608c9cab64a5d648cd3eee，而非文档分支。该版本 B6 账本将 PBS RUNNING 回执标为 17:45 SGT；该时间比撰写时会话时钟稍晚，按源中时间记录而非本次实时 qstat 确认。Codex 首次接力必须先回读实际 owner、job、更新和租约，不能依据历史 R 盲目再次提交。

| 项目 | 实际状态 | 结论 |
|---|---|---|
| V6.5 1K / 3K | 正式训练与各10格完成；未晋级 | R4-200 继续作父；不自动5K/10K |
| V7-A | 双500、完整GPU资格/冷恢复、20格全部结算 | 两臂不晋级；冻结条件诊断仍可复用 |
| B6 Shared-L2SP | CPU/H200资格2工程步，独立Core恢复；原账本已提交双100 PBS 738836.hopper-m-02 | 接续实际作业，不重开；100端点收益待测 |
| A93 + R3 | 93完整+3恢复后缀，7916+115=8031窗；固定旧统计 | B6 不热换 |
| 实际Recovery | 14次已知尝试、4条/155窗，四次早期新尝试准入0；R3仍只有3条晚期 | 触发器改为事前碰撞路径预警，原安全不变 |
| P0-S 状态 | 全量分箱/物理跨度及新全局分位反事实完成 | 下一步验证新表达的净收益，不直接替换现役统计 |
| 物理语义回放 | 3回合241窗9460动作点；44窗输入入口一致 | 不是全库认证；补旧52 reader和高风险尾部 |
| E0 / L0 | 12条轨迹、138718请求、同状态前向；第251步真实梯度 | 不将单批/被动匹配当全局因果 |
| B6行为锚点 | 12个TRAIN回合审计；正式准入0 | 先参数锚定，继续收集直接消费证据 |

主要证据：
- [E-A 完整退化与预处理](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_FAILURE_CAUSE_PREPROCESSING_AUDIT_20261128.md)；[A完整训练与闭环](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_A_CONTEXT_500_ACTUAL_20261128.md)。
- [E-B6 最新资格/派发/状态跨度/提前量](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_PARENT_ANCHOR_ACTUAL_20261128.md)；[B6现场独立登记](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/docs/protocols/SLA_V7_B6_SHARED_PARENT_ANCHOR_V1_20261128.json)；[真实校准](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_PARENT_ANCHOR_CALIBRATION_ACTUAL.json)；[独立Core恢复](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_PARENT_ANCHOR_CORE_RESTORE_ACTUAL.json)。
- [E-S 状态物理跨度](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_STATE_BIN_PHYSICAL_SPAN_ACTUAL.json)；[E-D 后继早期失败](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_D0_FOLLOWUP_TERMINAL_ACTUAL.json)；[静态提前量](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_EARLY_CONTACT_QUERY_LEADTIME_ACTUAL.json)。
- [E-L 真实参数梯度](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_L0_GRADIENT_AND_RETENTION_ACTUAL_20261128.md)；[真实251步分项](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_L0_CONTROL_STEP0251_TERMINAL_ACTUAL.json)；[阶段实际访问](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_TRAIN_PLACE_RELEASE_EXPOSURE_ACTUAL.json)；[E-E 逐层分叉](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V7_E0_FIRST_DIVERGENCE_ACTUAL_20261128.md)。
- [V6.5 结算](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2bf6ab7a663099be27bc85a9b424256eed03920b/reports/SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6/V65_RECOVERY_ADAPT_1K_ACTUAL_20261127.md)；[Core原工厂](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f67fe887a2cc3e3b9309c122db6773febbefb697/reports/SLA_CORE_EMBODIMENT_FACTORY_V5/V64_CORE_SELECTED_REFERENCE_ACTUAL_20261127.md)。

### 0.1 固定原生证据
| 端点 | N02 HOLD | 独立 HOLD | PLACE grasp | lift≥20mm | in-zone | full PLACE | PLACE接触STOP |
|---|---:|---:|---:|---:|---:|---:|---:|
| R4-200 | 2/2 | 2/4 | 4/4 | 3/4 | 2/4 | 0/4 | 0/4 |
| V65-1K | 2/2 | 1/4 | 3/4 | 1/4 | 1/4 | 0/4 | 3/4 |
| V65-3K | 2/2 | 1/4 | 0/4 | 0/4 | 0/4 | 0/4 | 4/4 |
| A-Control500 | 1/2 | 1/4 | 2/4 | 2/4 | 1/4 | 0/4 | 3/4 |
| A-Treatment500 | 1/2 | 1/4 | 2/4 | 0/4 | 0/4 | 0/4 | 2/4 |
A两臂20格有效、518303 tick/1622前向。独立HOLD均保留VAL01_202、丢N03_202；两臂均在P01_701/P02_701抓持，均丢P01_702/P02_702。Treatment同两格最高抬升4.74/9.88mm，Control33.56/38.85mm；低要求出现在原P10。固定TRAIN查询72同RGB/state/上一命令/noise，R4/Control/Treatment原首缝26.32/50.97/52.97 mm/s，原安全40；L1是否修复另记，不能把原始越界当新STOP。两臂同一旧安全状态变坏，证实局部函数漂移；不证明全部退化来自某单一损失。3K没有正确抓持，不能由完整PLACE0推断条件释放能力。源MAE原告警未触发，不能代理目标闭环。

## 1. 研究目标、边界与八类Gap
总目标：冻结WBC下，先保住R4安全接近/抓持/抬升/HOLD，再实现真实完整PLACE（含合法承托、释放、稳定撤离），最终在独立布局、不同指令与未见任务验证跨本体能力。R4本就0/4完整PLACE，不把它的后半程失败当应保持行为。
不变合同：原WBC/OCS2/PD/腿策略、L1、2ms积分/50ms实际消费/H40预测/P10执行/40mm/s G7、完整H40检查、硬STOP后零积分/安装/许可、原接触与评分。禁止通过截短后缀、延时、放宽门、GT学生、任务脚本或教师接管伪装自主成功。教师可以用登记的仿真特权状态只做采集；部署输入须来自原生可用传感/控制接口。保护DEV/VAL/CONF/TEST按原角色，只回归和诊断；超参/权重/触发阈值只在独立TRAIN拟定，episode/scene家族分割，失败保留全部分母。新实现走独立版本/capsule/owner，不能热改活动执行或删除历史。

| Gap | 证据等级 | 可证伪假设和实验 |
|---|---|---|
| G1 Retention | 同状态原动作漂移直接证实，R4闭环退化 | B6参数锚定／合格P10功能保持是否减少漂移且不抑制新阶段 |
| G2 State representation | 全8031分箱确认，Y/X物理跨度约9mm | S1状态量程单变量；S2连续状态补充；旧状态输入迁移副作用需实测 |
| G3 Control observability | 参考残差与实耗历史可读，因果未证 | A2-TRACK、A3-ARM/BASE、A4短历史分别测 |
| G4 State distribution | R3全晚期，早期14尝试总4准入 | D0预警接管，C1同槽普通早期 vs 学生纠正 |
| G5 Objective | 第251步 Recovery/gripper内积正，单批 | 多批真实Adam方向；B2/B3/B4/B5择一 |
| G6 Horizon | H40/P10合同，后缀单批梯度大 | B2与RTC/事件辅助各自匹配，不删完整H40 |
| G7 Spatial adaptation | 官方与现役训练/冻结范围/EMA差异 | SFA0/1/2三臂逐变量；不是G1已证根因 |
| G8 Embodiment dynamics | 参考先分叉，底座与手臂几何均有例 | E0逐层诊断；A2/BASE/ARM；不默认解冻WBC |
评价Gap贯穿以上：固定MAE/H40包络合法不等于抓持/接触/任务成功；本计划只将代理用于筛选。

## 2. P0-B6：优先接续已实际派发的父参数锚定
### 2.1 精确复用已登记实验，不重新标定
当前 [实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/scripts/sla/v7_parent_anchor.py) 为共享可训练Action Expert全部继承叶的**平方和**（不是逐层均值），
L=L_current+lambda*sum_shared ||theta-theta_R4||²。
原合成梯度之后、原global clip及AdamW之前加2lambda*(theta-theta_R4)；原私有IO与VLM冻结范围不变。neutral previous-command 两臂；两臂原500步bank（a8fde923b6453be823fd8097b8440a7ad8780d6d484bc3d882df8a5b070e64ff），每步96Go2=48旧+46新+2R3、Robo16、LIBERO16、aux4旧2新2、0.375；原noise和LR完整500日程的前缀（peak8.481070829156122e-7，warm50，cosine500至0.1peak），不能把100步改成100余弦。固定R4共享父、原冻结指纹、相同新Adam/RNG起点；R4 parent receipt f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb，按真实现场再核验。Control已存在的500端点只作历史参照；本轮新LAMBDA_ZERO与SHARED_L2SP配对不能被旧控制端点静默替换。
校准来自Control250真实第251步训练批次：原共享加权梯度0.07164487987756729、原锚定梯度0.774902880191803；lambda=0.1*前者/后者=**0.009245659257303812**。该系数只在该点对应10%梯度比例；R4处罚项及梯度零，真实第2步加权锚定梯度5.64e-6，早期可能很弱。保持此系数到该对照终态，不能因100步效果小而在同臂加大。若需更强系数或按层归约，另立实验。
资格PBS738822 F/0、2工程步，冷恢复checkpoint ac6282a63f8035df110ebd0967c86996897f221b5bf4ea794ca3d8013bfd4a8e、四组Adam2、36叶。正式738836原账本为R，双100（200计划更新），当前终态未知。**先核现役实际终态，不重复派发、不重开资格、不自动延长。** 当中断时保存已实耗和未入账更新，按原规则合法恢复。

### 2.2 恢复与接力的真实代码缺口（审查发现）
[现役 formal worker](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/9ada3206728ffc10e6608c9cab64a5d648cd3eee/scripts/sla/v7_parent_anchor_formal.py) 逐臂 run.mkdir、创建R4新learner、range(end_step)；未见从该轮100 checkpoint恢复优化器/游标的显式分支。不能直接把end_step改500运行并记成100→500续训。立即准备独立续训入口（如 v7_parent_anchor_resume.py；名称为建议，不声称已实现），复用原v4恢复/保存函数；以下全部成立才接力：
- 原双100完整、每臂checkpoint/terminal/ledger/原source bank/父及freeze身份验证；两臂各自共享/三私有/新增条件/optimizer/RNG/cursors及step100恢复，不将资格2步端点带入正式。
- 新capsule明确原terminal的SHA，append-only接力账本；历史目录只读。逐臂从真实step+1读取原500步bank和已登记rates，按实际noise数组匹配原控制账本；不可重跑1–100、重置warmup或取错Adam state。中断不同步时只从各自最后合法完整端点接续且记录无法回收的实耗；对照终态只比同step。
- 先CPU/零持久更新的下一行恢复与原始随机流回归；任何实际优化器更新均按原工程/正式口径计，不能以shadow隐藏；若必须新增资格且余额不足，标BLOCKED，其他工作继续。旧实现磁盘free检查还应核个人实际quota及完整checkpoint≈6.04GiB、同时保留所需端点/15min收尾，不以共享free替代个人quota。旧checkpoint不删。
- 原代码保护deadline−900；继续保持单H200、12CPU/225GB或现场合法已批资源，原期限不变。若活动作业存在，先等待其真实终态而非另起重复。

### 2.3 100→250→500有限决策（候选新规则，冻结在读取新端点评价前）
100为首个科学机制筛选端点。先独立Core恢复，并封存一个**独立TRAIN诊断面板**：真实R4成功状态（含已知query72、84，仅作诊断而非已准入蒸馏），普通教师PREGRASP/CLOSE/LIFT/HOLD及后半程合法监督、R4早期失败状态，按episode/phase分层；全量可用则用全量，否则事先固定至少4场景/每阶段可用数量并报告缺口；不得选保护格调参。对R4、LAMBDA_ZERO、SHARED_L2SP使用相同RGB、原始state/token、实际命令/noise，比较raw归一化H40、物理P10 XYZ/SO3/夹爪、参考与L1修复、原门、条件进展、源固定DEV、分层参数漂移。固定两次相同backend/dtype重复测量误差；缺真实标签只诊断相对输出，不冒充误差或闭环。
报告原共享/Go2 IO相对父L2与逐叶相对距离、锚定梯度与合成梯度比例（每阶段p50/p95/max）、clip前后与Adam方向、总体/分项FM、aux、P10/后缀，保留失败状态、实际物理可执行性；不能单靠参数距离变小判好。
100后判断：
- 技术/安全/数据身份、冷恢复或随机流失败：不延长，修具体缺陷后按原权限新登记，不抹失败。
- 正臂在预封TRAIN安全状态新增原始非法/显著更弱抬升、错误目标倾向或旧阶段明显劣于零臂，且不属于测量容差：暂停该配方，不用更多步掩盖。
- 诊断显示锚定有非零作用，原安全状态相对零臂不退化且存在实际有利迹象（如共享累计漂移减少≥5%，或相同安全状态原首缝/P10物理偏差相对R4改善超过重复误差；同时不压制合法新动作）：在实际余额与恢复资格允许时先双250，再复查同一TRAIN面板。5%是方向性筛选，不是晋级。
- 若100处锚定尚很弱但安全无退化、原日程和已封校准点250附近说明需要观察，且已记录非零锚定和合理的更新尺度，可**一次**条件双250观察；若仍无方向性收益，停止，不自动调lambda/延长。250后仅在正臂有一致机制改善、未压制合法新阶段且配对完整、资源可用时双500。250仅中期TRAIN诊断；保护十格不得用于选步。若现场已完成更长合法端点，直接结算，不能倒退重跑。
- 100/250/500间的推进由两臂共同决定，不能因看到某臂表现好而只延长该臂。若选终点，在原生之前封存该唯一端点、参数/数据/原生名册。
### 2.4 原生评价与对照
旧50张额度已知46张有效已用（A2506+1K10+3K10+A20；实际启动/技术失败另核，余额最多4）。若无新增授权，B6只能做**4张方向性筛选**：预登记每臂同一 N03_202（R4成功、A两臂丢失）与 P02_702（R4抓持成功、A两臂丢失），原场景/seed/noise/安全/预算，合计2×2=4；不挑新好格。仅在被选同一完整端点、资源/槽正式映射后执行；也可保留槽等更有依据候选，但不得将两个单卡结果宣称晋级。若只余不足4不拆不对称；有完整配对名额后仍执行原N02 2+独立HOLD4+PLACE4/臂。若之前4张完全相同端点/名单/合同且有效，可计入20格的对应4格，需额外最多16；否则需20新槽。完整面板门维持N02 2/2、独立≥2/4、PLACE grasp4/4、lift≥3/4、in-zone≥2/4、无新增硬安全STOP、源告警不触发、完整≥1/4优先或前项不退化的真实阶段进步。相对零臂的机制改善与相对R4的能力晋级分别报告。小样本不是统计显著性；保护原格不作调参。新独立确认用于泛化。

## 3. P0-S：状态编码与归一化，两个问题分开
当前量化证据：旧52 Y低于下界241/4450，新41 208/3466，R3 79/115；Recovery X上界代码78（64真正超过上分位），物理483.770–492.756mm跨度8.986mm；Y同下界-75.277至-66.016mm跨度9.261mm；夹爪27同上界仅41.99858–42.00012mm，优先关注XYZ/旋转而非把所有上界都视为严重。新增rotation归一化最大24.90/13.20，突出样本实际相对原统计中心SO3 20.83°。全量TRAIN新1/99状态分位反事实：R3 Y下界79→18，X上界78→43，仍有合并；没有改任何现场统计。参见E-S。
### 3.1 立即零优化审计
- 固定全8031状态原文件、原统计、codec/tokenizer/float32和episode-fixed mount。按任务/episode/phase输出原/新q01/q99、物理量程、下界/上界分开（上界bin不是所有都超界）、每bin物理跨度、SO3实际距离、全部token变化率/长度、旧52安全阶段token变化。对稀有场景按独立episode而非重复窗报告。
- 重算**state-only**统计：全TRAIN1/99作为已实测候选；再准备一个按回合/阶段平衡的固定分位或有物理边界余量的候选作为离线比较，不用保护集拟合、不按闭环好坏搜索。保留ACTION.json原SHA、解码、原七维语义与32维输出不变。分箱覆盖提升需同时报告在常见区间分辨率降低的代价；阈值事先按TRAIN测量和物理量纲确定。
- 旧52 reader补train/provider输入一致性（当前44窗未覆盖），物理标签回放扩大至全部96回合可读部分，优先外侧箱、旋转尾部、尾窗、关键CLOSE/LIFT/LOWER/OPEN及源域独立抽查。原3回合241/9460的独立对照已通过；50ms起点 vs 终点必须同口径，不重复重锚。若发现全局语义错误先停对应训练修复；否则不无限重验无变化文件。
### 3.2 S1：状态量程单变量
对照共同R4或已完整晋级的父、原数据/真实noise、相同可训练scope/Adam/步数/aux，Control旧state量程，Treatment一个预封TRAIN新state量程；两臂同ACTION统计及物理编译。新state统计会改变step0 token/预测，**这是处理本身，不可声称初始函数相同**。先冻结权重做原/新token的同状态前向，量化安全旧态损害；如果初始明显恶化，优先S2，不直接训练掩盖。正式候选每臂250（若机制强、预算允许才500），先完整三路径/恢复资格；独立TRAIN比较分箱边界辨别、物理动作与原门。不得与A2/新数据/EMA/视觉范围同时改。Go2私有统计更新不改变源域。
### 3.3 S2：保留离散语言，补连续七维
若S1旧token迁移代价大或稀有状态仍合并，选连续状态旁路：保持原token、原action输出、原视觉；从真实原七维（XYZ/rotvec/实测夹爪）使用TRAIN固定物理尺度的无裁剪连续映射和有效位，经Go2私有零初始化投影到Action Expert（可比V7-A接口，但维度/尺度另登记）；Control同结构同参数量，中性无信息；Treatment连续真实状态。必须检查旧路径step0完整H40相等，非零注入能影响FM/部署/aux，Control零条件梯度与Treatment梯度，实际状态dropout及mask一致。先不加入参考残差（A2是新信息而非精度修复）。仅使用当拍合法信息，不泄漏teacher phase或未来，源码需确证真实用到，而非reader存在字段。实验对照同上，S1与S2择一优先，不同时启动两项。

## 4. P0-E / A2-A4：真正的控制状态与全身动态
R4逆重锚后，在同固定坐标中平移标签 a=r_future−r_current（Codec先相对实测p编码，重锚逆变换抵消p）；SO3按矩阵组合，不用分量相减。模型未显式看到当前accepted reference；3回合残差中位16.21/10.51/12.79mm，最大41.03/29/32.15mm，已知286/286查询可因果关联上一完整P10接收边界。依赖并不证明相同可见输入存在冲突标签；教师未来参考也会随状态变。
立即E0：复用12轨迹138718请求、四格首查P10差0.067–0.199mm、1mm阈值accepted差先于TCP的结果。前80共同50ms区间319/320三向请求数/新参考偏移一致，不把切换延迟当主要原因。对A/R4/3K扩同状态交叉表（普通教师、R4安全、当前学生早偏、真实恢复），固定实际RGB/state/noise；按原raw H40→全块检查/L1→accepted→inferred selected/独立真值→实际arm/PD→base/q/qdot/TCP→首接触。被动推断不是直接真值，静态置换不是动态修复。四个1K抓前碰撞中P02_701/N03_201底座置换改善、N03_202/VAL01_202手臂置换才改善，不能推广所有新失败。P02_701的250ms accepted下降1.73、实际TCP11.17、底座12.73mm，R4对应1.49/.95/.01mm；基于原始实际时钟验证新卡。错蓝块接触做独立TRAIN相同状态noise的红/蓝语言和目标方向，已有响应幅度1.95→1.19→1.03mm仅两个诊断观测，不定性语言遗忘。
A2-TRACK候选：在S1/S2独立选择后或证据强时单独测试同拍 accepted TCP 与 measured TCP 的有符号平移残差（m）与 SO3 log 旋转残差（rad），明确世界/固定mount表达，previous confirmed boundary 必须 query之前真实存在；首query用可验证暖启动参考，否则共同mask，禁止用未来endpoint、事后inferred-selected或GT。全8031窗口覆盖需实证，286诊断窗不能当全库；缺证按共同可用bank预登记，不静默删。先分析状态残差与正确参考增量的条件方差、分阶段输出和原门；Control同零入口中性，Treatment真实残差，保持现有7维token；零初始三路径一致、实际条件敏感性和梯度、保存/独立恢复。两臂同数据、步数、LR等；每臂250候选。A3-BASE（真实估计姿态/高度/速度）、A3-ARM（实际q/qdot或低维编码）、A4-HISTORY（两个已完成query及实耗动作）后续择一，先证明实际部署源和因果性。原WBC冻结；不存在教师也能重复完成任务时才另研究低层能力边界，非本计划改WBC权重。

## 5. P0-D：早期纠偏数据工厂，从G7拒绝扩展到碰撞前路径
原R3 115窗来自3个接管起点，CARRY26/LOWER50/OPEN12/WITHDRAW27，PREGRASP/CLOSE/LIFT为0；正式V65已6,000次R3 FM访问而无早期覆盖。实际14次尝试4准入155窗，前两张早期0准入（蓝自主抓持抬升50.94mm入区但未触发；红教师首包577后误588应578，CPU修复后仍需真接管），后两张43617tick/130前向均接触STOP、教师intent=null。G7触发不足：蓝无超限，红raw超限3但经L1无最终拒绝。不要仅重复同入口。
### 5.1 新事前触发器 D1（采集教师用，不是新安全门）
E-D两张TRAIN静态逐query前视证据：5mm阈值两卡未出现；10mm蓝未出现红提前6.840s；15mm蓝0.582s/红12.340s；20mm蓝2.582s/红14.340s，蓝最后查询净空11.83mm至STOP仅82ms。仅静态历史关联，未证动态接管/延迟/误触发。当前query每0.5s，5mm可能太晚。
- 先在**独立TRAIN**已关闭的成功与失败轨迹上用原时刻及最近过去qpos（最多2tick差有证，缺证不补），计算目标/错误物/桌面相关link几何净空、变化方向、实际参考残差、底座/臂状态及阶段停滞。教师允许特权几何，仅用于接管决策；学生输入不变。量化每episode触发数、相对原首STOP提前量、成功轨迹误触发、实际教师计算/安装延迟、是否在合法完整P10边界。不得用保护格选阈值，也不得用STOP后状态复活。
- 首个前瞻候选固定15或20mm之一（先在独立TRAIN比较成功误触发和失败提前量，优先20mm作为初始保守候选以覆盖蓝卡；须报告过早接管代价）。仅对禁接触link/object及合法相位使用，加入因果下降趋势/低净空与阶段停滞时需事前固定规则，不在看到新卡结果后改阈值；接管门不替代原4mm/接触硬STOP。若0.5s边界不足以安全接管，先证明现有时钟下可提前触发，不缩P10/加中途积分改变合同。保留原G7拒绝路径作为另一触发类型，不计新门通过即成功。
- 新前瞻TRAIN固定R4-200、合法独立reset/noise/场景家族（可已有教师合格几何但明确不把同几何不同seed当独立布局）；两张首波：低净空PREGRASP一张、LIFT停滞一张，原可用Recovery槽最多6，核实际账本再启动。教师仅在未硬STOP、完整P10、暂停身份/时间/owner一致、零未确认命令时接管；若G7拒绝，必须预安装零ACK/零消费。首包继承上一实耗gripper序号+1、真实命令/时钟；原577→578修复必须经动态验证。新教师动作完整H40合法、原G7 40mm/s、真实2ms/50ms、原评分。任何硬STOP后零积分、原学生接管仍算未自主完成；teacher-assisted单列。
- 首波结算后，剩余最多4张：按真实故障选择第二波2张，并至少保留2张给下一版合格学生状态；没有合法候选则保留，不凑数。若原额度变动以实际最小余量为准。若动态接管失败，区分未触发、晚触发、技术错误、原安全、教师规划、实际跟踪；修具体原因后只在新登记合法名册验证，旧失败保留。
- 新采集记录全部学生prefix及teacher suffix、干预tick/真实观测/noise/上一命令/accepted/selected/实际消费/原安全。支持label只由真实名义消费、原逆重锚、mask；完整成功后缀按原准入；短安全纠正另立版本待资格，不暗中混旧成功库；失败prefix不得当正向BC。同步获取B6安全阶段anchor直接selected/消费证据；独立TRAIN场景分割。每波实际manifest/hash/reader独立核验，冷交接，不热换B6。
### 5.2 C1：早期纠正 vs 普通早期教师
具备至少两个不同独立早期接管起点、明确PREGRASP/LIFT支持及匹配普通教师池才进入候选（不足只数据准备，不反复曝光窄库）。每步48旧+44普通共享+2原R3+2对照槽+16Robo+16LIBERO=128；Control两个普通早期教师，Treatment两个学生早期纠正，匹配phase、有效H、scene/task分层，aux原共同4、相同权重。两臂共同新bank和实际noise；不能复用不同bank的A/B6 control。250/臂初始候选，保持对照父与其他输入；若真正有效，再独立研究Recovery总4→8→16，不默认上16。更新后由已合格当前学生继续新TRAIN DAgger，独立起点与干预率/后续实际成功为指标；不只采R4反复场景。长期覆盖目标按新权限/实际数据：PREGRASP/低净空/SEAM各5、LIFT5、CARRY扩至10、OPEN5/WITHDRAW5；非当前额外额度。
### 5.3 B6行为锚点后继（不与当前L2SP混）
独立TRAIN成功安全局部P10、实际selected/消费证据、原完整实际noise、真实RGB/state/时钟/阶段成功；当前12回合/1126完整P10仍0准入，缺直接selected依据。新采集主动记录，禁止将accepted或inferred冒充实耗真值，也不把R4失败CARRY/OPEN固定。RETENTION_ANCHOR_TRAIN与BC分开，原学生失败prefix默认不准入。物理位置/SO3/宽度归一化的P10保持，原H40完整生成、只算真实消费支持；applied vs raw可能不同，初始父loss非零需记录。两臂相同anchor/noise/额外前向（系数0/正），候选每步2个锚点，阶段与独立回合平衡。λ仅TRAIN固定校准非零漂移点，不能在R4零梯度除零。缺足量合法anchor先留B6 L2SP结果。若纯功能蒸馏用父raw作为目标必须单列，不把未执行后缀当专家真实标签。

## 6. P0-L：优化目标、近期执行与实际梯度
第251步真实Go2分项：old/new/R3 gradient L2 .03307/.09517/.00550；P10 .01626 vs suffix .09552；XYZ/rot/grip/pad .04466/.07847/.02599/.00435；提议Adam对grip +4.15e-6、R3 +8.94e-8，单批一阶不等同真实训练后loss或任务因果；源共享cos -0.00195/-0.00243接近正交。907归档输出G7非零35、首缝argmax20/35、P10内部8、suffix7，首点梯度21/907，不能断言max总在后缀。真实训练每臂Go2 48000中四种PLACE真P10张爪过渡429、aux2000中仅6；红右276/3；R3 aux=0。任务episode红右27含3R3、蓝左5、蓝右1、红左1；重复采样不等于多样性。
立即多批零持久更新诊断：固定现役模块指纹和3–5个预先封存TRAIN batch/端点（已有251，另选早期/中/晚的完整checkpoint邻接行，有真实Adam和noise才复演；缺则只梯度不冒充Adam）。分旧/新/R3/源/aux、P10/suffix、物理7/pad25、XYZ/rot/grip、phase重构原loss和梯度及容差，实际合成、global clip、Adam proposal，dot和在内存重评目标的真实差（不持久化；实际优化器更新/资源按原口径计）；报告冲突频率、分位和缺证。不将梯度范数当可加权重，不默认PCGrad。
候选按证据择一：
- B2 P10-weighted FM：2:1候选，逐窗有效 N，分母32*(N+10)仅当至少前10真实有效且连续；一般用sum(mask*w)*32。完整40时25→40%，仅P10窗不变；251真实批均值28.47→42.92%。保留全H40及原aux。先同batch shadow重评P10/grip/R3方向与后缀/旧安全风险；若多批仍表明近期监督弱，再匹配250/臂。不把梯度改善直接当任务成功。
- B4 Recovery/event-aware Aux：固定FM bank、总aux4和0.375，仅替换1–2个槽为合法完整H40 R3或确有真实近端OPEN启动/增宽的样本；Recovery与关键过渡是两个候选，不混作一项。相同可读、真实noise/采样、其他三个槽保持；若无合法完整支持先重封双方bank，不造后缀；统计OPEN启动、充分张开、持物不误开、下放承托/撤离。不能强迫每个P10完全张开，真实教师跨多查询渐进OPEN。
- B3 Physical Delta：首点previous在预测/标签相减中抵消，不能代替条件输入；复用原函数第二项、真实上一命令及支持，λ在固定TRAIN批按梯度比例标定；保留原G7/width/XYZ，观察首缝/内部及任务进展。
- B1 Active/padding: 原7/32、25/32重构等价后，仅在多批真实物理梯度受压证据时降低padding相对权重；现有单批padding小，低优先；32输出合同不变。
- B5 Go2-only：只有多批显示源域冲突时，在相同target/aux曝光及相对系数/Adam/LR下移除源梯度，报告全局clip差和有效batch差，不用Go2填空源槽。源私有IO不更新按预定合同。PCGrad/GradNorm另项，不与去源混合。
- R-EMA：现役端点无独立EMA证据，官方EMA .999；在新独立课程可记录普通与EMA两套推理参数，EMA只改变读出，不改变训练梯度。短500从R4初始化父贡献约60.6%；先比较同step普通/EMA旧能力和新进展，不能用父权重主导的保持假装学会新技能；原已有checkpoint不能逆推出真实逐步EMA。新增checkpoint体积/存储/恢复先核，不改当前B6。
- RTC：冻结R4/有代表端点，同观测/noise的离线因果连续引导，查真正可用的历史命令与未来承诺prefix；同步物理暂停模式不存在异步推理期间执行前缀，不能伪造延迟或强制重放已消费P10。原H40、G7、L1保持，报告raw/处理后合法率、延迟、近端进展、误停；过度保守不晋级。训练prefix和异步部署另立实验。

## 7. P1-SFA：按官方SLA后训练路线恢复视觉空间适配
官方版本固定 db09d6141f58b113261aaba840a835df5883c2cc。[官方README](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/db09d6141f58b113261aaba840a835df5883c2cc/SLA/README.md)、[LIBERO](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/db09d6141f58b113261aaba840a835df5883c2cc/SLA/src/openpi/external_config/train_libero_plus.py)、[RoboTwin](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/db09d6141f58b113261aaba840a835df5883c2cc/SLA/src/openpi/external_config/train_robotwin.py)、[Piper](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/db09d6141f58b113261aaba840a835df5883c2cc/SLA/src/openpi/external_config/train_real_world_handover_vegetable.py)、[Pi0Align](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/db09d6141f58b113261aaba840a835df5883c2cc/SLA/src/openpi/models/pi0_align.py)、[normalization](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/db09d6141f58b113261aaba840a835df5883c2cc/SLA/docs/norm_stats.md)、[checkpoint EMA](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/db09d6141f58b113261aaba840a835df5883c2cc/SLA/src/openpi/training/checkpoints.py)、[tokenizer](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/db09d6141f58b113261aaba840a835df5883c2cc/SLA/src/openpi/models/tokenizer.py)。主要公开配置 action FM+0.5空间对齐、state dropout .25、EMA .999；RoboTwin VLM LoRA与Expert可训练且视觉模块未被过滤器冻结；LIBERO/Piper默认无VLM全冻结，H10/H50、步数30K/20K、batch512/64，RoboTwin H50 30K/256。不是所有下游必有align（RoboDojo-Sim论文例外），官方也使用离散状态。当前冻结VLM/LoRA、G7/width aux、原统计及FM train=False。配方差异是可验证假设，不是已确认退化根因。
准备不等待B6：检查现有官方SLA training_component/OmniVGGT许可、固定checkpoint/模块SHA、实际资源和GPU可用性；用真实同RGB/mask、无GT的对齐特征cache试验，官方视觉hidden capture位置(第12层及RoboTwin适应层配置)和padding/multi-view correspondence。若引入增强，必须同输入坐标/空间teacher语义一致，单独验证，不能把颜色扰动和解冻同时作为主变量；禁止在冻结上游只训projector然后称恢复视觉表征。确认梯度到达预定VLM LoRA/视觉范围，不把官方0.5盲目当最佳。
条件正式SFA：共同已验证父（无晋级则R4），同Go2数据、统计、输入、H40/P10、原FM和G7/width/aux、noise/LR/step；三臂F=原冻结，L=开放原有VLM LoRA（保留已训练值），LA=同L加实际OmniVGGT空间对齐（独立trainable projector；为公平L也创建同projector但λ0/不影响动作；F/L训练参数范围不同是此对照变量）。先F/L单对、L/LA同完整条件时复用L，不能用不同父/bank/步数的旧控制。每臂250候选，真实资格和独立恢复后才派发；原视觉dense保持冻结是本项目保守候选，不宣称复制官方全部训练范围；如需开放视觉dense另对照。记录视觉对齐梯度和动作/旧能力/目标颜色语言响应，固定独立TRAIN数据选择系数；EMA、状态、数据新增不混。无组件或资源则BLOCKED，其他工作继续。
Source608：公开多源只是先验，当前本体问题不靠盲加外部数据解决。保持现有16Robo/16LIBERO作为对照；后续在目标能力保持、五域实际联合前反向与冷恢复合格后，按原SOURCE608资格独立扩充；任务/本体分域H/时钟/动作语义/私有IO/统计保持。优先与目标抓放/容器/插入相关的独立多样高质量数据，按合法license和reader抽查，不能把原始窗口总数当独立任务。源域保持不替代目标成功。Core A93→100中Container/Push、N2/N4独立资格及真实教师生产继续在原权限内，不抢评价；Vanda暂停。

## 8. 长程闭环学习与完整任务（P2，条件开启）
当前先验证前段保持与早期恢复。对已到达区的回合诊断实际物体全部角点入区余量、目标下放、承托、开爪启动、连续增宽、脱离稳定及撤离；未到阶段条件成功率记NA。1K一次充分开爪仅H40末端不能普遍解释；A-Control入区后无充分OPEN，P02_701旧40.44mm因误用44mm机械比例已撤回，正确H40 38.76/P10 20.27，非合法释放。严格区分2–42命令与44机械。
在已有独立稳定自主完整PLACE后，才进入基于真实成功/失败/干预经验的RECAP/Value/Residual RL：先CPU schema和TRAIN value校准，原安全/terminal、实际实耗和奖励，避免刷阶段或提前松手；GT用于仿真离线奖励/特权教师，不进actor输入。Residual Policy仅在独立登记、冻结原SLA/WBC、小型有界残差、全H40/G7/L1原门验证，不能伪装原VLA本身能力。无稳定自主阳性不启动全模型RL；无额外权限不自动派发。参考 [DAgger](https://proceedings.mlr.press/v15/ross11a.html)、[HG-DAgger](https://arxiv.org/abs/1810.02890)、[IWR](https://arxiv.org/abs/2012.06733)、[L2-SP](https://proceedings.mlr.press/v80/li18a.html)、[RTC](https://www.pi.website/research/real_time_chunking)、[RECAP](https://www.pi.website/blog/pistar06)、[HIL-SERL](https://hil-serl.github.io/)。借鉴方法与实证分开。

## 9. 自主决策树与长程排程
T0为接力读取本PR后实际时间，非承诺耗时。现役Hopper优先B6实际作业/冷封存；Corelab独立推进审计、预警teacher资格及原生产。只使用已有执行者/owner，不新建重复调度器。
| 时间/条件 | Hopper | Corelab/CPU | 决策 |
|---|---|---|---|
| T0–6h | 核738836实际终态；完整双100，准备精确resume，不重复 | 固定TRAIN面板、S状态量程/旧reader、D1提前量验证、E0/L0扩展 | 100只在实际条件满足后双250 |
| T+6–24h | 条件双250/500；核实际资源/原噪声/恢复 | D1新两卡（合法slot）；B6同状态诊断；A2全量来源，RTC离线 | 选唯一B6原生端点或停止；如无完整名额只方向性筛选 |
| T+24–72h | 根据表择一S1/S2、A2、C1、B2/B4或B6保持后继；先做完整资格 | 继续独立故障采集、selected真值anchor、候选训练bank；SFA组件资格 | 仅条件匹配短训；缺评价/工程资源时继续数据/代码，不能冒充晋级 |
| 第4–7天至原截止 | 选择有实际正面证据的单变量组合、小规模独立确认；准备SFA/Source608 | 扩教师任务、独立场景/视频/失败证据，按真实学生再采 | 只有前段保持且完整操作阳性才研究长程value/跨本体；原期限到停止 |

选择规则（在读取新端点后按实绩，非同时跑全表）：B6改善并保持→选最强状态/数据缺口；B6参数变小但功能仍坏→合法anchor功能保持或部分Expert/分层LR择一；若S1能恢复精度但旧token迁移失败→S2；若新条件可降低真实参考残差相关歧义→A2；若早期纠正真实准入≥2独立起点→C1；若多批P10/关键通道冲突仍重复→B2；若生成辅助覆盖不足→B4；若前段保持但后段不进展→阶段曝光与释放对照；若固定图像语言目标响应明显退化且数据/控制已区分→SFA；没有直接证据时继续诊断，不用网格搜索替代原因。任意新组合须各独立因素先证明作用并配对；阶段未到达不当后半程失败归因。 
所有候选需符合既有权限、完整数据真实可用、代码CPU负例/完整GPU资格和独立冷恢复。正式训练前必须有合理配对评价/机制筛选计划；若无评价余额，不盲目耗完训练预算。原50张学生额度的4张不足以给所有后继训练晋级。可在已合法范围内推进筛选、数据和实现；不足才一次性列清明确资源缺口，不为普通代码/文档/只读诊断反复请示。每个工作包有实际有限成本、超时或无改善停止规则；所有无效结果保留并转独立工作，禁止无限重复。原320 GPUh、12月5日12:41:52 SGT截止，未获新权限不延长、不从其他站点偷偷借额度、不抢现役owner。

## 10. 预算审计与后继资源决策
以下为已知实耗/上界，实际现场最新账本较小余额优先，健康作业优先，不重置：
| 类别 | 当前已知 | 计划占用 | 余额上界 |
|---|---|---|---|
| formal | 4600（旧500+V65逻辑3000+100未封存实耗+A1000）；B6已派发200但实际终态待回读 | B6双100→4800；双250→5100；双500→5600（续训增量，不重训） | 双100后3200；双500后2400/8000 |
| engineering | 17旧+4 A+2 B6=23/24 | 最多1剩余，不为重复资格花费；新的scope如需更多资格先列缺口 | ≤1 |
| student native | 已知有效46/50 | B6两臂各2最多4方向筛选（合法映射后） | ≤4，若实际其他启动则更少 |
| Recovery | 14/20已知尝试、4准入 | D1先2，再条件2、保留2给下一版 | ≤6 |
| GPUh/时间 | 原320 GPUh及原截止 | 现场实际qstat/project award/个人quota核对 | 未取得实时余额 |

B6双500且无其他实耗后，formal剩余最多2400。**后续示例性分配**：S1或S2双250=500；A2双250=500；C1或B2/B4择一双250=500；SFA或保持后继择一两臂250=500；保留400给合法确认或未封存实耗。四项不是要求跑完，不自动授权，三臂SFA若需750另重排。实际工程/原生名额优先制约。
若B6取得方向性改善，完整20格若复用4张同身份有效筛选需补16，不能复用则20。下一项匹配双10格另20；最小独立确认建议12张（事先定不同场景/seed/任务分层，不拿训练窗作独立样本），实际资源申请需列每个endpoint与名册、预计成本、现存余额；新架构通常还需2–4工程资格步/实验，旧余1不足时需核明确新增权限。**这些是拟议资源需求，不是已经批准的额外名额。** 没有新增授权就只做可合法的4张筛选/6次Recovery以内/零更新诊断与准备，不能宣称全科学结算。所有分母计真实发生，包括技术失败和未封存更新按原额度口径；如有不同扣费定义，保留原规则并单列。原条件15000非默认授权。

## 11. 验收、实际交付、长程科研终点
每一波必须封存：代码/父/数据/统计/模型/实际noise/优化器/cursor/实际GPU和PBS回执、明确有效/未完成/技术中止及原因；实际原生逐格含query次数、积分、接收/选择/实耗、G7 raw与L1后、contact link/object、首失败阶段、硬STOP后0；报告held/gained/lost而非仅总数。接触早停删失后续G7，分别报告每100query与episode级分母及到阶段数量；固定状态指标与闭环分开。源域原10%告警不改，不能代理目标能力。新任务真机推广另行验证，此计划全部原生阳性指MuJoCo，不声称实体机器人验证。
明确状态枚举 PREPARED/CPU_VERIFIED/GPU_QUALIFIED/FORMAL_RUNNING/ENDPOINT_SEALED/ACTUAL_CLOSED/BLOCKED；区分科学负结果和工程问题。每波/最长12h更新实际回执，先收集最新增量，不重复通读历史；阻塞只暂停对应包。完成定义：
(1) R4旧前段安全保持；(2) 可见状态与标签合同及监督分辨率有明确对照；(3) 多故障独立早期Recovery有效；(4) 至少一组完整匹配原生达到原门且真实完整PLACE阳性；(5) 独立新布局确认，继而扩未见技能。前四项未完成不得以代理或训练量宣布成功。
## 12. 发布前对齐审查（本次文档审核，不是运行资格）
核对当前A已结束（不再重复）、B6已登记双100与固定lambda/共享平方和/原500日程前缀（不重标定），明确当前worker无resume路径；区分100/250/500逻辑步与实际续训增量，旧预算与新需求分开。核对状态跨度实测和fresh分位反事实，禁止将统计反事实当已部署；核对D1触发提前量是静态历史而非在线资格；原G7/硬STOP、split、WBC/L1、原H40/P10全保留；官方配置差异及空间教师不当已证根因。三份文件在同一提交，注册依赖与配方、名额算术和链接逐项回读。现场资源与远端Codex执行仍需实际确认，本次不提交PBS、不训练、不启动仿真、不传输/删除、不修改现役代码。
