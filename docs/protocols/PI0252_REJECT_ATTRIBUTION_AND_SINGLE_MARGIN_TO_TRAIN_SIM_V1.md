# PI0252 — 单例REJECT归因 → 单一对应约束对照 → 原M64门 → TRAIN闭环

日期：2026-09-28（Asia/Singapore）  
Authorization: `PI0252_REJECT_CAUSE_TO_TRAIN_SIM_GO`  
Decision: `GO_READONLY_REJECT_ATTRIBUTION_THEN_CONDITIONAL_SINGLE_MARGIN_PILOT`  
发布状态：`AUTHORIZED_NOT_EXECUTED_BY_GPT`  
审核源码ref：`41487312f2f914fc79086a65449df3246fb8019a`。Issue #11是canonical；PR #77只pointer，不merge。

## 0. PI决定：不把未知REJECT猜成线速度，不继续无差别加权位置

接受PI0251 C/E final2000有效双FAIL，旧结果不改。C的all64/old16、B0/B1四组位置与姿态聚合门已经全部通过；仅all64/B0的一条REJECT阻止整体资格。E同样有一条all64/B0 REJECT，且该组XYZ p95为0.035450747m，超过原0.035m。没有理由现在解冻骨干、扩大数据、继续增加超额位置loss或直接绕门进入WBC。

本Goal立即授权G0：用已存原件定位两条拒绝及专家可行性，不加载GR00T、不新推理、不训练。当前canonical没有披露这两条拒绝的issue代码、source、segment或幅度；本PI不预先认定它们是同一窗口、线速度或任何已确认机制。

本Goal同时预注册一个有条件的完整推进路径：若G0证实C仅有一种、且属于第3节白名单的物理动态/起点连续性违规，并满足专家margin可行性和身份要求，则自动运行A/G各2000步。A保留C目标；G只添加与实际违规类型对应的一个标量约束项。不是先试四种loss，也不是agent自行发明配方。准入类型在首次训练前由确定规则冻结；不匹配则只交付明确归因并停PI，不消耗训练预算。

通过后按原四组联合门冻结唯一development_checkpoint，再按第8节运行有限TRAIN A–D frozen-WBC one-chunk及receding。当前旧C/E仍不能直接进入sim。成功交付large_scale_init_checkpoint；本Goal不执行FULL、CALIB、正式selector、DEV/F、DAgger、真机或PR merge。

## 1. 来源、当前事实和证据边界

必读：
- Issue #11 `5861040750`（PI0251终态）、`5861042782`（清理）；PR #77 `5861044965`。
- PI0251不可变协议 `e2b86ef2c2d3ee2fb8cab05d24de24a43cd0f0a6` / `docs/protocols/PI0251_XYZ_EXCESS_RISK_TO_TRAIN_SIM_V1.md`，canonical `5858091531`。
- PI0250 `8501ffcf66ebec423e732abeb99131e0533bd63f` / `docs/protocols/PI0250_STORAGE_SAFE_CONTINUATION_TO_SELECTED_SIM_V1.md`（持久存储与同状态工程恢复规则）。
- PI0226 canonical `5733559806`及其原始TRAIN sim场景、reference生命周期、时钟和scorer绑定。
- 实际PI0251 C/E代码、cards、训练/online检查、checkpoint收据和screen命令；本次读取了上述审核ref的ce_train及gate源码，不将静态阅读冒充完整远端执行审计。

本GPT没有SSH/live qstat、原256 NPZ复算、HF完整权重读回、项目模型加载/训练/仿真。Windows与Corelab报告路径不是当前会话可读文件。以下结果来自本轮实际读取的canonical；执行agent需以原件绑定，不把文本摘要当原始tensor。

本轮读取的canonical列出的all64结果：

| 模型/noise | REJECT | median XYZ m | p95 XYZ m | median SO3 rad | 未通过项 |
|---|---:|---:|---:|---:|---|
| C2000/B0 | 1 | .017162076 | .034029621 | .016757384 | REJECT |
| C2000/B1 | 0 | .016324175 | .032613481 | .018856803 | 无 |
| E2000/B0 | 1 | .016459762 | .035450747 | .017663806 | REJECT、p95 |
| E2000/B1 | 0 | .015903903 | .034305307 | .020276540 | 无 |

old16四组均PASS。old16是all64子集，不能加成额外独立样本。按这些集合关系，两臂拒绝各自位于new48，但不能推断二者source相同。C all64/B0的p95距阈值约0.970mm；E超限约0.451mm。接近门不允许舍入、调阈值、删例或改seed。

每臂128 first-valid cells、独立CPU复算128/128及2000步在线随机身份通过属于已报告事实。完整报告SHA：`36360f2d6dcca7fe5031a7973e306ac52883a3d14e5b721b5787a396df27f39b`。

本次读取的checkpoint绑定：
- C2000：revision `7017368c42800927a17be5561056ceff58585de7`，tree `dbc5c5ff07ac73cd4bf448b5e679d0641a51ea4b33695246db79d0a5767a8144`。
- E2000：revision `616c52bbbf8789682205fc4e85097734eaff3bfe`，tree `924304dc0b4d833bc996688a520af3d80bdb56b960e25d6af134b2299ba32cbd`。

原HF repo沿现有receipt解析，不凭名称猜prefix。若有获准迁移，必须完整旧→新内容映射。若canonical、报告或receipt出现不同版本/绑定，不静默选择其中较好结果：G0先发布来源差异与正确原件映射；无法唯一确定本次C权重/256格原件则停止训练。不得根据此文件数字倒填报告。

状态：selected_checkpoint=NONE；原训练、完整性通过不等于WBC成功。现有101.9GiB空闲是执行侧快照，quota UNKNOWN；不开展新的大清理。

## 2. G0立即执行：零新模型调用，交付真正的拒绝原因

### 2.1 原件绑定，不再重做整库

读取已发布md报告与两份screen/CPU报告、score cards、每个REJECT对应record.json/tensors.npz/实际state/timestamps及gate配置。建立一个小型可读索引，列实际路径、源码commit/hash、命令、model revision/tree、输入/noise hash。复用既有独立CPU检查；只对本次拒绝和正例开展下面的针对性复算。

对C与E分别列出：完整source_episode_id、old16/new48、noise、全部issues及severity、observed/threshold/provenance、所有相关峰值及segment索引、初始位置/姿态差、now/observation_timestamp/dt、模型与专家首末点。不能只保留第一个issue或打印一个REJECT字符串。

对同source的B1和另一臂B0/B1作已存数组比较；不同source如实标注。把gate重新执行结果与原record逐项核对。Gate多条件取OR，不将XYZ/SO3平均过门视为每点或每segment都过门。

使用原专家标签、同state、同timestamps和同gate作CPU阳性对照。对原64专家同时记录第3节被选约束的逐窗最大值；全套专家其他issue保留说明。专家自身违反对应门，不证明不存在别的可行轨迹，但说明不能直接把该标签拟合作为修复答案。

### 2.2 分解与判定

- 若是LINEAR/ANGULAR_SPEED：列39段原物理速度、峰值及距原门余量；不只列全轨迹平均。核对实际dt=.05，区分生成误差与时间戳错误。
- 若是INITIAL_POSITION/ROTATION_DISCONTINUITY：用该cell实际achieved state比较首点，不能用专家首点替换机器人state。
- 若是STALE/OUT_OF_ORDER、workspace、invalid SO3/shape/非有限、目标/参考系或其他问题：保存明确证据；不采用第4节训练去遮盖契约问题。
- 对拒绝窗口报告其XYZ/SO3 RMSE和完整残差曲线；对全64位置p95沿原定义报告。跨窗口p95与H40尾段是不同概念。
- 先检查原件合法性和专家可行性，再讨论模型拟合；当前没有证据将问题归因于全局容量不足或数据量不足。

报告为 `PI0252_REJECT_ATTRIBUTION_V1`。G0不需要找到神经网络内部唯一机制，但必须找到确切gate issue及对应物理值。没有保存的细节写UNOBSERVED，不新做模型推理补历史。

元数据/路径/报告工程问题可自主修复；只在独立新位置写erratum，保留旧输出和旧FAIL。若原gate判定无法由相同数组/配置重现，先停 `PI0252_GATE_EVIDENCE_CONFLICT`，不把重新算出的较好结果自动升级为旧PASS或直接进sim。

## 3. 条件训练准入：只选择一个实际对应约束

初始化固定为第1节C2000，不是E；C已经所有位置与姿态聚合项达标。E只用于归因参考，不允许根据G0再改选E或旧1000点。

A/G训练仅在以下全部成立时准入：
1. C原128格、权重、实际decoder/gate和输入绑定完整，原all64/B0拒绝数量确为1，且该cell的REJECT issue集合恰好只有下列一个代码；WARN允许保留。
2. 代码属于：LINEAR_SPEED_REJECT、ANGULAR_SPEED_REJECT、INITIAL_POSITION_DISCONTINUITY_REJECT、INITIAL_ROTATION_DISCONTINUITY_REJECT。
3. 原有效门分别为2.0m/s、4.0rad/s、0.10m、0.70rad；从score card/record实测确认。若实际配置不同，不自行代入新值继续，提交绑定冲突。
4. 该C拒绝source的专家经相同gate无REJECT；所有64专家对所选单一约束均满足本节90%训练margin。其他专家gate问题如实侧报，不能隐去；若显示真实标签/frame冲突使目标无效，则停止。
5. 当前C及专家相应物理量合法/有限，旋转矩阵遵循原合法域，不加identity fallback。原监督与该margin不存在已观察的同类冲突。

确定映射：

| 唯一C REJECT代码 | 唯一G primitive | 原reject门 | 新训练margin（不是验收门） |
|---|---|---:|---:|
| LINEAR_SPEED_REJECT | peak_linear_speed | 2.0m/s | 1.8m/s |
| ANGULAR_SPEED_REJECT | peak_angular_speed | 4.0rad/s | 3.6rad/s |
| INITIAL_POSITION_DISCONTINUITY_REJECT | initial_position_jump | .10m | .09m |
| INITIAL_ROTATION_DISCONTINUITY_REJECT | initial_rotation_jump | .70rad | .63rad |

这不是四臂搜索。G0产生唯一primitive并封存一次；A/G共两臂。不得根据训练成绩切换primitive、同时叠四项、修改margin或用失败后另一行作备用实验。G0不匹配、混合REJECT或专家对应margin不满足时，终态为 `PI0252_ATTRIBUTION_COMPLETE_NEEDS_PI_CONSTRAINT_DECISION`，训练/新查询为0，交付已明确的原因及约束冲突；不可称本轮学习假设FAIL。

90%是预注册的保守开发margin，不是硬件标定，也不是已证明最优值。原SafetyGate全部门保持。

## 4. 精确训练目标：原C目标＋一个对应的无量纲超额项

两臂保留原C（亦原X）目标：

`J_A = L_native + .4*ell_pos + .1*ell_delta + .1*ell_rot`。

`J_G = J_A + .1*ell_gate_margin`。

不再包含PI0251 E的窗口位置超额项。原pos/.05m、delta/.05s/1mps、rot chordal/.05rad、mask与所有native有效10维均保留。新项来自同一次label-free完整S4生成的物理p_hat/R_hat，不增加第二rollout或另采noise。

定义rho=.9。针对G0冻结的唯一primitive：

**a. peak_linear_speed**
`z_j = ||p_hat[j+1]-p_hat[j]||² / (dt_j*2.0m/s)²`；
`ell_gate_margin = relu(max_j z_j - rho²)`。

**b. peak_angular_speed**
令b_j=dt_j*4.0rad/s，要求0<b_j<pi；
`z_j = ||R_hat[j+1]-R_hat[j]||_F² / (8*sin(b_j/2)²)`；
`h_j = sin(rho*b_j/2)² / sin(b_j/2)²`；
`ell_gate_margin = relu(max_j (z_j-h_j))`。
这利用合法SO3的 `||R1-R0||_F²=8*sin(theta/2)²`，在原角度域内与角速度margin同序；不通过acos求新训练梯度。验收仍原geodesic，不替换gate角速度定义。

**c. initial_position_jump**
`z = ||p_hat[0]-p_achieved||² / (.10m)²`；
`ell_gate_margin = relu(z-rho²)`。
p_achieved为该训练样本真实当前EEF位置，不是专家第0点。

**d. initial_rotation_jump**
`z = ||R_hat[0]-R_achieved||_F² / (8*sin(.70/2)²)`；
`h = sin(rho*.70/2)² / sin(.70/2)²`；
`ell_gate_margin = relu(z-h)`。
R_achieved为原state/原row-rot6d合法还原。

dt_j沿实际原timestamps并核验为.05s；不改变时间轴。取全部39个有效transition的最大值，不按39平均，不额外除3。起点项不与39段混合。保持全40点与原mask；此cohort有效维度应完整，缺失则输入契约错误，不能过滤问题点后继续。

对并列max采用最低segment index的确定性子梯度；ReLU精确边界采用0次梯度。这是本轮实现约定，须CPU fixture覆盖。batch固定1；不得把不同窗口混成一个max来改变样本权重。原decoder clamp/zero-range/相对frame/SO3合法域不变，不增直通梯度或非物理fallback。

新项约束的是生成轨迹自身的绝对速度/起点连续性，区别于原ell_delta对专家增量残差的均方监督。原均方已经提供监督，但并不数学保证每个segment的最大值过门。新项不是让模型静止：原位置/姿态/native目标全部保留，最终仍要求原精度。lambda=.1与rho=.9固定，不保证改善、不扫值。

## 5. G1及正式训练：复用成熟链，不新造框架

仅为被选primitive实现薄loss适配；其余映射无需先实现成通用平台。复用已有物理XYZ和rotation可微镜像，原部署decoder/gate不改。

CPU检查：
- 由已存合法数组计算新primitive及margin事件，与原gate物理量一致；rotation chordal恒等式和非平凡anchor/row约定正确。
- 新lambda=0时与原C值及梯度一致；阈值上下/并列max/精确边界、mask/分母测试；非奇异非边界fixture有限差分通过。
- 已有四步梯度和mode检查按身份复用；真实首更新在正式预算内验证，不增加模型梯度诊断。
- 实际entrypoint调用的online guard故意注入错误source/tau/epsilon必须在optimizer前拒绝；不只写一个训练不调用的测试。
- 以合成完成状态核对新family1000/2000的保存/sidecar/finalizer；报告缺项准确返回，不反复伪造旧兼容PASS。

A/G同C2000 weights，新AdamW/cosine2000/warmup10，各2000更新，LR1e-5、WD1e-5、betas(.9,.999)、eps1e-8、maxgrad1、batch1/accum1/workers0；原trainable/frozen、参数dtype/autocast、四步重计算不变。不是旧C optimizer的exact延长。

原64/2000曝光表按hash复用，48窗31次、16窗32次，不根据REJECT source加采样，不叠shuffle。全部窗口同一个选定margin项；不只对B0或那一条source打补丁。训练不使用B0/B1噪声packet作目标搜索。

共同RNG：ASCII `PI0252|22301|stage=M64_AG|optimizer_step=<本phase十进制i>`；seed为SHA256前8bytes大端 & ((1<<63)-1)。沿已测隔离边界覆盖native/S4/backward，原Beta/Gaussian/cast不变；key不含arm/机器/attempt/时间，恢复不重复前缀、不在checkpoint重计算递增key。

A先保存真实step/sample/key/tau/epsilon/stock记录。G每step实际检查同索引输入；最迟backward/optimizer前拒绝失配并记录guard事件。不能用训练终点POSTHOC重生预期替代；不要求trainable projector输出、预测或梯度跨臂相同。

A完整后即使质量PASS/FAIL也必须完成G，不能根据A成绩改变G。保存1000/2000完整Trainer；1000仅恢复、不screen。生成每臂独立完整性与新paired receipt。G0/G1通过自动开训，无需另一次例行GO。

## 6. 固定质量评估：只新增256格

C2000作为起点的旧128 cells通过G0绑定后复用。新A2000/G2000 × 原64 × 实际B0/B1 =256格；无新seed、无1000点评估、无旧模型重采样。

Corelab执行，逐文件验证最小推理视图和原processor/statistics。A128可以与G排队/训练并行；结果不改变G配方。每cell first-valid，合法REJECT或高误差不重试。报告/上传故障仅补报告/传输。

独立CPU从保存的完整数组复算模型/条件/noise身份、finite、XYZ/SO3、全部gate issues。不得仅复读summary。所有数值门与metric函数不变：all64/B0、all64/B1、old16/B0、old16/B1四组分别全有效、零REJECT、median XYZ≤.020m、NumPy linear p95≤.035m、median geodesic SO3≤.050rad。

新90%margin只作训练目标与诊断，不增设为最终资格硬门。报告选定primitive的max/margin分布、各issue及位置姿态代价；不隐藏变成另一类REJECT的情况。old16是重叠子集，noise不是独立训练重复。

两臂完整后：仅一臂全四组过则冻结它；都过预先优先简单A；都有效FAIL则 `PI0252_SINGLE_MARGIN_HYPOTHESIS_EXHAUSTED`，停止，不换primitive/lambda/seed或追加步数。invalid/incomplete/配对失效不称科学FAIL。

## 7. 开发候选冻结，不冒称正式selected

通过后写 `PI0252_M64_DEVELOPMENT_CANDIDATE_V1`，绑定唯一A或G final2000、完整祖先C与历史偏差、实际loss/primitive、weights/tree/revision、源码、S4、processor/stats、原输入/noise和四组结果。第一次sim前冻结，不试两臂后按sim挑选。

selected_checkpoint=NONE；formal_calib_selection_done=false。本候选不是通过正式FULL family selector的模型；PI0250对旧K/X的FULL条件没有成立，不将它转用于本新配方。

## 8. 重新条件放行PI0251的有限TRAIN sim pilot

只在第6节原门和独立/paired完整性通过后运行学习策略。G0归因完成或单条改善都不够；旧C/E不能直接运行。

训练期间CPU准备原PI0226 canonical5733559806/artifacts的TRAIN A–D C0场景、场景及query RNG、WBC源码/权重、StateV1/frame/TCP、持续reference入口、cadence/prefix、物理时钟、tick/query cap、termination/cleanliness和world-TCP scorer。原值唯一可恢复即可按此授权运行；冲突则SIM_BINDING_CONFLICT只阻塞sim，不从README猜值、不新建更容易benchmark。

最多4个native expert控制episode，完全同运行身份已有控制可复用。控制失败先区分工程环境与controller语义，不能修改WBC救学习模型。

固定development_checkpoint按A→B→C→D各一次one-chunk，最多4个模型query。真实RGB/StateV1→原S4 dense-relative输出→原SafetyGate→TrajectoryBatch→frozen WBC。REJECT不执行、不重采样。success按原实际achieved world-TCP距离≤.035m及原完整运行/cleanliness，不用离线RMSE或reference命中代替。

4/4通过才执行A–D各一次progress-preserving receding，每任务最多64query并遵守原更严格cap，总≤256。每次使用前段实际执行后的fresh观测，reference进度/时钟连续；不reset、不重复播放旧预测、不换候选/seed、不teacher前缀/RTC/平滑/retime/DAgger。保存实际noise tensor，不仅seed；不把同步sim控制时钟描述成真机实时能力。

记录每次输入/noise/输出/gate、参考与实际TCP、first-hit/min/final距离、tracking、时钟/终止原因和video。同次记录用于区分生成不到目标与控制未跟上，不增加模型调用或额外优化。

有效任务FAIL停止 `PI0252_TRAIN_SIM_FAIL — STOP_FOR_PI_POLICY_OR_CONTROLLER_REVIEW`，不换另一臂再试。全部通过才写 `PI0252_TRAIN_SIM_READY_FOR_LARGE_SCALE — STOP_FOR_PI_FULL_RECIPE_FREEZE`，交付large_scale_init_checkpoint及完整训练/恢复/仿真证据。该初始化候选不保证扩量或泛化成功；正式selected、FULL/CALIB与DEV/F仍未执行。

## 9. 预算、资源与恢复

- G0/G1：新项目模型query/训练更新0；CPU数组分析和数学fixture不算项目模型反传。
- 若条件准入：A/G新canonical更新各2000，共4000；final-only新screen256；非选分完整S4 parity最多4，优先复用同语义证据，无额外optimizer smoke。
- 条件sim：模型最多260query、native expert最多4episode。前级失败不消耗后级。另记native/head/rollout/recompute/actual attempts，不以期望数字冒充实测。
- 工程小修、同完整状态恢复沿PI0250自主，不设任意次数上限；实际失败/重算计raw。不得延长canonical终点、重跑有效科学FAIL或用新optimizer冒充stage内恢复。报告/归档失败不重训。
- PC控制/代码，Corelab CPU/screen/sim；确需可微大模型训练才使用已验证Hopper大显存路径及合法PBS auto，不pin hostname、不重复4080S已知训练OOM、不在H200调试普通路径。
- 保持scratch当前Projects布局与兼容链接。空间够用就不清理；quota UNKNOWN如实标注，保存前按实际高水位复查。保护C/E完整durable来源、原U里程碑、当前resume与唯一证据；HF元数据或历史PASS不代替当前可恢复源。
- 邮件/monitor继续关闭；不新增清理或权限范围，不在公开日志写token。PR #77不merge。

## 10. 交付与启动正文

第一成果必须是小型REJECT根因表：两臂source/issue/metric/threshold/segment、同source B1、专家同gate与margin可行性；不能再用“只差一条”代替原因。其后才是被条件准入的唯一primitive、两张sealed cards、新配对训练及最终256格、唯一开发候选和条件sim视频。

执行 `PI0252_REJECT_CAUSE_TO_TRAIN_SIM_GO`。完整读取本文件、5861040750和原PI0251协议。先G0零模型定位C/E原REJECT；不假定同source或线速度，不重跑旧256，不改原FAIL。

只有C是第3节单一白名单违规、相关专家可行且身份绑定通过，才冻结唯一primitive并自动A/G各2000；不匹配就交付具体原因停PI，不用猜测开训。两臂同C2000权重、同64/2000表、新optimizer/LR1e-5。A保留native+.4pos+.1delta+.1rot；G仅加第4节固定.1 margin项。90%margin只用于训练，原gate/四组质量门不变。不要一次叠四项或根据结果改primitive。

新RNG key与真实pre-backward配对沿第5节执行，工程与收口复用成熟链。1000只恢复；仅评A/G final2000原256格，原C128复用。两臂都完成，均过优先A；都FAIL不追加。

通过冻结唯一development_checkpoint，按第8节原TRAIN WBC one-chunk4/4后receding，失败不换模型/seed。成功交付large_scale_init_checkpoint并停FULL配方审查。selected_checkpoint=NONE；不FULL/CALIB/DEV/F/DAgger/真机/merge。邮件关闭，不新建monitor，不为本Goal再大清理。
