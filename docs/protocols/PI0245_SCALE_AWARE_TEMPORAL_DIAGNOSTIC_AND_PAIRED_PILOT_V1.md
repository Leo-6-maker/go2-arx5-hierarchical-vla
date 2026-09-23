# PI0245 — 逐时间尺度感知的物理时序诊断与双臂微调

日期：2026-09-24（Asia/Singapore）  
Decision：`GO_SCALE_AWARE_NOISE_DIAGNOSTIC_THEN_GATED_PAIRED_PILOT`  
Authorization：`PI0245_PHYSICAL_TEMPORAL_PILOT_GO`  
发布状态：`AUTHORIZED_NOT_EXECUTED`  
代码导航基线：`a8c226587a88abfb6fea2467a8eb032100116b14`  
Canonical：Issue #11；PR #77只作pointer，不merge。

## 0. 本次决定与证据边界

用户报告的最新只读结果修正了此前讨论中“归一化动作误差差分等价于物理增量误差”的不完整表述：逐时间归一化尺度不同，该等价关系不成立。本协议明确以**原decoder还原后的物理位置残差差分**为新训练项；不修改statistics、decoder、H40或SafetyGate。

授权一次有限campaign：

1. P0：封存已有32条R0/R2只读时序分析；核验有效逐时间尺度、目标和新损失实现，零模型forward。
2. P1：原bridge4 × 两个旧最终权重 × 四个预定噪声条件，最多32个stock查询。
3. P0/P1完整且没有使试验无效的真实契约冲突时，直接运行P2：同一R0-final10000起点的native对照与physical-temporal实验，各2000个optimizer updates。
4. 固定输入/噪声进行P3评估、配对分析并交PI；不自动扩到M64/FULL/选模/WBC。

这是一份**新的科学授权**，不是把已耗尽的PI0244分支重新命名或延长。用户前一轮收到的32-query/2k建议此前未授权；本文件现在给出具体预算、单位、公式、输入和停止规则。准备通过后不需再次询问是否可以启动已列训练。

### 0.1 已读取的来源

- PI0244 R2终态：Issue #11 comment5798767358，PR pointer5798769580；原结果和原停止保持。
- 当前配对报告：`reports/PI0244_R2_SCREEN_CPU_VALIDATION_20260924.json`，SHA256 `d73ed22d90567ab5c4aaef0052fede9b5e1f9ac1a045b9976f237b4d348a8580`；CPU validator与Stage21/PI0243 screen路径沿原记录复用。
- NVIDIA/Isaac-GR00T pinned commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`：`gr00t/data/utils.py`、`gr00t/data/state_action/state_action_processor.py`、`gr00t/data/state_action/action_chunking.py`、processor与native flow源码。它们支持逐时间(T,D)统计参数，unapply先反归一化再还原相对动作。min/max逆变换代码含clip；mean/std逆变换另有zero-std规则。实际生效分支以本项目processor的norm_params/config为准，不从文件名猜测。
- PyTorch reproducibility官方说明仅用于界定跨环境seed证据，不能替代实际噪声和条件张量。

### 0.2 用户最新结果的来源等级

用户提供本地脚本：`D:/Users/GR00T project/.pi0244_r2_screen_verify_20260924/scripts/pi0244_temporal_npz_readonly.py`。本次读取GitHub的当前execution branch时，该路径的仓库文件返回404，最新ledger检索也没有返回这一新分析的canonical条目。GPT未读取该本地脚本或远端新报告，不能声称已独立验证其数字。

本协议将如下内容记录为**用户报告的已完成只读发现，P0先物化证据**：R2物理峰值16/16在37→38，中位跳变13.33cm，同段专家位移近零；normalized raw XYZ最大跳变通常更早（第2段），两组32条raw XYZ未触发裁剪；逐时间反归一化尺度放大尾段残差，复算与sealed decode吻合；35–39五点占12.5%点数，贡献R2约31.5%位置平方误差；诊断性去掉这五点后median XYZ RMSE仍约42.7mm。

不要把“原32条未触发clip”写成“decoder没有clip操作”，也不要把尺度放大观察写成“已经证明唯一根因”。31.5%的分母及聚合方式、raw第2段的编号、没有裁剪的具体范围，应由原脚本/报告明确，不能自行补定义。

GPT本次只读了GitHub/上游源码及官方说明，另在本地运行4项无项目数据的NumPy数学恒等式/反例检查，全部通过。没有SSH、Hopper/Corelab访问、项目原始NPZ重算、模型load/forward/训练或调度操作。数学fixture不是项目runtime测试。本次GitHub发布也没有执行上述campaign。

## 1. 保留的科学状态与假设

R0最终：5REJECT/11WARN、median XYZ47.100mm、p9554.982mm、median SO3 .03745rad。R2最终：15REJECT/1WARN、median XYZ48.746mm、p9559.655mm、median SO3 .03623rad；旧独立验证错误数0。selected_checkpoint=NONE。

现有证据不支持继续XYZ权重扫描；FP32参数/Adam状态使旧P32分支不适用。参数更新、数据身份、native/helper parity和F(F(h))修复的已闭合证据沿hash复用，不重新全库审计。

本轮检验的假设是：**同一normalized误差在不同horizon位置会对应不同物理误差，因此物理增量残差的监督可能比只改XYZ维度权重更针对速度门。** 噪声残留、位置表示与尺度可以共同作用，P1不要求硬选唯一根因。

禁止因尾段集中就裁掉35–39点、单独降低37段权重或改dt；不把未触发clip的旧输出外推为未来局部重构也不会饱和；不修改TRAIN stats为统一尺度来制造更平滑输出。时间常数归一化可以成为未来另一项研究，但不在本轮。

## 2. 数学修正、精确目标与单位

### 2.1 两条时间轴和变量

- k=0..39：机器人trajectory时间索引；物理采样间隔DeltaT=0.05秒。
- tau∈[0,1)：flow-matching去噪时间；每个chunk沿用native sampled tau，不能把它当DeltaT。
- a*[b,k,d]：同一次真实训练collator的normalized目标动作。
- epsilon、x_tau、v*=a*−epsilon：同一次native forward采样和构造的张量。
- v_hat：该forward的有符号predicted velocity，不能从已平方的action_loss还原。
- x_tau=(1−tau)epsilon+tau a*。
- a_hat=x_tau+(1−tau)v_hat：局部干净动作估计，仅用于训练loss/teacher局部诊断，不是自由生成policy output。

因此a_hat−a*=(1−tau)(v_hat−v*)。禁止使用x_tau+v_hat或直接对v_hat相邻差分作“轨迹平滑”。

### 2.2 逐时间尺度为什么必须进入差分之前

在当前XYZ逆变换的未裁剪仿射支路，可表示为：

p[b,k]=A[b,k] a_xyz[b,k]+b[b,k]。

A包含实际有效scale及确有的state/reference旋转；b包含offset与anchor。不得擅自加world变换。若本项目该XYZ分支只是逐维缩放和相加，A就是对角scale，保留其真实实现。

令normalized残差r_k=a_hat_k−a*_k；则物理位置残差E_k=A_k r_k。比较预测增量与专家增量时，同一个k的offset会抵消：

DeltaE_k = A_{k+1} r_{k+1} − A_k r_k
         = A_{k+1}(r_{k+1}−r_k)+(A_{k+1}−A_k)r_k。

旧normalized差分遗漏第二项，且第一项也有实际单位尺度。只有A随k不变时，两者才是固定线性映射关系；仍不应称数值/单位相等。

反例：相邻normalized XYZ残差都为0.1，normalized差分为0；若一轴scale从0.1m变0.5m，物理残差从1cm变5cm，相邻物理速度误差为0.04/0.05=0.8m/s。这里不需要任何decoder计算错误。

### 2.3 采用实际decoder的物理残差，不假定所有新预测都处于仿射区

定义D_xyz(a;raw_state,stats)为本项目**原完整XYZ decode语义**在训练loss中的可微镜像，包括实际relative分支、逐时间参数和原已存在clip/zero-range规则。它只用于新增loss，部署decoder不改。

E[b,k] = D_xyz(a_hat)[b,k] − D_xyz(a*)[b,k]。

D_xyz(a*)必须和原物理专家XYZ在既有数值容限内吻合；有意的原target clipping若造成超容限差异，记录为实际target/decoder不一致，P2不放行，不能临时以clipped target取代expert评价。

min/max实际逆变换的scale=(max−min)/2，offset=(max+min)/2，params可能是(40,D)，并且源码执行clip[-1,1]。mean/std以实际std和mean处理，std=0时沿源码特殊规则。必须从有效norm_params提取，不重新计算TRAIN统计、不只读absolute stats而漏relative override。

可微镜像若需要torch实现，只新增XYZ薄函数，并对官方NumPy路径做CPU parity。原min/maxclip保留为真实torch.clamp，记录饱和率及其局部梯度；不使用straight-through或自动取消clip。旧32条不clip不能成为新a_hat不clip的假定。native loss继续约束raw flow residual，不让饱和后的辅助loss被单独解释为成功。

若P0发现XYZ路径不是上述可复现的逐步decode，或必须改表示/frame才能实现，则停止loss准入，列出实际操作；不要另造近似decoder并冒称完全等价。

### 2.4 本轮唯一新增训练项

令m[b,k]=1仅在k与k+1的三个XYZ槽均是原mask有效点，k=0..38。有效样本为原完整H40，预期每窗39条有效transition；不排除37段或尾部。

v_err[b,k]=(E[b,k+1]−E[b,k])/DeltaT。

固定v_ref=1.0 m/s作为单位参照，不是新gate阈值：

L_phys_delta = sum_{b,k} m[b,k] * ||v_err[b,k]/v_ref||_2^2
               / sum_{b,k} m[b,k]。

三维使用L2平方求和，不再除3。分母为有效transition数量，不是40点/132槽；分母为0或mask语义冲突应报错。此loss无量纲，lambda=0.1是**针对本定义新预注册的数值**，不是把旧normalized代理项的lambda直接搬来。没有lambda搜索、gradient-balancing自适应调参或新单位尺度校准。

实验loss：L_T = L_native + 0.1*L_phys_delta。
对照loss：L_C = L_native。

原native loss是所有有效10维的原生flow regression，不沿用XYZ4。保留它的reduction和全部rotation/gripper学习。新的物理项只监督XYZ增量残差，不额外加入position重加权、hinge gate、acceleration、末点特殊权重或新旋转loss。

时序项对常量物理位置偏差不敏感，所以不能替代位置目标。42.7mm的非尾部误差仍要靠原native及新的训练实测来判断；本方案不保证一次修复两类问题。监督专家增量，不惩罚预测运动本身；禁止把静止轨迹当最优解。

## 3. P0 — 证据物化与窄实现准入（无模型forward）

### 3.1 先封存用户已做的工作，不强制重跑

在execution worktree读取用户实际本地脚本和输出报告，将脚本、JSON/CSV报告、原R0/R2输入manifest/hash、command、runtime版本和计数提交GitHub，Issue #11记一条canonical。存在的正确结果先复用；仅补缺失的字段，不重复完成整套旧审计。

补齐原统计的定义：35..39是零起始五点；31.5%是pooled SSE比例还是逐窗比例聚合；“第2段”对应精确zero-based index；0裁剪是仅raw XYZ还是其他维度。保留实际值，不把摘要数值强塞进脚本输出。

从原processor有效参数导出norm_scale_offset.json，记录(40,3) scale/offset、mask、relative reference分支和每个源文件hash；若跨样本实际存在state旋转，用每样本A/b或原操作chain表达。用现有32条raw重算physical XYZ、相邻位移，核对用户报告。已保存的完整实测证明可直接引用。

当前decoder数学吻合时，归类为表示尺度对物理残差的放大，不定性为decoder bug。无需再测全9792视频。可以读必要TRAIN metadata/statistics，禁止重拟合stats或做新CALIB/DEV读取。

### 3.2 CPU和synthetic autograd准入

新增薄loss及observer，复用成熟trainer/loader/screen和corrected-once特征路径。建议测试覆盖：

- 时变A下constant normalized residual仍产生physical increment residual；constant physical bias的temporal loss为0；不能遗漏DeltaA*r项。
- scale常量时退化到正确的固定映射；offset在残差中消去；非单位参考姿态不改变定义。
- clean oracle v_hat=v*时E和L_phys_delta为0；对运动专家亦成立，静止输出不能规避native loss。
- differentiable XYZ decode与原decoder输出一致：真实32条、合成非零anchor、time-varying scale、clip边内/边外、zero-range/zero-std。沿既有XYZ无损容限1e-5m，不在看结果后放宽；记录误差分布，clip kink处不强行要求唯一数值导数。
- 原有效mask与transition mask语义；padding/NaN/错shape应fail-closed。
- lambda=0时输出loss及参数梯度与未修改native在同输入下相同；新项有正确梯度，不detach，不从平方loss取sqrt猜符号。
- 使用同一次native forward的epsilon/tau/x_tau/v_hat；observer不修改输出、原noise/time调用次数不变、features只处理一次；不得第二次forward补loss。
- 数值小单元用float64作有限差分/gradcheck，实际新辅助项的残差与reduction显式用float32、autocast禁用局部范围；原模型/BF16autocast及native loss数值路径不变。两臂均执行同一observer/诊断计算，C的优化目标严格为native，不让额外随机调用改变训练流。

这些是synthetic/CPU验证，不声称已做真实model backward。不要创建额外训练smoke权重：真实第一更新计入该臂2000步。

P0没有真实语义矛盾且上述代码证据齐全，封存run_card/resolved_diff/data/decoder binding，然后进入P1。不能把“源文件尚未上传GitHub”解释为模型失败；用本地现有证据闭合即可。

## 4. P1 — 固定32-query的噪声与物理尺度交互定位

### 4.1 科学集合

原PI0242/R1 bridge4的四个source IDs，不重新挑选；模型为R0-final10000与R2-final10000。输入使用这四窗已有封存RGB/state、当前批准的label-free C_T eval处理；保存实际条件张量，不再追历史P1精确重放hash。

在独立CPU准备进程生成一次完整noise bank，固定torch.Generator('cpu')，每份shape[1,40,132] float32：
- N0：seed22301。
- N1：seed22302。
- N2：seed22303。
- N0_roll11：对N0沿horizon轴循环正移11点，全部132维一起移位。

按正式model的原noise dtype一次cast，并同时保存原float32和实际注入dtype的无损bytes/hash。此噪声不是历史screen未保存噪声的恢复。同一bank用于所有四窗和两个模型；N0_roll11只是相关的置换反事实，不计作第四个独立seed，也不要求模型输出按11点严格平移。

4窗×2模型×4条件=32个唯一stock查询。顺序按model R0→R2、source A→D、noise N0→N1→N2→roll11冻结。只在原sampler initial-noise入口注入，后续t/step-count/dtype/每步更新、inactive反馈及decoder保持；不roll state/action labels/统计尺度/位置编码。

第一既定cell可验证hook/内存，结果直接封存。工程失败没有有效输出才恢复相同cell；不得换噪声到通过。不额外扫采样步数、噪声强度或尾段截断。

### 4.2 输出与解释

每query保存condition、初始完整noise、每个原生去噪step的x与predicted velocity、tau/整数t、最终raw/decoded/expert/mask、gate。中间去噪态不是机器人执行轨迹，其physical decode只作标明的可视诊断，不拿中间gate当policy gate。

同时报告normalized residual、E_k物理残差、DeltaE/DeltaT、各轴scale、raw最大跳变位置和physical最大跳变位置、tail SSE占比、clip计数。必须在未改统计下观察噪声效应；不重新归一化raw后声称物理变平滑。

解释有三种可并存结果：峰值随noise显著变化；在原scale/位置绑定下持续集中尾部；混合且四窗证据不足。不能只凭roll后峰值不动就证明“全是scale”，或峰值动了就证明decoder无贡献。所有噪声结果都报告，不择优替代R0/R2旧FAIL。

### 4.3 P2自动准入

P1完整有效且无以下positive evidence时可直接P2，不需等待再次GO，也不要求P1先证明唯一根因：错误样本/时间/labels、无法复现原XYZ decode、未授权processor drift、hook改变stock算法、真实nonfinite/身份冲突。

若P1仅显示“噪声作用有限”或“尺度与噪声混合”，仍按预注册计划完成双臂；这是对新目标的试验，不是把某个诊断描述作为预设结论。P1工程问题修复后恢复，不用无限新矩阵取代训练。

## 5. P2 — 两臂各2000步的条件授权微调

### 5.1 控制与实验

| 字段 | C：native continuation | T：physical temporal |
|---|---|---|
| Family | PI0245_C_NATIVE_M16_2K_V1 | PI0245_T_PHYSDELTA_M16_2K_V1 |
| 起点 | 同一R0-final10000完整权重 | 同左 |
| Loss | 原L_native | 原L_native+0.1 L_phys_delta |
| 更新数 | 2000 | 2000 |
| 数据 | 原16TRAIN、相同新2000项schedule | 同左 |
| 优化 | 新AdamW，LR1e-5，WD1e-5，maxgrad1；cosine2000，warmup10 | 同左 |
| Batch/accum/workers | 1/1/0 | 同左 |
| seed | 22301，重新建立相同初始训练随机状态 | 同左 |
| 保存 | checkpoint1000、checkpoint2000完整Trainer state | 同左 |

原beta/epsilon/parameter groups、trainable范围、冻结模块、FP32可训练参数/Adam状态、BF16autocast、dropout/augmentation、图像规格、noise/time分布、H40表示和统计保持R0实际resolved配置。只有表中low-LR 2k新调度相对旧R0变更，且两臂相同；本轮比较可归因的因素仅为额外物理时序目标。R2不是任何臂的initializer，不叠加XYZ4。

R0-final作为共同warm start是本次预先选择的起点，不能说它已通过或是正式policy；两臂均新建optimizer/scheduler而不装R0旧optimizer。arm内部工程恢复才用本arm完整Trainer state exact resume，保留sample cursor/RNG/moments。不能借恢复换seed或从表现好的中期重开。

### 5.2 配对训练随机性

2000条曝光表从原16 ID和既有确定性sampler规则产生，每窗125次；能使用原R0 schedule前2000项且满足该计数时直接复用，不能换数据挑更容易的窗。两臂在任何模型运行前封存同一表hash。

两臂同设备类别、driver/runtime/source、初始model bytes及RNG起点。新增loss与observer不得调用额外随机函数；记录每次实际source_id、native tau及noise hash、输入augmentation hash/可保存的trace，不增加noise来对齐。如果实际随机消费不同，保留证据、降低配对解释强度；若因实现错误导致多次采样/错数据，技术修复并独立处理受影响轨迹，不能按成绩重跑。

按C→T串行执行。C正常有限的成绩差不阻止T，因为control failure是实验的一部分；若C暴露实际契约/数值异常，使两臂设计无效，则停P2。不要同时跑多份C/T碰运气。

### 5.3 实际计算图和日志

允许薄、独立的native-forward输出观察/overlay，暴露同一次调用的有符号pred、velocity target、x_tau和tau，或用已验证等价的scope hook；旧源图freeze，新增项在输出loss处相加。保存science diff；lambda0 parity先通过。不能再沿仅有action_loss²的XYZ4 wrapper推断残差。

第一正式更新审计实际总loss、native loss、L_phys_delta、有效39transition、scale hash、tau、clip/saturation计数、参数/梯度/moments dtype及一次optimizer更新。后续记录预注册milestones1/10/100/1000/2000的详细字段，常规loss日志按现有频率；不为了分量梯度norm新增整套真实backward诊断。真正计算出的nonfinite是停止证据，不能把模型输出重采样到finite。

控制臂和实验臂均保存native与physical指标用于对照，不能比较不同总loss的大小就宣布哪臂更好。不得在看过初期数据后改变lambda、v_ref、warmup、LR、增添位置loss或scale归一化。

### 5.4 预算

总新增canonical optimizer updates最多4000；两臂分别只判final2000，不挑checkpoint1000，不追加步数。工程恢复无任意次数上限，但不得覆盖first-valid科学结果或以报告/HF失败重训。

## 6. P3 — 固定评估、噪声复核和最终判定

### 6.1 主screen：80个唯一逻辑cells

共享warm-start权重step0一次 ×16；C1000/C2000/T1000/T2000各×16，共80。

主screen保留原seed22301/原stock配置和原16观测。共同step0使用原stock entry，**在初始噪声产生处立即保存实际noise和完整实际条件张量**，不从P1的CPU noise bank冒称历史主seed噪声。随后四个checkpoint使用逐source同一已保存condition/noise进行paired screen；不改sampler，checkpoint差异不消耗不同input构造随机流。它是新的paired评估身份，不覆盖PI0244历史screen。

共享step0可复用只有在所需的实际noise/condition确实已保存且完整绑定时成立；仅旧seed/raw hash不够。无法精确复用就执行该预算中的新step0，不追旧hash重试。

每窗原gate+XYZ/SO3/gripper/首尾/velocity、scale-aware residual、tail SSE全部保留。mode、raw/state/frame对应不变。指标由现有CPU validator复算，decision枚举/finite/shape/hash明确。

### 6.2 最终跨噪声诊断：32个cells

C-final2000/T-final2000 ×同bridge4×P1同四份noise=32；与P1完全相同packet/bank。它用于检测是否只修好了主seed、是否把尖峰移动到其他位置。三份独立noise和roll11分开汇总，不把roll11作为独立随机样本增加显著性。

全部新增stock/采样查询上限：P1 32 +主screen80 +终局诊断32 =144个唯一逻辑cells。若准确同身份重复可复用并记录，不强求花满预算；内部head/denoising/backbone forwards分开计数。没有额外32/48/64的direct-flow重放，本轮不重开旧未完成侧报。

### 6.3 主验收与解释

各臂只以final2000在主screen的16窗，按原四项联合门计算：
- SafetyGate REJECT=0，WARN保留；
- median逐窗XYZ RMSE≤0.020m；
- p95逐窗XYZ RMSE≤0.035m；
- median逐窗SO3 RMSE≤0.050rad。

逐窗XYZ为sqrt(mean40(sumXYZ error²))，SO3为sqrt(mean40(geodesic²))，quantile linear。不能替换为尾部剔除指标、teacher估计、loss、reference target distance或只报15个最好的窗。

记录两臂各自四项PASS/FAIL及16窗配对效果，不因一臂通过就隐藏另一臂，不选择中期。最终noise复核全报告但不混入主16门；即便主16通过，noise复核差也必须明确，不能称稳健或已泛化。

解释分类：
- 两臂都改善且差异小：low-LR/额外微调的收益，不证实辅助项必要。
- T相对C降低physical increment error/违规且位置姿态保持：所测配置支持新目标的作用，仍需更大支持和噪声验证。
- T降低速度但位置/姿态更差，或产生过度clip：权衡/退化，不晋级为成功。
- 两臂均未通过：保留剩余偏差/残差/尾部与noise证据，不能继续自动加2000步。
- P1/noise复核结果不足：写清支持范围，单seed/四窗不能证明普遍因果或泛化。

终态统一：`PI0245_PAIRED_PILOT_COMPLETE — STOP_FOR_PI_SCALE_OR_RECIPE_DECISION`，含各臂质量门与诊断结论；真正契约/外部blocker另标类型。此处不设置自动“选最佳模型后跑sim”。本协议只批准诊断和pilot；M64/FULL/selector/WBC/sim/DEV-E/F/DAgger/硬件和PR merge全部保持锁定。selected_checkpoint=NONE。

## 7. 资源、存储、恢复与交付

本地PC负责git/CPU tests/SSH；Corelab负责P0、真实Linux集成、能装下的P1/screen；训练容量证据成立再用H100/H200，复用既有pinned运行环境和已证明的内存要求，不重试已知OOM。Hopper login只做允许的轻量准备/传输/调度，不在H200反复pytest或排CPU探测。

已有项目数据/initializer/checkpoint/归档不删除。各臂输出及HF prefix独立：建议`pi0245/c-native-m16-2k-v1`和`pi0245/t-physdelta-m16-2k-v1`；实际repo由现有合法publisher绑定。按同时在用的checkpoint、每臂两份full Trainer state、原子写入/归档/screen峰值计容量；调用已有逐checkpoint持久化与fixed-revision验证。当前reader/恢复点不可回收，归档失败只修上传。

同一阶段唯一owner，两臂串行；后续CPU准备可并行，不能提前运行未准入的训练。阶段测试/配置status从实际检查推导，不写死PASS，不依赖all(empty)。修复不重新构建第五套框架，不扫描全部9792视频或重跑历史80cells。

最小交付：
1. 用户temporal只读脚本/报告与来源、normalization/decoder尺度binding、CPU数学/gradient测试；
2. 一份PI0245 run card/experiment matrix，两个arms的resolved_diff与2000schedule；
3. 实际conditions/noise bank、32-query P1、各臂训练/消费/update/loss与完整checkpoint receipts；
4. 主80screen和终局32noise复核、独立CPU验证、各臂门与paired closure。

未运行阶段写NOT_RUN及精确原因，不能把发布计划当执行完成。新证据Issue #11先canonical，PR #77只pointer；不新建定时监控。工程修复次数不限，科学范围与以上预算不可自行扩大。

## 8. PC agent启动正文

执行PI0245_PHYSICAL_TEMPORAL_PILOT_GO。完整读取本文件immutable commit；旧R0/R2 FAIL和selected_checkpoint=NONE保留。此文正式授权P0、32-query P1以及通过窄准入后的两臂各2000步，不必再次询问已列训练授权。

先从用户本地路径读取并封存pi0244_temporal_npz_readonly.py及现有输出，把报告范围和31.5%/42.7mm分母说清；复用旧32NPZ结果，校验实际norm_params逐时间scale/offset和原XYZdecode。GPT没有替你读取该本地文件，不要以协议正文当原始证据。

数学修正必须实现：先D_xyz还原物理位置残差E，再算(E[k+1]-E[k])/.05；不使用Delta(normalized residual)冒充物理增量。原minmaxclip/meanstd特殊分支、state anchor和frame照原decoder，TRAIN stats不改。CPU验证可微镜像、时变scale反例、clip/zero-range、oracle零误差、mask、lambda0值/梯度一致。

P1按原bridge4、R0/R2 final两权重、N0/N1/N2/roll11共32query。完整noise用独立CPU generator seeds22301/22302/22303生成并保存，roll11仅horizon置换全132维。保存实际输入/noise及原sampler trace，不追旧noise/hash，不按表现换seed。完成且无真实目标/decoder/输入契约冲突后直接P2。

P2两臂从同R0-final10000 warm start，新optimizer/scheduler，同16/2000曝光表每窗125次、LR1e-5、cosine2000/warmup10、其余R0配置相同。C=native；T=native+0.1*L_phys_delta，v_ref=1m/s，L2平方按有效transition平均，不除3。辅助项使用同次nativeforward的x_tau、tau、有符号v_hat；不加XYZ4、不改统计/精度/采样器/阈值。C→T串行，第一正式update兼真实审计，计入预算。

每臂保存1000/2000 full checkpoints。主screen共享step0+两臂1000/2000×16=80唯一cells；主seed22301实际noise在共同step0捕获并固定。最后两臂final×bridge4×P1四noise=32，合计最多144个stock cells。只判final2000原四项门，另报噪声鲁棒性和paired差异，不挑中期、不延长训练。

本地/Corelab测试和小任务，有容量依据才上H100/H200；工程自主修复无次数上限，旧数据/权重/报告不动，first-valid结果固定。无M64/FULL/选模/WBC/sim/CALIB/DEV/F/DAgger/硬件授权，不merge，不新建monitor。完成PI0245_PAIRED_PILOT_COMPLETE后交PI_SCALE_OR_RECIPE_DECISION，或精确typed blocker。

## 参考（只用于已说明的方法事实，不代替项目冻结内容）

- 上游pinned归一化： https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/data/utils.py
- 上游pinned动作处理： https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/data/state_action/state_action_processor.py
- 原生flow： https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/model/gr00t_n1d7/gr00t_n1d7.py
- PyTorch复现： https://docs.pytorch.org/docs/stable/notes/randomness.html
