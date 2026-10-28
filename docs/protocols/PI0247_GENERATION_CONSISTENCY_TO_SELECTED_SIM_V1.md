# PI0247 — 生成状态一致性、终点监督与 selected-policy sim 长程 Goal

日期：2026-09-26（Asia/Singapore）。

Decision: `GO_BOUNDED_GENERATION_HYPOTHESES_TO_SELECTED_SIM`  
Authorization: `PI0247_GENERATION_TO_SELECTED_SIM_GO`  
Publication: `AUTHORIZED_NOT_EXECUTED_BY_GPT`  
Repository: `Leo-6-maker/go2-arx5-hierarchical-vla`  
Audited execution parent: `65a6f4bb986e6307fce2c52ea3ac50448cff6063`  
Canonical ledger: Issue #11. PR #77 pointer only; NO MERGE.

## 0. 本轮明确放行及完成定义

用户授权验证上一轮提出的机制假设，并争取取得正式selected checkpoint及仿真成功。本协议因此是**新的科学授权**，不是把已耗尽的PI0246预算重新打开，也不是纯工程补丁。

自动连续路径：

`D0旧证据/条件/生成状态只读分解 → D1冻结权重2×2生成干预 → D2固定16复核 → 必要时L/U各2K终点监督对照 → M64/5K → FULL9792/10K → 原规则完整CALIB选模 → frozen-WBC A–D C0 one-chunk → progress-preserving receding`

D2若已得到满足本轮M16准入的P权重+新runtime组合，可跳过L/U小训练，直接M64；否则执行已定义L/U对照。通过各阶段即可自动进入下一阶段，不需要每次再询问训练、选模或sim是否授权。有效科学终点失败按本协议停止；工程问题自主最小修复、回归、恢复，无任意次数上限。

**最终成功必须同时交付：**
- 新FULL family由原TRAIN-CALIB selector选出的唯一checkpoint，以及不可变selected-policy manifest；
- 该winner自己的TRAIN检查合格；
- 同一weights/processor/sampler经原SafetyGate和frozen WBC，在原A–D C0 one-chunk和receding任务上全部成功；
- 完整任务、观测、reference、world-TCP、时钟/进度证据。

成功终态：`PI0247_SELECTED_POLICY_TRAIN_SIM_PASS — STOP_FOR_PI_GENERALIZATION_REVIEW`。

取得M16合格配方、S4选出checkpoint、sim任务成功是三个不同里程碑。不得给M16权重贴上FULL正式selected标签，或把离线reference命中称为机器人成功。旧FAIL不撤销，不保证本协议必然产出PASS。

本协议仅在列明条件下解除PI0246对新诊断/训练/扩量/CALIB/选模/sim的停止限制。DEV-E/F、Final-F、DAgger、真机、生产部署、WBC权重/内部控制器、原gate/scorer阈值及PR merge继续锁定。

## 1. 已核对的来源与证据边界

必读并复用：

1. PI0246终态canonical `5832231367`、配对closeout `5832495497`；执行parent下 `docs/evidence/PI0246_R1_PAIRED_TERMINAL_CLOSEOUT_20260925.json` 及 `PI0246_P_SCREEN_AND_PT_VERDICT_20260925.json`。
2. 原PI0246章程、PI0245物理decoder数学与P1 noise bank；P1 closure SHA `77712a2598ae7f8d8a076139c4160ec480c4d9ff624b2e45be9a0a51a55cc70b`。
3. `scripts/pi0246_physical_position.py`、`pi0245_physical_delta.py`、`pi0246_paired_screen.py`、既有独立CPU validator、实际训练/归档入口。
4. pinned GR00T source `51d4c89f72fda44cbf77285c6a8114b52676b8a1` 的 `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` 和 `modules/embodiment_conditioned_mlp.py`；实际runtime源hash须绑定，不能换上游最新版。
5. `scripts/pi0228_train_calib_checkpoint_selection.py` 的选择函数及目标authority；原PI0243协议 `5d6d843235625c416d372cdbed01b185a1e6da5f` 的第5–9节，仅作本协议明确继承的扩量、选择和sim规则来源。
6. PI0226 canonical `5733559806` 的native continuous reference、one-chunk/receding和原artifact身份；原handoff `5741204004` 仅作历史导航，不把其中旧live状态当当前状态。

本次GitHub检查未返回晚于2026-09-25 12:38:09 UTC终态记录的新执行comment。GPT未SSH、未运行项目模型/测试、未检查实时scheduler、未下载权重、未启动训练或仿真。本次发布只改变文档和授权。执行agent先检查后续ledger与实际owner。

现有证据：C/T/P/PT各2000 updates，PI0246共8000；208个logical stock cells。C/T/P/PT最终均FAIL，selected_checkpoint=NONE。配对终点（median XYZ/p95，单位mm；median SO3，rad）：

| 模型 | REJECT | XYZ median/p95 | SO3 median |
|---|---:|---:|---:|
| S0 | 6 | 47.2025 / 54.9386 | .037188 |
| C2000 | 9 | 44.1649 / 52.8444 | .031881 |
| T2000 | 1 | 62.3276 / 81.1081 | .036843 |
| P2000 | 1 | 40.5363 / 52.0768 | .033320 |
| PT2000 | 1 | 40.5403 / 51.8974 | .040726 |

P/PT表明联合位置与时序辅助项未带来足够最终质量；不证明所有物理loss无效，更不证明架构不可能拟合。原生loss按有效mask求均值，不把122维描述为loss分母稀释。H40在去噪循环中联合生成，不是沿物理时间逐点自回归。

**索引更正不触发重跑：**旧P/PT verdict JSON中P2000的records/queries/CPU-report三项误抄了PT值；paired closeout里的P2000分别为 `10192e32...c5b92e28`、`96c6a338...8beb58`、`523b17d8...c41fe1ad`。D0从原已封存文件读出完整hash，追加erratum，不覆盖旧artifact、不据此重训。

## 2. 新假设、干预和不可扩大结论

### H1：未监督维度的生成状态偏离训练支持

训练损失mask仅覆盖active0:10，而采样器在完整132维中采噪声并更新，action encoder读取完整向量。由此提出：inactive10:132的自由演化可能通过encoder影响active输出。

这只是源码支持的候选机制。mask=0不证明标签填零，更不证明这些维度有实际因果影响。必须先核对编码后的padding值和实际trace。不会以“padding无用”为由直接全部清零。

### H2：当前去噪积分分辨率影响最终误差

比较原Euler步数与固定更细步数。变化的是去噪时间步长，不是物理轨迹dt=.05s。更细网格改善不等于证明纯数值截断误差；也可能来自学习向量场的时间依赖和分布误差。失败也不证明任意求解器都无效。

### H3：局部teacher状态上的辅助目标与自由生成终点存在差距

在同一起点/相同预算下，比较局部物理目标与真正无专家输入的完整生成终点目标。不是再换局部loss系数。不能在任意模型生成状态上把 `a_star-epsilon` 当成唯一oracle velocity。

### H4：输入歧义、尺度/clip与单例gate问题

仅先用现有数据/配置/trace进行窄检查。报告实际condition是否足以区分样本、原training augmentation/dropout是否启用、每点scale/饱和和剩余同一REJECT的issue。精确来源冲突或确定的输入-标签不一致应停止受影响实验；相似图像或未测容量不足不是确诊。新的augmentation、state-history、backbone解冻、gate修改不在本授权内。

## 3. D0 — 复用旧artifact的机制分解，零模型调用

输入只限原TRAIN支持和已封存P1/S0/C/T/P/PT结果、日志、模型配置/metadata及编码代码；不读取CALIB/DEV/F，不重新初始化全视频树进行无关扫描。

一次交付 `pi0247_mechanism_inventory.json`：

- 纠正索引身份后复用原结果，核对最后一个P/PT REJECT的准确source和全部issue；用同一原gate比较对应专家，不改标签或删例。
- 沿两个轴分析：物理horizon k=0..39和去噪步骤j。报告raw/decoded误差、位置bias与残差、相邻物理增量、每点scale、尾部贡献、clip；旧R2尾部统计不能充当P/PT统计。
- 从全132维trace报告active/inactive能量、随tau的变化；若padding目标为零，比较inactive状态与 `(1-tau)*epsilon_inactive` 的偏差。这是分布诊断，不是off-trajectory oracle误差。
- 逐原16窗检查encoded clean action的inactive值和mask；再用已绑定pad实现证明这些值来自固定的结构性padding，不含expert信息。允许D1的M干预的必要条件是inactive10:132均为**结构性零padding**。否则M=`NOT_APPLICABLE`，不把真实未监督动作当零；D1仍做不含M的积分对照。
- 核对实际sampler N、time buckets、RTC是否关闭、precision、active slots、原image/state处理；原方案宣称N=4，实际必须匹配才执行本次4/16矩阵。真正不同且无法由正确binding解释时停 `SAMPLER_CONFIG_CONFLICT`，不悄悄重定义实验。
- 对已封存条件的精确重复/源映射冲突作检查；只有明确缺失的必要state/标签语义才停并说明。不得以新root-cause猜测无限延长D0。
- 对旧loss日志报告tau、native/aux量级、saturation；缺项=`UNOBSERVED`，不新增模型查询补历史。

以上只补解释，既有identity/decoder/P1不用再跑。D0有效且无真实语义冲突，直接D1；不要求已找到唯一根因。

## 4. D1 — 冻结权重的2×2生成对照

### 4.1 固定矩阵

- 权重：原R0-final10000、PI0246 P-final2000。P是本轮预先选择的机制代表，不是正式winner；不加入T/PT以追最佳结果。
- 窗口：原P1 bridge4，A–D各一窗，原顺序；不得挑最易通过或最后那个失败窗替代。
- 噪声：原P1已保存N0、N1完整 `[1,40,132]` 张量。实际dtype转换前/后的tensor均存hash。
- 条件：从对应原S0可读取的condition packet取得；如需跨根复制只做内容核验，不重新构造条件冒充精确重放。
- runtime：以下四项；每个cell同checkpoint/condition/noise/device。

| ID | 去噪步数 | inactive状态 |
|---|---:|---|
| S4 | 4 | 原stock自由演化 |
| M4 | 4 | 下述结构性padding路径 |
| S16 | 16 | 原stock自由演化 |
| M16 | 16 | 下述结构性padding路径 |

总计 `2 weights × 4 windows × 2 noises × 4 runtimes = 64` unique scored queries。M不适用时只执行S4/S16的32；不用省下的预算增加seed。一个query内部4/16次head调用须另外计数，不能统称64次总forward。

### 4.2 M的精确定义

只对D0确认的结构性零padding维I={10,...,131}执行：在每次head调用前令

`x_j[...,I] = (1-tau_j) * epsilon[...,I]`, `tau_j=j/N`。

active0:10仍使用原head预测及原Euler更新。每次更新后inactive按下一时刻解析值重设，最后tau=1的inactive为0。初始完整噪声仍不变；不是从起点把inactive噪声置零。不得对active XYZ/rotation/gripper做裁剪、平滑、retime或teacher替换。

这个规则只用本次随机噪声、时间及静态embodiment padding契约，不读取专家未来/target。它是**新scientific sampler variant**，即使有效也必须标成model+runtime组合；旧stock FAIL保持。

N=16仍采用原left-endpoint Euler、原离散time embedding公式和原per-step模型参数，不换Heun、不加RTC、不调guidance。物理H40/最后点1.95秒不变。

### 4.3 实现准入与计数

尽量复用原stock循环，以薄sampler options或共享step函数增加两个明确开关；不另造模型。S4 adapter必须在相同权重、dtype、device、condition和实际noise下重现原调用；保留attention/mask kwargs，不借此统一训练与推理图。

本campaign最多额外8个**非选分**完整rollout parity执行，用于stock-adapter和后续可微loop。它们独立记账，不能成为第二套可挑选结果。CPU toy测试先行；同语义同数值profile原则上要求逐step exact。确有运行库数值差异时只接受预设小界：normalized raw max abs≤1e-4、decoded XYZ max abs≤1e-5m，原gate决定/issues完全一致；报告差异而不称bit-exact。超界属于实现/binding问题，不放松阈值。

所有中间x、pred_velocity、time IDs、initial noise、active/inactive norm、raw、decoded、gate、条件hash封存。不要复发F(F(h))：raw和processed-once分别有独立身份，不通过可变别名重复预处理。缓存仅限确实相同的冻结特征，且不把含noisy expert action的hidden state当条件。

### 4.4 解释与预先确定的runtime选择

报告配对factor effects：M4−S4、M16−S16、S16−S4、M16−M4，逐window/noise列出。噪声条件不是独立新窗口，不声称普遍因果或统计显著。

只在P权重的8个fixed diagnostic cells上决定后续R*，不对R0/P二者挑模型。变体相对S4满足以下全部条件才eligible：
- 无invalid输出；reject count不高于S4；
- median XYZ至少改善10%，且至少6/8 paired cells的XYZ RMSE更低；
- p95 XYZ不高于S4；median SO3≤.050rad。

10%及6/8是本轮**操作性预注册路由条件**，不是因果证明或显著性阈值。多个eligible时按median XYZ、p95 XYZ、reject count、较少head steps、优先无M、ID字典序排序选唯一R*。无eligible则R*=S4，记 `NO_SUPPORTED_RUNTIME_GAIN_IN_D1`，不追加变体。

R*在D1结束即封存；此后不得因D2/训练/CALIB/sim成绩再换runtime。所有旧对照继续保留。

## 5. D2 — 原16窗口完整确认，不凭bridge4晋级

固定P-final2000；比较S4和R*（相同则去重），原16窗口，两个实际噪声packet组：
- B0：原S0逐source保存的实际noise；
- B1：原P1 N1，同完整tensor给每个source，并显式承认其跨source共享。

最多 `2 runtimes ×16×2 =64` scored queries；R*=S4时32。只复用完全同身份结果，不把旧seed标签当tensor身份。

分别对每个16窗noise组应用原四门：REJECT0；median XYZ≤.020m；p95 XYZ≤.035m；median SO3≤.050rad，全部40点、原linear quantile。

**本次M16晋级新增一个明确的稳健性要求：两个noise组都满足四门。** 不重写旧PI0246的主门定义，不把两个组池化成32样本稀释坏结果。这仍不是独立泛化测试。

R*下P同时通过两组：冻结 `PI0247_M16_RUNTIME_QUALIFIED`，跳过第6节L/U，直接第7节M64（训练recipe使用P原native+.1局部物理位置项）。S4与R*均保留，R*身份不再改变。

R*任一组有效失败：这是runtime-only路线未达标，按本协议直接进入L/U验证H3，不重选R*、不改noise。源/decoder/ABI真正冲突或invalid算术则先按typed failure处理，不能用训练掩盖错误。

## 6. 条件L/U：局部监督 vs 完整生成终点监督

### 6.1 明确的新实验问题与共同配置

仅D2有效失败后执行。两个arm均从**同一P-final2000 weights**开始，fresh AdamW/scheduler；不是继续旧PT，不重训旧P。本轮起点选择已固定，不依据D1再挑R0或PT。

共同配置：原16窗口；原PI0246的2000项schedule，每窗125次；train/data seed22301；LR1e-5、WD1e-5、maxgrad1、batch1/accum1/workers0；cosine2000、warmup10；原trainable/frozen、precision、augmentation、TRAIN stats不变。完整保存1000/2000。L→U串行，两臂都执行完，不能L先PASS就隐藏U。

- L family：`PI0247_L_LOCAL_M16_2K_V1`
- U family：`PI0247_U_ROLLOUT_M16_2K_V1`

固定R*用于两臂最终screen；U训练中的生成loop也必须是R*。

### 6.2 相同物理目标，差别在动作估计来自哪里

`D_xyz`为既有可微原decoder镜像，原scale/offset/anchor/clip/mask不变。

`ell_pos(a)=mean_valid_points ||(D_xyz(a)-D_xyz(a_star))/(.05m)||²`

`ell_delta(a)=mean_valid_transitions ||((E[k+1]-E[k])/.05s)/(1m/s)||²`, `E=D_xyz(a)-D_xyz(a_star)`。

三XYZ平方求和，不额外除3；native loss保留全部有效10维。

L：原同一native forward构造 `a_local=x_tau+(1-tau)*v_hat`；

`loss_L=L_native + .1*ell_pos(a_local)+.1*ell_delta(a_local)`。

U：从**纯噪声**通过完整R*采样循环产生 `a_free=S_R*(epsilon, condition)`；

`loss_U=L_native + .1*ell_pos(a_free)+.1*ell_delta(a_free)`。

U的循环所有active动作由模型产生；不能以a_star、x_tau、专家首点或target初始化或修正循环。专家仅在末端监督loss里出现。M若启用，仅使用结构性padding规则。

这是“局部条件目标监督”对“部署式自生成终点监督”的对照；不先声称局部fit已经充分或采样器一定有问题。两者都包含相同系数的两个物理目标，不能临时改成XYZ4或只监督末端目标距离。

### 6.3 同步随机源、条件与梯度

每个训练step仍先执行一次原native forward。捕获其epsilon/tau和**action混入之前**的纯观测条件。U可复用该epsilon作为纯噪声起点，避免多消费全局随机流；L也保存其hash以配对。保存随机状态/真实消费身份，不仅记录seed。

U的rollout使用label-free deployment条件，action-head处于eval/dropout关闭模式但保持autograd开启；native分支仍是原train mode。必须在退出后恢复模块mode，不能`no_grad`包住U或detach每个Euler步骤。backbone/VLLN保持冻结；复用processed-once的VL与纯state特征时，保留state encoder应有梯度，不复用包含teacher action的Transformer hidden states。严格剥离`actions`字段，避免误触RTC。

**U明确新增每step一次完整的可微生成rollout**，不是继续声称“一个head forward”。记录native forwards、rollout count、每rollout的4/16 head evaluations、反向重算与实际optimizer steps分别计数。L/U此差异是授权的一部分。

无梯度模式的U kernel须在上节8次parity总预算内与同runtime参考比对；真正训练的首个更新兼验证有符号梯度、所有Euler依赖、mode恢复、loss组成和冻结范围。真实首更新计入2000，不另开hidden optimizer smoke。CPU toy模型先做finite-difference/链式梯度测试，真实模型不额外跑梯度实验。

### 6.4 显存管理的有限授权

U需保存多步计算图，推理能装下不代表能训练。本协议**新允许**仅对U的确定性Euler/head子图使用activation recomputation/checkpointing，`use_reentrant=False`、保留RNG、无in-place别名；先验证同模型forward/gradient语义，不升级Torch/Transformers。该实现选择在首次U更新前写入card，不能因成绩切换。

不允许stop-gradient截断、改N、减H40、量化optimizer、改batch或只对最后一步反传来“适配内存”。CPU静态规划后用已验证大显存平台。若完整合法图仍不能容纳，停 `ROLLOUT_GRADIENT_CAPACITY_BLOCKED`，不得以近似梯度冒充U。旧4080S的AdamW OOM不重复。

### 6.5 L/U固定评价与选择

共享初始P权重、L1000/L2000/U1000/U2000，共5profiles×原16×B0/B1，最多160逻辑cells；D2的同身份P/R*32直接复用。每组各16分别报四门，不池化。

只final2000决定；要求两noise组均通过四门。一个PASS选该臂；两个PASS按B0的p95、B1的p95、B0 medianXYZ、较简单L的固定顺序选配方。中期不能替代终点；两臂都失败则 `PI0247_M16_HYPOTHESIS_SET_EXHAUSTED`，停止扩量，不另扫lambda/seed或延长。

比较L/U相对共同P起点、U相对L的同source/noise差值；报告线速度、SO3、所有gate issues、全H40及尾部误差、clip和日志。即便U更好，单个training seed不能证明普遍显著性。

M16通过只得到`fit_recipe_manifest`（weights+loss+R*+processor+noise证据），selected_checkpoint仍NONE。本协议随后明确条件放行扩量，不再停一次形式上的scale确认。

## 7. M64与FULL：只有一个已经过门的配方扩量

### 7.1 M64 / 5000

Family `PI0247_M64_<qualified_recipe>_V1`；来自D2的P或L/U获准的最终权重。新optimizer/scheduler。继承胜出loss和R*，**各stage LR固定1e-5**，WD1e-5/maxgrad1/cosine(stage_steps)/warmup10/batch1；这是本PI新声明的课程超参数，不沿用旧PI0243的1e-4冒称相同。

原64TRAIN支持与旧16子集，按原(target_family,phase,source_episode_id)排序；独立PCG64(22301)逐轮全排列拼接至5000，实际每窗78/79次。表在训练前封存，模型RNG不受表生成影响，不叠加DataLoader shuffle。

full checkpoints1000/2500/5000；primary B0式固定packet screens0/1000/2500/5000×64=256，final5000再做第二noise组64，共320逻辑cells。原16条件/噪声同身份则复用，不重新挑选。

final5000要求：两个64窗noise组各通过原四门，且其原16子集各通过同四门。全部first-valid；没有PASS就 `SCIENCE_FAIL_M64_SUPPORT_EXPANSION`。不得用M16 memorizer绕过M64送WBC。

### 7.2 FULL9792 / 10000

仅M64 PASS后，family `PI0247_FULL_<qualified_recipe>_V1`，从M64-final5000 weights，新optimizer/scheduler。继承上述loss/R*和LR1e-5。原9792TRAIN、原stats/split，不碰CALIB训练。

原window-addressed方案：按原episode_index/source_ID排序，用独立PCG64(22301)逐轮排列至10000，第一遍9792加第二遍208；每个window仍取原observation index0和H40。不是改source内部起点或创造新数据。每个更新保留实际sample/noise/目标身份。

保存完整2500/5000/7500/10000四份。固定64TRAIN primary screen在0/1000/2500/5000/7500/10000共384；四个正式候选step各再64个第二noise结果，共256，总640。1000只作封存model-only临时screen，不冒充full resume point。

继承旧明确失稳早停：1000及2500两次完整64primary均64/64 `LINEAR_SPEED_REJECT`，在2500暂停后停止，不事后追认。为此正式训练在2500设screen结果barrier，验证后从完整同一Trainer轨迹恢复；不是改scheduler或重建optimizer。其他普通中期未过门不自动提前终止。

没有早停则完成四候选family。CALIB选出的唯一winner必须在其对应step的64两noise组及原16子集满足质量门，才能送sim；不先用TRAIN结果删掉候选改变原selector排名。

## 8. 正式selected checkpoint：新family一次完整CALIB选择

只有FULL四checkpoint和输入/runtime family完整后解锁CALIB评价，不解锁用CALIB优化训练。之前反复使用的TRAIN-CALIB是校准/选择集，不是最终盲测泛化证明。

固定顺序：原PI0226 adapted baseline1000（只作对照）→新FULL2500→5000→7500→10000；每份原1088CALIB，共5440。同R*、processor、dtype、目标authority、gate；baseline每source实际初始noise封存，候选复用，不让runtime选择发生在CALIB之后。

复用成熟evaluation/provider/target-map，只新增新runtime/family的薄入口；`scripts/pi0228_train_calib_checkpoint_selection.py`的admissibility/ranking函数单独hash锁定，不调用已经出错的历史helper入口代替成熟runner。

原排序不变：完整且没有任何REJECT的四候选 → median reference min-target-distance → worst source-rollout-cell median → 较低step。baseline不参选。无admissible候选即 `NO_ADMISSIBLE_CHECKPOINT`，不选最不差，不移除坏行。已知专家gate不一致历史证据保留，不临时修标签来救selection。

S3完整provenance，S4唯一冻结，S5只读解释。S4通过才允许selected_checkpoint记录唯一FULL权重+step+family；selected_policy必须同时包含R*、padding定义、去噪步数、processor/TRAIN stats、完整权重与源码hash。

然后检查该winner对应step的M64两noise/16保持门。失败则保留S4冻结结果，`sim_admitted=false`并停止，不重新挑runner-up。通过则 `sim_admitted=true`。无论sim后来成功或失败，都不根据sim分数重新选checkpoint。

## 9. frozen-WBC仿真条件授权

### 9.1 先绑定旧任务语义，不重建接口

在扩量阶段就CPU准备 `pi0247_sim_binding.json`。从PI0226 canonical5733559806及其封存artifact（A/B canary、A–D receding summaries）机械恢复原A–D C0场景/seed、WBC权重、source frame/TCP、原reference lifecycle、原cadence/prefix/tick cap与clock/progress规则；引用旧artifact hash和实际命令。

不得把史上StructuredAction、world-target未成立的prefix实验或默认README值混入当前dense-relative入口。若必要的原数值从未封存或互相冲突，明确 `SIM_BINDING_CONFLICT`，前面的已授权训练可继续，sim不能凭猜测运行。这不是要求找到参数后再申请GO：原值可唯一恢复即自动准入。

新外层科学上界：4个one-chunk model查询；每个A–D receding任务最多64次model查询，总≤256。实际run cap必须沿旧canonical，且能在该上界内完成完整任务；不能偷偷截断旧episode后称同benchmark。若旧完整cap超过该上界，sim前停并报告具体冲突，不以无限rollout运行。

允许最多4个A–D原native expert控制episode核验当前sim恢复；同场景/同控制器，不训练、不写入模型观测。已有完全同运行身份的有效control可复用。专家control失败先修工程/核对环境，不拿它证明学习策略失败。

### 9.2 One-chunk → receding

S4冻结、winner TRAIN门、runtime/资源/原任务绑定通过后，自动执行A–D C0各一次one-chunk。真实RGB/StateV1→冻结model+R*→原dense-relative decoder→原SafetyGate→TrajectoryBatch→frozen WBC。REJECT不执行、不重采样；原成功半径.035m、world-TCP scorer、termination/cleanliness不变。

4/4 one-chunk全通过才进入A–D各一次progress-preserving receding。每次query必须读取前一次执行后的fresh观测，reference时间/进度连续；不重置episode、状态或reference来消除失败，不把一次预测反复播放充当闭环。

R*更慢时只使用原已支持的同步暂停仿真/模型控制时钟，不改变物理dt、过期检查、cadence，不宣称达到真机实时频率。模型/IsaacGym环境保持分离；核验共驻显存，必要的进程顺序/同机服务复用既有支持，不新造未经验证的远程实时架构。

记录achieved world-TCP min/final/first-hit、tracking、每次输入/噪声/输出/gate、native时钟、实际执行reference、termination和任务video。四个任务均成功才报最终PASS。有效任务FAIL立即停止下游，保留selected模型及失败，不换seed/调gate/换候选/DAgger。

## 10. 原四门与新的边界一览

原四门始终：零REJECT（WARN可保留）；median per-window XYZ RMSE≤.020m；linear p95≤.035m；median per-window SO3 geodesic RMSE≤.050rad；原完整40点、原mask与有限性。不得把均值位置距离、reference hit或局部loss替代RMSE。

本PI新增：M16/M64以及送sim前的TRAIN检查要分别在两份冻结噪声组过门；新增padding生成路径、4/16去噪干预和完整rollout监督，仅限上述阶段。它们是明示科学改变，不是工程同义修复。原physical ABI/gate和WBC不变；旧结果不重算成新模型结果。

## 11. 预算与阶段停止，不能无限探索

最大新optimizer预算：L2000+U2000+M645000+FULL10000=19000。D2直接通过时跳过L/U，最多15000。不同arm/阶段的optimizer均按各自定义，stage内恢复必须exact Trainer resume，不拿新optimizer冒充续跑。

离线model query上界（完全同身份可复用，以下为保守上界，不是必须执行量）：

| 阶段 | 上界 |
|---|---:|
| D1 scored | 64 |
| 全campaign额外kernel parity，不参与选分 | 8 |
| D2 | 64 |
| 条件L/U主screen，两noise | 160 |
| M64 screens | 320 |
| FULL TRAIN screens | 640 |
| CALIB五profile | 5440 |
| 合计 | 6696 |

仿真最多4 one-chunk model calls+256 receding calls，另最多4 expert-control episodes（无GR00T）。不把两份noise当不同数据集，不重复充值旧208cells。内部head evaluations、训练额外rollout及activation重算单独计数；U若获准扩量，最多17000个训练rollout、每个最多16次head eval（不含重算），不隐藏计算成本。

D1某假设不支持可继续预定义下一阶段；D2模型失败可进入L/U。L/U都有效FAIL、M64最终FAIL、FULL失稳、no-admissible、selected TRAIN FAIL、one-chunk/receding FAIL是本协议科学停止点，不自动增加新arm/步数/seed。

缺少历史漂亮日志不是新停止门。真实decoder/label/ABI冲突、错误family、非法输出或unsafe reference必须typed stop并保留证据。工程恢复没有固定尝试次数，但不能重跑有效负结果或进行未经授权的科学改动。

## 12. 资源、归档和可持续执行

- PC：开发、git、SSH控制；Corelab：CPU数学/真实输入测试、D1/D2与screen、native sim；训练优先已有验证的大显存路径。模型名不构成机器必须条件，但当前entry/device guard须与受测能力一致，PBS auto可合法分配，不硬绑定某一hostname。
- 4080S已知全训练AdamW OOM不重复；推理曾成功不等于可微多步训练能跑。U先估算activation/optimizer峰值，H100仅在实测容量及运行契约成立时用，否则H200。不能为了省显存改变已冻结科学配置。
- 普通路径/HF/pytest在CPU/Corelab成熟，正式计算节点仅复验特有路径、容量和GPU身份。沿当前成功scratch/runtime/cache/FFmpeg binding，不让已缺失的Project_Storage挂载重新成为隐含依赖。
- work目录、完整Trainer归档、inference view、selected deployment分别有scope receipt。按实际full-checkpoint大小、最新恢复点、写入临时峰值、reader、传输加20%或10GiB取大的新增余量做阶段容量；df/statfs不是quota，UNKNOWN如实记录。不要为每份报告重复写几十GiB探针，复用未过期的同fileset证据并核对新增占用。
- 完整源安全保留时，可把已逐文件验证的推理视图先送screen，不必把optimizer搬到推理GPU。完整fixed-revision归档之前不得删除唯一源；只回收本PI已归档且无reader/resume依赖的重复副本，不扩大旧数据清理权限。
- 复用已授权HF仓库/可达durable receiver，PI0247独立prefix/family；不上传凭据或整套私有runtime。网络/认证/归档/报告失败只恢复传输和只读收口，不重训。
- 一个stage一个有效owner；失联先查原job/uploader再重提。不得关闭host-key、猜地址或绕过MFA/权限。确需用户会话或管理员处理时给出精确外部条件。
- 独立execution分支；文档分支不merge进正在按hash验证的worktree。机器可读manifest自动生成表格、hash和Issue摘要；旧artifact不覆盖，新erratum关联来源。
- 不新建定时monitor；若现有执行循环有状态读取，复用一个owner，只报阶段变化/科学结果/明确blocker，不每几百MB刷一次科学账本。

## 13. 最少交付物和直接启动正文

每阶段run card在该阶段首个模型调用前生成并封存：精确输入/模型/runtime/hash、逻辑矩阵与内部计算预算、owner、存储/恢复、判定式。按本文确定性公式机械填值即可，不要求每个card再次人工批准。不可解析的真实科学冲突例外。

交付集中在：机制inventory、D1配对factor表、R*冻结、D2结果、必要的L/U训练与配对结果、单一fit_recipe_manifest、M64/FULL lineage、S3/S4/S5、selected_policy_manifest、sim_binding和A–D实际闭环报告。首批成果应是D1的有效假设对照，不再只报compile/PBS提交成功。

### PC agent启动正文

执行 `PI0247_GENERATION_TO_SELECTED_SIM_GO`，完整读取本协议及引用canonical。PI0246已有效耗尽，旧8000 updates/208cells和全部FAIL保留。本协议是新授权，允许以下固定机制干预、条件训练、扩量选模及sim，不是无限试到PASS。

先检查后续ledger/actual owner，从65a6f4b仅作阅读起点。D0复用旧trace/配置作零模型分解，纠正摘要hash误抄但不重跑旧实验。D1按R0/P×bridge4×N0/N1×S4/M4/S16/M16做64，inactive只在结构性零padding被证明后按(1-tau)epsilon闭合；不影响active输出，不借口padding修改loss分母。按预定规则冻结唯一R*。

D2用P-final、原16、B0/B1复核。R*两noise组过原四门直接M64；否则跑同P起点的L/U各2000，新optimizer、同曝光/LR1e-5。L用局部PT目标，U对无expert输入的完整R*生成终点反传，均native+.1位置+.1增量；不是再加局部loss，也不能截断Euler梯度。按预算完成paired screen，均FAIL停止新假设；任一合格锁定配方。

合格后自动M64/5K→FULL9792/10K，继承唯一loss/R*，本PI各stage LR1e-5。FULL四checkpoint完整后才读1088CALIB×五profile，原selector不改；S4唯一selected后检查winner自身TRAIN门，才执行冻结WBC的A–D one-chunk，全部通过再receding。M16不冒充selected，selected不冒充sim成功。

每阶段PASS无需再等例行PI GO；工程问题自主修复不限次数，遵守首个有效结果/实际更新计数和原源码身份。Corelab诊断/screen，训练用实际容量够的合法PBS auto资源；不再盲投4080S全训练、不在H200调试普通路径。完整归档与推理视图分开，报告或上传失败不重训。

不得改物理gate/scorer/WBC/H40/dt/target边界、删尾或换seed追分。DEV/F、DAgger、真机和PR merge继续锁定。Issue #11先canonical，PR #77只pointer。最终交付唯一selected-policy及真实TRAIN sim结果，或本协议明确科学停止点的证据；不能保证PASS，不新增未定义实验。

## 14. 外部方法参考（非项目结果）

- Lipman et al., Flow Matching for Generative Modeling, arXiv:2210.02747：区分固定条件概率路径上的向量场回归与ODE采样。它不证明本项目inactive状态或积分就是根因。
- PyTorch官方torch.utils.checkpoint文档：activation recomputation及RNG注意事项。本项目继续锁定其实际Torch版本，不按新文档升级runtime。
- 项目实际语义以第1节pinned源码、run card、实际tensor和验证结果为准；本文新增公式和路由规则是PI预注册设计，不伪称已有实测支持。
