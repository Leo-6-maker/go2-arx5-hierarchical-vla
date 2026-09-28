# PI0252-R1 — 修正 gate 绑定，峰值速度对照贯通 FULL 微调、选模和 TRAIN 闭环

日期：2026-09-28，Asia/Singapore。

Authorization: `PI0252_R1_SPEED_MARGIN_TO_FULL_FINETUNE_GO`
Decision: `GO_BINDING_ERRATUM_AG_2K_THEN_QUALIFIED_TRAIN_SIM_FULL10K_SELECTION_POSTFULL_SIM`
发布状态：`AUTHORIZED_NOT_EXECUTED_BY_GPT`。
Issue #11 是 canonical；PR #77 只追加 pointer，保持 draft/unmerged。

## 0. 本次 PI 决定：纠正文字绑定，不改变实际 gate；成功路径不再停在扩量之前

用户最新要求：更新计划，跑通大规模微调训练。本文件显式修订原 PI0252，而不是将原协议未授权的 FULL 视为自动获准。

1. 接受 PI0252 G0 的有效归因及正确停止；它是协议绑定停止，不是新的学习假设 FAIL。接受 PI0251 C/E 原有效双 FAIL，旧输出、旧报告及旧协议均不覆盖。
2. PI0252 中我写入的初始位置/旋转拒绝值 `.10 m/.70 rad` 与实际执行配置不符。本次读取的源码默认值及 G0 实例证据都是 `.75 m/1.50 rad`。这是上一版 PI 文档的绑定错误，不是执行方擅改 gate，也不能用改变实际 gate 的办法消除。
3. 只将当前离线 screen 的绑定纠正为实际配置；线速度 `2.0 m/s`、角速度 `4.0 rad/s` 不变，原四组质量门不变。完整 config/调用参数仍绑定原实例，不从默认值猜未披露 override。
4. G0 已确认 C 的唯一拒绝类型是 `LINEAR_SPEED_REJECT`，专家 margin 可行。直接复用已封存 G0，不再执行一次归因、不重跑旧 256 格、不要求下一次例行 GO。仅实现原 PI0252 已定义的峰值线速度项，执行 A/G 各 2000。
5. 新成功路径：`A/G 最终256格 -> 唯一 development_checkpoint -> 原 TRAIN one-chunk/receding -> large_scale_init_checkpoint -> 单个 FULL9792/10000 family -> CALIB 原规则选择 -> selected 自身保持检查 -> post-FULL TRAIN 闭环`。
6. 原 `STOP_FOR_PI_FULL_RECIPE_FREEZE` / `READY_FOR_LARGE_SCALE` 改为中间里程碑，不再是本 Goal 停点。下列具体 FULL 配方已经前瞻冻结；机械条件满足即继续，不只交付计划、qsub 或初始权重。

科学 FAIL、真实输入/评分/恢复状态冲突仍停止相应依赖链；没有承诺一定成功。DEV-E、Final-F、DAgger、真机、WBC/gate 修改及 PR merge 不开放。这里的 FULL 是现有单任务 TRAIN 支持集扩量，不是 9792 个独立技能或新任务泛化。

## 1. 已核对来源与证据边界

本次实际读取：
- G0 canonical：Issue #11 `5862242865`，PR pointer `5862244746`。
- G0 报告的 runtime ref `317a2e85419cacd396f4f989a4ce83f4776bb85b` 下 `src/go2arx5_vla/safety/trajectory_gate.py`、`scripts/pi0247_m64_screen.py`、`scripts/pi0247_d1_screen.py`、`scripts/pi0228_train_calib_checkpoint_selection.py`。
- PI0251 训练入口 `scripts/pi0251_ce_train.py`，ref `41487312f2f914fc79086a65449df3246fb8019a`。该文件可复用生命周期，但当前仍硬编码 C/E、2000、X initializer；不得把它原样当 A/G 或 FULL 入口。
- PI0251 协议 commit `e2b86ef2c2d3ee2fb8cab05d24de24a43cd0f0a6`，`docs/protocols/PI0251_XYZ_EXCESS_RISK_TO_TRAIN_SIM_V1.md`。
- PI0252 原协议 commit `b5530e1ba4c5e27f0c1b911245648700668ffa1c`，`docs/protocols/PI0252_REJECT_ATTRIBUTION_AND_SINGLE_MARGIN_TO_TRAIN_SIM_V1.md`。
- PI0250 FULL/存储/选模规则，commit `8501ffcf66ebec423e732abeb99131e0533bd63f`，`docs/protocols/PI0250_STORAGE_SAFE_CONTINUATION_TO_SELECTED_SIM_V1.md`。
- PI0226 TRAIN 仿真 canonical `5733559806`。该历史模型本身没有达到 35 mm；引用它的是场景、控制器、时钟与评分身份，不是冒称历史学习策略成功。

本 GPT 没有 SSH/live PBS、远端 quota、完整权重下载、G0 JSON 原字节读取、项目模型 forward/backward、训练或仿真。Windows/Corelab 路径不是本会话可读文件。以下 G0 数值与 hash 来自本次读取的 canonical；执行 agent 用实际原件绑定，不根据本文件摘要倒填收据。

### 1.1 直接复用 G0，不再猜测原因

原 source：`pi0227_persistent_train_a_j03_yp015_s04_ym050_r03_r03_227079:obs-0073`。

| 对象 | 峰值线速度 m/s | 结果 |
|---|---:|---|
| C2000 / B0 | 2.503784 | 唯一 REJECT 为线速度；第7→8点，.35→.40s |
| E2000 / B0 | 2.377422 | 同 source、同 B0，亦仅线速度 REJECT |
| C2000 / B1 | 1.889792 | WARN，无 REJECT |
| E2000 / B1 | 1.893384 | WARN，无 REJECT |
| 同 source 专家 | 1.672041 | 实际 gate 无 REJECT，满足1.8 margin |
| 全64专家对应速度 margin | 最大1.672041 | 64/64满足1.8 |

C 初始位置差 `.138151 m`、姿态差 `1.157715 rad`；E 为 `.136742 m/1.114646 rad`。实际 gate 下非拒绝，不是零跳变，也不是硬件安全结论。专家 side issues 包括 61 个初始旋转 WARN、1 个初始位置 WARN、4 个线速度 WARN，全部保留。少数已观察噪声对照支持存在条件相关的局部峰值，不能推出所有噪声下的可靠性或神经网络内部唯一机制。

G0 收据 SHA256：`e184f76f8155a0445640b675ea7b559d99ceaed29cd92a722e54ab5f85bbae5d`。
G0 审计脚本 SHA256：`ba75cb9c040ba9c06168fd8ba2290c2394d248fe2381d4834816e125aa8e8729`。
Gate 源 SHA256：`cd1d2667b5e8f67a99e08baca28ffc1abc492a34dd0ef585f3739259b359d1c4`。
Screen runner SHA256：`1f668589f3ad9d8004c35de0a2ea3550102d7074390e06e081367f5ae83de5cc`。
原件目录：`/home/liuyu-nfs/Projects/go2-arx5/PI0252/evidence/`。

初始化固定 C2000：revision `7017368c42800927a17be5561056ceff58585de7`，tree SHA256 `dbc5c5ff07ac73cd4bf448b5e679d0641a51ea4b33695246db79d0a5767a8144`。完整 path、repo、文件集由实际 durable receipt 解析，不能猜 latest/prefix。E 只作已有诊断对照，不是备用 initializer。

## 2. 绑定勘误与分层 gate：只解决已经查明的冲突

发布小型 `PI0252_R1_GATE_BINDING_ERRATUM`，包含旧协议、G0、实际 source/config/调用参数、原128格身份及下表，不改旧报告：

| 当前 screen 的拒绝项 | 本次采用的原执行值 | 处理 |
|---|---:|---|
| linear_speed_reject_mps | 2.0 | 不变 |
| angular_speed_reject_radps | 4.0 | 不变 |
| initial_position_reject_m | .75 | 纠正文档中的.10 |
| initial_rotation_reject_rad | 1.50 | 纠正文档中的.70 |

该勘误是本 PI 对已知冲突的明确解决。完整有效 config 必须从 G0 原实例/实际 runner 导出，连同全部 WARN、workspace、path、age、provenance、frame/TCP、时间参数封存并比较。没有在本文件列出的字段不等于已确认默认值或授权调整。数值/身份与原件一致即可自动放行，不再以这两个已经解决的差异返回同一个 STOP。

区分 `offline_screen_profile`、`calib_profile`、`sim_execution_profile`。离线 gate 是测试/benchmark 配置；源码明确速度/workspace 默认值不是实机标定值。不得因为某个离线 profile 允许 .75m/1.50rad，就将其复制到历史 sim 或真实控制器以放宽约束。

Sim 采用 PI0226 原 artifact 中实际执行 profile。若本来与离线 profile 不同，应显式记录两个 profile 的作用域，而不是要求跨作用域数值相同。只在同一个 profile 的实际输入/配置无法唯一绑定时报告冲突。不能凭旧 PI 的 .10/.70 字面值另造更严或更松的 sim gate。

初始姿态 WARN 在后续 screen 与 sim 中完整可见；actual achieved state、参考坐标与 frame/TCP 使用原契约。非有限/参考系/时间等真实契约错误不由本勘误豁免。旧 C/E 仍然因2.0m/s门有效 FAIL，不能直接下发 WBC。

## 3. 立即可执行：薄适配、增量测试、A/G 2K

### 3.1 唯一干预

复用原 PI0252 训练定义，但其他三个待选 primitive 从本次执行范围移除；G0 已经选定 `peak_linear_speed`，不再运行约束选择器。

`J_A = L_native + .4*ell_pos + .1*ell_delta + .1*ell_rot`

`J_G = J_A + .1*ell_speed_margin`

对同一次 label-free 完整 S4 rollout 的物理解码位置，

`z_j = sum_xyz((p_hat[j+1]-p_hat[j])**2) / (2.0*dt_j)**2`

`ell_speed_margin = relu(max_{j=0..38}(z_j) - .81)`

实际 dt 必须由原时间戳确认 .05s。覆盖39个有效 transition，取单窗 max，不按39平均、不除3、不裁点、不根据 source 或 B0/B1 条件分支。并列 max 取最低 segment index；ReLU 精确边界取0次梯度。batch1，不把跨窗口 max 混入。

margin1.8m/s和系数.1不扫描。原 pos/delta/rotation 的物理归一化、合法 rotation 镜像、active10/padded132 mask、decoder clamp/zero-range 与所有 native 损失不变；E 的位置超额项不加入。原 .1*ell_delta 是增量残差监督，新项是生成轨迹自身的峰值约束，两者不是同一个量。

完整 S4 保持可微，最大值不能经 NumPy/item/detach 计算后冒充梯度；不增加第二 rollout、不引入目标 XYZ/专家 actions 到模型特征、不做 inference smoothing/retiming/限速裁剪。

### 3.2 只做变更所需检查

继承 G0 完整性及既有 S4/processor/decoder 证据。新 CPU fixture 覆盖速度值与原 gate 一致、margin上下/并列max/边界/分母/mask、lambda0与原C值及梯度一致、非奇异点有限差分。loss 镜像应使用与原训练一致的 dtype 规则，不因为测试偷偷修改部署解码。

A/G 薄入口更新 authorization、family、C initializer、G目标和新 card；复用成熟的 lifecycle、online guard、archive、screen、CPU replay。新增 FULL 入口在 CPU 合成状态测试10000终点/2500等保存与恢复 cursor，不得沿用旧入口中 C/E、X 或2000的硬编码。不要将旧 finalizer 的兼容字段伪造为 PASS；新 schema 必须如实表达新 family。

实际入口的配对 guard 必须通过错误 source/tau/epsilon 注入，在 backward/optimizer 前拒绝。复用真实通过的未改语义 fixture，不再审计整库或反复运行无关 smoke。真实首更新在正式预算内验证；非选分 S4 parity 总预算见第10节。

### 3.3 两臂严格共同输入

两臂均从同一 C2000 weights warm start，重新创建 AdamW/cosine2000/warmup10，各2000更新；不称 C optimizer exact续训。LR1e-5、WD1e-5、betas(.9,.999)、eps1e-8、maxgrad1、batch1/accum1/workers0；原 trainable/frozen、参数dtype/autocast、四步梯度重计算不变。visual/LLM/VLLN不解冻。

原64/2000曝光表按hash复用，48窗31次、16窗32次；不额外shuffle、困难例重采样或用评估noise作训练搜索。全部窗口接受相同G项。

保留原 PI0252 尚未执行的 RNG namespace：ASCII `PI0252|22301|stage=M64_AG|optimizer_step=<1-based十进制i>`；SHA256前8bytes大端 & ((1<<63)-1)。key不含arm/主机/job/attempt/时间，隔离 native/S4/backward 的原随机边界；checkpoint重计算不能重新推进step key。

A先记录真实 sample/key/tau/epsilon/stock 消费，G对每步真实数据最迟在backward前对照；不能仅从seed重生期望值完成事后配对。模型输出和梯度允许跨臂不同。A最终表现不改变G训练。

各保存1000/2000完整 Trainer。1000只恢复，不screen。阶段内恢复须包含weights/optimizer/scheduler/RNG/样本cursor/审计前缀，同key且不重复canonical前缀；原未丢失有效产物不重跑。两臂新完整性和paired receipt都完成才晋级。

## 4. M64质量门：256格，不新增小样本搜索树

A2000/G2000 × 原64 × 原B0/B1 =256新格；旧C128复用。Corelab用当前逐文件验证的最小推理视图评估。逐cell first-valid；合法REJECT/高误差不是重试理由，上传/报告故障只补传输/报告。

每臂全64/B0、全64/B1、old16/B0、old16/B1均须：完整有效、零REJECT、median XYZ≤.020m、NumPy linear p95 XYZ≤.035m、median geodesic SO3≤.050rad。完整issues与初始连续性WARN保留。1.8margin是训练目标，不成为额外验收门。

独立CPU从保存数组重算128格/臂的身份、finite、XYZ/SO3和gate，不能只复读summary。old16是64的子集；noise不是独立训练重复。

只有一臂四组全部PASS则取该臂；均PASS预先优先简单A；均有效FAIL则 `PI0252_R1_SPEED_MARGIN_HYPOTHESIS_EXHAUSTED`，不换lambda/seed/primitive/1000点或增加步数。invalid/incomplete/配对冲突不得称科学FAIL。

冻结唯一 `development_checkpoint`，绑定arm、C祖先、weights/tree/revision、objective、S4、processor/TRAIN stats、source/noise/gate profile与全部质量证据。此时 formal selected 仍NONE，不根据sim再换另一臂。

## 5. 扩量前最小 TRAIN 闭环：合格后直接接 FULL

在A/G训练期间并行准备原 PI0226 A–D C0 的场景/seed、WBC源码和权重、StateV1/frame/TCP、持续reference入口、cadence/prefix、tick/query cap、物理时钟、termination/cleanliness及world-TCP scorer。原artifact唯一可恢复即可运行；不照README猜值，不混入已失败的旧world-target接口。

最多4个 native expert 控制episode；完全相同runtime身份已有控制可复用。它们验证当前环境/接口，不把PI0226旧学习策略结果当阳性。控制失败只做原语义的工程修复，不修改WBC救模型。

development checkpoint 按A→B→C→D各一次one-chunk，最多4模型query。真实RGB/StateV1 -> 原S4/dense-relative decode -> 原sim gate -> TrajectoryBatch -> frozen WBC。REJECT不执行，不改gate、不重采样。用实际 achieved world-TCP≤.035m及原完整时长/cleanliness判成功，不能用reference命中或离线RMSE替代。

4/4通过后，A–D各一次progress-preserving receding，每任务最多64query且遵守原更严格cap，总≤256。每次使用真实前段执行后的fresh观测，reference时钟/进度连续；不reset、不重复播放旧预测、不teacher前缀、不RTC/平滑/混合/retime/DAgger。

保存输入/实际noise/输出/gate、reference与achieved TCP、first-hit/min/final距离、tracking、时钟、终止原因和video。沿原同步仿真控制时钟，不能假称真机实时。有效模型任务FAIL停止 `PI0252_R1_PRE_FULL_TRAIN_SIM_FAIL`，不改选备用模型或追加样本。

全部通过写 `large_scale_init_checkpoint` 与 `PRE_FULL_TRAIN_SIM_PASS`，**立即进入第6节**。这不再是STOP_FOR_PI_FULL_RECIPE_FREEZE；无需用户再次批准标准扩量。

## 6. FULL9792/10000：本文件已批准一个完整微调 family

### 6.1 数据与模型

只运行一个 family：由已冻结arm唯一决定 `PI0252R1_FULL_A_V1` 或 `PI0252R1_FULL_G_V1`。从第5节合格的同一weights初始化；新AdamW/scheduler，不用E/旧X代替，不称阶段内resume。

使用既有封存9792 TRAIN窗口，沿原source-rollout split、A–D范围、H40/StateV1和相对参考表示；原1088 TRAIN-CALIB不进入optimizer、统计估计或训练选配方。恢复原dataset/split/tree/processor/TRAIN statistics seal，避免把metadata的窗口数叫独立rollout数。不得重新划分数据、访问E/F、添加人工专家或借扩大数据重建原scope。

在CPU预先按原episode_index/source_ID确定排序，以独立PCG64(22301)逐轮排列：9792完整一轮，再第二轮前208，共10000行；封存表及hash。不叠加dataloader shuffle，不强化old16或坏source。数据镜像允许路径映射，必须保持原逻辑内容及seal；数量/真实split不一致不能临时凑成9792。

仅验证增量变更与现有seal；不重新解码全部视频或重新生成专家数据。G的margin仍是固定软目标，不事后按FULL专家分布换系数、裁样或增加一个“所有9792必须小于1.8”的新准入门。若发现真实frame/dt/非有限/标签与契约冲突，应如实停该输入问题，不用目标惩罚掩盖。

### 6.2 精确优化设置

- Canonical optimizer steps：10000；batch1/accum1/workers0。
- AdamW：LR1e-5、WD1e-5、betas(.9,.999)、eps1e-8、maxgrad1。
- Cosine10000，warmup10。
- A继承J_A；G继承J_G，包括同一个.1峰值速度项和1.8margin；不在扩量时自动加减loss。
- 原trainable/frozen和参数dtype/autocast保持；S4完整四步可微，原native随机采样、physical decoder/mask、无目标特征边界不变。
- 新stage key：ASCII `PI0252R1|22301|stage=FULL|optimizer_step=<1-based十进制i>`，同SHA256前8bytes大端和63bit规则；使用独立新stage，不重复M64 keys。单臂不伪造paired receipt，记录实际forward-backward消费与可恢复状态。

正式启动前CPU验证实际入口、loss选择、scheduler终点、sample表、保存点、runtime import来源与回调都引用FULL card，不仅检查声明字符串。stage card可依据已经固定的arm、原seal和本协议机械填入，不再设置第二个人工审批。

### 6.3 checkpoints与并行screen

正式候选为2500/5000/7500/10000，四者均保存完整Trainer并永久durable保留；1000只允许model-only观察快照，不作为选模或完整resume点。禁止只跑到1000/2500便以“已启动大规模”宣称完成。

使用PI0250既有FULL评估矩阵：全64/B0在step0/1000/2500/5000/7500/10000共384逻辑格；正式四candidate各全64/B1共256，总640。step0/B0的64从合格development screen按全部身份复用，最多576新格。old16由同一结果提取，不新增query。

屏幕和归档在Corelab/传输端进行，可与后续训练并行；2500是预注册失稳检查barrier并保存完整状态。只有1000和2500的B0均出现64/64 `LINEAR_SPEED_REJECT`时触发该既定质量早停，状态 `FULL_SYSTEMATIC_SPEED_INSTABILITY`。普通中期联合FAIL按原计划记录并继续，不临时换LR、checkpoint schedule、目标或数据；数值非有限/实际训练语义错误按完整性规则停。

不得依据TRAIN排序删掉某个候选改变后续选择；早停时如实报告未完成FULL family，不能假造7500/10000或执行缺候选的selector。报告/归档失败不重训已完成canonical更新。

## 7. 完成FULL后：原CALIB规则、唯一正式selected

FULL四候选与完整性/归档证据闭合后，才用原1088 TRAIN-CALIB做评估选择。它是开发校准集，不是未见泛化盲测；模型输入不得包含用于scorer的目标坐标。

固定比较：原PI0226 adapted1000 baseline（只作对照）及本FULL2500/5000/7500/10000，每份原1088，最多5440格。S4/processor/TRAIN stats/target-authority/数据身份/gate profile按原契约封存；原seed22301与实际初始noise由基线逐source记录并供候选配对复用。只有全部身份相同的已封存baseline结果可以复用；不把其他family的候选得分当本次得分。

复用 `pi0228_train_calib_checkpoint_selection.py::select` 的实际admissibility/ranking，绑定函数hash；用薄profile适配原S4推理，不原样调用旧模型加载入口导致sampler/processor漂移。补充校验每份必须恰好1088唯一原source、finite和完整gate数组，不因原函数只比较两个count字段而允许降低expected count。

选择规则保持：完整且零REJECT的候选 -> 最低median reference min-target-distance -> 最低worst source-rollout-cell median -> 较低step。baseline不参选。WARN不偷偷改成ACCEPT，但原规则允许WARN；没有合格候选就 `NO_ADMISSIBLE_CHECKPOINT`，selected=NONE，不挑“最不差”。

S3 provenance锁定、S4唯一selection freeze、S5只读解释；CALIB不产生新的梯度更新或配方分支。

有winner后记录正式selected。再检查该winner既有全64/old16的B0/B1四组保持结果；不满足则保留真实selection记录但 `sim_admitted=false`，不改选runner-up。不允许扩量前pilot的通过代替FULL模型自己的保持结果。

## 8. post-FULL TRAIN闭环与交付终态

只让第7节唯一selected按第5节同一场景/时钟/控制器/scorer运行第二轮one-chunk A–D，然后4/4通过才receding。最多4+256模型query；最多4个native expert控制，实际同runtime且未改变的控制可复用。扩量前后模型不同，学习策略输出不能复用。

选模冻结在此轮物理评价前；失败不换checkpoint/seed、不调WBC/gate、不通过teacher前缀重试。记录模型reference与实际TCP两条证据，区分生成未到目标与控制未跟上，但不额外分支诊断消费模型预算。

必须区分：
- `FULL_TRAINING_COMPLETE`：10000真实canonical更新、四个完整durable checkpoint及恢复证据完成。这是训练工程终态，不代表质量或闭环成功。
- `FULL_SELECTION_COMPLETE`：原CALIB冻结唯一selected；没有admissible如实写NONE。
- `PI0252_R1_FULL_FINETUNE_SELECTED_TRAIN_SIM_PASS`：selected保持与post-FULL A–D闭环全部通过，才可称本次贯通成功。此后STOP_FOR_PI_GENERALIZATION_REVIEW；DEV/F与真机仍未授权。

交付一个小型可读汇总，包含真实run/job/step、dataset窗口数与split seal、loss/范围/精度、四候选固定revision/tree/完整恢复源、FULL640保持矩阵、CALIB选择、前后两轮sim的实际TCP和视频、raw/canonical资源账本。任务不以“代码已准备”或“作业在Q”完成；WAIT_GPU和外部阻塞明确报告。

## 9. Scratch、PBS和持久化：不再把整理作为主任务

保持 `/scratch/e1192548/Projects/go2-arx5` 和兼容链接，不搬第二遍、不新建同内容目录树。旧smoke删除已记账，不重复计算回收。约101.9GiB只是此前快照；实际输出fileset的当前free与高水位需核对，不能写成quota headroom。

**PI0249 X 20.88GB仍保护**：用户报告它未完成完整性封存且被后续链引用。本Goal不删除它、不假造seal、不以C已训练成功代替其可恢复性。C/E来源、原U正里程碑、唯一证据和所有当前/排队job依赖同样保护。新训练直接消费C权重；祖先记录如实继承，不要求为元数据追溯反复物化每个祖先的全部weights。

当前环境找不到qstat表示 `JOB_STATE_UNKNOWN`，不表示无作业。允许在已有授权Hopper登录/PBS初始化环境中恢复合法PATH/module或使用已验证的PBS可执行文件路径；查询本用户所有活动/排队/hold作业及提交脚本引用。状态未能确认时不重复qsub、不删除或移动checkpoint；CPU适配、fixture、传输规划可继续。不能用旧“无作业”快照代替当前结果，也不能因为客户端缺失就绕PBS在登录节点跑GPU训练。

训练使用已验证Hopper大显存路径和合法auto队列；不pin hostname，不重新尝试已知4080S训练OOM。PC/Corelab负责代码、CPU、screen和sim。只有现有调度/资源能力实际可用才提交；一次活动phase一个生产job，qsub超时先查job token再补交，不重复申请。

不新增大清理。采用PI0250现有durability规则管理本Goal新生成的A/G/FULL产物：完整checkpoint当前可读并完成全文件bytes/hash验证后，才可退休不再被job/评估引用的scratch副本。四个正式FULL候选全部durable保留，但不要求四份同时常驻scratch。至少保留最新完整本地resume源，写下一份原子checkpoint成功并验证前不删除前一份。Corelab只物化需要的推理文件，不下载optimizer用于screen。

高水位按实际实现计算：`required_free = next_complete_checkpoint + overlapping_atomic/upload_temp + planned_downloads/cache_growth + logs + reserve`，reserve至少10GiB；已在磁盘的当前源不重复加入free需求，但禁止删它来制造空间。沿现有max(100GiB, required_free)操作目标，使用新产物滚动转存避免空间增长；不根据过期df盲删旧资产。现有源码若有多份同时保存，应实际计入，不只按单checkpoint估算。

quota可查询则分别检查；不可查询则标 `SPACE_READY_QUOTA_UNOBSERVED`，允许按既有规则在真实输出fileset做小型非稀疏写/fsync/读回、规划高水位并逐保存点复查后继续。小写入不证明大文件quota充足。真实EDQUOT/ENOSPC时停止新写入、保护最新完整恢复源；不要删除唯一源重试。

仅用既有已授权HF仓库或已验证持久Corelab存储；传输/凭据/匿名可读性分别记录，不因公开访问失败重新训练，不擅自变更仓库visibility或公开新数据。历史metadata/readback PASS不足以批准删除唯一有效本地副本；需当前完整内容可读。归档队列落后就在安全保存barrier等待转存，不让checkpoint无限累积。

工程路径、报告schema、传输、同完整状态恢复可自主最小修复并回归；不能放宽loss/输入/gate/选择规则来换Exit0。邮件/定时monitor保持关闭，不新增工具或权限范围。

## 10. 统一预算与执行边界

| 阶段 | 新canonical optimizer上界 | 新模型query上界 |
|---|---:|---:|
| G0复用/勘误/CPU准备 | 0 | 0 |
| A/G正式训练 | 4000 | 训练内native/S4单独计实际调用 |
| A/G最终screen | 0 | 256 |
| pre-FULL TRAIN sim | 0 | 260，另expert≤4episode |
| 唯一FULL family | 10000 | 训练内native/S4单独计实际调用 |
| FULL保持screen | 0 | 576新格，640逻辑格 |
| CALIB baseline+四candidate | 0 | 5440，严格身份复用扣减 |
| post-FULL TRAIN sim | 0 | 260，另expert≤4episode，可同身份复用 |

总新canonical训练上界14000；离线scored query上界6272；sim模型query上界520；native expert最多8episode、优先复用同身份控制。非选分完整S4 parity另最多6（A/G至多4、FULL至多2），只用于必要推理/梯度语义对齐，不进入checkpoint选择。模型load/head denoising/recompute/native forward与高层query分别记录实测，不能把期望计数冒充实测。

纯工程故障从最近完整源恢复的丢失后缀允许自主重算，不设任意次数上限；所有真实失败/重复更新累入raw账本，canonical阶段终点不延长。有效科学FAIL、已封存有效cell不重跑；没有完全状态的warm start不能冒充stage内exact resume。

原PI0252 G0新训练/新模型query均0；本文件不能把它改记为训练尝试。每阶段子状态单独维护，前级条件不成立不消耗后级预算。无需为已验证的成功阶段再申请例行批准，不能自动改写本文件寻找另一条科学路线。

## 11. 给执行agent的启动正文

执行 `PI0252_R1_SPEED_MARGIN_TO_FULL_FINETUNE_GO`。先完整读取本文件及G0 canonical5862242865；原PI0252/PI0250规则只在未被本文件明确修订处继承。

已解决的冲突：screen初始拒绝绑定纠正为实际.75m/1.50rad；不是改gate。复用G0原receipt与C/E归因，不重跑旧256或归因。只实现peak_linear_speed；C2000共同初始化，A/G各2000，原目标及同RNG，G仅加.1*relu(max速度归一化平方-.81)。测试和真实pre-backward guard通过就自动开训。

两臂final256和独立CPU/paired完整性通过后，原四组门取唯一development候选，均过优先A。按原TRAIN WBC one-chunk4/4后receding；通过即接一个FULL9792/10000，不停在large_scale_init或等新GO。

FULL依第6节固定配方、四个完整checkpoint和滚动归档；完成后原CALIB1088×(baseline+四candidate)、原排序冻结selected，再检查winner保持并做post-FULL闭环。科学FAIL不换配方/seed/checkpoint追成功；工程同状态恢复自主，raw如实记账。

不要再大清理scratch或删除X；qstat缺失先恢复合法PBS查询，不当作无作业、不重复qsub。quota未知可沿现有有风险说明的写入/高水位协议推进，不无限等一个查不到的quota。邮件/monitor关闭；不FULL以外额外扩数据、不DEV/F、不DAgger、不真机、不WBC/gate改动、不merge。
