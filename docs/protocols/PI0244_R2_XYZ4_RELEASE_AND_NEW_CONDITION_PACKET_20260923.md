# PI0244 — R2/XYZ4放行与新条件诊断快照

日期：2026-09-23（Asia/Singapore）  
Decision：`GO_R2_XYZ4_WITH_NONBLOCKING_NEW_CT_DIAGNOSTIC`  
Token：`PI0244_R2_XYZ4_RELEASE_GO`  
发布状态：`AUTHORIZED_NOT_EXECUTED`  
源码导航父提交：`48c732d2195af2820a804baaede68c96a0390dc3`  
Issue #11为canonical scientific ledger，PR #77只作pointer，不merge。

## 1. 本次PI决定

1. 保留R0的有效科学失败与全部80个有效cells，不重新训练/筛查R0，不将11条WARN视为全模型通过。
2. 接受现有记录对R1/P32不适用的支持，跳过R1。它不是失败的实验，也不释放一份可挪给其他配方的10K预算。
3. **明确放行已预注册的R2/XYZ4。** R0的XYZ门失败且P32不适用，已满足章程中的分支条件。准备、单元测试和容量准入完成后直接训练，不需要再询问是否允许backward。
4. 旧P1条件精确重放的本次路径以 `HISTORICAL_CT_REPLAY_UNAVAILABLE_FROM_SEALED_ARTIFACTS` 结案。保留原hash不相等的失败证据，不能删除检查后宣称精确重放。
5. 允许用固定原4窗、原处理语义建立**一份新的eval-mode C_T条件快照**，在原未消耗的48-cell范围完成新配对诊断。它明确不是历史P1重放，结果不能覆盖P1或R0 stock。该诊断是解释侧报，**不再是R2训练或其主判定的阻塞门**。
6. 如果新快照仍无法在下面明确的小范围内合法构造，保留精确缺项为 `NOT_RUN_NONCRITICAL_DIAGNOSTIC`，R2继续。若发现的是会直接破坏R2本身的真实模型/数据/标签/ABI矛盾，则停止受影响主线；不能把历史证据缺失与当前训练契约错误混为一谈。

本文件仅补充commit `4117f401778067e183385508c668686d1798120f` 的 `PI0244_GATE_MODEL_EXPLORATION_AND_TRAINING_CHARTER_V2.md`（Issue5782066001）。明确覆盖其中将完成旧48-cell历史重放解释为R2前置条件的执行顺序；R2训练配方、最终质量门、下游条件和原科学预算不变。不新开第三个配方，不叠加P32，不给R0加步数。

## 2. 本次依据、边界与应保留的状态

已读取：Issue5787181985、5787779075、5787781428（SHA更正），PR pointer5787782402；上述章程的R2公式与条件分支；48c732d的sidecar实现diff。查询最新相关ledger未发现这之后的新执行结论。

GPT本次没有SSH、实际qstat、运行pytest/模型、下载大checkpoint或原始NPZ，没有提交/取消PBS或启用监控。数值和dtype证据来自用户与canonical ledger；源码发现仅限读到的代码，不声称独立复核了Hopper数据。执行agent核对相关小manifest后复用，不重做全库和全数据审计。

### 2.1 R0与P32

R0科学screen为628527，账本报告80/80完整，compute/archive相同。最终：5REJECT/11WARN，五个REJECT均有LINEAR_SPEED_REJECT；XYZ median0.047100m、p950.054982m均不通过；SO3 median0.037452rad通过。R0保持 `SCIENCE_FAIL_M16_10K_BUDGET`，selected_checkpoint=NONE。训练与评估完成性由各自实际结果和内容证据判断，不能仅凭单个PBS状态码推断归档失败或成功。

关键SHA256：
- R0 closure：`4a64898d05377ad23a7022ae16921ceb2bb2d80e330cf7433e7c3595a5de9851`
- R0 training manifest：`62487ffff722eb61e41e621044c40e8292d291aeb1583d3c48073f0620cf7b9a`
- step10000 records：`4c502ab316ba9b4b800bfc743c9c5ecb1668a14d8b998f4c7a0f101cf5ed0d76`
- step10000 binding：`c6a2ac11c9c54c594f9d1bbb4a44d82a3e2c0645cb6cd1616ab1e7a08dd5dc9c`
- update/dtype audit：`eb775e0e00108cb240e2060e052e2132d179436e4ef4ddd6d7f8a28aa8eaa7a5`
- 新sidecar原始代码SHA（更正后）：`d938e401f6de9b54c89e3f28de2b1a1d387c1352fcb504c2f8fd047752d89086`
- PI0242 helper SHA：`7c03c5a2377d301011383cb04ee4f58e54253aabe2f45a3048745627d6a1ca1a`

471个可训练张量FP32的审计，加checkpoint2500 optimizer pickle中的FloatStorage证据，支持不再运行“改成FP32权重/状态”的重复arm。存储类检查不是逐参数exp_avg/exp_avg_sq映射审计，也不证明所有前向算子为FP32。记录这个证据等级即可；不要为跳过R1再整套复制optimizer。R2首个正式更新后按参数名记录实际参数、梯度、exp_avg和exp_avg_sq dtype，与R0实际路径匹配；有新矛盾再报告。

### 2.2 条件hash阻塞的真实含义

sidecar的make_plan调用旧 `_build_p1_conditions`，再把新生成的condition_hash强制与历史P1 C_T hash比较。历史P1只存了条件hash，而未保存对应pixel/token等实际条件张量或构造时RNG状态，故现有文件不足以支持精确重放。相同seed本身不证明相同随机流消费、预处理模式、运行版本与输出bytes。

这不是模型forward结果，也不是专家标签已经改变的证据。normalized expert hash匹配；mask float64→float32后数值相同，应保留两个原始dtype/hash及无损cast映射，不改写旧hash。任何新条件都必须用新身份登记，不伪装成P1。

初始化9792-row数据集、读取10个shard的元数据，不自动等于读取了9792个视频或重新跑全量审计；目前不夸大旧尝试的读量。但四窗诊断应优先直接消费已封存的screen观测和labels，避免再次走全量dataset初始化。允许读取定位所需的既有manifest/metadata和TRAIN统计，不允许重新拟合统计、重编号样本或为此遍历全数据视频。

## 3. 主线：R2/XYZ4现在可执行

### 3.1 固定run card

- Family：`PI0244X_XYZ4_M16_10K_V1`；独立execution分支、output、HF prefix和receipt，不覆盖R0。
- 起点：与R0相同的PI0226 adapted baseline1000权重；**不是R0-final10000、PI0243-A5000或历史最佳模型**。
- 新optimizer/scheduler/RNG；原16TRAIN、原R0 10000条曝光顺序、seed22301、每窗625次、batch1/accum1/workers0。
- 原AdamW、peak LR1e-4、WD1e-5、max_grad_norm1、cosine10000/warmup10；beta/epsilon/decay groups从R0实际resolved配置绑定。
- 训练范围/冻结模块/实际参数与Adam状态dtype/autocast/原augmentation与dropout/图像规格/stock sampler均继承R0。R0实际可训练FP32不等于本arm额外应用P32。
- 原C_T训练模式与原label-free eval模式分别保持。新增诊断快照不用于取代训练输入或改变正式screen。
- 原StateV1、dense-relative H40、universal132/active0:10、TRAIN stats/relative_stats、mask、frame/TCP、row-rot6d、dt0.05与target隔离全部保持。
- 保存完整Trainer checkpoints2500/5000/7500/10000；正式screen0/2500/5000/7500/10000×16=80个唯一cells。首个既定更新和screen可兼功能检查，不加隐藏调试updates/seed。

### 3.2 唯一科学变化

沿同一次原生forward返回的可微 `action_loss = mse(pred_v,target_v)*action_mask`，设置XYZ slots0..2权重4，其余权重1：

`L_XYZ4 = (w * action_loss).sum() / ((w * action_mask).sum() + 1e-6)`。

不得第二次采样noise/time或调用另一张teacher图；不得detach后造新tensor；不得在已masked action_loss上引入不同的额外mask/有效点定义。权重tensor与原输出的设备和数值路径匹配，不借loss改动切换原生计算精度。

全部10个有效维度等权时总权重10，XYZ4时总权重19。因此它不是总loss乘4：忽略同一个极小epsilon，XYZ元素相对原native梯度系数约40/19，其他元素约10/19，XYZ对其他维度的相对权重为4。姿态目前通过不保证新训练仍通过；最终SO3和gate仍是硬门。flow velocity包含噪声，不是物理相邻动作速度，不在这里直接平滑它。

CPU synthetic-autograd必须验证：w=1时原值/梯度一致；mask无效槽无贡献；XYZ与其他有效组的解析梯度比例；有穷值；分母；真实loss作为Trainer返回值而不是仅改日志。日志同时保留native总loss与XYZ/rot6d/gripper组统计，新加权loss不与旧native loss直接比大小。

只增加薄loss适配器、对应配置及测试，复用现有trainer/loader/persistence。原R0代码和card不改；新run_card与resolved_diff完整列出唯一loss差异及新的非科学identity/路径。

### 3.3 执行优先级

先核对live owner，复用已关闭R0；不重提628527。完成小范围代码/CPU测试和实际容量准入后，直接开始R2。诊断packet、旧48-cell重建、整理历史日志不占据R2关键路径。

不设工程恢复次数上限；每次有根因与针对性测试，合法结果first-valid。原recipe的exact resume保存optimizer/scheduler/RNG/数据cursor和真实消费记录；报告/归档失败只补报告或上传，不重训。有效科学终点FAIL不延长本arm。

## 4. 辅线：新的C_T条件快照，不是历史重放

### 4.1 输入与处理身份

使用原已固定bridge4的相同A–D source IDs，不重新hash选新窗，不按误差挑选。优先从R0固定screen已经封存的NPZ中取得同source的body/wrist RGB、state和独立专家/原mask；校验源文件及逐数组hash。源窗口/时间必须与原bridge4对应，不能只凭A–D相同就替换。

用**R0正式screen实际采用的、已批准的C_T-derived label-free eval processor**生成一次条件；绑定真实resolved kwargs、TRAIN stats、modality、pinned代码和eval mode。不是调用旧PI0241 train-mode builder后假定等价，不在诊断中新增augmentation。若processor入口有随机操作，在独立CPU准备进程按seed22301、固定A→D顺序执行一次并保存RNG前后状态。训练进程的RNG不得被修改。

新identity建议 `PI0244_CT_EVAL_PACKET_22301_V1`，明确字段：`historical_p1_exact_replay=false`、`source_condition=R0_APPROVED_EVAL_CT`、`purpose=new_paired_diagnostic`。同seed只定义本次准备，不宣称历史一致。

### 4.2 保存实际张量，而不只保存hash

保存完整模型消费的pixel/grid、tokens/attention mask、state及mask、embodiment等实际叶节点；keys按pinned模型拆分逻辑获取，不猜字段。专家未来/target/source ID/phase留在诊断metadata，不进入policy conditioning。

建议用safetensors加JSON保存tensor/结构；NumPy可用无pickle NPZ，BF16等应采用能无损保存其原dtype的格式。每个叶记录shape/dtype/content hash，整包记录hash、源观测hash、processor与模式、RNG和版本。新GPU执行只读取这份包，**不能每次重新跑builder再只比较seed**。各checkpoint用独立只读副本，避免可变alias和F(F(h))复发。

normalized expert和mask保留原权威来源；binary mask先校验形状与数值，再按原规范做已记录的dtype转换。保留cast前后hash，不把float64 bytes与float32 bytes不同认作标签变化。新expert本身变化或mask非二值等真实矛盾应停止相应比较。

### 4.3 只替换原尚未执行的矩阵，不加实验臂

原R0解释侧报仍是 initializer / PI0243-A5000 / PI0244-R0-final10000 × bridge4 × 原t0/q10/q50/q90 =48个direct cells；在新packet上重新编号，不与历史P1数字拼成同条件差值。噪声沿已验证P1实际noise数组读取并核验，time/离散化按原native来源；不需要还原旧条件即可复用已完整保存的noise。noise若也缺失，不自行新增seed臂，记录该辅线缺项；不阻塞R2。

R2原32-cell侧报（initializer/final×bridge4×四t）可用同一新packet，以便解释R0与R2差异；共享的initializer cells在完整身份相同时复用。总预算仍为原48+32个计划scope，精确重复项不重复计算；绝不新增stock查询用于追历史hash。direct模式保持no_grad、原单次feature处理和已验证native helper，teacher-forced误差不当policy成功。

这份新packet改变的是诊断的条件实现身份，不改变R0旧数据和结论、R2训练输入、正式screen、损失权重或排名。新的direct结果只支持这四窗/四t上的解释，不能单凭其表现改R2配置。

优先在Corelab或R2计划中的screen GPU分配内顺序执行，不让历史诊断另排H200阻塞R2。CPU打包/小测试可以并行；GPU只读诊断与训练隔离设备/进程/RNG，不抢占同卡训练资源。新packet未准备好仍可训练；终局报告如缺侧报要明确标未完成，不假称归因已经闭合。

## 5. 验收、下游和停止

R2最终10000必须满足：80完整合法cells、对应的16个最终rows及训练/数据/更新证据有效；原gate零REJECT；median逐窗XYZ RMSE<=0.020m；p95<=0.035m；median逐窗SO3 RMSE<=0.050rad。WARN保留。逐窗RMSE全40点，percentile linear。只判最终点，不挑2500/5000/7500，不换seed，不加步数。

R2 PASS：封存fit_recipe_manifest（仍不是WBC selected policy），自动继承XYZ4及R0实际训练精度进入章程原M64/5K→FULL9792/10K→统一五profile原selector→winner TRAIN准入→frozen-WBC one-chunk→receding。每一级原数据、budget、旧16保持、早停、scorer与失败门继续适用。不要R2通过后又悄悄改回native loss，不同时启动旧B或另一recipe。

R2有效FAIL：研究树R0失败、R1不适用、R2失败，终态 `PREAPPROVED_RECIPE_SET_EXHAUSTED — STOP_FOR_PI_NEW_HYPOTHESIS`。没有新增P32+XYZ4、20K/50K、速度loss或修改sampler的授权。完成可获得的既定解释证据并指出未知；不无限等待缺失历史条件。

当前支持集训练/筛查不自动解锁CALIB或sim；仅到达原FULL/selection/控制阶段门后才可执行其既定范围。DEV-E/F、Final-F、DAgger、真机和PR merge保持锁定。没有合法最终winner时selected_checkpoint=NONE。

## 6. 资源、存储、监控和交付

PC负责git、CPU单测、SSH；Corelab负责Linux集成、小诊断/screen和获准后native sim；H100/H200只用在成熟且确需容量/规模的训练。不要为了显示Corelab优先重复已知OOM，不在Hopper登录节点做重视频扫描、全模型加载或大数组计算。

R2容量按R0实际完整FP32训练checkpoint大小、四份保存、原子临时文件、screen、cache及余量计算；复用成熟可见镜像和已验证HF publisher，独立prefix/固定revision。不动唯一恢复点、旧initializer或旧数据，归档错误不重训。metadata入口、run card、输出、finalizer、screen都沿共享runtime mapping，CPU做端到端测试，避免H200反复发现路径拼错。

监控与科学执行分开：本次GitHub发布不启用任何定时服务。用户要求的10分钟频率只能由有相应能力的PC agent既有本地监控实现；这里的ChatGPT定时任务最低间隔是1小时，未创建10分钟自动化。PC端若启用，先列出现有实例，复用唯一owner，每600秒只读检查其任务/产物增量；没有活动job时不反复查询已终结R0，不因轮询本身自动qsub。工程恢复依据本授权的实际根因、测试和resume边界进行；到科学终态/显式BLOCKED退出。不得声称GPT已在后台监控。

最小产物：
1. 分支决定小JSON：R0有效FAIL、R1 NOT_APPLICABLE及证据等级、历史重放缺项、R2明确GO；引用已封存hash，不重算旧大审计。
2. R2 run_card/resolved_diff/loss autograd tests/实际launch与容量binding；训练和screen沿原产物格式。
3. 可选新condition packet及48/32侧报（准确记录READY/COMPLETE/NOT_RUN及原因），不冒充历史重放。
4. R2五行screen表、最终联合判定、各主要issue、真实update/dtype/moments和对应HF receipts；PASS后的原条件推进或FAIL后的单一下一假设建议。

Issue #11先canonical，PR #77只pointer；新execution分支最小diff，不覆盖旧报告。此次审核未执行上述训练、测试、诊断、监控或远端写操作；仅发布协议与账本。

## 7. PC agent启动正文

执行PI0244_R2_XYZ4_RELEASE_GO。完整读取本固定文档、原章程4117f401778067e183385508c668686d1798120f及R0 closure5787181985、blocker5787779075、erratum5787781428。

R0有效FAIL保留，P32按现有FP32证据NOT_APPLICABLE，不重跑R0/R1、不加R0步数。旧P1精确条件重放缺证已由PI裁定，不再是R2训练门。先封存分支决定并复用已有数据/参数/归档/原80cells，禁止全库重审。

主线直接准备并执行PI0244X_XYZ4_M16_10K_V1：同原PI0226 initializer，新optimizer/scheduler，原16/10000 schedule/seed22301、原训练precision和C_T modes不变；只把同一次native forward的XYZ loss权重改为4，其他1，按正文weighted-mask分母归一化。CPU验证w=1原值/梯度一致、mask/解析比例/finite、实际Trainer用新loss；首个正式step记录named moments和param dtype，不另加科学updates。

四full checkpoint2500/5000/7500/10000，80个原定stock cells，final10000原四项门；全部source保留，不挑中期、不改gate/时序/输出。工程自修，准备合格直接开训，不再次等训练授权。

辅线允许同bridge4从当前封存观测建立一份新的eval-mode C_T packet，独立seed22301一次生成，保存全部真实条件张量/kwargs/mode/RNG/hash。历史hash不必相同，但旧检查/失败记录不覆盖；新packet显式不是P1 replay。运行只加载packet，不再每次初始化9792 dataset。原48+R2原32 direct scope不扩展、不新增stock，噪声复用可验证原数组；该辅线不抢占、不阻塞R2，无法合法完成明确记录未完成。

R2 PASS自动走原XYZ4 M64→FULL→selection→frozen-WBC one-chunk/receding；R2FAIL研究树耗尽停PI。选模前selected_checkpoint=NONE，DEV/F/DAgger/硬件/merge不开放。PC/Corelab测试和小任务，容量需要才上H100/H200；复用成熟存储/归档，不重训补报告。

用户所说的10分钟监控由PC端既有单实例只读轮询处理，不创建重复monitor、不让轮询绕过准入或触发重复提交；科学终态退出。回报实际job/step/有效cells与证据，不只报已读或排队。Issue #11先canonical、PR #77pointer。
