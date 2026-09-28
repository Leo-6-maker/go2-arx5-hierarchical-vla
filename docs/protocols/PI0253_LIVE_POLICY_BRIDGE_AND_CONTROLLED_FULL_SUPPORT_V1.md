# PI0253 — 离线合格策略到真实输入的桥接审计 → 单一 FULL 支持扩展 → 正式选择与 TRAIN 闭环

日期：2026-09-28，Asia/Singapore。
Authorization: `PI0253_LIVE_BRIDGE_TO_FULL_SUPPORT_GO`
Decision: `GO_BOUNDED_BRIDGE_THEN_CONDITIONAL_FIXED_FULL_G_NO_NEW_M64_LOSS_SEARCH`
发布状态：`AUTHORIZED_NOT_EXECUTED_BY_GPT`。
Issue #11为canonical；PR #77只追加pointer，保持draft/unmerged。

## 0. PI决定及本次明确改变

接受PI0252-R2的G2000 M64四组PASS，也接受其TRAIN-A/C0 one-chunk实际FAIL。两种结果检查不同条件与指标，不互相撤销。原 `PI0252_R1_PRE_FULL_TRAIN_SIM_FAIL` 及全部first-valid输出保持，旧receding/FULL条件未成立。

本Goal不增加M64 loss分支，不要求把原A控制臂修到PASS，不进一步提高速度惩罚。立即授权：复用旧数组完成D0窄诊断；D1只做两种输入×两种真实噪声×两个实际推理入口的8格桥接。排除真实契约错误、确认专家及数据适用后，显式放行一个未改G目标的FULL支持扩展实验。

**新的科学依赖决定：本Goal不再要求旧G先在该新鲜输入上达到任务成功，才允许用更广TRAIN支持训练。** 否则会形成“未学会扩展支持，先要求扩展支持成功才能扩量”的循环。替代的FULL前置条件是：G原资格与恢复来源成立、离线/部署策略计算可比、输入与标签/坐标/时间语义成立、当前WBC有相同场景的正确专家控制、现有FULL数据包含适用任务和起始阶段支持。没有放宽模型质量、SafetyGate或实际任务成功门；只是允许训练去检验支持扩展是否有效。

这是新预注册的支持扩展研究，不是声称旧协议已经允许忽略sim FAIL，也不证明数据量是唯一根因。新增数据曝光、10000更新和新optimizer共同构成本次实际干预；没有等预算M64对照，因此不得将改善纯归因于数据规模。

目标链：
`D0原件/坐标/时间/输入支持审计 -> D1固定8格入口桥接 -> D2机械准入 -> 单个FULL_G/10000 -> 原CALIB选择 -> winner自身保持 -> 原TRAIN one-chunk -> 条件receding`。

只要既定条件通过，agent自动推进，不再申请例行GO。真实语义/恢复/评分冲突或各阶段有效FAIL停其依赖链。DEV/F、DAgger、新专家采集训练、真机、WBC/gate修改和PR merge仍未授权。

## 1. 审核来源与当前证据

必读：
- 最新执行canonical `5869849519`，pointer `5869857834`。
- 执行及报告commit `e1f3ab667ba0c89d6929822de4135877532f656a`。
- `reports/PI0252_R2_PRE_FULL_TRAIN_SIM_REPORT_20260928.md`；同commit的 `reports/evidence/PI0252_R2_PRE_FULL_SIM/`。
- 同commit `scripts/pi0252_r2_sim/pi0252_g_model_worker.py`、`pi0252_g_one_chunk.py`、`pi0252_expert_control.py`、`pi0227_expert_reference_binding.py`及实际shell入口。
- 执行依赖ref `317a2e85419cacd396f4f989a4ce83f4776bb85b`下 Stage21 worker/inference/native_model_receding、`pi0247_m64_screen.py`、`pi0247_d1_screen.py`与实际G screen adapter。
- PI0252-R2原协议commit `a5ccfba37a54e53b1ad1818a095edf67319aba9c`及canonical `5866382788`；P0 `5868341369`。
- PI0252-R1 commit `cef7dadc2fb60304d39057dc3e720bc5b6210061`的FULL、选模、恢复和资源定义。本文件只在明示处替代依赖和namespace，不重复发放旧M64预算。
- PI0226 canonical `5733559806`和实际封存的TRAIN场景、cadence/clock/scorer，而非README猜测值。

当前G：HF revision `efecea485ae77000599d19171b02491f8d514f42`，完整Trainer tree `cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226`。

最新已报告：
- 正确目标绑定后的native expert A–D 4/4：actual minimum 8.712/2.692/4.514/9.828mm；同为97tick完整clean episode。
- G TRAIN-A/C0：gate ACCEPT；reference minimum .10497794769237719m；actual minimum .15649767760855235m；actual final .20134563374433626m；无hit。
- reference final .17647578043478232m；world同tick tracking mean .13742363742946653m、max .16681461002383685m、final .1406965106780276m。
- 初始位置跳变 .12097060081762022m；峰值线速度 .793706544077868m/s；本次不是速度REJECT。
- 1次真实模型query、4次S4 head、0 optimizer，未运行模型B–D/receding/FULL/CALIB。
- 原native control .02s、H40 .05s/1.95s；原完整hit标准保持，不将终点必须停留35mm变成新门。专家B–D终点较远仍是原定义下PASS。
- HF证据固定revision `d0cdfac5297c10fb8dd4ccc71f3e60cf578dbaa7`，prefix `evidence/PI0252/r2-pre-full-train-sim-20260928`，61文件/11,621,636bytes，tree `b0b15ccba077a015ce2b289adcda775dc573017fb8091f724c723f3c5af11497`。
- generation binding SHA `74b8cce6872af2026f3b87f47246b5a814a852cf1207d9f6b0328039cd26fcb7`；G manifest `f136ec0213bda0d8598f9fd64509b15c3c0f4bd4f6172a15e0b206e44854f739`；实际rows `f84e85461cc3c1ddf846c67347a939e446165f8a9272dc019063a34d99d71256`；CPU replay `fae083942291b9598a1a8729ee96d3e9a284b90a6b0379d681588aa6136dc87b`。

本PI读取了上述GitHub报告与源码。当前会话尝试读取HF原NPZ时遇DNS失败，未成功下载原数组；HF连接返回repo元数据不构成原件读回。本GPT未SSH、查询live PBS、执行项目模型或仿真；不能将执行侧CPU/HF PASS说成本轮独立重做。

## 2. 判断边界：不把一个数字差值当成根因

1. 原M64指标是相对专家的整窗XYZ/SO3误差及gate；sim是新鲜输入上相对实际目标的任务成功，两者不是相同随机试验。任务属于TRAIN A不等于该reset图像/state/window已经在原64中被监督。
2. reference离目标105mm说明轨迹生成/解释层至少存在任务缺口；它不支持“仅WBC跟不住”。expert成功只证明当前控制器能完成那些专家参考，不证明它可跟踪任何gate ACCEPT的轨迹。
3. `156.498-104.978≈51.520mm`是两个不同曲线极小值之差，最小值时刻不同，不是可相加的tracking因果分解。跟踪必须沿真实同tick desired/achieved计算；不得做DTW或事后相位移动来改善正式分数。
4. report中的旧 `oracle_vs_model_position_rmse_m` 和unbound oracle距离已明确作废，不得重新用其.2397m作为专家拟合误差。
5. 当前1次open-loop one-chunk失败不等于已经测得闭环分布漂移或多任务泛化失败。receding未运行。H40是局部预测窗口，是否应在本窗口命中应与同起点、同时间采样的实际专家定义核对；不能临时延长episode救分。
6. 初始跳变约121mm和137mm平均tracking值得分析；同样，峰值速度已低于1m/s，无证据支持继续增大speed loss。也不能仅靠平滑、夹断首点或低速化修复目标选错。

## 3. D0：只读桥接诊断，不重建全部实验

### 3.1 绑定原始数据并用正确坐标/时间比较

只下载/读取本阶段小型证据，不重新下载大权重或回放旧256格。主对象：
- `G/A/mailbox/requests/model-0000.npz`，SHA `07883fc3422c9df1e3ed6c8afa0e32de47577a75c7bf6df22117ff68535db524`；请求JSON及实际worker返回。
- response NPZ SHA `00a63737df8e59bb0850ac31cd4616f05eefe72ccfe9b741227f7739c2f09b4b`。
- noise NPZ SHA `12323ebb5a91d4a2696c076f700bd5f6b0252b7d6beb3d90973cc18937b13a44`；真实BF16 raw tensor SHA `585570a70dbb0cfb597bfd49c6ffce1e77b598c744f2a89eb3e890455c9a0852`。
- 合法专家A是 `controls/A_attempt2/actual_expert_reference.npz`，SHA `5df0c2c2e75d7204baebacfaa7f0415a41e972867ccb9d772fe36103d6dfb140`；不是A_attempt1的旧anchor模板。
- 对应actual rows、initial root/dof/anchor/history、目标位置及原相机数组。

用原轨迹的真实timestamps和原TrajectoryBatch/采样函数，将正确目标绑定的专家参考采到模型的0:.05:1.95时间点。不能按相同array index比较101点专家与40点模型，也不能将目标坐标灌入模型。若专家与G的initial state不一致，报告不匹配，不做虚假的same-condition误差。

分别交付：reference-target曲线；actual-target曲线；同tick actual-desired tracking；模型与合法专家的同时间残差（只有绑定成立才计算）；初始jump、目标方向投影、窗口内minimum/endpoint及时间；旧指标原样保留。

可用已公布bounding box作待验证线索：anchor_y=.0322408415，decoded local y最大=-.00834157877，按当前translation-only映射预测world y最大约.0238993，目标y=.1，至少相差76.1mm。只读原数组确认该推论；它说明目标方向覆盖不足，不证明y sign bug、视觉失明或具体神经机制。

### 3.2 审核真实输入，而非再检查一次文件名

比较live与实际M64入口：body/wrist顺序、RGB/BGR、尺寸/裁剪/缩放、视频解码与直接渲染、processor eval状态、文本tokens/attention/image masks、StateV1局部/世界参考、rot6d行约定、gripper、TCP和H40 anchor、dtype/device/autocast及action-head配置。

分别保存processor之后、实际get_action之前的完整leaf inventory。检查target actor实际world pose、render完成顺序与相机可见性；原request含target元数据允许scorer使用，但policy张量不得含target/source/phase/expert/future字段。可使用已存图片和几何投影，不新跑VLM/视觉embedding模型。

同一seed不替代实际噪声。本次live按BF16 randn产生噪声；历史screen使用冻结float32 packet再按runtime转换。分布/具体样本不同不自动是bug，也不能说seed相同故noise相同。D1控制实际tensor解决此问题。

Stage21 worker与screen使用不同入口是已读源码事实，不是已确认它们结果不同。不要未经对照修改precision、scheduler、norm、action或solver。

### 3.3 支持覆盖和专家监督含义

沿原source64/old16和原9792 TRAIN的metadata、数值state/action行审计，不重解码全部视频、不读取CALIB/DEV/F、不生成新训练数据。

记录live观测是否属于原64的精确输入；祖先已有数据暴露按manifest继承，不宣称G一生只见64窗。统计原64与FULL中相同任务/目标、reset或早期phase的独立rollout数与window数；报告initial-state距离、姿态差、目标相对方向和窗口阶段分布。近邻选择用预先固定的物理量（XYZ欧氏距离、SO3角度分别报告），不用模型输出挑“最像成功例”。范围内不等于已证明充分覆盖。

确认FULL中存在该任务及起始阶段、标签采用与已验证expert reference一致的动作语义。若只有后段/终点样本或根本无该视觉/任务支持，不能靠执行更多epoch宣称扩量有效，应停 `PI0253_FULL_SUPPORT_OR_LABEL_SCOPE_GAP` 请求新数据协议。不得重采样一个失败source作为全量数据替代。

D0输出小型 `PI0253_LIVE_CONTRACT_AND_SUPPORT_AUDIT_V1`，区分OBSERVED / INFERENCE / UNOBSERVED。已有控制4/4和742文件WBC读回身份未变则复用，不要求再跑4套控制。

## 4. D1：固定8格，不是新的seed或solver搜索

### 4.1 两输入、两真实噪声、两入口

权重仅G2000，不比较其他模型。

输入：
- L：原失败live request的两路RGB和StateV1，文件固定。
- M：从原64完整source_episode_id（含obs suffix）按UTF-8字典序选第一条，独立于分数，使用其原观测和B0收据。不得用旧碰撞cell_key去索引。

噪声：
- N_live：原失败的实际BF16 initial noise。
- N_M：M源历史B0的实际initial noise，不是只取seed或未cast的source packet。

入口：
- Q：G四组资格实际调用的screen get_action/processor/decode路径。
- S：本次Stage21 sim worker实际路径。

全矩阵 `L/M × N_live/N_M × Q/S = 8` 个逻辑cells。严格相同元组和策略身份的既有输出可复用；诊断新增模型query最多8，不为补充trace重跑已有有效元组。已存输出缺中间trace记UNOBSERVED，而非必须重新推理。每格均保存raw/decoded/gate/输入/实际noise，真实新调用额外保存四步状态/velocity。

固定噪声只注入模型初始动作随机张量，保持shape/dtype/实际数值，不改变conditioning/solver/输出；只对精确action随机调用生效，不能拦截所有随机过程。其他随机状态与mode按原入口隔离。若模型runtime的required dtype不同，记录差异并先定位，不能悄悄cast后称字节相同。

### 4.2 比较与允许的机械修复

同输入/同实际noise比较：真实policy leaves、模型配置与参数dtype、4步过程（已有则用）、raw、物理解码位置和SO3、完整gate decision/issues。不同输入或不同噪声的差异只能侧报敏感性，不当成入口错误或泛化置信度。

本次新bridge一致性容限（不是质量门）：raw绝对/相对allclose各1e-5；decoded XYZ逐点欧氏max<=1e-5m；合法SO3逐点角max<=1e-4rad；gate decision与issue code/severity一致，observed数值按同单位既定容限。若原同入口封存测试有更严且适用约束则保留。不要边测边放宽容限。

若超限，定位最早差异。仅允许恢复Q已经资格化的语义：实际输入cast、processor kwargs、错误/重复特征处理、mask、路径或遗漏配置等可证实的worker偏离。不得换权重、改变S4、norm/statistics、reference frame/TCP、采样分布或人工变换动作来追目标。修复前后均保留独立身份；旧sim FAIL仍有效于原执行组合。

修复只重做受影响桥接元组，属于相同测试的工程回归，不新增输入/噪声搜索；首次有效结果和所有raw尝试保留。无法在既定语义和容限内闭合则 `PI0253_INFERENCE_BRIDGE_UNRESOLVED`，不开训练。若需改变已合格Q本身，属于新策略实验，返回PI。

Q在live输入上没有命中目标不阻止桥接PASS；入口一致性不是task success。N_M使live成绩更好也不能改部署seed或把原fail升级。

## 5. D2：明确的FULL准入，不重复“先让旧G成功”的门

全部满足则写 `PI0253_FULL_ADMISSION_V1` 并自动开始第6节：
1. 原G M64资格、完整weights/processor/stats和真实训练来源可恢复；没有新证据推翻原独立CPU结果。
2. D0无未解决输入/坐标/时间/标签冲突，当前任务的专家控制在同runtime可行；对正确时间采样的专家H40语义已确认。若one-chunk命中要求与训练标签任务定义实质冲突，停PI，而不是自动增加终点loss或换任务时长。
3. D1 Q/S身份闭合，或机械修复后同义闭合；保持Q策略语义不变。
4. 原FULL数据seal与scope真实成立，包含对应任务及起始阶段的有效支持，且无DEV/F或CALIB进入训练。
5. 完整持久保存、实际fileset容量、合法GPU资源和阶段恢复路径准备好。资源等待不阻塞CPU及归档工作，权限未知不伪造PASS。

此准入不要求从一个失败cell证明全局“过拟合”或“数据不足”，也不要求G在此时重新达成sim目标。允许检验的假设是：**当前G的局部资格能否通过固定目标下更广的TRAIN曝光转化为部署输入上的任务能力。** 训练结果可能支持，也可能否定这次扩展。

本Goal不再次执行旧G的one-chunk/receding，不把桥接时更好的噪声输出拿去WBC。科学结论是新FULL结果，不是追一次原G成功。

## 6. 单一FULL支持扩展实验：不改loss、不另开2K树

Family: `PI0253_FULL_G_SUPPORT_V1`。
初始化：唯一G2000 weights；新的optimizer/scheduler，是新stage warm start，不称G旧optimizer exact续训。

固定目标与G一致：
`J = L_native + .4*ell_pos + .1*ell_delta + .1*ell_rot + .1*ell_speed_margin`
`ell_speed_margin = relu(max_j(sum_xyz((p[j+1]-p[j])**2)/(2.0*dt_j)**2)-.81)`。

原H40/dt=.05、active10/padded132、rot6d/decoder/minmax/relative stats、S4四步完整反传、trainable/frozen与参数dtype/autocast全部保持。不再增加首点惩罚、target吸引loss、E超额项、训练noise个数或解冻视觉语言骨干。目标XYZ不进入policy输入。

数据：原9792 TRAIN窗口，原source-rollout split及A–D范围；数值、视频、统计身份沿seal。按原episode_index/source_ID排序，以独立PCG64(22301)逐轮排列9792+第二轮208共10000曝光；不强化失败source，不叠dataloader shuffle、不改变old16比例。原1088 CALIB不用于梯度、统计或调系数。

优化：10000 canonical updates；batch1/accum1/workers0；AdamW LR1e-5、WD1e-5、betas(.9,.999)、eps1e-8、maxgrad1；cosine10000/warmup10。

新stage RNG：ASCII `PI0253|22301|stage=FULL|optimizer_step=<1-based十进制i>`，SHA256前8bytes大端 & ((1<<63)-1)，沿已测试的forward/backward随机隔离、Beta/Gaussian与checkpoint重计算语义。不含job/机器/attempt/时间，不复用M64的step namespace。单臂保存实际random/消费证据，不伪造配对臂。

10k约是该表1.02轮，不称其为充分收敛的保证；本轮只检验该冻结预算。不得因看到趋势好临时再加1000或改LR。

正式完整Trainer checkpoint：2500/5000/7500/10000全部durable保留。1000仅model-only观察快照，不参与selection或冒充完整resume。各候选保存模型/optimizer/scheduler/RNG/游标与append-only消费/梯度/mode证据。

CPU先测试实际新入口的10000终点、G初始化、完整目标、scheduler、保存/恢复cursor和finalizer；不能只改family字符串而沿用2000硬编码。新card加入本授权，已完成G card/日志不改。旧恢复fixture能复用就复用。

工程故障同完整状态恢复允许；无完整中间恢复源时仅允许从该stage确定起点重算已丢失且未封存有效产物的后缀/前缀，canonical终点不变，所有raw更新计账。不以新optimizer冒充checkpoint resume，不重训已完整完成的stage。

## 7. FULL固定评价、选择和停止

### 7.1 保持矩阵与live侧报

原M64保持矩阵继续：全64/B0在step0/1000/2500/5000/7500/10000，共384；四正式candidate各64/B1，共256，总640逻辑格。step0/B0的64按完整身份复用G旧结果，最多576新query；old16从这些结果提取。仅路径/worker薄修且Q计算不变，旧Q结果仍可复用。

2500保留原系统失稳barrier：1000与2500两次B0均64/64 LINEAR_SPEED_REJECT才触发该既定质量早停；普通中期联合FAIL只记录并继续，实际NaN/非法数据/恢复错序单独按完整性停。不能把某一个候选FAIL错误解释为整个family必须停。

全部训练完成后，四candidate各对原L输入/N_live进行1次固定Q离线预测，共4次，G在D1相同格作为起点复用。侧报目标距离、正确专家误差、initial jump、完整gate和速度；不得进入梯度、改变训练终点、排序、候选集合或追加seed。这个已用于研究的live-A不是新的盲测。

所有有效cell first-valid；报告/传输失败只补报告/传输，不重新随机采样。独立CPU从实际数组复算，不能只复读summary。

### 7.2 正式CALIB选择

完整family和保持证据闭合后，才读取原1088 TRAIN-CALIB用于本轮选择。基线为原PI0226 adapted1000，只对照、不参选；加四candidate，每份1088，最多5440 query。完整身份一致的旧baseline可复用，否则运行已批准基线；不是读取E/F。

保留原S4、TRAIN statistics、target authority及CALIB gate scope；实际noise从基线逐source封存后给candidate复用。目标仅供scorer，禁止policy读取。成熟selector `pi0228_train_calib_checkpoint_selection.py::select` 的admissibility/ranking不改：完整零REJECT候选 -> 最低median reference min-target-distance -> 最低worst source-rollout-cell median -> 较低step。必须1088个唯一完整source及finite，expected_count不得降低。

一个candidate失败不否决其他合格者；没有任何admissible才 `NO_ADMISSIBLE_CHECKPOINT`。不得新增“live-A好看才可参选”过滤。S3 provenance、S4唯一freeze、S5只读诊断。该CALIB是开发选择集，不称最终泛化测试。

正式winner必须自身all64/old16的B0/B1四组满足原零REJECT、medianXYZ<=.020m、linear p95<=.035m、medianSO3<=.050rad，才进入sim。保持失败时保留selection记录，`sim_admitted=false`，不得改选runner-up。若无candidate，保留完整family和解释，不添加训练分支。

## 8. 新FULL winner的TRAIN闭环

沿本次e1f3实际有效专家控制与PI0226任务绑定，保留相同场景、目标、reset/query seed、WBC与原gate、StateV1/frame/TCP、reference生命周期、cadence和物理时间。用D1闭合的Q语义部署入口，旧未绑定oracle只保留历史不用评分。不同gate作用域分别封存，不把screen宽门复制到sim或反过来。

同runtime有效expert A–D控制可复用；只有实际环境身份改变需验证时最多4个控制episode，不调整目标去迁就旧模板。invalid engineering attempts与canonical控制分开记；既有A_attempt1不会消失。

唯一selected按任务A->B->C->D各一次one-chunk，最多4个新模型query。这是新FULL权重的前瞻性评价，不是重试原G。原criterion：完整clean episode内实际world-TCP曾达到<=.035m，终点距离单独报告；reference是否命中是诊断，不擅自新加endpoint hold条件。gate REJECT不执行、不重采样。

4/4 one-chunk通过后才运行原progress-preserving receding：A–D各一次、每任务<=64query且服从原更严格tick/query cap，总<=256。fresh观测、连续reference时钟/进度，不reset、不teacher prefix、RTC、smoothing、retiming、target injection或DAgger。

同时保存reference/desired/achieved、同tick tracking、目标距离与first-hit时间、实际噪声、终止状态和视频。同步暂停仿真不冒称真机实时。有效任务FAIL准确停后续链，不换候选或seed、不直接调WBC。

终态分开：
- `PI0253_FULL_TRAINING_COMPLETE`：10000更新+四完整durable产物。
- `PI0253_FULL_SELECTION_COMPLETE`或`NO_ADMISSIBLE_CHECKPOINT`。
- `PI0253_SELECTED_TRAIN_SIM_PASS — STOP_FOR_PI_GENERALIZATION_REVIEW`仅在winner保持与闭环均通过时成立。

本Goal没有证明多任务或未见技能泛化，也没有新DEV/F、真机资格。

## 9. 预算与工程自主权

新canonical optimizer最多10000，只一个FULL family，无新M64更新。

| 范围 | 模型query上限 |
|---|---:|
| D0及CPU数学/来源检查 | 0 |
| D1固定桥接 | 8逻辑格，严格旧元组复用扣减 |
| FULL保持 | 576新增，640逻辑格 |
| 四FULL在原live输入的侧报 | 4 |
| CALIB | 5440，合法复用扣减 |
| selected TRAIN sim | 4+256=260 |

离线新query总上界6028；sim<=260；native expert最多4个按身份必要的控制episode。必要FULL非选分S4 parity<=2，优先复用语义未改的既有证据；没有额外optimizer smoke。native/head/recompute/model loads与query分开实测。

仅已定位工程修复可对同一失败/变更身份元组回归，不开新input/noise或挑最好输出；旧有效成绩始终保留且不重选。所有工程attempt和raw重算完整记账，不设任意次数上限逼迫反复等PI；无法闭合的科学语义冲突仍停。完整报告或上传失败不重训。

Corelab用于CPU、无梯度bridge/screen/sim，GPU0共存评估需满足已有用户授权和容量，不结束无关进程、不做Corelab训练。Hopper合法PBS承担需大显存的训练；不在H200反复调CPU路径，不重复qsub，未知job状态先恢复合法查询，不当作无作业。

保持scratch Projects布局与兼容链接，不再大清理；G、X保护源、唯一原证据/统计/dataset不动。新产物按当前完整读回后的滚动归档退役，原完整恢复源保护。100GiB是准备目标，不是每次写入后必须保持的free下限；保存前检查下一完整写入、重叠temp、cache/日志及>=10GiB reserve。quotaUNKNOWN如实报告，实际EDQUOT保护恢复源停写，不能用df或小write假称大文件quota已证实。

邮件/monitor关闭，PR #77不merge。

## 10. 交付顺序与启动正文

首要成果不是又一份loss方案，而是：原live与M64的真实输入差异、正确专家H40对照、Q/S同噪声桥接结果及明确FULL准入决定。随后取得实际FULL训练、family和新模型sim结果。

执行 `PI0253_LIVE_BRIDGE_TO_FULL_SUPPORT_GO`。完整读取本文件、5869849519及执行commit e1f3ab667ba0c89d6929822de4135877532f656a。旧G M64 PASS和sim FAIL都保持，不再要求原A通过、不增加M64训练或速度loss。

先D0只读原件，正确使用A_attempt2目标绑定专家和真实timestamps，区分同tick tracking与两个minimum差值；比较live输入/processor/frame/noise和原64/FULL起始支持。禁用已作废oracle side metrics。

D1按固定两输入×两实际noise×Q/S共8格，旧同元组可复用；只修可证实偏离资格化Q的worker实现，不改算法、gate或噪声追成绩。无语义冲突且第5节条件闭合即启动一个FULL_G10000；不再要求旧G先通过pre-FULL sim，也不等待新GO。

FULL继承全部G loss和固定9792/10000曝光、优化与保存定义；不加首点loss或视觉解冻。原640保持矩阵、4个live侧报、5440 CALIB按协议，唯一正式选择后保持检查再原TRAIN one-chunk4/4->receding。

工程同状态恢复自主，raw全部记账；有效科学FAIL停对应依赖，不换seed/候选/终点。selected在正式CALIB前仍NONE。不要重做清理、旧256 screen、全库视频审计或额外模型诊断；不DEV/F、DAgger、真机、WBC/gate修改、merge或monitor。
