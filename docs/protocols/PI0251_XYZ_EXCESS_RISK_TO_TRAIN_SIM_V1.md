# PI0251 — 困难窗口位置误差对照 → M64原门 → TRAIN仿真 → 大规模初始化候选

日期：2026-09-28，Asia/Singapore。

Authorization: `PI0251_XYZ_EXCESS_RISK_TO_TRAIN_SIM_GO`
Decision: `GO_G0_G1_CE_PAIRED_2K_FINAL256_THEN_QUALIFIED_TRAIN_SIM`
Status: `AUTHORIZED_NOT_EXECUTED_BY_GPT`
Issue #11是canonical scientific ledger；PR #77只追加pointer，保持draft/unmerged。

## 0. 本轮决定及明确改变

接受PI0249/PI0250终态：K/X各2000更新和各128个有效screen完成；两臂old16的B0/B1均通过，全64均未通过。旧 `PI0249_XYZ_PRIORITY_HYPOTHESIS_EXHAUSTED` 保留，旧C/R的tau配对失败和本次K/X的事后检查限制也保留。不得把“hypothesis exhausted”扩大为所有位置监督方法均无效。

下一假设：**当姿态、零REJECT和旧16保持已经恢复，而全64的剩余失败集中于跨窗口XYZ p95时，保留原位置均方目标，再对当前生成误差超过固定35mm容限的训练窗口增加一项有限的超额误差惩罚，是否比相同预算的原X继续训练更有效？** 这是可证伪的新候选，不是已证明的根因或最优算法。

新授权链：

`G0旧证据/执行来源及误差分布窄核验 → G1新增标量目标与在线随机身份检查 → C/E各2000更新 → final-only256格 → 唯一M64-qualified开发候选 → TRAIN A–D C0 one-chunk → progress-preserving receding → large_scale_init_checkpoint`。

本轮有两个前瞻性科学改变，必须显著披露：
1. C/E共同从X2000权重出发，使用新的优化块；E增加第4节的窗口级超额平方误差项，C不增加。
2. **将合格开发模型的有限TRAIN仿真提前到FULL训练之前。** 这是新的开发可行性pilot，不是旧FULL/CALIB selected-policy benchmark的完成。不修改旧门、不豁免当前失败模型、不偷用DEV/F；正式 `selected_checkpoint` 仍保留给未来完整family的原CALIB选择。

只有G3的原四组联合门全部通过后才允许运行学习策略sim；不把当前仍FAIL的X直接送WBC。两臂有效FAIL则停止本假设，不追加系数/训练或执行sim。G0/G1无真实语义冲突且通过后自主开训；合格后自主进入本文件规定的TRAIN sim，不再等例行GO。

**本Goal终点是具有M64保持与有限TRAIN闭环证据的大规模初始化候选。此Goal不启动FULL、CALIB、正式selector、DEV-E/F、DAgger、真机或PR merge。** PI0250对合格K/X的FULL条件未成立，不是可转用于任意新配方的空白预算。

## 1. 审核依据与证据边界

必读来源：
- [最新终态5857965791](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5857965791)；PR pointer5857967492。
- PI0250授权5855202629，commit `8501ffcf66ebec423e732abeb99131e0533bd63f`，`docs/protocols/PI0250_STORAGE_SAFE_CONTINUATION_TO_SELECTED_SIM_V1.md`。
- PI0249协议 `17f0514e7cb091816a4323958966ac83331f58e1`，`docs/protocols/PI0249_R_ANCHORED_XYZ_PRIORITY_PAIRED_PILOT_V1.md`；K/X旧随机规则和原64曝光表。
- PI0248-R1终态5852051721、原tau异常5850988664；不重算其历史为新PASS。
- PI0226 native持续参考/仿真canonical5733559806及对应原始artifact；PI0250第8节的原任务约束。
- 在审核ref8501ffcf中读取的 `scripts/pi0247_u_rollout_overlay.py::_physical_losses`、`scripts/pi0243_staged_learning_to_sim.py::_screen_metrics`；已知 `scripts/pi0247_m64_screen.py::inputs/group`、`pi0248_rotation_cpu.py`与当前实际K/X执行适配器由G0绑定。

最新终态发布于2026-09-27 17:09:10 UTC，即2026-09-28 01:09:10 SGT。随后Issue增量读取无更晚记录。PR元数据仍open/draft/unmerged。本GPT没有SSH、live qstat、远端容量测试、HF权重下载、原256 NPZ重算、模型加载/训练或仿真；只核对了来源、相关底层代码，并用公布数字重新计算各门真假和差距。最后Corelab SSH超时不是已完成CPU复算被证明失败，也不是可假装最新远端核验成功。

本轮可读的相关PI0249/PI0250分支指向协议提交；终态给出远端报告路径/hash但未提供完整实际执行代码绑定。不能据此断言代码不存在于其他分支，也不能把旧底层代码当最新staged运行的全面审计。G0只需补齐实际源码/命令/收据索引，不再重建全部证据体系。

### 1.1 已公布final结果与审计后的含义

| 模型/组 | REJECT | XYZ median m | XYZ p95 m | SO3 median rad | 剩余失败 |
|---|---:|---:|---:|---:|---|
| K all64 B0 | 0 | .020316 | .043039 | .016014 | median、p95 |
| K all64 B1 | 0 | .020317 | .042327 | .017308 | median、p95 |
| K old16 B0 | 0 | .016364 | .030553 | .014834 | 无 |
| K old16 B1 | 0 | .016761 | .033201 | .017294 | 无 |
| X all64 B0 | 0 | .018867 | .040995 | .018293 | p95 |
| X all64 B1 | 0 | .018385 | .039688 | .019154 | p95 |
| X old16 B0 | 0 | .014882 | .028264 | .015689 | 无 |
| X old16 B1 | 0 | .016824 | .033709 | .019154 | 无 |

X的p95距离35mm尚有5.995/4.688mm，需要相对当前p95再降低约14.62%/11.81%。不是舍入误差，也不是完全没有学习。当前零REJECT和SO3通过只说明模型输出层的冻结测试过关，不是实测WBC tracking或生产硬件安全保证。

old16通过不证明其每个窗口都≤35mm；原p95允许极少数较大值。仅依据64项和16项的linear quantile，可推导每组all64至少4项超过35mm、old16至多1项超过35mm，所以new48至少有3项超过35mm；这只是数学上下界，具体数量、身份和原因必须读逐窗结果，不能编造“固定4个坏样本”。

跨窗口p95与H40第35–39点的时间尾段是不同维度。不得把旧R2的时间尾段机制直接移植到当前X。更早C/R数组未裁剪的结论，也不能未经新数组检查就宣称当前X训练/评估均无饱和。

本轮64仍是同一视觉到达任务的TRAIN状态/目标支持，不是64个不同技能；两组noise不是独立训练重复，old16是64的重叠子集。不断使用这些TRAIN结果选择开发方案属于明确的开发调参，不是未见集泛化证据。

## 2. G0：一次性窄核验，不能再陷入清理与整库审计

### 2.1 输入权重与真实来源

唯一新训练初始化：PI0249 X-final2000。
- 旧发布repo `leoyuer/pi0246-gate-pass-campaign`；
- 固定revision `c289992952c08161b8c37e215ceef98b6cb38b70`；
- 完整tree SHA256 `e8c8ac97f6b455204decc93ec753ab90e0bbe8f380260488fdfdc20c50b86467`；
- 已报告9文件、20,882,646,355 bytes。

实际path从durable receipt解析。若按PI0250已迁移到备用repo，允许使用原完整逐文件hash相同且有old→new映射的固定新revision；不凭repo名称或latest选择。先验证当前可恢复性。模型初始化仅加载权重，optimizer/scheduler本阶段重新创建；祖先X的posthoc收据、原始callback-incomplete审计与恢复来源完整保留。

保护：X2000、K2000、原U-M16合格里程碑、必要祖先和唯一完整证据。scratch已报139.3GiB，quotaUNKNOWN；不新增大清理。接手复查作业owner和实际高水位，目录使用现有 `/scratch/e1192548/Projects/go2-arx5` 布局，旧兼容链接不移动。

### 2.2 终态报告与训练身份检查的范围

读取并复用：
- `/home/liuyu-nfs/Projects/go2-arx5/PI0250/evidence/PI0250/PI0249_KX_PAIRED_VERDICT_20260928.json`，SHA `e444dcde07e8a15285fd9a0e0ffa79a5084562a0d6717850c59489828beb1f8b`；
- K/X screen SHA `3707da9e6b9822f878c84298953300b08b8b1aaa88b807487e6981d71cf5640a` / `b49840c1827579467643a572d51ef42272396d0d0a08b9f04f87ec83222f55f2`；
- K/X CPU replay SHA `ec129a0eec71e9c45c3437aabcbd63d348eb1fe5d3174b32a1636543931c3662` / `e56384f35a2a6186ca94fac9437383dbd1a2d40d5e74b049c444ddfc7c2c4c6a`；
- X pair receipt SHA `2106860f734aa35f33f1b7fab95a058f1bbb9881d0fdff30b0724ece14774b59`；原raw audit SHA `49ab5f12400bdd6b0299b3b0dd85fbea1ce23739008d5cd8114ca91bdf454295`。

发布小型 `pi0251_source_index.json`：上述文件可读位置/固定revision、bytes/hash、实际训练/恢复/收口/screen/CPU命令和执行源码hash/commit或补丁。不得通过先修改代码来倒推原运行内容；独立worktree补交实际源码即可。

特别记录：K/X的 `POSTHOC_CPU_REPLAY` 不是在线pre-backward防护证明。核对它比较的是已保存的实际输入/tau/epsilon/stock tensor记录，还是仅从key重生的期望值。前者可支持事后随机输入一致；后者只证明重放算法一致，不能补出历史实际消费事实。使用 `observed_posthoc` / `expected_regeneration_only` / `unknown` 准确标注，不把任何一种写成online。

此程序偏差本身不撤销独立固定screen的有效负结果，也不要求重训K/X。X祖先作为共同已封存权重可以被新实验使用；不完整的旧对照归因不得传递成新实验的既成事实。若缺失是报告字段，可从原件追加修订；若真实权重、数据、目标、梯度或评分契约不成立，停止新训练，不能豁免。

### 2.3 只读定位剩余位置风险

原X128为主要对象，K128及旧R作已有对照，均不新增模型调用：
- 按B0/B1、old16/new48、既有A–D/phase列出每窗XYZ/SO3/gate、超35mm的完整ID、排名及跨noise/模型重合；报告最高值而不是只报p95。
- 用原H40位置残差分解 `mean||r||² = ||mean r||² + mean||r-mean r||²`，列XYZ轴、首点/终点、全时间误差和35–39点SSE比例；区分恒定bias与时间形状误差。
- 检查实际raw越界/decoder裁剪、专家往返与对应source标签/观测绑定。只检查原64，不重建9792数据或扫描全部历史。
- 可对已存原始/processor输入做重复/近似条件与不同标签检查；没有额外视觉embedding forward授权。相似条件不自动证明标签歧义，不删除疑似困难窗口。

这份分解可以暴露真正的标签/观测冲突；没有唯一根因不阻止下文既定实验。不得由分解结果临时调阈值、系数、sample权重或挑noise。已有CPU128/臂完整验证直接复用；只因报告字段变动，不重跑GPU评估。

## 3. 待检验假设和不作出的结论

观察支持：平均位置质量、姿态与旧16保持已经改善，当前X的资格短板是跨窗口位置分布高分位。可能机制包括困难条件欠拟合/优化资源分配、标签或条件信息不足、模型表示限制；目前不能凭汇总表区分。

均方误差本来就会惩罚大残差，不能说它“完全忽略困难样本”；但均值降低不保证有限样本p95过门。因此本轮测试一个明确的有限超额风险代理，并保留原目标续训对照。它不是CVaR估计，不直接求batch内p95，不承诺消除所有尾部。

C回答额外共同训练块的效果；E回答在此起点/数据/seed下新增超额项的增量效果。两臂均失败则这项尝试结束；不据此否定所有大模型、所有数据规模或全部采样器。

## 4. 精确目标：在同一S4生成轨迹上增加窗口超额误差

固定batch1。令p_hat_k和p_star_k是同一原物理decoder对完整S4自由生成和专家动作的XYZ，m_k是原有效XYZ mask。定义：

`q = sum_k m_k * ||p_hat_k-p_star_k||² / sum_k m_k`，单位m²；
` s = q / (0.05 m)² = 原ell_pos `；
` c = (0.035 / 0.05)² = 0.49 `；
` ell_excess = relu(s - c) `。

两臂目标：

```text
J_C = L_native + .4*ell_pos + .1*ell_delta + .1*ell_rot
J_E = J_C + .4*ell_excess
```

- C是本PI0251的原X续训对照，不是PI0248的C；family使用 `PI0251_C_X_CONTINUATION_M64_2K_V1`。
- E family `PI0251_E_WINDOW_XYZ_EXCESS_M64_2K_V1`。
- ell_delta沿原.05s/1m/s相邻残差；ell_rot沿PI0248的物理row-rot6d decoder矩阵弦距、theta_ref=.05rad；native全部原有效10维不改。
- q等于原 `_screen_metrics` 中XYZ RMSE的平方（全40点有效时），不是对120个标量再多除3；不得只用末端或max-point误差。
- 超额项以**当前训练step新生成轨迹的整窗误差**决定是否激活，不读取B0/B1 screen或预先按成绩选择source。原64全部保留均匀曝光。
- s<.49时新增项及梯度为0；s>.49时对位置误差的标量导数从.4升到.8；精确边界采用原torch ReLU的0次梯度，不添加平滑或直通估计。目标连续但导数分段。
- .4是本轮固定试验强度，不是最优值；不得扫.2/.8、改35mm或降低旋转系数。它并不保证p95下降，更不能保证噪声泛化。
- ell_excess与原position来自同一未detach tensor；只增加标量运算，不增加第二次rollout、额外噪声或额外模型backward。保留S4四步完整梯度与原重计算。
- expert仅在监督loss内出现，模型condition/prefix/RTC/状态注入均不得使用expert或target XYZ。原clamp、statistics、相对frame、H40/dt和推理解码不变。训练loss阈值不替代SafetyGate或task scorer。

## 5. G1与正式训练：成熟入口、小改动、实际检查

### 5.1 共同配置

两臂各2000个新的canonical optimizer steps，都从上述同一X2000 weights开始。新AdamW/cosine2000/warmup10、LR1e-5、WD1e-5、betas(.9,.999)、eps1e-8、maxgrad1、batch1/accum1/workers0；原trainable/frozen名称/shape/dtype、autocast、processor与S4固定。

原64/2000曝光表按内容hash复用，48窗各31次、16窗各32次，不把额外曝光指定给old16或困难窗。不得新加DataLoader shuffle、改图像增强、从C终点启动E或换更大视觉骨干。

共同step RNG沿PI0249已测隔离方法，新key写死为ASCII：

`PI0251|22301|stage=M64_CE|optimizer_step=<本phase十进制i，无前导零>`

`seed_i = int.from_bytes(SHA256(key).digest()[:8], 'big') & ((1<<63)-1)`。

key不含arm、job、attempt、时间或结果。隔离覆盖实际native forward、同noise S4与backward；原CPU Beta/Gaussian/cast路径不变，重计算不重新递增key。保存CPU/本作业CUDA、numpy/python、optimizer/scheduler及消费cursor等实际恢复需要的状态；新stage不是旧X optimizer的exact延长。

### 5.2 仅必要CPU回归

在本地/Corelab复用已有真实entrypoint结构：
- 新项权重设0时与原X目标数值及梯度一致；非饱和合法fixture中 `J_E-J_C=.4*relu(s-.49)`，阈值两侧梯度/有限差分正确，原mask/分母/单位不变；在边界验证规定subgradient，不用有限差分要求不可微点平滑。
- 以已存合法decoded数组确认q的CPU计算与旧XYZ RMSE平方一致；不加载项目模型。原rotation镜像和U链式梯度测试按身份复用。
- 同key及1000点恢复后的样本/tau后缀一致、外部随机状态恢复；故意注入错误source/key/tau/epsilon，**实际将被Trainer调用的检查函数**必须拒绝，而不只是另写一个不会执行的fixture。
- 用合成完成状态走实际finalizer/sidecar路由：新family/card/路径在1000、2000点正确，缺收据返回准确状态，不新造旧兼容PASS。

G0/G1通过后可自行启动；不重做整库测试或单开H200调试作业。

### 5.3 C→E与真实在线身份

C先完成。C每步保存真实输入身份、key、tau（实际值/原dtype身份）、epsilon/stock噪声hash、loss各项/超额是否激活、mode与optimizer cursor；比较依据是实际捕获，不是key重生的期望值。

E每步与C同step核对source/固定输入、tau、epsilon和stock初始噪声；训练前先核source/key，随机采样后最迟在该步backward/optimizer前拒绝不一致。检查要记录实际执行事件和允许的当前step，与optimizer更新记录关联，不能最终只给POSTHOC替代。不要要求经过trainable projector的features、模型输出或梯度跨臂相同。

复用成熟梯度/mode记录。第一正式更新在预算内验证全部四步生成图的有效梯度和实际loss组成。新增loss不需要再做额外GR00T梯度诊断。必要的原S4无梯度parity最多共4个完整rollouts，单独计数，不能产生隐藏评分候选。

保存两臂1000/2000完整Trainer状态；1000只恢复，不screen、不晋级。原K/X不重训。每臂独立完整性与新跨臂receipt分开，失败不靠篡改旧schema解决。

## 6. G2/G3：只做两个final的原256格

共同X2000的128格在G0身份确认后直接复用。新增：

`C2000/E2000 × 原64 × 原实际M64 B0/B1 =256 cells`。

使用成熟screen/decode/scorer，新增外层本stage身份映射；不得把新card常量塞进旧card覆盖。推理视图固定revision/full inventory并绑定原processor/stats；HF不可用但有验证过的完整持久源时沿PI0250真实local-durable适配，不伪造HF PUBLIC。

Corelab优先，C完成可先做其128格，与E排队/训练并行；C的质量不得改变E配方或决定是否执行E。每个有效cell first-valid保留；报告/传输异常只读修复，不重采样。

独立CPU从保存数组复算各128格，覆盖source/noise/模型身份、全H40 finite、metrics、gate及issues。不能把重新读取同一summary当独立复算。

每臂final2000的四组：all64 B0/B1、old16 B0/B1，分别要求：
- 原64或16集合完整、全40点、原mask、全部有效；
- 原SafetyGate REJECT=0，WARN保留；
- per-window XYZ RMSE median≤.020m；
- NumPy linear p95 XYZ RMSE≤.035m；
- per-window geodesic SO3 RMSE median≤.050rad。

不池化noise、不把old16当额外独立样本、不对阈值先四舍五入、不删最差source。new48/A–D/phase与超35mm数量/最大误差均报告但不新增资格门。分析C/E相对共同X及E−C，分别呈现位置、姿态、速度和旧能力。

全部完成后：仅一臂通过即冻结该开发候选；均通过优先简单C，同时报告E也通过；均有效FAIL即 `PI0251_XYZ_EXCESS_HYPOTHESIS_EXHAUSTED — STOP_FOR_PI_NEXT_HYPOTHESIS`，不sim、不加步数或改阈值。incomplete、invalid或配对失效不是科学FAIL。

若通过，生成 `PI0251_M64_DEVELOPMENT_CANDIDATE_V1`，固定权重、完整祖先、loss、源码、processor、S4、输入/noise与四组结果。在第一次sim查询前写死唯一 `development_checkpoint`。`selected_checkpoint=NONE`，`formal_calib_selection_done=false`。不得试两个候选后按sim成绩再挑。

## 7. G4：本轮新放行的TRAIN仿真pilot（FULL之前）

### 7.1 目的、准入与原语义绑定

这是验证大规模初始化候选是否已能与冻结WBC形成有限TRAIN闭环的开发pilot。它不能证明未见任务泛化或正式FULL selector成功。**仅G3全部资格成立且新配对/完整性有效时允许运行学习策略。** 当前X2000和其他旧FAIL不能借本节直接绕门。

训练期间可在CPU准备绑定，不阻塞训练：机械恢复PI0226 canonical5733559806及原封存的TRAIN A–D C0场景、场景seed、模型query RNG规则、WBC源码/权重、state/frame/TCP、原持续reference API、cadence/prefix、物理时钟、tick与query cap、termination/cleanliness和world-TCP scorer。引用原实际命令与artifact hashes。

不混用更早world-target未成立或StructuredAction实验；不从README猜默认值。若不能唯一恢复，记录具体 `SIM_BINDING_CONFLICT`，只阻塞sim，保留已经合格的模型。若原完整episode所需查询超过每任务64的本轮上界，sim前返回该冲突，不截断episode后声称同benchmark。

本pilot采用恢复出的原场景和随机规则，并在首查询前封存完整运行card；没有择seed、择初态或试跑到成功。每次模型调用保存实际noise tensor，不仅seed。S4四步、H40、dt=.05、原decoder/SafetyGate不变。

### 7.2 环境控制与one-chunk

允许最多4个原A–D native expert控制episode；完全同runtime身份已有效的控制可复用。控制不用于训练或向模型灌expert参考。控制失败先处理实际环境/绑定问题；修复若改变科学controller语义则返回PI，不能修改WBC让模型过。

在控制及身份成立后，A→B→C→D C0各一次one-chunk，最多4个学习模型查询：真实RGB/StateV1 → 已冻结development_checkpoint → 原S4/dense-relative decode → 原SafetyGate → TrajectoryBatch → frozen WBC。gate REJECT不下发、不重采样；合法失败保留并停下一阶段。

成功按原achieved world-TCP到任务目标≤.035m、原termination/cleanliness及完整运行定义，不是训练RMSE，也不是reference自行命中。记录reference质量与achieved-minus-reference tracking，区分“生成参考不到目标”和“参考对但执行未跟上”。这种诊断用同次轨迹/状态，不增加优化或模型调用，不自行设置新的成功标准。

4/4一次性通过才进入receding；不因为单次one-chunk失败就擅改为长episode追成功。

### 7.3 Progress-preserving receding

A–D各一次，按原更严格episode/cadence界限，每任务最多64次模型query，总≤256。每次查询读取上一次实际执行后的fresh RGB/StateV1，保持reference进度和时钟连续。每个task只在开始时按原条件初始化，不在task内部重置episode/reference/robot来消除失败。

禁止反复播放同一预测冒充反馈、teacher前缀、target XYZ进入模型、RTC、平滑/blending/retiming、改变WBC/gate、换候选或噪声、DAgger。模型和IsaacGym环境沿成熟分离方式运行，允许原同步仿真控制时钟，不宣称满足真机实时性。

保存每次输入/noise/输出/gate、reference/实际TCP、first-hit/min/final距离、tracking、时钟/更新、终止原因及video。全部任务按原标准完成才PASS。科学失败保留唯一候选及结果并停 `PI0251_TRAIN_SIM_FAIL — STOP_FOR_PI_CONTROLLER_OR_POLICY_REVIEW`；不能用C/E另一臂再试、扩大任务时长或重新训练。

### 7.4 成功交付与名称

通过后：
`PI0251_TRAIN_SIM_READY_FOR_LARGE_SCALE — STOP_FOR_PI_FULL_RECIPE_FREEZE`。

交付 `large_scale_init_checkpoint` 完整weights/Trainer归档、配方/源码/runtime、M64四组门、固定TRAIN sim原件/video、真实随机/恢复规则。并显式写：

```text
m64_joint_gate_pass = true
train_sim_pilot_pass = true
formal_calib_selection_done = false
selected_checkpoint = NONE
generalization_demonstrated = false
hardware_admitted = false
```

该模型适合作为**下一次受控大规模训练的候选初始化**，不是保证在9792或未见任务上成功。下一PI决定FULL预算/配方；原正式selector仍只能在完整family上运行，本pilot不能冒名替代。此处停下是交付用户要求的规模训练前checkpoint，不是重新回到纯准备状态。

## 8. 工程自主恢复、资源和存储

沿成熟路径做薄适配，不再新建总调度/验证平台。先检查真实job owner和staged来源，健康运行不原位改文件，后继新目录与新工程子card追加，不覆盖旧失败。

本地PC：SSH控制/代码/小CPU回归；Corelab：证据索引、CPU验证、无梯度screen及sim；大显存可微训练沿已验证软件和设备能力走PBS auto、project CFP03-CF-004，不pin某台host，不用H200试普通路径错误，不重复已证实4080S全训练OOM，不擅改精度或世界大小适配资源。

storage事实为最新已报139.3GiB、quotaUNKNOWN。接手只核必要输出高水位和当前可恢复源，足够就不再清理；不因“archive”或“FAIL”名字批量删除。保护现有唯一原件、X基线及新C/E完整1000/2000，沿固定revision全内容读回/已核验持久本地接收端，只有当前可恢复后才退休本地副本。旧兼容链接和作业引用保留；不把tmpfs或同scratch第二目录当持久备份。

保留PI0250已授权的主HF repo和备用 `leoyuer/go2-arx5-checkpoint-archive-v1`；读/写/匿名公开三种证据分开，不扩大账号权限，不输出token。HF发布异常不改变科学结果；安全持久输出存在时可继续独立计算。权限或SSH确实不可用时准确阻塞依赖子任务，不能伪造远端验证。

工程修复与完整状态恢复不设任意次数上限；按PI0250已允许的规则，从最后完整checkpoint重算丢失且无有效封存结果的相同step后缀，raw实际更新全部累加，canonical终点不延长。不重训已完成有效模型，不为更好成绩重做科学失败，不新optimizer冒充stage内恢复。无法闭合身份/配对/状态则停科学完整性门。

完成训练后仅callback/报告/上传失败，优先只读收口或重传；原posthoc兼容receipt不得写成历史实时证据。邮件保持关闭、不建monitor、不扩大scratch/home整理、不merge。

## 9. 预算与最少交付

| 阶段 | 新canonical更新 | 新项目调用上界 |
|---|---:|---:|
| G0/G1旧证据与CPU fixture | 0 | 0；无GR00T加载/梯度诊断 |
| C/E训练 | 2000+2000 | 每更新原native＋一次4步可微S4；重计算另计 |
| 必要非选分S4 parity | 0 | ≤4完整rollouts |
| 固定screen | 0 | 256完整S4 queries；X基线128复用 |
| TRAIN expert环境控制 | 0 | ≤4episode，无学习模型query |
| 学习策略one-chunk | 0 | ≤4 |
| 学习策略receding | 0 | ≤256 |

新canonical训练最多4000，离线评分256，sim模型查询最多260，非选分parity最多4。各阶段条件不成立则不花下游额度。实际native/head/recompute、工程失败及重复恢复量分开实记，不能把理论4×步数当观察日志。CPU标量/解码数学测试不是GR00T梯度查询。

最少交付：一个可读源码/证据索引和G0误差分解；新目标与在线检查CPU证明；两张sealed cards和共同表/key；两臂完整模型及实时/事后分离收据；256格和独立复算；唯一开发候选；条件sim binding、原件/video和large_scale_init_checkpoint，或具体科学/工程停止原因。

没有新增数据集、CALIB/DEV/F、额外solver、困难例抽样、更多loss系数、骨干解冻、WBC训练、真实硬件或FULL训练权限。旧K/X已经终止的假设不会因新goal重复开跑。

## 10. Codex启动正文

执行 `PI0251_XYZ_EXCESS_RISK_TO_TRAIN_SIM_GO`。完整读取本文件及最新5857965791、PI0249/PI0250协议。

接受K/X all64有效FAIL，保留old16四组PASS、全部零REJECT与姿态进步；不重训K/X，不再清理已足够的scratch。G0复核X2000固定revision/tree与实际代码/报告来源，区分旧POSTHOC_CPU_REPLAY到底比较了实际记录还是期望重生；不伪造online。只从已有NPZ定位跨窗口>35mm、old16/new48、bias/时变/饱和，不改sample或阈值。

新C/E同起点X2000，新optimizer，各2000，原64/2000曝光表、LR1e-5、原S4/完整梯度/参数精度/decoder不变。C保持native+.4pos+.1delta+.1rot；E仅加.4*relu(ell_pos-.49)。ell_pos是整窗物理XYZ均方欧氏误差/(.05m)^2，不多除3、不用最后一点、不读取评估noise来挑样。

CPU数值/梯度与真实入口在线检查fixture通过后自动开训。按PI0251共同step key记录实际sample/tau/epsilon/stock；E在backward前比对C，不再只在终点声称检查通过。首更新及所有重算计真实预算。保存1000/2000，1000不screen。

只新C2000/E2000×64×B0/B1=256格，X原128复用。两臂都完成并独立CPU复算，四组原四门不变。均过优先简单C；单臂过固定该开发候选；两臂均FAIL停新假设，不追加。

本PI明确允许M64合格候选在FULL之前做TRAIN sim pilot。先封唯一development_checkpoint和原PI0226场景/WBC/frame/cadence/clock/scorer绑定，最多4expert控制；A–D各一次one-chunk4/4过后再progress-preserving receding，各≤64query。fresh观测、原gate和实际world-TCP .035m/cleanliness不变；失败不换候选/seed、不调WBC、不DAgger。

成功交付large_scale_init_checkpoint及完整TRAIN sim证据，停FULL配方冻结；selected_checkpoint仍NONE，不冒称正式CALIB选择或泛化。FULL/CALIB/DEV/F/真机/merge未放行。工程小修和同状态恢复自主，raw计算全部记账；Corelab先成熟代码/screen/sim，大显存训练才PBS auto。邮件/monitor关闭，Issue #11追加canonical，PR #77只pointer。
