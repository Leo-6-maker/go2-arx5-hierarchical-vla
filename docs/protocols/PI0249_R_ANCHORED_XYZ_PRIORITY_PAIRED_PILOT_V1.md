# PI0249 — 从R2000出发的位置优先对照：保持姿态，检验位置误差能否过门

日期：2026-09-27（Asia/Singapore）。

Authorization: `PI0249_R_ANCHORED_XYZ_PRIORITY_GO`  
Decision: `GO_G0_G1_THEN_AUTONOMOUS_K_X_2K_AND_FINAL256`  
发布状态：`AUTHORIZED_NOT_EXECUTED_BY_GPT`  
审核执行ref：`317a2e85419cacd396f4f989a4ce83f4776bb85b`。  
Issue #11为canonical；PR #77仅pointer，保持未合并。

## 0. PI决定与本Goal终点

保留PI0248-R1的C/R final2000模型质量双FAIL，以及原 `STOP_UNPAIRED_NATIVE_TAU_IDENTITY_MISMATCH`。不放宽原门，不把R2000称为合格模型，不把旧1000点改成晋级点。

下一问题限定为：**从已经达到姿态与零REJECT要求、但位置尚未合格的R2000出发，在相同新训练块内，保持原目标继续优化，还是仅增强自由生成XYZ位置监督，能够达到原M64及old16四组联合门？**

明确放行以下完整链路：G0旧证据/执行源码窄收口与零模型误差分解 → G1增量目标及恢复安全随机流测试 → K/X各2000新更新 → 两个final2000的固定256格及独立CPU复算 → 候选/负结果终态。G0/G1通过即自主继续，不需再次等待PI确认；真实身份/语义冲突、训练完整性失败或预算用尽则按本文件停止。

K保持R目标；X仅将XYZ位置辅助项系数由0.1改为0.4。两臂都从同一R2000模型权重开始，使用新optimizer/scheduler、同一原64/2000曝光表与本轮共同的step-addressed RNG。RNG改变是显式共同设计，不冒称旧R训练的exact续跑，也不修复旧C/R配对。

这是一次有固定控制臂的假设测试，不是loss/seed/LR搜索。通过后只形成M64 fit-recipe候选并停止FULL准入；没有FULL/CALIB/selector/WBC/sim/DEV/F/DAgger/真机/merge授权。

## 1. 审核依据、结论与边界

### 1.1 必读来源

- [PI0248-R1终态5852051721](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5852051721)，[PR pointer5852052190](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/77#issuecomment-5852052190)。终态发布时间为2026-09-27 02:44:12 UTC，即10:44:12 SGT；本次随后读取的Issue增量没有更晚记录。
- [PI0248-R1授权5851408869](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5851408869)及不可变协议commit `2f75aec82f9d0fe4d72c154615f339f50c7eb41c`，路径 `docs/protocols/PI0248_R1_AS_EXECUTED_QUALIFICATION_AND_SCREEN_RELEASE_20260927.md`。
- 原PI0248协议commit `81e3faa4eb23e6ce0ed521e6b8dd5b7206506771`，路径 `docs/protocols/PI0248_M64_POSE_REPAIR_PAIRED_PILOT_V1.md`。
- 原配对STOP5850988664及CPU RNG审计5851180653/5851224835/5851302733；这些不是本轮配对PASS。
- 审核ref下 `scripts/pi0248_cr_screen.py` 的scope准入、screen、cpu_replay；`scripts/pi0247_m64_screen.py` 的inputs/group；`scripts/pi0248_cr_train.py` 的初始化/曝光表/rotation adapter；`scripts/pi0247_u_rollout_overlay.py` 的native采样捕获、S4、_physical_losses及forward；`scripts/pi0245_physical_delta.py::decode_xyz`。
- 沿用并在执行时核对 `scripts/pi0248_rotation_cpu.py`、原冻结processor/统计与GR00T固定source `51d4c89f72fda44cbf77285c6a8114b52676b8a1`。

GPT本次读取了GitHub账本、上述相关源码与分支/PR状态，并按公布数字独立重算八组final门的真假。没有SSH、live qstat、HF大权重下载、原512个NPZ的独立复算、GPU训练或新模型查询。以下CPU复算PASS属于执行侧报告；G0必须补齐可访问的原件和实际执行来源。不能把本次文本审核写成GPT亲自验证了远端tensor。

### 1.2 八组final门复核

| 模型 | 集合/noise | REJECT | median XYZ m | p95 XYZ m | median SO3 rad | 未通过项 |
|---|---|---:|---:|---:|---:|---|
| C2000 | full64/B0 | 1 | .019577 | .044023 | .102845 | REJECT、p95、SO3 |
| C2000 | full64/B1 | 1 | .018398 | .045389 | .103933 | REJECT、p95、SO3 |
| C2000 | old16/B0 | 0 | .015481 | .032828 | .104028 | SO3 |
| C2000 | old16/B1 | 0 | .017025 | .039145 | .096504 | p95、SO3 |
| R2000 | full64/B0 | 0 | .023054 | .050053 | .019809 | median XYZ、p95 |
| R2000 | full64/B1 | 0 | .023522 | .049620 | .019723 | median XYZ、p95 |
| R2000 | old16/B0 | 0 | .019223 | .035828 | .017517 | p95 |
| R2000 | old16/B1 | 0 | .019169 | .041233 | .019457 | p95 |

原门为REJECT=0、median XYZ≤.020m、linear p95 XYZ≤.035m、median SO3≤.050rad。公布数字与双FAIL一致。R的full64 p95分别超出15.053mm、14.620mm；old16分别超出0.828mm、6.233mm。不是统一的四舍五入临界失败，不能用旧16/B0接近阈值来豁免其它组。

R1000→R2000的full64：B0 median .029240→.023054、p95 .055473→.050053；B1 median .032218→.023522、p95 .065027→.049620。R1000的full64 SO3已经为.030270/.033556，且零REJECT。旧1000训练前缀的配对证据有信息价值，但不晋级、不外推普遍因果显著性。

据此选择R2000作为本轮开发起点：其已观察到的剩余门集中在位置，而C尚有明显姿态和REJECT问题。这不证明唯一原因是位置loss权重不足、梯度冲突或训练不足；K用于检验额外训练块，X用于检验位置监督的相对权重。两noise并非独立训练重复，64均为训练窗口，old16是重叠子集。

## 2. G0：旧证据与源码一致性窄收口；零新模型调用

### 2.1 本次发现的具体待核对项

审核ref的 `pi0248_cr_screen.py::check_as_executed_evidence` 已将新收据hash写入 `integrity['_integrity_file_sha256']`，screen和推理视图使用该值；但 `cpu_replay()` 末尾生成result时仍使用：

```python
"training_integrity_sha256": sha(metadata_root / "pi0248_training_integrity.json")
```

该路径指向旧命名文件，与本轮新scope存在潜在不一致。尤其R旧完整性PASS未写出，不能仅从仓库这段代码推导“R新收据已正确写出”。但真实运行可能使用独立CPU验证器、隔离metadata视图或尚未推送的窄修复；因此本PI**不据此直接指控分数错误，也不撤销已有负结果**。

执行agent须提供实际CPU命令/脚本路径、源码hash和commit或补丁、metadata_root解析结果，以及四份cpu_replay的实际 `training_integrity_sha256` 对象。核对它们是否分别绑定C/R as-executed收据，并保留旧scope与deviation。若用的是另一个独立验证器，应提交其实际源码及执行证据，不能只把仓库代码修到“看起来能跑”后倒推旧运行。

若只是末尾收据字段/报告保存错误：在新输出位置修为当前scope的真实hash并从**已有NPZ**重做CPU报告，保留旧文件、旧hash及old→new映射。不能伪造R旧PASS，不能为获得新hash重跑GPU cells。若发现实际模型/输入/noise/评分语义不一致或缺失原始结果，停止为 `PI0249_BLOCKED_BASELINE_EVIDENCE`，不要启动K/X。

这项核对可以自主完成；纯元数据缺陷修复后不需要另一个PI审批。

### 2.2 原始证据索引

一次性发布小型、不可变的 `pi0248_r1_evidence_index.json`：四份screen_report、四份CPU复算、两份as-executed收据、deviation、实际执行源码、128唯一source-noise身份交叉核对及cell文件清单。每项有可定位的路径/固定revision、bytes和SHA256；只列hash而没有可读位置不算完整交付。已有大NPZ保持原保存位置，索引绑定其逐文件hash；必要时沿既有HF evidence通道归档并读回，不建立新平台。

关键旧hash必须核对，不覆盖：

| 对象 | SHA256 |
|---|---|
| C as-executed integrity | f96ff07585f3be61af4c34a3da53e0d5ae0d6b5ba451d11e938b27a0834406bd |
| R as-executed integrity | ab5dd8b62335cc22cf00492c8ec627d0c43b32a4b568dde8fe2bcda27b7e9611 |
| deviation | 4b97f8effdb9a45b5ee6abc8b3850baf296a63ce1646334a3d2bce0826e598f7 |
| C1000 report / CPU | e0fdadaf5067d08243d6f9df78e10d2d03fa89cbb898b7aa6c51415b22577aa5 / 8272fb2e581329434c1d12ce78ca0dc5db8250000f596ddc3a14b53b0859e9a5 |
| C2000 report / CPU | 8020ffc7c57cfc865ccebf32b9b1de124849cae4b4807806dd9c91be9d4d79ba / 7fef451a5b72f65d410f32405e706d77c4292d9996933c5e68b7db7f4fa1a559 |
| R1000 report / CPU | f9bae00b4a7259866ccbe26317079cc3730e8b47addde02b8f8e6dd25663f713 / 67560422311455e87cd14ea8cb3b8307889f98d6b04b2c3846331b43a4012d66 |
| R2000 report / CPU | 21bcb08a54190a7c078eb4aceacc460f84714c11a67a9f9d480379cce8073fcd / a875b4a7df6a26ccca4b9f6ddd2d7c3163e1c37d033799c0f6a5fde7060ddccb |

若报告只能做来源可追溯的修订，新hash另列，不强行让新文件匹配旧hash。既有512个first-valid逻辑cell和原C/R4000实际更新仍永久计数。

### 2.3 位置误差分解，不变成第三个实验

从已有R1000/R2000为主、C及共同起点为辅助的数组与日志，输出一个 `pi0249_position_breakdown.json`。不用新模型forward/backward，不读CALIB/DEV/F。

按noise、old16/new48、既有A-D/phase列出XYZ分布、超过35mm窗口的完整source ID及同ID变化、SO3与全部gate原因。对H40残差r_k计算恒等分解：

`mean_k ||r_k||² = ||mean_k r_k||² + mean_k ||r_k-mean_j r_j||²`。

同时列出各轴误差、H0–39曲线、既有35–39点SSE比例、线/角速度和inverse-normalization饱和情况。这里的“跨窗口p95尾部”与“轨迹末端35–39点”是两个不同对象，不能混称尾部修复。

源码decode_xyz在minmax路径使用clamp(-1,1)。报告raw XYZ越界比例、对应物理误差及可由CPU decoder Jacobian核验的零梯度坐标；不能把提高系数当成能穿过clamp的保证。CPU decoder数学/梯度检查不属于项目模型反传；未保存的模型参数梯度冲突标为UNOBSERVED，不补做真实梯度诊断。

该分解用于解释和发现真实契约冲突，不用于选择新seed、调0.4系数、重排/加权困难样本或新增隐藏门。没有确定唯一根因不是停机理由。若真实标签/decoder/统计冲突使比较无效，则停受影响实验；不能借G0临时换科学路线。

## 3. 固定起点及两臂唯一目标差异

两臂共同起点为旧R2000：

```text
HF repo: leoyuer/pi0246-gate-pass-campaign
revision: d089c09380c568e720c3dfa6bec5aa064acd7c4f
full checkpoint tree SHA256:
4422a24d2117206681c423a0f8d3137dff7eb614c990cb0334a8cdaf73900298
```

路径从这个revision对应的durable receipt解析，不凭latest或文件夹名称猜测。完整Trainer原件永久保护；新阶段初始化只取已核对的model/config及原冻结processor/statistics，不载入旧optimizer冒充续跑。K与X独立从该同一权重开始，X不得从K终点开始。

沿用原完整S4自由生成结果a_free、原native flow loss及原所有active0:10的监督：

```text
J_K = L_native + 0.1*ell_pos + 0.1*ell_delta + 0.1*ell_rot
J_X = L_native + 0.4*ell_pos + 0.1*ell_delta + 0.1*ell_rot
J_X - J_K = 0.3*ell_pos
```

Family：`PI0249_K_R_CONTINUATION_M64_2K_V1` / `PI0249_X_XYZ4_PRIORITY_M64_2K_V1`。这里XYZ4指本文件的位置辅助项四倍权重，**不是旧PI0244名为XYZ4的其它实验或坐标变换**；run card须写全family与目标，禁止按旧名字复用错误recipe。

ell_pos严格沿 `pi0247_u_rollout_overlay.py::_physical_losses`：有效点上三维欧氏平方和除以(.05m)²，再除以有效点数；不是再除以3，不是只看最后一点，也不改为p95/Huber/CVaR。

ell_delta沿原相邻位置残差差分，dt=.05s、v_ref=1m/s，原有效transition mask。ell_rot沿PI0248原可微真实decoder镜像和row-rot6d→SO3、theta_ref=.05rad的chordal surrogate：有效点平均 `||Rhat-Rstar||_F²/(2*.05²)`。原clip、zero-range、anchor、合法域以及部署scorer不改。

0.4是本次预注册的有限机制扰动，不是从历史成绩推导出的最优系数。代数上它相当于保留0.1但将位置参考尺度减半；实现仍保留.05m分母，只改变系数，避免同时改动单位。四倍loss系数不等于AdamW参数步长四倍，尤其原global-norm clip仍然存在。

不降低rotation系数、不加旋转保持teacher、不改native active mask、delta、LR或冻结范围；不引入动态loss balancing、分组PCGrad、困难例重采样、tail loss、平滑/裁尾/retime、padding干预或新solver。专家仍只作监督，不进入生成condition/RTC/前缀。

## 4. 共同训练配置与曝光表

每臂2000个新的optimizer updates；新AdamW/cosine2000/warmup10；LR1e-5、WD1e-5、betas(.9,.999)、epsilon1e-8、max_grad_norm1、batch1、accum1、workers0、基础seed22301。

原64源、排序和PI0248的2000行曝光表按内容身份直接复用；每臂仍为48个窗口各31次、16个窗口各32次，具体ID由原表决定。不得把额外一次人为转给old16，不叠加shuffle，不根据本次G0成绩调整采样。再次使用这张表是新phase的共同设计，不声称全局训练中样本此前未见。

H40、dt=.05、132维/active0:10、observation绑定、原两路RGB、语言、state、TRAIN statistics、embodiment与decoder保持。保存实际输入/label/mask身份，不只检查source名称。

实际trainable/frozen membership保持R原件：已报告471个trainable tensor，其中diffusion456、projector15；559个frozen tensor。须核对名称/shape/dtype与实际optimizer membership，不仅填这些计数。原参数精度、autocast、action-head mode与S4四步重计算不变；不能把部署BF16加载方式误当训练参数dtype指令。

使用原已成功的Hopper软件版本和执行链，不为了本轮升级PyTorch/Transformers/GR00T。新stage内部恢复完整optimizer/scheduler/模型状态/消费cursor及所有必要RNG状态；本轮step-addressed规则也必须一致。恢复cursor是本phase 1–2000，不是祖先checkpoint标签2000加上前向次数。

## 5. G1：先封存恢复安全的共同随机流，不再依赖一次全局seed

### 5.1 本轮共同随机化规则

本轮明确引入 `PI0249_STEP_ADDRESSED_TORCH_RNG_V1`，作用于两臂相同的实际训练step边界。每个optimizer update i∈[1,2000]的key固定为ASCII字符串：

```text
PI0249|22301|optimizer_step=<i的十进制整数，无前导零>
seed_i = int.from_bytes(SHA256(key).digest()[:8], 'big') & ((1 << 63) - 1)
```

封存全部2000行key/seed与样本表关联。key不含arm、job、attempt、机器、walltime、loss或结果；不是每步反复设同一个seed。

在batch/固定输入准备完成后、实际model training forward之前，以隔离的CPU和本作业CUDA RNG上下文设置seed_i，覆盖本次原native forward、同噪声S4及其实际backward；退出恢复上下文外的状态。复用 `torch.random.fork_rng` 等等价的最小实现，必须只隔离/触碰本作业设备，兼容已有mode和gradient-checkpoint wrapper。

原 `sample_time` 的CPU Beta(1.5,1.0)、noise_s=.999及dtype/cast路径，原native Gaussian采样的设备/shape/dtype，以及 `stock_noise = native_epsilon.to(vl_embeds.dtype)` 的关系全部保留。不得自己换成均匀tau、CPU生成CUDA epsilon、额外独立stock noise或改变四步solver。梯度checkpoint重计算仍使用原preserve_rng_state机制；不能在每个重计算forward重新递增key或重新抽样。

该规则改变了相对旧PI0248的训练随机轨迹，属于**本轮两臂相同的显式设计**。K对照回答“在这个新的共同随机化与optimizer块中保持原R目标”，不是旧R原优化器的无改动恢复。R2000的历史tau无需被新序列重现，旧C/R偏差保持不变。

### 5.2 回归要求与实际证据

先做零GR00T模型的CPU fixture：模拟实际恢复生命周期/迭代器建立/跳过已消费项/额外无关CPU随机调用；比较不中断与1000点恢复后的1001–2000 keys及原Beta/cast结果，必须完全相同，前缀不重复。测试上下文退出后外部RNG还原、同key同值、不同key实际变化、错误cursor拒绝。不能仅重复旧“从rng_state独立重放Beta”的测试。

允许在已授权H200作业内、模型加载前做小型CUDA随机tensor/context回归，不单独申请GPU调试作业，不做项目forward/backward。比较的是同一固定runtime/device上的CUDA重复，不要求CPU随机序列等于CUDA序列。

同时做CPU增量目标测试：系数回到0.1时与原R目标数值/梯度一致；在非奇异、非饱和fixture上 `J_X-J_K=.3*ell_pos` 且梯度差一致；原mask和全40点分母不变；饱和点保持原clamp梯度语义。复用已有rotation decoder测试，不重新实现它。

正式训练第一更新在预算内保留全四步有效梯度、实际loss分项、mode和membership证据。每步记录phase step/key、实际source/input、native tau、native epsilon、stock noise的值/内容hash与dtype/shape、mode、实际optimizer/global_step。

K先完整运行；X每次消费时对照K同phase step的实际样本、tau/epsilon/stock-noise及条件身份，最迟在该步backward/optimizer前拒绝不一致。不要等X全部2000完成才发现首次失配。CPU audit比较不增加项目模型调用。审计只能使用真实捕获tensor，不从seed表填“期望值”冒充实测。

冻结骨干条件特征仅在其确为相同冻结子图输出时要求一致；不要要求经过trainable projector后的特征、预测或参数梯度跨臂相同。若实际实现存在其它随机消费者，须在封存前纳入相同边界并做回归，不在看到科学结果后临时改变规则。

随机输入一致与完整数值bitwise再现不是同一声明。本轮须保证配对采样和可恢复的训练状态，不凭这项fixture承诺不同平台/库版本的权重逐bit一致。PyTorch 2.9官方文档说明fork_rng保存/恢复CPU及指定设备状态，同时说明跨版本/平台的完全可重现没有普遍保证：[RNG API](https://docs.pytorch.org/docs/2.9/random.html)、[reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html)。这是API依据，不是本项目回归PASS证据。

G1未通过可自主工程修复/CPU回归；不能拿旧as-executed豁免再次跳过新配对要求。无真实语义冲突且G0/G1通过后立即推进K/X，不继续无界追查旧CPU RNG唯一consumer。

## 6. 训练、保存与中期观察的边界

K→X串行；两臂各2000，即使K后续质量失败也不改变X配方。两臂各完整保存phase1000/2000的Trainer状态，固定2000为唯一资格终点。

phase1000只用于恢复和已有训练日志趋势，不调用screen，不临时晋级。不能因为训练loss好坏改变2000步数、学习率曲线或另加一个checkpoint评估。

目标项使用同一次native forward和同一次完整S4自由生成，保留四步全梯度。每臂第一次正式更新可沿原实现执行2个非选分完整S4 parity rollouts，合计最多4；其余复用已有同语义证明，无额外optimizer smoke或模型梯度诊断。真实parity调用独立计数，不冒充zero-query测试。

每臂完整性独立收口；另有配对receipt核对全部2000条实际同索引随机输入。旧PI0248 schema/card及FAIL不修改；新stage/run cards以本授权绑定原R2000祖先。新配对失败不得自动转为as-executed资格或静默重训；先保留失败，按实际剩余预算和无科学语义改变的恢复规则处理，无法闭合则停止PI。

## 7. 固定最终screen：新增256，不重跑旧512

只新增：

```text
K phase2000 × 原64 × 原M64 B0/B1 = 128 cells
X phase2000 × 原64 × 原M64 B0/B1 = 128 cells
合计256 cells
```

共同起点R2000的原128 cells在G0核验后直接复用。C/R旧512只可从保存数组继续分析，不新增forward；不重跑M64旧baseline、不评估新phase1000、不换B0/B1或新增seed。

训练后在Corelab优先执行无梯度screen；原 `pi0247_m64_screen.inputs/group`、D1 run_one、decoder、gate和独立CPU复算逻辑复用。增加明确PI0249 card/family/配对receipt准入，不能改旧PI0248训练card来满足入口常量。每个有效cell第一次结果即保留；合法REJECT/大误差/合法负结果不重试，工程中断仅补缺失结果。

模型从固定HF revision物化必要model/config视图并逐文件核验，原processor/统计另外绑定，不把optimizer下载给推理GPU，不经本机PC中转大权重。评估必须用原实际source/condition/noise packets，不只复用seed编号。

每臂128条独立CPU复算：完整source/noise集合、hash、全H40、finite、decoder、XYZ/SO3、gate全部issues、速度和记录身份。沿既有单元逻辑复算，不把另一个读同一汇总JSON的程序称独立tensor复核。保留全部WARN和REJECT原因，实际C/R偏差不能污染新配对收据。

## 8. 原门、候选规则和科学解释

每臂phase2000的full64/B0、full64/B1、old16/B0、old16/B1四组分别全部满足：

| 检查 | 固定要求 |
|---|---:|
| 数据完整 | 64或16个指定窗口、全40点、原mask、全部有效/有限 |
| 原SafetyGate | REJECT=0，WARN保留 |
| per-window XYZ RMSE median | ≤.020m |
| per-window XYZ RMSE linear p95 | ≤.035m |
| per-window geodesic SO3 RMSE median | ≤.050rad |

`np.quantile(...,.95,method='linear')`及原metric定义不变；不池化noise，不合并old16和full64来增大n，不舍入后过门，不删困难窗口。new48/A-D/phase是描述性分层，不新增隐藏资格阈值。

两臂完成并独立复算后：仅K过→记录K；仅X过→记录X；两者都过→预先优先未改变相对loss权重的K，同时保留X合格记录。两者都不通过→`PI0249_XYZ_PRIORITY_HYPOTHESIS_EXHAUSTED — STOP_FOR_PI_NEXT_HYPOTHESIS`。缺失/invalid/配对失败不当成有效科学FAIL。

通过终态：`PI0249_M64_POSE_POSITION_QUALIFIED — STOP_FOR_PI_FULL_ADMISSION`。候选manifest包含完整祖先R2000、旧PI0248 as-executed及配对偏差来源、新stage和RNG规则、weights/revision/tree、recipe、四组指标、独立复核与实际预算。`selected_checkpoint=NONE`、`sim_admitted=false`。候选不是部署资格，原U-M16合格里程碑和旧M64/C/R负结果全部保持。

主比较为X−K在同一新训练块内的结果，以及各自相对共同R2000的变化。若只有X过门，支持本次位置加权设计在这个起点/seed/训练集合下可行，不证明普遍优势；若K过而X不过，说明在本次实现中更强位置权重不是必要改进。即使位置改善，也要如实报告姿态/速度/零REJECT是否牺牲，不把单项降低当成联合成功。

未通过时不自动开放0.2/0.8、更多步数、tail loss、重采样、解冻或新solver。该组预注册假设完成后必须带真实负结果回PI，而不是连续扫到通过。

## 9. 预算、资源及恢复权限

新增optimizer更新上限4000，K/X各2000；旧PI0248的4000和512格、所有更早历史计数均不充值。实际失败attempt中的optimizer更新也记raw；不得通过丢弃失败目录把已消耗更新变为免费预算。仅保存/归档/报告失败且已完成更新时，修复保存/报告，不重训。

stage内可恢复完整状态及正确cursor，已完成前缀不重复。若中断后唯一可恢复checkpoint落后于已消耗更新，任何需要重复的实际优化仍占本轮4000总额度；不足以完成固定两臂即报 `PI0249_BLOCKED_UPDATE_BUDGET`，不擅自扩额或缩短某臂冒充final2000。工程修复不设任意次数上限，但没有免费更新授权。

科学评估固定新增256逻辑cell、最多4个非选分完整S4 parity rollouts。另记工程失败forward attempts、复用128、实际新增、模型load、native forward、S4 rollout、head与backward重计算，不能把期望常数当观测。4000正常训练rollouts的四步forward期望16000次head，四步重计算期望另16000次；256正常screen期望1024次head，parity另计。没有额外项目模型诊断/梯度查询预算。

Corelab用于CPU与可容纳的screen；S4可微训练沿已经验证的大显存H200路径及既有PBS project/资源配置，合法自动调度，不绑定hostname，不再重复已知4080S完整训练OOM。启动前接手检查实际job owner/产物，避免重复qsub；本GPT未查询live scheduler。

沿成熟HF写入/固定revision读回/rolling archive链，源checkpoint和推理视图分离。新K/X prefix隔离、认证写读preflight通过后启动；在1000点预先检查sidecar使用的是新stage/card而不是旧C/R常量。只按已有授权退休已durable验证的本轮局部副本，保留旧唯一完整Trainer源、合格U及失败模型；不新增home/scratch大清理或删除权限。真实fileset quota以实测为准，不能凭df截图或旧空闲量保证保存。

继续邮件关闭、`Mail_Points=n`，不新建monitor/automation，不修改健康staged目录，不改WBC/SafetyGate/metric，不merge PR。工程代码放独立execution worktree，旧card/原件/失败不覆盖。

## 10. 最少交付与PC agent启动正文

交付集中为：旧证据可访问索引及CPU收据路径核验/必要修订；一个位置误差分解；两张sealed card/共享曝光与RNG key表/CPU回归；真实2000+2000更新、独立及配对完整性和四个durable Trainer receipts；256格及独立CPU复算；四组×两臂质量表、相对R2000与X−K的解释、候选或明确STOP。不要为了这些小表重新建设调度/存储/验证平台。

### 启动正文

执行 `PI0249_R_ANCHORED_XYZ_PRIORITY_GO`。完整读本文件及5852051721、5851408869、原PI0248协议。

接受旧C/R质量双FAIL，保留原tau配对失败和selected_checkpoint=NONE。先G0核对实际CPU验证代码/命令与新收据hash：仓库cpu_replay末尾仍引用旧pi0248_training_integrity.json，不能靠假设消除此差异。发布原report/CPU/收据/deviation可访问索引；元数据问题自主修复，仅从已有NPZ生成新CPU报告，不重跑旧512。真实身份/评分冲突才停，不因缺少唯一根因无限审计。

从已有数组做窄位置bias/时变/跨窗口p95/轨迹末端/饱和分解，不改变采样或目标。

新K/X均从固定R2000 revision d089c09380c568e720c3dfa6bec5aa064acd7c4f、tree4422a24d2117206681c423a0f8d3137dff7eb614c990cb0334a8cdaf73900298开始，新AdamW/cosine2000/warmup10，LR1e-5，原64/2000表。K保持native+.1pos+.1delta+.1rot；X仅把pos系数改为.4。原S4、完整梯度、decoder/mask/统计/精度/冻结范围不变，不叠加新loss或重采样。

两臂共同按本协议step key隔离实际训练CPU/CUDA随机流，保持原Beta、Gaussian、cast和同noise S4。先真实入口顺序CPU恢复回归与增量目标数学测试；G0/G1通过自动训练，不再次等PI。正式首更新/每步留真实tensor证据，X在线核对K同step样本/tau/epsilon/stock身份，失配在该步optimizer前拒绝，不等2000再发现。

每臂完整保存1000/2000，但1000不screen。两臂均完成；Corelab仅新K2000/X2000×64×B0/B1=256格，R2000原128复用。first-valid和独立CPU复算保留，四组原门不变：零REJECT、medianXYZ≤.020、p95≤.035、medianSO3≤.050。

都过优先K，单臂过记录该候选；合格停FULL准入，均不合格停新假设。无额外步数/系数/seed搜索，无FULL/CALIB/selector/WBC/sim/DEV/F/DAgger/真机/merge。新增实际更新≤4000，parity≤4，新固定screen256；历史计数不充值。工程小修与同语义恢复自主，邮件关闭，不建monitor、不扩清理。下一交付是可解释的K/X最终联合质量结果，不是准备完成状态。
