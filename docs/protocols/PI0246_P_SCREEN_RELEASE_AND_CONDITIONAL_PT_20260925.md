# PI0246 — 已完成 P 训练的评估放行与条件 PT 续行

日期：2026-09-25（Asia/Singapore）  
Decision：`GO_EXISTING_P_SCREEN_THEN_CONDITIONAL_PT`  
执行标识：`PI0246_P_SCREEN_AND_CONDITIONAL_PT_GO`  
原科学授权：`PI0246_GATE_PASS_CAMPAIGN_GO` / `PI0246_ENGINEERING_TO_GATE_PASS_CONTINUE_GO`  
发布状态：`PLAN_PUBLISHED_REMOTE_EXECUTION_NOT_PERFORMED_BY_GPT`  
代码审核基线：`35532eb8c8dbc8b929404d493bc498f6e8e70bfd`。  
Issue #11 为 canonical；PR #77 只作 pointer，保持不合并。

## 0. 决定：下一目标是 P 的科学判定，不是再开训

**认可已封存的 C/T 有效科学 FAIL；认可 P 已完成 2000 步并完成只读更新核验。放行现有 P1000/P2000 的固定 32-cell screen。若 P-final2000 联合 PASS，冻结 M16 合格模型并停止；若 P 有效 FAIL，自动执行原章程已批准的 PT2000，不再申请例行 GO。**

本文件不增加科学配方、样本、seed、训练步数或 gate 门槛。P/PT 的目标、系数、初始化、数据、最终判定点沿原 PI0246。已完成的 C/T/P 不因网络、HF、退出码、报告或代表性参数探针重训。

当前关键外部前置条件是恢复持有 P 唯一完整源文件的访问路径，确认原 uploader 状态并完成可信物化；不是缺少 P 训练授权。工程问题自主修复、回归、恢复，无任意次数上限；真实权限/MFA/网络前置条件不能靠重复命令或换科学配置绕过。

## 1. 本次实际审核与状态修正

### 1.1 读取依据

- C/T 主 screen closure：[5825455989](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5825455989)。
- C/T 最终跨噪声 closure：[5825569583](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5825569583)。
- P 准备/card/归档操作：[5826223695](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5826223695)；P job631951 的预检与实际更新记录5826240327、5826259897。
- P2000训练完成、抽样假阴性reconciliation与归档失败：[5826650715](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5826650715)。
- 最新源端访问与HF状态：[5826886505](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5826886505)，该comment更新于2026-09-25 04:46:27 UTC，即12:46:27新加坡时间。
- 已封存 S0/C/T 与物理decode、原PI0245/0246章程及R1/后续清单。
- 当前 `scripts/pi0246_physical_position.py`、`scripts/pi0246_paired_screen.py`；P新增commit `cf4252552a6c3d3d14270a366f0a972bb623a6f9`；reconciliation/screen补丁 `319c029ba4e56a6fb144337f5069e0f99bcfc9b9`；当前campaign-state索引。

GPT本轮读取GitHub源码、账本与状态文件，没有SSH/live qstat、没有独立读取或重算远端原始NPZ/权重、没有项目测试/模型调用/训练/归档上传/删除或scheduler变更。认可的是这些来源支持的状态和下一执行范围，不冒称已独立逐文件复验远端产物。

### 1.2 最近确认状态

| 对象 | 已确认内容 | 尚未完成或不能推断 |
|---|---|---|
| P0/P1 | 尺度/decoder证据及P1 32/32已经完成 | 不再重跑；P1完成不是模型PASS |
| C | 630383完成2000步；1000/2000完整；15/15 projector全张量变化已只读确认 | C-final主门FAIL，不推广为整体优化器未更新 |
| T | 631721在H200完成2000步、同forward日志与完整checkpoint；已归档；final主门FAIL | 4080S首个AdamW OOM是被保留的工程attempt，不是这条有效H200训练轨迹 |
| C/T主screen | S0/C1000/C2000/T1000/T2000，共80/80，独立CPU复核 | 不重跑、不挑中期 |
| C/T跨噪声 | 32/32；C 9 REJECT/7 WARN，T 3 REJECT/13 WARN | 这是4窗×4噪声/模型，不是另16个独立窗口或泛化测试 |
| P | 631951已完成2000；完整1000/2000；reconciliation `b819f23221e33c9af890cb32d92e711797193fa124fdabe0afe2eba39e8fc2ff`，两点15/15 projector变化 | P没有正式screen结果，不能称PASS或科学FAIL |
| P archive | 上传曾reset；最后检查HF public revision `b0e3fa8a29cfd7de37222f6eb74cc01a3e2a14e0`无P路径/receipt；原件未删 | HTTP可读/目标repo存在不等于P已归档 |
| 访问与owner | 最新记录VPN断开、Hopper DNS失败、Corelab SSH超时 | 当前远端job/uploader状态UNKNOWN，不能沿旧qstat推断仍无任务 |
| 下一科学动作 | P1000/P2000 ×原16，32唯一cells | 不是再训练P、再运行P1或直接启动PT |

P run card SHA256：`7aa496963ee99ed241bad1506779a63e1f0383617a008f4ac09cfefde04041ce`。
P runtime binding：`bc714188a2795233139b8b703f8476414e1b1918004a5fe137689683a1bd5b61`。
P训练前source/runtime身份沿其原card与manifest；新文档commit不得覆盖已运行代码。

### 1.3 主结果与含义

| Profile | REJECT / WARN | median XYZ RMSE | p95 XYZ RMSE | median SO3 RMSE |
|---|---:|---:|---:|---:|
| S0 | 6 / 10 | 本次引用的closure未提供 | 同左 | 同左 |
| C1000 | 6 / 10 | 中期只报告，原记录保留 | 同左 | 同左 |
| C2000 | 9 / 7 | 0.0441649 m | 0.0528444 m | 0.0318809 rad |
| T1000 | 5 / 11 | 中期只报告，原记录保留 | 同左 | 同左 |
| T2000 | 1 / 15 | 0.0623276 m | 0.0811081 m | 0.0368430 rad |

T比C拒绝数少，但位置median与p95分别更高约18.16mm与28.26mm。其表现是“准入拒绝减少与位置拟合退化并存”，不是已经接近联合PASS，也不证明所有时间一致性目标都无效。上述主closure没有给每个reject的全部issue分类，不能仅凭计数断言所有变化都由线速度单一指标造成。用已有NPZ/issue记录补分解，不新采样。

C/T成功执行、参数变化、较低loss都不替代动作质量。原四项门下两者都是有效FAIL；两者XYZ仍失败，正好满足P的事前准入条件。P已经实际跑完，无需再次“放行P训练”。

## 2. Gate A — 恢复源端访问和单一owner，停止无关的预检循环

1. 复用现有SSH alias、校园/VPN/jump合法路径。仅做用户态、可回滚的连接诊断；不得关闭host-key校验、猜IP、批量扫网段、修改系统路由/DNS、复制私钥或绕过MFA。
2. 若确实需要用户建立已授权VPN/MFA会话，给出唯一明确的外部请求，状态记`EXTERNAL_SOURCE_ACCESS_BLOCKED`；不要把重复HF whoami、GPU排队或重训作为替代。
3. 连通后先确认P训练terminal、原HF uploader及transfer owner、scheduler和Corelab进程；上一次提交/上传是否受理不清楚时先查，再决定恢复，禁止双投。
4. 已完成P的两份checkpoint、原manifest、reconciliation、audit与日志保持。先抓小证据与实际路径，不重hash全历史数据或全仓库。
5. 禁止为这一步启动H200。当前P1 wrapper再次通过不是里程碑；除非P screen真实依赖发生变化，不再重跑P1 preflight、模型预检或32-query诊断。

最小产物是一次新鲜、无秘密的owner/access快照。HF目标公开可达但P对象不存在时，根本缺项是P源文件尚未送达，不能持续把它误报为HF凭据问题。

## 3. Gate B — 物化现有P权重；训练归档与推理视图分开记账

### 3.1 完整训练源

确认P1000/P2000属于相同P card，step与训练消费符合原2000 schedule。复用完整本地inventory、原训练manifest和已完成reconciliation；发现真实内容矛盾才进一步核验。

原Exit1来自抽样projector探针不代表训练失败。原manifest保持，独立reconciliation与其绑定；不能重写旧JSON的PASS字段、删除训练目录或通过再次2000步“修复”记录。

### 3.2 上传、接收与固定revision

优先恢复原HF上传，确定没有并行上传owner；传输残片/对象进度可以恢复，不覆盖sealed权重。上传成功后记录正确repo、path prefix、模型revision、receipt revision、逐文件sha/bytes与完整tree定义。

分别验证：完整Trainer payload、模型index覆盖所有shards、config/control文件、optimizer/scheduler/RNG/Trainer state完整；大文件采用内容身份，缺远端SHA则固定revision读回核验。不能把尺寸匹配、目录存在或upload函数返回当完整收据。

原档案中若tree字符串有不同scope（manifest-relative paths、额外receipt等），明确算法和文件集合。优先以相同scope下逐文件内容确定身份，不将不同定义的tree字符串相等当要求，也不猜测它们可互换。

### 3.3 不让optimizer上传成为无关的GPU等待

**允许P screen与完整归档在有条件下并行，且不新增科学预算：**

- P完整源已按原manifest/step/card/reconciliation验证，原完整源安全保留；
- 目标执行端已取得该checkpoint的全部推理必需文件，逐文件与完整源inventory完全匹配；
- 独立`INFERENCE_VIEW_VERIFIED` receipt绑定原family、step、source manifest/tree、文件scope和实际path/revision；
- S0条件/噪声、processor等原screen依赖完整；
- 存储/网络资源不影响唯一源保存、数据读写或正在运行任务。

这种情况下，不必等11GB量级optimizer传完才能调用只依赖模型文件的screen。推理视图绝不称`FULL_TRAINER_ARCHIVE_VERIFIED`，也不能作为exact resume或允许删除唯一完整源的证据。

若成熟screen reader只认full receipt，允许新增薄的具名inference-view reader adapter，以内容证明绑定；不得伪造receipt、仅跳过断言或把partial目录整体当checkpoint。实现新adapter的成本高于完成现有上传时，直接沿既有完整归档路径完成，不另造通用存储框架。

本次不扩大历史删除权限。P两份原始完整checkpoint至少保留到对应完整持久归档通过、reader/恢复依赖清楚。网络阻塞不能成为删除源文件、重训或改输出内容的理由。

## 4. Gate C — 精确执行P screen，不重开旧主评估

### 4.1 固定范围

- P1000 ×原16、P2000 ×原16，共**32个唯一新主cells**。
- 使用实际C/T已采用且独立复核的S0 condition/noise packets；不重新构造S0、固定seed不能替代实际tensor。
- 共享同一eval-mode processor、TRAIN statistics、BF16推理、原sampler步数、H40/dt、frame/TCP、SafetyGate和metrics。
- 不训练，不运行C/T/S0/P1的新query；旧合法结果按完整身份复用。

当前`pi0246_paired_screen.py`已经支持P1000/P2000及P card SHA，无需新增另一个评估器。该版本`inputs()`会读取C/T/P三个小card；准备好这些既有小文件，不因此复制C/T optimizer或重跑C/T。

`run()`目前用`cards[arm]['family'] in restore['prefix']`作一部分绑定。如果实际HF前缀是`arms/P/...`，需要先确认恢复manifest究竟如何记录family；不能只看目录名。潜在路径/前缀不一致属于执行核对，不是已证实的新故障。正确适配是显式绑定family/card/step/源inventory，不擅自改科学family或删掉身份保护。

### 4.2 只补实际入口所需测试

本地/Corelab CPU预检加载实际stage-root、P card、S0 packet、P模型index与receipt，验证文件/shape/hash/source集合和CLI分派，确认缺shard/错误step/family/noise会fail-closed。已有有效证据复用，不为“测试完美”扫描所有9792视频。

S0在旧、新输出root之间若有metadata搬迁，用已有内容hash、实际condition/noise和迁移receipt建立对应，不根据不同JSON path hash重新跑一次S0。无法证明身份一致时明确受影响范围，先恢复原packet，不追新seed结果。

每profile仅一个owner。P1000中期出现REJECT、高RMSE属于结果，正常保存后继续P2000；只有实际数据/模型非有限或契约异常才停受影响执行。首次有效结果冻结，不能为通过而重采噪声。

### 4.3 独立CPU复核和主验收

直接复用成熟CPU validator，覆盖原16 ID、完整NPZ/record、shape/dtype/finite、model/source/condition/noise、dense-relative decode、gate decision及issues、全40点指标。只比较模型日志无法作为最终判定。

主门仅由P-final2000决定：

1. 原SafetyGate REJECT=0（WARN保留）；
2. median per-window XYZ RMSE ≤0.020m；
3. p95 per-window XYZ RMSE ≤0.035m；
4. median per-window SO(3) RMSE ≤0.050rad；
5. 输入、训练更新、模型、原16窗口及上述独立复核有效。

XYZ逐窗=sqrt(mean_40(sum_xyz(error²)))；SO3逐窗为geodesic RMS；quantile保持原linear方法。不能把“linear p95”误解为速度p95。P1000不作为替代终点；不得去尾、换指标或用aux loss替代screen。

补充一张S0/C2000/T2000/P2000表，P相对C/T的同source差值、bias/残差、尾部SSE、每段速度和所有gate issues。所有分析来自已有轨迹，不增加query。主16 PASS仍不代表一般泛化或sim成功。

## 5. Gate D — P结果决定下一步；PT为原授权，不是新方案

### 5.1 P联合PASS

关闭未执行PT。执行原章程的P-final × bridge4 ×原四noise条件，**16个已授权的noise复核cells**；复用P1实际bank和相同输入。

这些结果完整报告，不改主16 PASS定义，差时标记`NOISE_ROBUSTNESS_WEAK`，不挑noise重训。冻结`fit_gate_champion_manifest.json`、完整checkpoint与可信持久归档、恢复说明、主32cell结果和noise结果。

终态：`M16_GATE_PASS_MODEL_FROZEN — STOP_FOR_PI_SCALE_CONFIRMATION`。

`selected_checkpoint=NONE`。这是M16 TRAIN-fit模型，不是正式CALIB winner、WBC模型或general model；本轮不自动扩64/FULL/sim。

### 5.2 P有效FAIL

**自动放行PT2000，不再请求一次例行PI确认。** 实际P source/inventory不完整、模型非法或报告未闭合不算这个分支的有效FAIL。

| 项目 | 冻结PT配置 |
|---|---|
| Family | PI0246_PT_PHYSPOS_PHYSDELTA_M16_2K_V1 |
| 初始化 | 原sealed R0-final10000 weights，不是C/T/P终点 |
| Optimizer/scheduler | 新建；按原2000-step配置 |
| 数据 | 原16，原同一2000曝光表，每窗125次，seed22301 |
| 超参数 | LR1e-5、WD1e-5、maxgrad1、batch1/accum1/workers0、cosine2000、warmup10 |
| 其他配置 | 原R0实际参数/moment/autocast精度、冻结范围、augmentation、sampler、stats、H40保持 |
| 保存与主screen | full1000/2000；PT1000/PT2000各原16，新增32主cells |
| 终点 | 只按PT-final2000的原四项联合门 |

同一次native forward：

`a_hat = x_tau + (1-tau)*v_hat`

`E = D_xyz(a_hat) - D_xyz(a_star)`

`L_pos = mean_valid_points ||E/(0.05m)||²`

`L_delta = mean_valid_transitions ||((E[k+1]-E[k])/0.05s)/(1m/s)||²`

`L_PT = L_native + 0.1*L_pos + 0.1*L_delta`。

两项必须共享同一a_hat、同一noise/tau、原scale-aware decoder与原mask。三XYZ平方求和，不额外除3；原clip/zero-range/reference语义保持。native loss保留全部有效动作维度，不加XYZ4、加速度loss、gate hinge或末段特权权重。

**PT不能直接把P和T两个会各自patch `head.forward` 的overlay叠起来。** 以一个薄observer捕获一次native输出，再计算两项与一次loss组合；检查observer cardinality和RNG不额外消耗。当前P helper可复用数学函数，构图和metadata作为具名PT实现单独封存，不伪装成运行过的P代码。

CPU回归仅覆盖PT新增组合与准入：lambda为0的原值/梯度、oracle、常量物理偏差、时变尺度、mask、finite、clip、一次native调用/noise/tau、ARM/profile分派与恢复cursor。真实第一forward/optimizer核验计入正式PT2000，不另增加隐藏科学smoke。

PT代码/card/reader可在P归档或screen等待时进行CPU准备，状态`READY_NOT_STARTED`；P最终有效FAIL前禁止PT模型调用/backward。

PT联合PASS：按原16noise复核、归档与冠军规则交付。PT有效FAIL：

`PI0246_GATE_PASS_TREE_EXHAUSTED — STOP_FOR_PI_NEW_HYPOTHESIS`。

不自动延长、扫lambda、换seed/精度/sampler或解冻backbone。此时交付完整C/T/P/PT结果和一项NOT_AUTHORIZED的新假设。

## 6. 剩余预算：不可重置

按本次最后确认：C/T/P各已2000，**已完成6000 canonical updates**；只有PT条件分支最多再2000。Corelab OOM等工程attempt的实际load/forward/backward单独记账，不把它们写成完成update，也不把发生过的计算都写成零。

stock逻辑cell预算：

- P1已有32；
- S0/C/T主screen已有80；
- C/T final跨noise已有32；
- 合计144已完成；
- 下一P主32 →累计176；
- P通过时冠军noise16 →累计192，PT关闭；
- P失败时PT主32 →累计208；若PT成为冠军再16 →最多224。

这些是原章程的逻辑范围，不是新增充值。实际first-valid重复/失败attempt应真实保留并解释；不能用重新运行同profile来填一个看起来更整齐的表。

## 7. 资源、归档与操作纪律

- Corelab 4080S继续承担已经证明可以运行的无梯度screen；每次只常驻一个checkpoint，实际GPU占用与站点预约先核对，不终止他人任务。
- 4080S T在首个AdamW step OOM已经提供训练容量证据。PT沿已验证H200训练资源路径，不再重复同样的4080S训练OOM，不为显存改recipe。
- PBS auto负责合法分配；当前trainer的设备准入必须与实际资源契约一致。不要把科学任务绑定成“某个特定host必需”，也不能仅删除H200 guard后就宣称任何GPU可运行。已有健康owner保持；节点特有故障按明确证据恢复。
- GPU前只复验实际节点必须的路径/容量/库；普通pytest、HF认证与CPU数据检查在本地/Corelab完成，不用H200反复发现同类工程错误。
- 输出fileset容量、完整Trainer state、原子临时保存、传输与screen读取峰值按实际来源测量。`stat -f`/`df`提供文件系统余量，不自动证明用户quota；未知就记录UNKNOWN，沿既有实际容量方法准入。不能仅因为50/60GiB字符串检查通过就写quota PASS。
- 归档副本与当前reader/恢复点共享磁盘或硬链接时，按实际占用/生命周期核对，不将逻辑目录大小相加当实际释放量。
- 断线后先查询original uploader/job是否仍运行，不重复上传或qsub。外部会话确需用户动作时明确阻塞，不能承诺无人值守永远恢复。
- 不扩大旧文件清理权限，不改科学标准、目标注入边界、WBC、数据、frame、dt、action representation；不读CALIB/DEV/F，不运行DAgger/硬件，不merge PR #77。

## 8. 修正执行索引，不再让旧状态覆盖新科学结果

审核基线的`artifacts/PI0246_CAMPAIGN_STATE.json`仍含早期`training.updates=0`、旧C归档active_stage、旧failed attempt，并且P1 closure字段结尾为`...cc70a`，与已封存P1`...cc70b`不同。该文件的历史snapshot日期也早于最新C/T/P账本。**这是状态索引过时/抄录不一致，不能据此推翻原始P1、C/T/P或触发重跑。**

追加一个内容来源明确的current snapshot：C/T/P真实完成步数、C/T主门与crossnoise、P screen pending、P archive/access状态及其观察时间、已知/UNKNOWN owner、next action、源comment/manifest hashes。保留旧snapshot及其时间，不擅自把未知远端状态写成空闲。

P1正确closure SHA：`77712a2598ae7f8d8a076139c4160ec480c4d9ff624b2e45be9a0a51a55cc70b`。只修索引记录，不更改原P1文件或再生成closure。

阶段推进应来自实际完成receipt与原始验证结果，不来自一个最新修改时间下夹带的旧字段。这个小索引修正可与归档/screen准备并行，不另设GPU停止门。

## 9. 本轮完成定义与下一报告

近期goal完成应给出：

1. P源端/上传owner恢复状态与完整源保护；可信的P1000/P2000推理物化及完整归档各自scope/状态。
2. 固定32个P screen cells、独立CPU复核、P-final原四项判定。
3. 若P PASS：冠军及16noise结果；若P有效FAIL：PT实际启动、完成或精确外部blocker，不能仅重复“等待PI授权”。
4. 最终冠军或研究树耗尽；所有状态和累计计数真实。

Issue #11先canonical，PR #77只pointer。精简重复轮询报告：记录状态转换、完整checkpoint、有效科学结果或明确新blocker，而不是每传几百MB再产生一份相同授权说明。不得新建定时monitor。

## 10. PC agent 启动正文

执行`PI0246_P_SCREEN_AND_CONDITIONAL_PT_GO`。先读本文件、原PI0246章程/R1，最新5825455989（C/T主FAIL）、5825569583（32跨noise完成）、5826650715（P2000完成及reconciliation）、5826886505（source access blocker）。35532eb是审核基线，不假定为当前live head。

C/T/P已经训练完毕，P0/P1与C/T评估已完成，不重训、不重跑旧cells。当前主任务是现有P1000/P2000各16的固定screen。

先恢复既有授权VPN/SSH并核对原job/uploader；没有源端访问就报精确外部条件，不重复查HF凭据、不猜IP或绕过host key/MFA。保护Hopper P唯一完整源；上传恢复而非重训。HF固定revision内容核验后物化；或在完整源仍安全、推理文件全部逐hash验证并有独立scope receipt时，允许screen与余下完整归档并行，不伪称partial为full Trainer。

复用已有paired screen的P1000/P2000和实际S0 packets，只执行缺失首次有效cells。预检实际card/family/step/HF前缀与index，不按路径substring代替身份。Corelab screen，不等完整optimizer进GPU。独立CPU重算32cells，仅final2000原联合门判P。

P PASS关闭PT，完成原16noise复核并冻结M16冠军；P有效FAIL自动原PT2000，无需再问。PT从原R0-final10000、新optimizer、同2000表开始，不能从P-final续训；loss为native+.1phys-pos+.1phys-delta，按原单位/系数/decoder一次forward共用a_hat，禁止叠两个重复patch/采样。PT代码与容量可CPU提前准备，P主FAIL前不backward。

工程最小修复和恢复不限次数，科学budget不重置：已6000updates，最多余PT2000；P主32、条件PT主32、仅冠军16noise，campaign最多224stock逻辑cells。已有有效负结果不替换。4080S训练OOM不重复；PT用已验证H200，auto合规分配。报告/归档失败不重训。

修正campaign-state旧快照与P1 hash抄录错误，原artifact/旧FAIL不改。P或PT最终PASS停scale确认，树耗尽停新假设。selected_checkpoint=NONE；M64/FULL/CALIB/WBC/sim/DEV/F/DAgger/硬件/merge继续锁定。Issue #11先canonical，PR #77仅pointer，不建新monitor。
