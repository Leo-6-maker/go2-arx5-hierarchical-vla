# PI0250 — 存储恢复、K/X收口与条件FULL选模到frozen-WBC仿真

日期：2026-09-27（Asia/Singapore）  
Authorization: `PI0250_STORAGE_SAFE_TO_SELECTED_SIM_GO`  
Decision: `GO_SCOPED_STORAGE_AND_HF_RECOVERY_CONTINUE_X_SCREEN_K_THEN_CONDITIONAL_FULL_SELECTION_SIM`  
发布状态：`AUTHORIZED_NOT_EXECUTED_BY_GPT`  
Issue #11是canonical scientific ledger；PR #77只追加pointer，不merge。

## 0. PI决定及优先级

本Goal不重新开始PI0249，也不新增K/X以外的小样本loss搜索。它授权：

1. Codex先核对真实scratch占用、quota与作业依赖；只按缺口执行下面的逐项清理，不为“整理”删除已有保护源。
2. 明确HF发布目标和备用目标，修复读/写/匿名访问的区别；无必要再等待用户选择仓库名。
3. 保护K2000和X1000；X从已封存1000完整状态只完成1001–2000；K的128格无梯度screen可在Corelab与X排队/训练并行。
4. K/X全部256格及新配对完整性通过后，按原规则决定唯一M64-qualified配方；旧R2000 baseline128复用。没有合格者就保留科学FAIL，不启动下游。
5. **本PI新增条件授权**：合格者直接进入一个FULL9792/10000新family、原TRAIN-CALIB选择、winner自身TRAIN保持检查、frozen-WBC A–D one-chunk与progress-preserving receding。通过每级无需再申请例行GO。

这是对PI0249“STOP_FOR_PI_FULL_ADMISSION”的显式条件解锁，不是假称旧协议已经授权FULL。旧gate、ABI、数据split、评分、WBC及所有已产生FAIL保持。DEV-E、Final-F、DAgger和真机继续锁定。

状态必须拆分：`storage`、`archive_access`、`gpu_queue`、`training`、`screen`、`science_gate`。一个GPU作业排队，不应让独立CPU/传输/Corelab评估都停掉；空间释放也不等于GPU可用。

## 1. 审核来源、已知状态与未知项

必读：
- PI0249原授权5852365229，commit `17f0514e7cb091816a4323958966ac83331f58e1`，`docs/protocols/PI0249_R_ANCHORED_XYZ_PRIORITY_PAIRED_PILOT_V1.md`。
- 旧清理授权5854739808，commit `a44ccfe099703d1fc39a5ab67bbece749e81b69f`，`docs/protocols/PI0249_STORAGE_RECLAIM_AND_HF_RETENTION_20260927.md`。
- 清理收口5854890159；K/X进度5854982211；HF访问勘误及分解5855071361。它们均在Issue #11。
- PI0248-R1质量终态5852051721及原tau配对停止5850988664，不能因新Goal覆盖。
- PI0247 `50e66efe6ccfbe814d04caf12499493821be7957`下主协议第7–9节和execution binding notes；PI0226 canonical5733559806及其持续参考/仿真artifacts。
- 参考实现 `317a2e85419cacd396f4f989a4ce83f4776bb85b`下 `pi0243_storage_retirement.py`、`pi0247_m64_screen.py`、`pi0228_train_calib_checkpoint_selection.py`；实际K/X执行源码需由Codex从当前staged目录绑定，不能假定这些旧ref就是634990的执行字节。

截至本轮读到的最新ledger（2026-09-27 10:33:41 UTC，即18:33:41 SGT）：

| 对象 | 证据支持的状态 |
|---|---|
| 上一次scratch清理 | 35,664,166,912 → 128,722,141,184 bytes可用，约119.9 GiB；filesystem delta93,057,974,272 bytes |
| quota | UNKNOWN；quota -v无报告、lfs quota不可用，不能把df余量写成quota余量 |
| K | 已完成2000，逐臂完整性与历史9/9文件HF读回PASS；尚无质量screen |
| X | 完成1000，完整恢复源保留；634990.hopper-m-02在small队列Q、无hold，请求1GPU/12CPU/225GB/16h |
| PI0249更新/screen | 3000/4000已执行；最终screen 0/256；selected_checkpoint=NONE |
| G0/G1 | 已完成：CPU恢复key测试及旧512数组分析复用，不重做 |
| HF勘误 | 旧repo匿名API/指定文件401，Hopper现有有效token对repo/revision404；之前“PUBLIC读回”目前不可由该路径重现 |

本GPT读取了GitHub记录、旧清理helper与既有FULL/sim协议；没有SSH、live qstat、实际du/quota、远端删除、checkpoint重算或训练。本轮HF连接确认账号为leoyuer，能获取旧campaign与两个既有archive repo元数据，但连接权限只有read-repos等读权限；**这不是Hopper credential的权限证明，不是固定文件读回或上传成功证据。** Codex负责接手时的真实机器检查。

旧清理收口列出的五组PI0246 P1000/P2000、PI0244x5000/7500/10000已退休payload不再计入待回收空间。HF勘误要求复查其可恢复性，但不能据此宣称已发生数据永久丢失。先查既有存储/凭据/镜像，不重新训练去“补回”。

## 2. 存储整理：先画实际依赖，再执行允许项

### 2.1 审核范围与实际清理目标

仅处理本用户拥有、属于此项目的scratch checkpoint/派生缓存payload。从旧inventory `/scratch/e1192548/Projects/go2-arx5-hierarchical-vla-hopper-state/inventory/pi0249_storage_reclaim_20260927.json` 和真实run cards列出的路径出发。不得因为出现checkpoint字样就跨到其他项目、home、共享缓存或别的用户目录。

记录：真实挂载点/fileset、账户、df bytes/inodes、quota命令结果、项目根分层du、checkpoint的大文件清单、逻辑/allocated bytes、device/inode/nlink、symlink目标、活跃及排队job引用。先stat/du分类，只对需要验证的完整源做大文件hash；不要遍历所有HF历史revision或重复hash全部TRAIN视频。

包括排队job在内，读取PBS提交脚本、Variable_List、workdir、card、resume路径、代码根、processor/data/cache路径，生成`protected_paths`。Q不是“没有使用者”：作业稍后仍会打开这些文件。当前job引用的symlink目标必须递归列入保护集合。

对每个候选给出精确绝对路径、family/step、类别、分配空间、是否当前依赖、保留源及删除依据。将候选manifest和保护表写入非删除范围并提交小型GitHub记录。**这是一项已经批准的执行授权：分类明确者可以直接执行，不必把整张清单再次发回请求批准。** 分类不明则隔离该项，继续其他安全项。

目标沿旧100 GiB工作余量，但以真实下一阶段高水位计算：

`required_free = next_checkpoint_write + simultaneous_atomic_or_upload_temp + planned_downloads + new_logs + reserve`。

reserve至少10 GiB；每项按实际实现/receipt量化。当前完整Trainer约20.9 GB只是历史量级，C取当前最大完整inventory，不硬编码文件数/大小。FULL同时保留的本地checkpoint、1000 model-only快照及上传buffer必须明确计入。滚动归档时优先保留最新完整恢复源和正在写的新源，四个正式checkpoint必须全部durable保留但无需全部常驻scratch。

`operational_target = max(100 GiB, required_free)`。若实际已满足目标，记录`NO_ADDITIONAL_DELETION_NEEDED`，直接继续。没有“必须再删出100 GiB”的额外指标。不为漂亮数字移除有用唯一源。

quota可观测时分别通过容量检查。quota工具不可用时不得无限轮询：记录`SPACE_READY_QUOTA_UNOBSERVED`，必须有当前用户在实际输出fileset的真实小型非稀疏写入/fsync/读回测试和可容纳最近/新checkpoint的容量规划，保留持久恢复源、每次保存前再检查。小写入不能证明20GB quota；该剩余风险明示保留，不写成quota PASS。实际EDQUOT/ENOSPC时停止新写入并保护恢复源，不删除它来碰碰运气。

### 2.2 允许直接删除：不要求上传坏权重

在不触碰保护集合、保留小型日志/审计/receipt后，以下payload允许直接unlink，无须上传：

- 已证实不完整/损坏/序列化失败/配额写残、且不是任何恢复或研究引用源的checkpoint。
- 已被完整后继覆盖的工程smoke/probe/debug模型及临时碎片；单凭PBS Exit!=0不能归类为坏权重。
- 同一fileset中确有另一个内容验证过的保留副本，且本副本不被job/manifest实际运行路径依赖的重复恢复/物化/下载副本。
- 不再使用的临时下载碎片、项目专属重复model view；共享HF blob是否仍被其他snapshot引用必须检查，不能整库清缓存。

损坏大文件可用已保存的错误与尺寸/缺文件证据确认，不要求为删除它先做完整多GB hash。已完成科学训练但成绩FAIL不属于“写坏的垃圾”。

### 2.3 有科学价值的完整权重：归档/验证后删除本地副本

PI0243/44/46/47/48的完整科学FAIL、必要中期对照、祖先和已合格里程碑，可不常驻scratch；但只能在下面的当前durability条件成立后退休本地副本。

优先回收无当前依赖的旧失败family和旧中期checkpoint；其次清除旧模型的重复推理视图。达到空间目标后停止，不逐层删到最后一份保护资产。

**HF访问勘误后的新规则：历史readback收据不单独构成本轮删除依据。** 在实际将用于恢复/消费的主机和授权身份下，重新确认repo type、固定revision、path、全文件集合与内容可读。对将删除唯一有用本地副本的源，必须实际流式读取远端全部完整Trainer文件并复算bytes/hash；仅metadata、LFS指针或HEAD 200不够。逐流读回可在Corelab或可信转存端进行，不把所有archive重新下载到scratch。

参考旧retirement helper的`verify_remote_tree`可能以LFS元数据hash判定，不自动等于完整内容GET；不要直接调用带PI0241硬编码列表的旧main。复用安全库存/验证小函数并作薄作用域适配，不重建通用存储平台。

如HF不可读/不可写，允许先转存到已授权且实测持久的Corelab项目存储，核验完整树及独立恢复视图后退休不被当前job引用的scratch副本，记`DURABLE_LOCAL_VERIFIED / HF_PUBLIC_PENDING`。不能把同一scratch的第二个目录、tmpfs、未证实持久的hpctmp或仅模型shards当完整恢复备份。不得利用协议授权规避真实配额/权限。

### 2.4 必须保留可恢复的资产

| 资产 | 固定身份/保护要求 |
|---|---|
| PI0249 X1000 | revision `bd5201ca5c4ee78cffce5f8fb2da3cb08552b8cb`；tree `fbd057360998f9957a94cbe7715529999cbea1c6edfebb5c960fd4b7c81c9f2a`；当前634990引用的本地完整模型/optimizer/scheduler/RNG/key/audit前缀不得移动或删除 |
| PI0249 K2000 | revision `b06dfcd14cc9d67881868fb5c239146a020d871d`；tree `e2c82f95d8a1f46af46742e200af3ee0121adff288df032c8afdfefd0ad50760`；完整源须当前可恢复，Corelab只需最小推理视图 |
| PI0248 R2000 | revision `d089c09380c568e720c3dfa6bec5aa064acd7c4f`；tree `4422a24d2117206681c423a0f8d3137dff7eb614c990cb0334a8cdaf73900298`；祖先及当前引用的initializer view保护 |
| PI0247 U-M16合格 | revision `0296f5dca5cb46daf05b51f90c53e1df905a7203`；tree `dace0dd423c4c1c6297b26bd6da09aee1e99edf4238638c5a317356e417a920f`；保留真实正结果 |
| PI0247 M64-final5000 | revision `093c450fd49052c8fd3285bf38111a30a1f59009`；tree `1811bfa5c29fb7ce2e911ade0a0b476607e32ac1c1322cce68b8dff04449d776`；失败祖先保留durable源 |
| PI0226 adapted baseline | 后续原selector baseline必需，未确认可恢复前不能随旧失败模型一起删 |

TRAIN/expert与split/meta、processor/statistics、B0/B1真实tensor、source/target映射、当前代码/cards/seals/环境、WBC权重与runtime/Cosmos依赖、唯一NPZ/video/日志/manifest/readback和G0/G1证据全部不属于本次清理对象。删除大checkpoint payload，不删除其所在run的科学历史。

### 2.5 删除操作安全和回收证明

只执行manifest中精确列出的payload文件或已经确认只含可删除payload的目录；拒绝根目录、用户根、空路径、路径穿越、父路径/挂载变化、越出授权根、链接目标和运行依赖交叉。禁止按FAIL名称glob递归删除，禁止跟随symlink去删其target。

实际unlink前再次lstat核对device/inode/size/mtime，重新检查作业状态与候选保护关系。旧脚本检查并不能代替本次revalidation。可优先逐文件删除model shard/optimizer，保留小型metadata。hardlink删除的表观bytes不等于释放bytes；记录nlink/实际df和du差异，不把两者混写。

默认不hold、不取消634990。仅当检查确认它会使用无效输入或不安全写入路径时，授权对**自己的尚未运行作业**临时hold防止启动，记录理由；修好并复验后release同一owner。不能为了常规盘点/排队抱怨hold，也不能重复qsub抢资源。RUNNING作业的staged目录不原位修改。

收口输出一份before/after、逐项删除和保护checkpoint的manifest，保留旧清理记录。当前HF可恢复性仍不明确的项列为`DEFERRED_ARCHIVE_RECHECK`，不是整项任务一律blocked。

## 3. HF发布路径：本PI给出明确选择，无须再问仓库名

### 3.1 主路径与备用路径

主目标继续 `leoyuer/pi0246-gate-pass-campaign`（repo_type=model）；不重命名、不删除、不改变整个旧仓库的可见性。

如果该repo从现有授权发布端恢复读写，原K/X归档按原card路径继续；新证据使用不覆盖旧对象的`evidence/PI0250/...`，FULL用`arms/PI0250/FULL_<K或X>/...`。

明确授权的备用公开模型仓库为：

`leoyuer/go2-arx5-checkpoint-archive-v1`

仅在确认执行身份为leoyuer且具有相应写/创建权限后，可以创建并用作本项目已授权权重/证据的公开archive，不需要用户再指定路径。若同名repo已存在，先确认归属及内容，不覆盖冲突prefix。只发布本项目已授权的权重和去除凭据的证据，不上传home、token、auth配置、未授权私有数据或无再发布权限的独立上游缓存。

建议namespace：`checkpoints/<family>/checkpoint-<step>/`、`evidence/PI0250/<run-id>/`、`migration/<old-repo>/<old-revision>/...`。保持checkpoint内部相对文件名不变；迁移后生成旧repo/revision/path→新repo/revision/path与逐文件hash映射，旧card与旧receipt仍保持。外层host/archive binding补充新位置，不能改旧科学card冒充原HF commit。

如果可用token只能写已有其他仓库，且不涵盖上面两者，不能凭“token有效”声称已解决；不得自动写入任意第三方namespace。本PI已经解决的是目标选择，不会凭空产生权限。

### 3.2 三类证明必须分开

分别记录`authenticated_read`、`write_probe_readback`和`anonymous_fixed_revision_readback`，加host/endpoint/repo_type/revision/time，不记录token值。

- whoami成功只证明凭据能识别账号，不证明目标repo可读/写。
- repo metadata成功不证明指定revision的文件可读。
- 认证下载PASS不等于PUBLIC匿名读回PASS。
- 401/404既不能单独证明repo不存在，也不能证明模型已丢失。

官方依据：HF user access tokens文档区分read/write与fine-grained资源范围；repository指南要求创建/修改权限。文档是API依据，不是本次远端权限测试：
https://huggingface.co/docs/hub/security-tokens
https://huggingface.co/docs/huggingface_hub/guides/repository

Codex仅检查本用户已配置的凭据来源与作用域（环境/标准HF登录），不搜索他人账号/浏览器秘密，不打印或写入GitHub/HF任何token。需要用户更新fine-grained授权时明确给出两个目标repo和缺失read/write/create scope，不能继续要求其回答已经指定的repo名称。

### 3.3 不让公开发布问题阻塞本地科学计算

旧27文件/4.8MB证据包优先保住本地源。必要的小报告、索引和代码可在本GitHub项目独立目录追加；这不冒充HF PUBLIC通过，也不替代大权重完整归档。

已经在Hopper/Corelab具有完整验证的输入和安全持久输出时，K screen/X恢复不必等待匿名公开问题解决。采用独立归档transport adapter；HF失联时写到已核验持久接收端，返回准确的本地durable状态，后台/登录端现时可用的传输流程再补HF。禁止把LOCAL状态伪装成旧HF PASS以绕过hardcoded assertion。

如果634990尚Q且其wrapper硬要求当前已失败的HF检查/仓库写入，先核实实际脚本；确认会失败时才按2.5的安全方式hold、使用新小型工程binding和独立worktree修复发布transport，测试后恢复。若无法修改同一job的提交环境且必须替换，只在确认旧job未启动并取消收口后生成唯一后继，保留lineage和原X1000恢复cursor；不能同时保留两个执行owner。

不把所有历史checkpoint重新下载到scratch再上传。先救保护表资产与正在进行family，再处理历史复现副本；若旧5个已删除payload的远端恢复暂不可验证，记录范围与凭据故障，不重训、不扩大删除。

## 4. 资源与任务并行：GPU排队不是全项目停止

634990在latest ledger为Q/无hold；接手先查询真实owner与产物，正常Q继续保留。清scratch不会改变`Insufficient ngpus`。

- 本地PC：代码、配置、SSH控制和小型CPU回归；不把数十GB权重经PC绕行。
- Corelab：已证明容纳的K/X无梯度screen、CPU复算、archive接收和后续native sim准备。
- Hopper：原已验证大显存训练与必要的大批量评价；PBS合法auto/项目CFP03-CF-004，不绑定某一hostname。不为普通pytest、路径检查、CPU hash分配H200。

当前X exact恢复保留原训练设备类别/精度/软件与sampler，不能只为排队改batch、精度、optimizer、分布式世界大小或上已证明OOM的4080S。任何其他GPU类别必须已有同契约容量与恢复一致性证据；没有证据不将跨设备变更装成纯调度优化，也不启动付费HF jobs或云服务。

资源不可用时，如实记录`WAIT_GPU`；可继续本地清理、HF传输和K screen。用户未要求新邮件/定时monitor，本Goal不创建它们；不承诺无工具的后台持续跟踪。

## 5. 先完成PI0249，不新增小样本训练树

K2000已完成，不重训。X只从当前完整1000点恢复到本phase2000；保留模型、optimizer/scheduler、RNG/key表、消费cursor及1000条审计前缀。G1已经PASS，复用原真实测试，只为新transport/恢复变更做相关回归。

K目标 `native+.1pos+.1delta+.1rot`；X目标 `native+.4pos+.1delta+.1rot`；LR1e-5、原64/2000表、S4/H40/dt、有效mask、processor/TRAIN stats、冻结和参数dtype均不变。X继续逐步对照K实际sample/tau/epsilon/stock身份，在违规step backward/optimizer前拒绝；不能用旧as-executed豁免放过新配对失效。

**明确允许K2000的128格screen提前在Corelab执行。** 若现有入口要求等待跨臂receipt，增加阶段限定的逐臂准入，只允许K已有完整性+原模型/输入身份；最终配对判定与下游仍等待X完整完成及两臂配对receipt。不得先看K结果后改变X/取消X或让K单独晋级。

K2000/X2000 × 原64 × 原B0/B1 =256格，总量不追加。已有first-valid全部复用；R2000原128 baseline复用，不评估K/X1000、不重新生成噪声。HF正常则必要推理文件直达Corelab；HF异常可由可信本地完整源生成同hash推理视图，附独立receipt，不冒充匿名下载成功。

两臂每个final2000都分别检查全64 B0/B1和old16 B0/B1：完整/有效、零REJECT、median XYZ≤.020m、NumPy linear p95 XYZ≤.035m、median SO3≤.050rad。独立从NPZ复算，原gate全部issues保留。old16是重叠子集，不增加样本数；不池化noise。

两臂完成后：只有一个合格取该arm；都合格优先K；都有效FAIL停止 `PI0249_XYZ_PRIORITY_HYPOTHESIS_EXHAUSTED`。前级科学FAIL时本PI的FULL/sim条件授权自动不成立，不再扫lambda/seed/步数。

## 6. 新条件授权：唯一合格K/X → FULL9792/10000

### 6.1 准入与继承

K/X获准者本身就是已验证的M64/old16保持候选，不再无意义地重训另一个M64。封存`PI0250_QUALIFIED_M64_RECIPE`，绑定winner权重、原R祖先及旧C/R偏差、K/X新配对记录、真实objective、S4、processor、统计、输入/noise和四组结果。

确认winner完整Trainer/推理配置当前可恢复，原9792 TRAIN dataset/split与窗口清单seal完整，CALIB未用于该训练，持久保存高水位可承载本轮family，才自动启动一个FULL run。不是无条件“必定开10k”。

Family `PI0250_FULL_K_V1`或`PI0250_FULL_X_V1`，由已冻结winner唯一决定。从该final2000的weights warm start；新AdamW/scheduler，不称旧optimizer exact续训。K或X的loss系数和真实rotation decoder原样继承；不自动切回历史仅XYZ的U loss。

固定FULL10000更新；LR1e-5、WD1e-5、betas(.9,.999)、eps1e-8、maxgrad1、batch1/accum1/workers0、cosine10000/warmup10。原trainable/frozen与参数精度/autocast/全四步梯度不改；只读准备在GPU外做。

原9792窗口按既有episode_index/source_ID确定排序，独立PCG64(22301)逐轮排列到10000，即完整一轮9792+第二轮208，不叠加shuffle、不重加权old16/困难例。沿原observation绑定与H40，不创造新数据或改target injection。训练前封存表和输入身份；如果seal数量/约定不一致，报科学绑定冲突，不自行改成“差不多一万条”。

沿PI0249 step-addressed算法，对这个**新stage**显式采用key `PI0250|22301|stage=FULL|optimizer_step=<i十进制>`，SHA256前8bytes大端并与(2^63−1)相与；不含job/attempt/时间。隔离CPU/本作业CUDA的同一forward-backward边界、原Beta/Gaussian/cast及重计算语义；不是续用PI0249的1–2000 keys。CPU恢复fixture扩到10000步及各checkpoint cursor，复用实现不另造RNG。

### 6.2 checkpoint、screen及早停

四个正式完整Trainer checkpoint：2500、5000、7500、10000。全部永久durable保留，以滚动方式移出scratch；1000允许model-only观察快照，不能当完整恢复点或selection candidate。

沿PI0247 FULL screen矩阵：原64 B0在step0/1000/2500/5000/7500/10000共384逻辑格，正式四candidate各再64 B1共256，总640。step0 B0的64格必须从已完成winner screen按身份复用；最多576新增。相应old16从同结果提取，不额外query。各candidate的noise、source、指标保持一致。

沿既有早停定义：1000和2500两次B0完整64中均为64/64 LINEAR_SPEED_REJECT才触发明确失稳STOP。2500设置结果barrier并保存完整状态；不是因普通中期联合FAIL随意停/调参。其余情况下完成原四candidate。屏幕与归档走Corelab/传输端，不把screen全串行塞入训练GPU。

尚未选定winner前，不能按TRAIN成绩删候选改变selector排名。所有完整family和对应保持数据闭合后进入第7节。

## 7. 原TRAIN-CALIB选择 → 唯一selected

只在本FULL family完成、输入/runtime/target映射完整后，放行原TRAIN-CALIB选择集评价；没有用CALIB优化训练的权限。明确它是开发校准集，不是最终泛化盲测。

固定baseline为原PI0226 adapted1000（只作对照），再FULL2500/5000/7500/10000。每份原1088，共5440；原S4、目标权限边界、gate/provider/统计保持，实际初始noise在baseline逐source封存、candidate配对复用。旧其他recipe分数不替代本family评价。

采用成熟评估入口和 `pi0228_train_calib_checkpoint_selection.py` 的原admissibility/ranking，单独绑定函数hash。原规则：完整且无REJECT候选 → median reference min-target-distance → worst source-rollout-cell median → 较低step；baseline不参选。不改admissibility使WARN变ACCEPT、不挑子集、不把“最不差”变成合格。

S3 provenance lock、S4唯一selection freeze、S5只读解释。没有admissible candidate：`NO_ADMISSIBLE_CHECKPOINT`，selected_checkpoint=NONE并STOP；不根据sim反馈换排名。

有winner则记录正式selected权重/step/family与processor/TRAIN stats/S4/源码/noise绑定。再检查该winner对应step的全64与old16 B0/B1四组保持门；失败保持selection记录但sim_admitted=false，停止，不改选runner-up。通过才进入仿真。

## 8. 条件frozen-WBC仿真：先真实one-chunk，再闭环

只在第7节通过后执行。不是把K/X固定64 qualification直接称正式selected，也不是清理完磁盘就运行WBC。

扩量期间在CPU机械恢复PI0226 canonical5733559806及对应sealed artifacts中的A–D C0场景/seed、WBC源与权重、frame/TCP、reference lifecycle、cadence/prefix/tick cap、clock/progress、termination/cleanliness和原scorer。生成binding manifest并列实际命令/hash。不混入更早world-target未成立的实验、StructuredAction入口或README默认值。

原值可以唯一恢复即可自动准入，无需新GO；若真实数值没有封存或冲突，`SIM_BINDING_CONFLICT`仅阻塞sim，前面独立合格的训练/归档照常。不能猜seed、放大episode时长或把截断episode称原benchmark。

最多4个A–D native expert控制episode核验环境；已有完全同runtime身份控制可复用。控制失败先核对环境，不将它归因于学习策略。

第一阶段：A–D C0各一次model one-chunk，共4次查询。真实RGB/StateV1 → selected GR00T S4 → 原dense-relative decode → 原SafetyGate → TrajectoryBatch → frozen WBC。REJECT不下发、不重采样。按achieved world-TCP及原0.035m成功半径和原cleanliness判定，不以reference命中或离线RMSE代替机器人成功。

4/4 one-chunk通过后，A–D各一次progress-preserving receding：每次用上一段真实执行后的新观测，保持reference时钟/进度连续。每任务最多64次模型查询，总≤256；实际tick/query cap仍沿原更严格bound。不能反复播放同一次预测、重置episode/reference追成功，不能修改WBC、gate、物理dt=.05、H40、目标输入、RTC、平滑/blending/retiming或seed。

沿原可用同步仿真控制时钟，不把慢推理伪装成真机实时。模型与IsaacGym环境分离、共驻显存按成熟方法核验；不临时引入未经验证的远程实时新架构。

记录输入/实际noise/输出/gate、reference时间、achieved TCP min/final/first-hit、tracking、termination及video。有效任务FAIL即停止下游并保留selected/失败；不换checkpoint/DAgger。四任务均成功的终态：

`PI0250_SELECTED_POLICY_TRAIN_SIM_PASS — STOP_FOR_PI_GENERALIZATION_REVIEW`。

这是TRAIN任务闭环证据，不是多技能/未见任务泛化证明。DEV-E/F、真机和PR merge继续不开放。

## 9. 预算、工程恢复与停止界限

| 阶段 | canonical optimizer | 固定模型评价上界 |
|---|---:|---:|
| 清理/HF/已有证据 | 0 | 0；无项目梯度实验 |
| 已有PI0249 K/X余程 | 只剩X1001–2000，最多1000；旧3000不重置 | 旧未执行K/X256，已有格扣减；baseline128复用 |
| 条件FULL | 新10000 | 640逻辑TRAIN格，其中step0的64复用，最多576新格 |
| 条件CALIB | 0 | 5440 |
| 条件sim | 0 | 4 one-chunk+≤256 receding；native expert≤4episode |

从当前ledger起最多新增11000个canonical optimizer steps、6272个离线scored cells、260个sim model queries。PI0249原≤4次非选分parity只有未用余额可用；FULL入口确需parity时新增最多2个完整S4非选分rollouts，不做额外optimizer smoke。内部head calls/重计算/native forwards与逻辑query分开，统计实测，不填期望值假装测量。

**工程恢复预算显式修订**：允许纯工程故障从最后完整checkpoint重新执行已丢失、没有封存有效产物的同一确定step后缀；不为这种恢复设任意重试次数上限。canonical唯一step范围仍严格止于PI0249各2000和FULL10000；每个真实执行/失败/重算update均累计raw账本，不称免费计算、不从旧计数抹掉。不得重训已经完整封存的K2000，不得重试有效科学FAIL或换seed得到更好结果。恢复需完整optimizer/scheduler/随机/消费状态及同key语义，不能新optimizer冒充resume；不能闭合真实配对/状态问题则停科学完整性门，而不是借恢复授权伪造PASS。

正常checkpoint写入/报告/上传失败优先只读收口、重新传输或保存当前完整状态；不要为Exit 0重训。健康运行的staged代码不修改，新增adapter绑定独立commit/工程子card，原科学card不覆盖。当前job运行状态未知时先查，不重复提交。

真正外部权限或GPU资源阻塞无法通过授权凭空消除；只阻塞依赖该资源的子任务，其余继续。工程问题可自主最小修复和回归；科学假设、门槛、配方、数据、controller变化需PI新决策。前级FAIL或无法恢复的证据/状态冲突即准确STOP，不默认整个goal成功。

## 10. 必须交付与简短启动正文

集中交付：实际空间/保护路径/删除清单与回收量；HF三类权限及迁移/当前可恢复收据；K128和X128质量表+新配对完整性；条件FULL完整family及四组保持；唯一selection manifest；条件sim结果与video。Issue #11逐阶段追加canonical，PR #77只pointer。保护小型证据，不创建新通知或monitor。

### PC/Codex启动正文

执行 `PI0250_STORAGE_SAFE_TO_SELECTED_SIM_GO`，完整读取本文件及5854890159/5854982211/5855071361、原PI0249协议。

先复查实际空间和634990依赖。旧清理已报119.9GiB，quota未知；足够就停止额外删除。按精确manifest自主删无用partial/smoke/重复payload；有价值/科学FAIL先当前固定源完整读回再退役。旧HF收据不抵消新的401/404。保护job引用的X1000、K2000、R2000、U-M16、M64祖先、PI0226 baseline、TRAIN/统计/代码及唯一证据；不要按FAIL名字批量rm。

HF主repo仍leoyuer/pi0246-gate-pass-campaign；备用明确授权leoyuer/go2-arx5-checkpoint-archive-v1，具备本人写/创建权限后可用，不再问仓库名。保留旧revision到新位置的内容映射；匿名公开/认证读/写probe分开，token不出日志。无权限时安全本地durable转存并继续独立任务，不伪造PUBLIC PASS。

保留健康634990，X从1000恢复只完成1001–2000。G0/G1复用，X逐步与K实际随机输入核对；不重训K。现在Corelab先做K原128格，X完成再补128；原真实B0/B1和baseline复用。四组原门，两臂完整通过后单一winner、都过优先K。

本PI明确条件解锁winner→FULL9792/10000→原5440 CALIB selection→winner自身TRAIN门→frozen-WBC A–D one-chunk→receding；按第6–8节配方、保存、早停、选择、sim绑定及预算，不自行补步骤。K/X都FAIL即停，不开FULL。正式选择之前selected_checkpoint=NONE。

工程小修/同状态恢复自主，raw计算全部累计；禁止有效负结果重试和假exact。Corelab CPU/screen/sim，Hopper只做需要大显存的计算，PBS auto不pin主机，不重复4080S已知训练OOM。邮件关闭，不新建monitor，不删唯一恢复源，不merge，不读DEV/F/DAgger/真机。不要以另一轮同样的队列快照代替可推进的K评估和存储/HF收口。
