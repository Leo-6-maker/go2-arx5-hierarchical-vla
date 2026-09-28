# PI0253 存储执行修订：淘汰失败权重双端清除，有用资产HF公开归档，保护正在执行的FULL

日期：2026-09-29，Asia/Singapore。

Authorization: `PI0253_STORAGE_TIERING_AND_RETIRED_FAILURE_PURGE_GO`
Scope: `STORAGE_ONLY_WITH_SCIENCE_STATUS_REVIEW`
Status: `AUTHORIZED_NOT_REMOTE_EXECUTED_BY_GPT`
Scientific authority remains: `PI0253_LIVE_BRIDGE_TO_FULL_SUPPORT_GO`，release `82554098fb55b39ceca4b955252f8952c91c0dc0`，Issue #11 `5870278574`。

Issue #11为canonical；PR #77仅pointer，保持draft/unmerged。此修订不新建训练family、不改变当前stage card、科学阈值、数据或预算。

## 0. 用户决定与本次明确修改

用户要求仔细审核进度，并将Hopper和Corelab之前明确失败的checkpoint完全删除，有用的保存到HF public。本次据此修订此前过度保留的规则：

1. **已经终结、明确淘汰、无当前或已授权后续用途的失败checkpoint，大型payload允许直接永久删除，不要求先上传HF，不要求一定保留另一份权重。** 即使曾是完整科学FAIL，也可以成为淘汰对象；不再把“完整科学结果”自动等同于“全部权重永久保留”。
2. 删除模型/optimizer等大型payload，不删除失败发生过的事实。保留原结果、既有hash/inventory、训练配置、来源、必要日志、原有效输出与评估收据。明确写明 `WEIGHTS_DISCARDED_INTENTIONALLY`；只有证据保留而权重不存在时，不宣称仍可重新加载模型复现。
3. 有用资产按固定revision完整归档到HF PUBLIC；两台服务器仅保留当前运行/恢复/近期评价确需的工作集。不得将20GB训练包从Hopper搬到Corelab后就称完成双端清理。
4. 当前PI0253的FULL2500虽然中期质量FAIL，仍是正式候选和恢复来源，不是被淘汰的旧失败。当前family的2500/5000/7500/10000全部受保护，不能按分数删除候选改变选择集合。
5. G2000虽有一次有效sim FAIL，仍是M64合格的当前initializer与桥接对照，必须保留可恢复的完整资产。
6. 本修订覆盖旧“所有科学FAIL须先完整上传再删”的默认要求，但不覆盖当前真实运行依赖、候选全集、必要baseline、唯一证据和凭据保护。失败分类不清则跳过该项，不把整项清理无限挂起。
7. “完全删除”在本授权中的范围是Hopper/Corelab本项目可安全删除的本地payload及其无用副本。**不删除既有HF仓库、HF历史revision、Git历史或第三方资产，不执行安全擦盘/覆写。**

对明确淘汰类，用户选择意味着接受失去该旧权重重新推理的能力；小型证据保存不能逆推出原权重。该不可逆边界必须在执行manifest中明确，而非以归档PASS掩盖。

## 1. 本次进度审核：真实FULL已开始，不是仍停在桥接

实际读取的canonical及来源：
- D0/D1 `5871378817`：两入口四组输入/noise配对一致；8逻辑格=2复用+6新query；原TRAIN支持/时间语义核验。
- FULL admission `5871805200`：新入口CPU检查与实际提交。
- 1000/2500固定检查 `5873978549`；最新续行addendum `5874169301`。
- 报告commit `9d713b2aab765d42367f0788b7d918167e521318`，`reports/PI0253_FULL_2500_FIXED_BARRIER_REPORT_20260929.md`。
- 训练执行commit `568fe6a16dfbf9aa6700375f668715efae655d7f`，`scripts/pi0253_full_train.py`。
- PI0253原release及其固定早停、选择、预算规则继续适用。

截至GitHub最新观察（2026-09-29 00:21:55 SGT，00:24:46发布）：

| 项目 | 已有证据支持的状态 |
|---|---|
| 桥接 | Q/S相同实际输入/noise的raw、XYZ一致；无证据要求修推理算法 |
| 数据 | 原9792窗口、612独立TRAIN rollout；有效reset/early支持存在，不等于已证明充分覆盖 |
| 636930 | 完成2500更新，在已批准barrier处有意停下 |
| 637034 | 恢复preflight拒绝仅metadata目录；0新更新，不是模型科学FAIL |
| 637051 | H200真实加载optimizer/scheduler/RNG并执行后缀；至少2701/10000，不仅是Q或R状态 |
| 完整checkpoint | FULL2500已固定revision全文件读回；5000/7500/10000未见完成记录 |
| 保持评估 | 新192/576格，均first-valid，CPU各自复算通过 |
| raw证据 | 795文件/241443594bytes，HF固定revision完整流式读回通过 |
| 正式选择/新模型sim | 尚未执行，selected_checkpoint=NONE |

当前质量并不理想：

| 组 | REJECT | median XYZ mm | p95 XYZ mm | median SO3 rad |
|---|---:|---:|---:|---:|
| G起点 all64/B0 | 0 | 14.808 | 31.454 | .016274 |
| FULL1000 all64/B0 | 0 | 67.649 | 156.129 | .041260 |
| FULL2500 all64/B0 | 0 | 54.544 | 194.305 | .050837 |
| FULL2500 all64/B1 | 0 | 57.874 | 196.700 | .054379 |
| FULL2500 old16/B0 | 0 | 50.803 | 95.426 | .059744 |
| FULL2500 old16/B1 | 0 | 51.871 | 101.532 | .059629 |

这是明确的固定支持能力回退，不应以零REJECT包装为质量PASS；原因未由这些汇总唯一确定。1000和2500的B0均为0/64 LINEAR_SPEED_REJECT，未触发原“两次均64/64”的系统失稳早停。**继续固定FULL是执行预注册方案，不是认定性能已好或保证最终恢复。** 不因本次清理暗改loss/LR、提前选模、追加步数或重训。

最新报告中的175811592192bytes（约163.7GiB）是Hopper提交前快照，next-save门为31620064634bytes（约29.45GiB）；quotaUNKNOWN。本轮没有Corelab当前容量数值，也未直接SSH/live qstat/du或重新读回权重。本GPT只作GitHub/源码审核和授权，不声称已清理、已释放空间或已现场验证最新job状态。执行agent接手必须更新两主机实际状态。

## 2. 三类资产；失败与用途分开判断

### 2.1 KEEP_ACTIVE：当前工作集，不能被清理脚本触碰

包括：
- 当前运行、排队、hold、待resume、上传、screen进程实际引用的输入、输出、缓存blob、恢复目录及硬/软链接目标。
- PI0253 stage card、封存代码、样本表/step keys、真实append-only日志、当前正在写的checkpoint、分片或上传源。
- `experiments/PI0253/initializer_G2000`等实际绑定输入，以及当前恢复所需路径。
- 原TRAIN/CALIB、processor/statistics、实际B0/B1、live L输入、WBC源码/权重/runtime/Cosmos缓存和唯一原证据。这些不是本次checkpoint删除范围。

默认清理进程不处理整个活跃PI0253运行根；已有成熟训练归档器继续按自身已验证的保存、加载ack、readback流程管理本run。不要为了新storage authority改写已封stage card或运行中的代码。

### 2.2 KEEP_HF：有明确用途，完整归档后可清除无用本地副本

最低保留集：

| 资产 | 用途及要求 |
|---|---|
| G2000 | 当前FULL初始化、M64合格里程碑、桥接/物理对照；revision `efecea485ae77000599d19171b02491f8d514f42`，tree `cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226` |
| 当前FULL2500/5000/7500/10000 | 正式候选全集与恢复来源；全部HF完整Trainer保留，即使中期保持FAIL |
| PI0226 adapted1000 baseline | 本轮CALIB必需原基线；原权重、processor和引用身份保留，不误删同号smoke |
| PI0247 U-final2000 | 首个M16正里程碑，HF完整归档即可，无需双服务器常驻 |
| PI0252 A2000 | 当前速度margin同期控制，已有固定HF产物可保留；无进一步本地使用时删除双端工作副本 |
| FULL1000 observation | 已形成的中期对照；保留已有HF model-only包，不伪称完整Trainer；screen结束后的双端推理视图可退役 |
| 其他实际必需initializer/baseline | 由当前card和已批准后续命令证明用途；不能只用“可能未来有用”永久保留全部历史 |

FULL2500身份：revision `289b3a8c98a8f8aa8cc62a94c26c3a827c154a4b`；tree `fadc5581ebfce50e79a94078d7d6434ee333a55f7e8c1f8e9350f475c5a580cb`；9文件/20882714202bytes。其中9只是本产物实际inventory，不是通用完整性判断。

795份证据的固定revision是 `5043765c8d3b35c1e7d036bf71bb96264e66dbc0`，prefix `evidence/PI0253/full-barrier-20260929`；它是证据包，不包含或替代每份20GB完整Trainer。

PI0249 X、其他旧祖先的处理：旧“尚未封存/后续引用”状态先核对，不默认仍然成立，也不因名字含FAIL立刻删除。**来源图中被提及不等于当前代码还要读取其大权重。** 当前实际依赖消除后，有明确保留用途则按KEEP_HF；完全无用途且属已终结失败，则可按PURGE_RETIRED处理并记录旧保护解除的具体依据。原card/receipt/lineage的小文件仍保留，不能为清理修改G祖先记录。

### 2.3 PURGE_RETIRED：可直接永久删除，无需先上传旧坏模型

一项同时满足以下条件可入此类：
1. 该阶段已有终结科学FAIL/明确废弃决策，或确证为损坏、不完整、被取代的工程smoke/重复产物；PBS Exit非0本身不足以证明损坏。
2. 非当前候选全集、非活跃/排队/恢复/评估/归档引用、非KEEP_HF明确用途，也不是唯一可用的必需基座/运行依赖。
3. 已保留最小失败证据与既有inventory/hash，实际绝对路径/属主/文件身份已纳入本次删除manifest。

授权内容包括模型safetensors/bin、optimizer/scheduler/RNG训练包、旧推理视图、恢复下载副本、同项目重复tar/zip、已停止传输的临时碎片，以及只被这些废弃资产引用的独占缓存payload。不能只删optimizer而留下大权重，也不能只删一台机器却保留另一台同内容备份后声称完全清理。

候选发现可从PI0241/43/44/46、旧PI0228失败family、PI0247失败L及历史废弃中间点等已闭合阶段inventory开始，但**阶段名称不是路径白名单，也不代表其中每份资产都无用**。实际用途优先。当前G、必要baseline和现有FULL一律不因一次FAIL成为此类。

没有HF备份亦可执行，记录 `archive_required=false, weights_discarded_intentionally=true, reexecution_from_weights_available=false`；已有HF者记录其实际身份但不重新上传、不删除HF。不要为删除一个坏分片先启动20GB全量GPU校验；用既有错误/大小/缺文件证据及实际path identity即可，缺失原hash如实写UNKNOWN。

## 3. 双主机精确清单，而非通配符rm或第二轮搬家

Hopper主范围：`/scratch/e1192548/Projects/go2-arx5`。
Corelab主范围：`/home/liuyu-nfs/Projects/go2-arx5`。
其他旧目录、`/var/cache/users/liuyu-nfs/`下项目专属缓存、HF缓存仅在既有card/inventory指向、确认为本项目且本用户拥有后逐条加入显式allowlist。禁止扫描后按“checkpoint”字符串跨项目删除；不扩大到整home、系统环境、其他用户或共享第三方模型。

先查询Hopper本用户全部active/Q/hold job及工作目录、script/Variable_List和resume路径；Corelab查询当前相关训练/worker/screen/上传进程与打开文件。qstat/PATH或SSH不可用记UNKNOWN，该主机不可证明安全的删除暂停；另一主机明确不相关的工作、CPU清单及HF归档可继续。不因为查询缺失就认为无job，更不取消健康637051或重复qsub。

生成一个集中manifest，按host和canonical absolute path分组，字段至少包括：
`asset_id, family, step, status, usefulness_reason, class, host, path, realpath, owner, dev, inode, nlink, logical_bytes, allocated_bytes, verdict_ref, protected_by, archive_repo/prefix/revision/tree, content_readback_receipt, action, irrecoverable_ack, pre_delete_identity, result`。

预览后将manifest与保护表存到非删除目录，追加Issue #11记录。**用户已授权：清单内分类明确、测试通过的项可直接执行，无须逐个等新确认。** 不明项逐项列出理由并跳过，不能用一项UNKNOWN阻塞所有已确认废弃项。

默认lstat且不跟随symlink；link对象与target独立记录，不能递归越过受保护边界。根布局与广泛兼容链接不再重建。退役checkpoint若产生已批准移除的无用alias，单独记录alias退役/原target说明；不能留下未注明断链，也不能把指向metadata的链接称可恢复checkpoint。

硬链接按(dev,inode)去重，不修改/截断共享inode内容。删除无用link不一定释放物理块；仍被G、基座、当前工作集引用的blob必须保留，计 `SHARED_BYTES_RETAINED`，不是继续删保护副本追求“完全”。HF cache只删除明确revision/文件集合，预览真实删除策略；禁止全局prune或整repo cache rm波及仍使用的revision。不升级冻结训练环境来做缓存清理。

实际删除前重新校验path/dev/inode/size/mtime与活跃引用；与预览不同或处于写入状态跳过。可用短期项目清理锁协调本清理脚本，但不能假定训练器也遵循该锁；活跃run根本就不由外部purger处理。不得结束无关进程来回收被占用文件。

## 4. HF PUBLIC归档与本地工作集分层

主repo继续 `leoyuer/pi0246-gate-pass-campaign`；先按已存在receipt的prefix/revision定位，不复制同包到新的随机目录。已验证过且身份不变的归档复用；既有备用repo `leoyuer/go2-arx5-checkpoint-archive-v1`仅按之前授权、实际账户写权限和明确内容映射使用。

对于KEEP_HF新归档或需要删除唯一有用本地源的项：
- 创建有用途说明的inventory，绑定模型权重及必要config、processor、TRAIN stats、action/frame/TCP/S4语义。
- full-Trainer资产须包含本checkpoint实际weights/optimizer/scheduler/RNG/Trainer进度及恢复审计前缀；不能用模型only包冒充exact resume。processor可在独立固定配套包，但必须有不可变映射与hash。
- 使用现有认证上传，记录实际commit/revision；不要在receipt/code/log/public包中泄露token、SSH key、认证目录、浏览器会话或无关数据。
- 对精确revision的每个必要文件作完整内容流式readback和bytes/SHA256核对；HTTP HEAD、LFS metadata或仅repo可见不够。
- 公开性与认证读写分开记录：匿名访问精确manifest及必要对象可读，并验证repo公开状态；401/403/404明确标PUBLIC_PENDING，不伪称PUBLIC PASS。不能仅用有token成功推导匿名可读。
- 读取验证优先逐流丢弃，不在scratch/Corelab再落一整套重复包；只验证本次资产，不重新遍历所有历史HF revisions。
- 同资产成功readback且没有后续错误证据时复用带时间的完整收据并做精确revision可达性检查；如有401、内容身份变化或恢复来源不明，重新完整验证。避免每次cleanup重复传输同一20GB已封包。

归档成功后，无活跃或近期明确使用依赖的本地权重、optimizer、推理view、下载副本可以双端清除。Hopper留训练实际工作集；Corelab只留当前screen/worker需要的模型视图及小型证据，不常驻整套Trainer家族。

HF权限失效不阻止PURGE_RETIRED的无备份直接删除；但阻止删除有用且唯一未归档的源。独立持久备份可避免阻塞计算，但必须标HF_PUBLIC_PENDING，不将临时转存称完成用户要求。无新HF云计算job授权。

## 5. 修正恢复状态表达，避免再次metadata目录误用

637034已经说明：目录存在、甚至有9个文件，都不意味着完整checkpoint。当前FULL guard正确拒绝了错误源，后来通过固定HF树恢复并在637051执行到真实2501。报告还明确，恢复后的本地源在加载ack后已按成熟流程退役。

本次不要求为了满足“至少一个本地副本”抽象规则再下载已退役2500，也不将当前已验证remote-only恢复源说成丢失。任何新删除都必须保持完整固定HF恢复源可用，当前使用路径与加载/归档生命周期由成熟程序管理。

新增存储索引明确区分：
- `LOCAL_FULL_VERIFIED`：实际内容与该inventory一致；
- `LOCAL_MODEL_ONLY_VERIFIED`：能用于相应推理，不是完整resume；
- `HF_FULL_VERIFIED_LOCAL_RETIRED`：完整恢复源在固定HF；
- `METADATA_ONLY_NOT_RESUMABLE`：本地只有sidecar，不可传给Trainer；
- `PURGED_RETIRED_FAILURE_NO_RESTORE`：旧失败权重按用户授权永久淘汰。

状态更正用伴随索引/tombstone记录，不改已封的历史card。下一次resume必须依原helper先做完整内容验证，必要时从固定revision恢复到正确路径，不得仅以exists()/文件数放行。首个有效suffix记录、optimizer/scheduler/RNG加载ack和后续保存点都保留。

CPU只需窄fixture覆盖：活跃恢复源拒删、当前失败候选受保护、唯一有用未归档源拒删、终结无依赖失败允许无备份删除、共享blob保护、metadata目录拒作resume。这里的fixture不加载项目模型、不消耗科学query或optimizer预算。不重写成熟训练/归档系统。

## 6. 空间目标及完成标准

Hopper/Corelab按实际mount/fileset分别记录df bytes/inodes、项目分层allocated du、quota结果、删除前后时间。不同主机的NFS路径若实际指向同存储，按mount/device映射避免重复计算回收。

区分三类数字：被unlink的逻辑大小、唯一inode占用量的变化、文件系统实际free变化。硬链接、进程仍打开的文件和共享文件系统其他任务写入可能使它们不同。仅rename/移动到同一filesystem下的archive/Trash不算释放；不要求shred或安全擦除。

Hopper现有100GiB是准备目标，不是每次写入后必须维持的恒定free。运行保存前检查 `free >= next_complete_write + overlapping_atomic/upload_temp + planned_download/cache_growth + logs + reserve`，reserve至少10GiB；现有完整源已占用空间不重复计入free需求。当前源代码只估算一个完整checkpoint加储备，agent应核实是否存在额外写入重叠，不盲称该估计覆盖所有峰值。quotaUNKNOWN仍是未知，小写入/df不能证明大文件quota。

**用户要求的已明确废弃项按本轮有界manifest完成双端清除，不因空闲已超过100GiB就永久留下一堆已决定淘汰的副本。** 但不为更大的空闲数字扩大到新的模糊对象或保护资产；清理不能抢占训练关键IO。低并发、优先关闭独占无用payload，归档与正在执行的计算协调。

完成报告必须包含：每台主机的删除/保护/归档/跳过清单、真实回收量、仍占用的共享或活跃字节、有用资产HF PUBLIC固定索引、resume状态索引及科学job最新快照。禁止只有“空间整理完毕”一句，也禁止没有实际删除就声称已释放XXGiB。

## 7. 科学继续推进，存储不成为新的研究分支

当前637051/原PI0253 family按live状态继续。保留原10000终点、原G目标、数据、RNG namespace与全部保存点。5000/7500/10000的固定评估、4格原live输入侧报、完整family后的原CALIB selector、winner自身保持门及TRAIN sim依原PI0253执行。

当前1000/2500保持FAIL值得如实汇报，但未达到已冻结的早停条件；不能删候选或偷偷更换loss来让表格好看。一个候选FAIL不否决其余候选；无admissible才NONE。若出现真实非有限/恢复/数据契约故障，按原对应科学/完整性门停，不用存储授权掩盖。

本修订新模型query=0、新optimizer=0、新物理episode=0；不重跑已封有效分数或失败，不延长科学预算。工程定位/传输/同状态恢复自主，实际raw全部记账。邮件/monitor关闭，不读DEV/F、不DAgger、不真机、不改WBC/gate、不merge。

下一正常交付是：双端实际清理闭合与有用HF资产索引，加原FULL下一真实checkpoint/质量结果；不是再搬一遍目录、再开新的M64实验或仅提交一个存储计划。

## 8. 启动正文

执行 `PI0253_STORAGE_TIERING_AND_RETIRED_FAILURE_PURGE_GO`。完整读取本文件，科学仍按PI0253原release。先更新637051和Corelab真实进程/路径，保护当前FULL与G、基线和唯一证据，不重启健康训练。

已终结、明确淘汰且无依赖的失败weights/optimizer/旧view/重复包按精确manifest双端永久删除，无须先上传坏或无用权重。保留小型失败证据，明确权重不可重载；不删已有HF历史。当前FULL2500是正式候选/恢复源，不是垃圾。

有用资产固定HF PUBLIC完整归档/readback，再清无用本地副本。G、FULL四候选、必需PI0226 baseline及列出的正里程碑受保护；历史祖先仅元数据引用不等于必须留20GB。X旧保护先按实际用途闭合，不能一律猜可删或永久保留。

检查硬链接、symlink、独占缓存与活跃文件，按inode去重计算实际释放；不用全局cache prune或通配符rm。区分LOCAL_FULL、MODEL_ONLY、HF_ONLY、METADATA_ONLY，不再将配置目录当resume。现有2500已加载后退役不为新规则重新下载。

分类明确项有权直接执行，无需逐项再审批；未知项跳过，其他继续。当前run由成熟rolling helper管理，不在线改sealed代码。清理完成后原FULL继续，不因中期质量FAIL删正式candidate，不重置预算、不开新monitor、不merge。

## 外部实现参考（不作为项目运行证据）

- Hugging Face官方cache布局/删除策略： https://huggingface.co/docs/huggingface_hub/main/guides/manage-cache
- Linux unlink语义： https://man7.org/linux/man-pages/man2/unlink.2.html

这些资料仅支持缓存共享和释放空间语义；主机权限、实际文件集及运行依赖必须现场读取，不从文档推定。
