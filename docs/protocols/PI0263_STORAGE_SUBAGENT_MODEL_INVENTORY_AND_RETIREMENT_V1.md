# PI0263-S：副agent逐模型占用核查与本地副本退役

Authority: `PI0263_STORAGE_SUBAGENT_MODEL_AUDIT_AND_RETIREMENT_GO_V1`

Publication state: AUTHORIZED_AUDIT_AND_BOUNDED_CLEANUP_NOT_EXECUTED.

## 1. 角色与权限范围

用户明确要求副agent负责清理空间，并仔细核查具体哪些model占用、哪些可以清理。此任务独立于主agent的PI0263科学执行，不是让主agent暂停训练、等待清理或接管清理工作。

- 主agent继续依其有效科学授权推进；本文件不改变模型、数据、配方、预算、评价、停止条件或正式选择结论。
- 副agent负责逐模型/逐副本核查、来源和当前依赖判断、已有归档核验、获准的归档和本地副本退役，以及实际空间收据。
- 辅助存储工作不成为主线的审批环节。某个候选与主线的使用关系无法闭合，只保护该候选及其共享依赖，继续核查其他历史资产。不得因主agent未回复而全局等待，也不得擅自取消、暂停、重启主作业。
- 此授权有限覆盖旧科学协议中对“广泛清理”的禁止：只允许此副agent在下述明确条件下处理本账号历史模型及冗余模型副本。它不授权一般性删除数据、环境、证据或当前实验内容。
- 不是一份远端清理完成证明。发布者读取了用户贴出的存储记录、固定科学协议及预检，但未SSH、重新扫描scratch或执行删除。

科学协议：`1c574b6e6e9d6a7851e64ca57e89c468a4c9eb28`，Issue #11 `5913659202`。预检参考：Issue #11 `5914444537`，它仅代表该次检查时间，不能代替副agent执行时的实时作业状态。

## 2. 现有输入与范围

本次主扫描根：`/scratch/e1192548/Projects/go2-arx5`。读取已有PC/Hopper索引及收据：

- PC目录：`D:/Users/GR00T project/scratch_cleanup_20260928/`。
- `STORAGE_MAP_20260930.md`。
- `hopper_fail_model_retirement_20260930.json`及前轮PI0253/home退役收据。
- Hopper对应inventory内的实际文件；从索引定位，不猜测未给出的绝对路径。

用户报告的最后状态：df可用155.542 GiB；本轮新增44.382 GiB与前轮26.997 GiB分开；个人quota UNKNOWN；最新账号目录du尚无删除后同步测量。不可把868.758 GiB估算当实时占用，也不能把旧PBS空列表当当前无作业。

以下旧目录计数是调查起点，不是本次剩余可删量：experiments约576.5e9 bytes，archive约176.5e9，runtime约123.8e9，data约102.9e9；以原表精确字节和新扫描为准。

本goal清理限本账号授权scratch内的历史模型、模型归档副本、模型下载/解压/staging副本。home、Corelab及PC可用于已有索引/归档的只读核查，但本goal不扩大到这些位置的删除。不得访问他人账号、改权限、认证、密钥、配额配置或共享安装。旧兼容链接保持，不迁移整个项目。

## 3. 必须交付逐模型账，不得止于目录du

对发现的每个checkpoint/模型实例和其本地副本，记录至少：

`asset_id, PI_or_stage, lineage, arm_seed_step, scientific_terminal, evidence_ref, artifact_kind, logical_bytes, allocated_bytes, all_local_paths, realpath, device_inode_nlink, symlink_or_shared_blob_refs, identity_manifest, duplicate_group, current_consumer, dependency_evidence, archive_repo_revision_path, archive_completeness, readback_receipt, public_release_permission, decision, estimated_reclaimable_bytes, unresolved_reason`。

artifact_kind区分完整Trainer、模型权重、adapter及其外部base依赖、optimizer状态、推理导出、tar/zip归档、解压/staging、HF cache及非模型文件。

必须分别汇总：

1. 每个科学模型身份在所有路径中的逻辑大小和实际分配占用；
2. 每份副本的额外占用和预计可回收空间；
3. 当前需要本地常驻、可仅归档保存、可直接退役、仍有具体缺口四类；
4. experiments/archive/runtime/data中已归类模型字节、非模型字节和仍未解释的较大对象。

不能把同一inode/兼容链接/HF共享blob重复计数。tar压缩包和解压副本内容相同但可能分别占物理空间，应分别计入副本账；不能因文件名相同就认定内容重复。共享快照/reflink的实际回收不明确时标记UNKNOWN，而非承诺释放。删除一个仍有其他硬链接的路径不等于释放对应块。

先做一次有限元数据扫描和已有manifest关联，再对必要候选核对内容；不为初步排行把整个近TB项目重复hash。扫描模型目录中的小分片/配置也要归组，不能只看单文件大小。对未归类的大归档可只读列成员和小型配置，不在scratch完整解压另造一份。不要加载模型、执行不可信pickle、运行forward或恢复训练来识别权重。

首份可行动输出为 `MODEL_OCCUPANCY_TOP30.md`（不足30则全部）和完整CSV/JSON，而不是再次给出六个顶层目录大小。后续清理按这份精确对象账执行，不等用户逐项回复已满足规则的候选。

## 4. 指定优先调查清单，不是直接删除名单

逐项解析实际大小、全部副本、使用依据及归档收据：

- PI0208 C80/S25：用户记录为通过模型；核查是否仅是历史baseline以及是否已有完整可恢复归档。PASS不自动要求永久留在scratch。
- PI0208 MCTRL各seed：核查是否只有已完成screen对照用途。保留对照指标不自动要求所有本地权重常驻；不得把失败MSUP结论套到MCTRL。
- PI0205 E：核对初始化链与当前是否还存在实际base/adapter/恢复依赖，祖先关系本身不是文件依赖。
- PI0254 final：从报告确定确切arm/step/质量和档案，不把所有“final”自动当golden或废弃模型。
- Stage45B/Stage45E golden及recovered-primary：区分真正受保护运行基线与历史复制包，核对公开许可与完整archive。不能因名称golden就不再调查，也不能无归档删除唯一有价值版本。
- PI0260旧Base/G终点及中间点、PI0261 C2000/中间点/其他副本：核查是否仅需旧预测数组、是否仍被当前card或待执行任务实际加载。不得笼统保护整个PI0260/PI0261目录，也不以FAIL自动删除。P2000指定本地donor除外，按第5节硬保护。
- archive中模型tar与experiments中相同权重/Trainer、重复下载与解压副本；runtime/data内真正模型cache也按模型依赖核查，不删除环境或冻结数据。

已明确删掉的PI0253分片、PI0208 MSUP7101/7102/7103和PI0240两个推理tar不再计作新清理收益；只有确认另有不同本地副本才建立新条目。PI0244–PI0251旧扫描未发现>100MiB文件的结论可复用，除非新目录映射发现实际大对象，避免重复空目录巡视。

“单任务微调1”保持未映射别名，不要求用户先解决这个名字才能调查上述明确身份。

## 5. 当前保护集与并行安全

立即保护：

- 指定PI0261 P2000完整donor：HF `leoyuer/pi0246-gate-pass-campaign@875abdde2cc7c9e6858fac008e53f2f4b3375255/arms/PI0261/P/checkpoint-2000`，tree `882931d754cb25653105a9213fe0dbeb4d2b14e463efb27e49c6101fd5fc5e4b`；用户记录本地`experiments/PI0261/P/main/training/checkpoint-2000/`。该指定副本不由副agent裁减为model-only；其他真正重复副本可按依赖单独审查。
- 用户指定唯一G2000：`experiments/PI0253/initializer_G2000/`及它确实依赖的共享文件。
- 当前PI0263及执行时后来有效主线的活动输出根、canary、F/X1000/2000、临时写入、完整恢复源、上传/读回队列。副agent不参与裁剪当前主实验检查点，即使它读到某个FAIL或stage名称改变。
- 当前实际base weights、processor/statistics、COMPACT数据、所需历史预测/噪声/评价原始数组；私有WBC/机器人文件；唯一配置、日志、实验结论和必要原始证据。

保护集从最新有效card、执行脚本、配置、恢复路径、运行/排队/hold作业及进程/传输信息解析。PBS空不等于没有本地进程；缺少qstat也不等于无任务。对计划未来会打开的输入也要保护，不能只检查当前open handle。

副agent在独立工作目录做低I/O并行调查，避免改主agent脚本、环境、锁和活动manifest。已有owner/任务标记可直接读取；不新建全局分布式锁平台或要求主agent停机。

每批真正删除前刷新该批依赖、路径身份和是否变更；stat/owner/mtime/inode/内容与批准候选不一致则跳过并重核该项。若主线无法读取或新活动范围不确定，对可能相交的分支保守保护，其余历史枝继续，不全局停工。不得杀进程来让待删路径变成“不在用”。

## 6. 决策与执行规则

每份本地对象只给一个明确处理状态：

- `KEEP_ACTIVE`：有具体当前/排队/恢复/输入依赖，或第5节指定保护。写出消费者与证据，不仅写“有用”。
- `DELETE_REDUNDANT_VERIFIED`：多余副本内容身份和保留的必要副本已核对，没有独立消费者和共享依赖受损。可直接移除精确副本。
- `RETIRE_FAILED_NO_CONSUMER`：有明确已失败且被放弃/退役的证据，当前无依赖，必要失败证据保留。沿用用户此前明确授权，权重可以无完整远端备份而删除；必须记录`weights_no_longer_restorable`及其范围，不称作可恢复归档。FAIL但仍作为donor/科学对照待用者不得归此类。
- `RETIRE_LOCAL_ARCHIVED`：历史有价值模型或已归档失败模型，不需本地常驻；固定归档与完整身份/文件读回已证明、恢复依赖明确，可以删除本地冗余权重/Trainer副本。保留restore pointer和收据，原结果不变。
- `ARCHIVE_VERIFY_THEN_RETIRE`：有价值但尚无完整恢复档案的历史非活动模型。只对用户已准公开的项目微调资产，使用已有允许HF PUBLIC仓库和独立prefix归档，核对许可、配置及内容无凭证/私有资产，完成固定revision全文件完整性读回后转上一状态。原完整Trainer不降格成只有weights却称“可恢复训练”。默认使用现有成熟上传机制，尽量流式/逐文件，不先复制完整本地树；已有等身份完整归档就复用收据，不重复上传。
- `HOLD_SPECIFIC_GAP`：科学身份、当前使用、公开许可或完整归档仍有具体不可解决缺口。列出已检索的卡/报告/路径、缺哪条事实以及需谁决定。此项保留，但不阻塞其他候选。不得把“golden”“可能有用”“归档不清楚”作为未经进一步查证的最终理由。

不将无许可或含私有WBC/机器人/供应商内容的包上传PUBLIC；这类资产保持本地或仅引用已授权安全归档，不能以用户要求省空间推定对外公开权。不得改/删HF远端归档、覆盖固定revision的证据、改旧scientific verdict。

先将精确realpath/file集合、身份、决定与预计回收量写入执行清单并完成规则校验，再执行已经授权的条目，无需逐条再次申请例行GO。本协议不提供宽泛通配删除命令。禁止递归删除整个实验/模型根目录；模型内容与证据共处时，仅删列出的权重和无需保留的状态。符号链接不得跨出允许根或进入保护目标；HF cache不得按snapshot显示大小盲删共享blob。凭据变化、无权对象或新不可逆范围不可自行扩展。

## 7. 空间目标和验证

目标是查清大体积模型与副本，并尽可能回收已获准空间，不是为了300GiB盲删。300GiB仍是副agent可争取的存储管理目标，既不构成主训练门槛，也不是迫使所有历史模型退役的删除指标。

完整调查范围和指定清单没有完成时，不得仅以“目前155GiB或许够训练”为由终止副agent任务。反之，查清并处理了合规候选但仍未达300GiB时，可有依据收口，列出具体保护项/缺口，不声称整理失败或让主agent等待。

每批分别记录logical_unlinked、inode-aware allocated estimate和df前后实测。共享文件系统并发变化不能全部归因于本次清理；du不原子则注明测量区间/UNKNOWN。以前27.0及44.382GiB不计本次新增。个人quota无接口就UNKNOWN，df及微小写入测试都不是quota证明；不要重复无效quota命令形成另一轮阻塞。

归档有空间压力时分批处理，不制造比清理更多的tar/解压/读回副本。单个上传/权限/依赖受阻，仅保留该对象并继续其他已授权候选。下载验证不可省必要完整性，但可复用未变身份的既有完整固定版本收据，不重复读回所有历史归档。

## 8. 必需交付与终态

在原PC/Hopper inventory的独立子目录交付，并更新索引不覆盖旧收据：

1. `MODEL_OCCUPANCY.csv/json`：完整模型/副本账与未归类大对象。
2. `MODEL_OCCUPANCY_TOP30.md`：按分配占用及可回收量排序的可读表；具体模型名、PI/arm/step、各份路径、总大小、HF状态、决定。
3. `PROTECTED_ACTIVE_ASSETS.json`：保护依据和检查时间。
4. `RETIREMENT_PLAN.json`、`RETIREMENT_EXECUTION.json`：精确前后身份、已归档/不可恢复区分、真实执行和失败/跳过记录。
5. `ARCHIVE_AND_RESTORE_INDEX.md`：仅本轮涉及模型的固定revision、完整性范围、恢复所需base/config；不冒称运行过模型恢复。
6. `STORAGE_MAP`增补：新du/df、实际净变化、已退役与具体仍待决对象。

向Issue #11发布简明canonical及PR #77 pointer，仅小型表和收据；不提交大模型或私有配置到GitHub。首份Top30可先交，不等所有上传完成才让用户看到实际占用。

终态：`STORAGE_MODEL_AUDIT_AND_BOUNDED_RETIREMENT_COMPLETE`，并分别列出已执行、仅归档、保护和HOLD。若无法访问Hopper，明确`REMOTE_AUDIT_NOT_EXECUTED`并保留准备文件，不用旧数字冒充新扫描。不能只交“空间大概够训练”的建议就完成此goal。

本goal模型加载/forward/training/backward、GPU作业、physics/WBC/TEST/DEV-F、主实验修改、广泛环境/数据清理、邮件/monitor/merge均为0。没有任何主线质量或选模授权变化。
