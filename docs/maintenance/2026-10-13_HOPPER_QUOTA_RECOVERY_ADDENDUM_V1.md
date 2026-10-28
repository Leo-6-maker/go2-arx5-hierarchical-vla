# Hopper Scratch 容量恢复补充 GO V1：从文件数量转向有效 GB 回收

**2026-10-13 · 对既有 `33bbc2933e8bd6554d49b04f536519c10d0b962f` Hopper 专项 GO 的有界实施补充；不是新的科学/模型/GPU/PBS 授权。**

## 0. 为什么要调整

- Stage22 后续报告 [#47/6140317229](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6140317229) 使用旧 CoreLab maintenance release `3835ecea`，实际退役了296源码、6旧启动器、失效handoff与重复tar，合计338文件、43空目录、**2,936,832 allocated bytes下界（约2.8MiB）**。备份通过，但**未解决Hopper模型空间问题**。不要重复清理/计数这批。
- 阶段性 scratch 个人配额曾报 **940.7 GB / 1.0 TB**（确切单位/时间/最新个人硬配额由Hopper现场读取），先前PI0292 S3要求新增峰值80GiB+20GiB保留；真实计算项目 `CFP03-CF-004` 结束，空间恢复**不等于**PBS权限恢复。[#11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6139813870)
- Stage22 培训侧进度196000/200000、旧Hopper运行Python3.12.3和官方GR00T SHA `51d4c89f72fda44cbf77285c6a8114b52676b8a1` 与PI0292仍有关。**保留当前196k完整训练恢复链、N1.7原权重/processor/源和数据**；PBS空队列不等于所有恢复状态可删。
- 新的 Hopper GO 已授权在核验后直接删除不再使用的质量失败checkpoint或缓存，不要求为失败全量模型再做一次同体积备份。旧 C1 模型有HF固定公共完整 Trainer读回、Hopper本地历史20.47GB，是第一个明确候选；C0之前已删不重复计数。当前Stage22及PI0292保护范围不因本补充改变。

**这次维护的成功指标不是删除文件数，而是Hopper本人scratch真实quota变动、满足PI0292现场容量 preflight，并明确记录没有合格回收候选时的扩容/合法迁移路径。**

## 1. 优先级与容量门

1. **立即现场复测**：hostname/UID、私人scratch真实根/挂载、`hpc quota`个人硬限额及字节单位、`df`（仅辅证）、本人PBS jobs、Stage22和PI0292科学lease。冻结维护T0与原 Hopper GO 的累计账本，**不得重新开4h/12core-hour/2GiB/2public/30GiB额度**。若原Hopper GO尚未实际采用，则第一次现场采用T0；本补充不触发独立计时。
2. **只对已知大目录做有界容量树**：复用私有inventory及checkpoint manifest，列出个人scratch下allocated≥1GiB的目录/产物，汇总按用途分类：full Trainer模型分片/optimizer、model-only、官方HFCosmos缓存、私有data、临时下载/解压、日志和其他；标志完全相同的SHA、hardlinks、现役访问与HF回读。首先列出`TOP_30_ALLOCATED_CANDIDATES`；不为了小型/tmp目录开始整盘百万文件递归hash。
3. **优先处理已证明的不再需要的大件**：C1 PI0286完整HF归档、科学DEV5/16不合格，复查最新消费者/本地身份/原固定HF14文件hash和可恢复读回，合格就按exact manifest unlink本地非唯一全Trainer；C0本地大件若不存在记ALREADY_RETIRED=0。接着处理历史明确失败且无人引用的model-only、full Trainer、不必要optimizer、旧临时checkpoint、重复缓存/解压物。**`selected=NONE`单独不够**，需真实冻结质量结论和引用核对。此前9份旧Trainer退役、PI0264/PI0277/PI0278已删除件不重复扫描或计数。
4. **拆开Stage22的两类依赖**：PI0292可能需要共享的Python3.12.3环境、固定官方源码和N1.7基础权重，但**未必使用Stage22每个历史checkpoint/optimizer**。建立`STAGE22_DEPENDENCY_MATRIX`：物理路径×对象身份×具体消费者×读写/恢复要求。保护当前196000/200000完整最新resume（模型+optimizer+scheduler+RNG+sampler+stats+processor+sidecars）及经实际证明的最小回退点；旧间隔checkpoint/重复完整快照/导出BF16、历史缓存仅在不被用作独立对照和恢复点、且满足既有GO核验门后处理。不能因为训练快完成就删除唯一resume；亦不能因目录名Stage22全树永久保护。
5. **执行阈值**：第一目标是个人scratch使用量从近期940.7GB降至**≤850GB**，或取得**≥120GiB真实可用的个人quota**；为下一步完整训练再用已实现的`main_capacity.py`以真实smoke/checkpoint尺寸、个人quota、文件系统与预留计算精确容量。850GB并不是S4模型/优化器两套checkpoint足够的保证。没有足够合格大件则STOP本地删件并生成扩容/替代存储申请，不循环继续删除MB级源码。
6. **存储不足的合法替代**：向HPC申请个人scratch quota增额，或使用已明确获准的项目持久存储作为**数据集只读cache、冻结checkpoint恢复归档或完成的model-only推理副本**；更换写入/加载路径需现场试写、权限、带宽、quota和hash，不能使用别人的scratch或共享`df`自由空间。保留活跃数据在可满足吞吐的节点；高价值完整Trainer如需长期恢复则优先私有HF固定revision/完整读取、**流式上传避免同scratch第二份暂存**；合规正面模型可选Public model-only。绝不把完整Trainer公开、无授权上传私有WBC/原始数据。

## 2. 两个科学上不同的保护边界

**Stage22**：196k是保存的训练状态，不是本次科学任务已达到200k；训练进程本次未启动且PBS空。用户明确保护其当前恢复链。唯一最新checkpoint不得删，不得只留下model-only充当恢复。可以核查更旧快照的必要性，逐件判定。

**PI0292**：当前实验有固定bank64/16、HF revision和官方processor；S3–S6被**存储+过期PBS项目**阻断，未发生神经训练。保留此科学lease与所有固定身份。即便quota解封，也不能直接`qsub`：必须由科学owner复核原T0/截止、剩余PBS提交额度、项目授权及真实S3/S4 gates；已过截止不能通过维护GO自动恢复。

## 3. 每一实际删除候选都要经过现场硬门

- 实际本人scratch的 `realpath/st_dev/st_ino/UID/nlink/size/mtime/st_blocks`，拒绝symlink逃逸/未知硬链接；目录内容由exact file list固定。
- 最新本人PBS job/FD/cwd、Stage22/PI0292当前依赖/leases均无该目标；最少保持冻结评分、训练代码身份、processor/stats hashes、失败收据及meta。
- 已有私有/HF全量归档者必须固定revision全成员或可靠可恢复凭据；对明确不需要的失败权重，用户授权**不新增同体积备份**，但必须可追溯失败质量结论且不是唯一预注册对照。
- 按单项清单逐个unlink、空目录rmdir；不要在scratch上rename到`.trash`，不对顶层PI根 `rm -rf`；对象数据变化则HOLD。
- 再读 `hpc quota`本人额度（最重要）、allocated blocks、open-deleted handles（只读）、`df`(仅辅证)；没有真实物理可用空间增长则不报达标。路径保留仍可转向下一个合格候选。

## 4. 独立执行交付

已有local-pc唯一owner直接接续 `33bbc293`，无需对已达门的失败/重复/临时件逐一再请示，但**不扩大原删除范围、维护额度或HF Public资格**。

在既有 `reports/maintenance/2026-10-13-hopper-scratch/<run_uuid>/` 输出脱敏：`CAPACITY_TOP30.json`、`STAGE22_DEPENDENCY_MATRIX.json`、`RETENTION_DECISION.json`、`DELETE_ACTUAL_RECEIPTS.jsonl`、`QUOTA_BEFORE_AFTER.json`、`HF_RELEASE_OR_ARCHIVE_READBACK.json`、`PI0292_CAPACITY_GATE_READONLY.json`、`CAPACITY_BLOCKERS_AND_FALLBACK.md`、`FINAL_REPORT.md`。完整私有路径/inode/FD及凭据仍在Owner私有机器；#47交真实删除、人员quota和未解决原因，#11只有PI0292能力门的只读更新。

绝不把本Github协议或报告发布写成Hopper已经删了几十GB。没有真实现场执行则`PUBLISHED_PLAN_ONLY`；不足容量但所有合格候选已处理则`STORAGE_EXHAUSTED_NEEDS_QUOTA_OR_AUTHORIZED_REBIND`。

### 来源
- [最新Stage22实际清理](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6140317229)
- [既有Hopper GB级专项GO](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/33bbc2933e8bd6554d49b04f536519c10d0b962f/docs/maintenance/2026-10-13_HOPPER_SCRATCH_RETENTION_AND_HF_PUBLIC_GO_V1.md)
- [PI0292 S3外部容量/项目BLOCKED](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6139813870)
- [PI0286负结果、HF归档](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/7d0c4af6fa4a055a72939c203481584a84b2552a/reports/PI0286/final/FINAL_SCIENTIFIC_REPORT.md)
