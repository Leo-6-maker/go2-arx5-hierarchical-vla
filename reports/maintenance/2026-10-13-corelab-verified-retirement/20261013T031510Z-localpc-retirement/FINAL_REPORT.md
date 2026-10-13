# Corelab 实际退役与统一入口验收

**PARTIAL_WITH_EXACT_HOLDS**。本批实际退役 **33 个源路径、6,648 个普通文件**，两节点均已重新确认源路径不存在。包括 4 个原生 worktree 移除、1 个独立 clone 的逐文件退役、22 项已有归档原副本退役，以及 6 项保留全部内容的冷态迁移。Home 可见项 **48 → 21**，Projects **13 → 7**；仍有现役依赖和受保护原件，Projects 物理上没有只剩一个目录。

采用 Issue #47 comment 6137256578 的固定 release `3835ecea4560ed523ce77dad4c3e9dd2e937a817`，完整执行文档 hash `8b38307c0b307a25409d09d68a3d9c523b402e73f0d2018c402b0ec98fa86e30`。单一已有 local-pc owner；T0=2026-10-13T03:15:10.665289+00:00，最终文件系统收据=2026-10-13T03:57:03.093844+00:00，没有重置 PI0292 科学时钟。沿用 448 项原审计、24 个两节点 Projects 观测和 44 个 Home review；24 个共享观测对应 12 个对象，本批合计 56 个候选。没有重新扫描 181 万文件。

## 实际代码恢复覆盖

| 原候选 | HEAD | 普通文件 | 恢复覆盖 | 实际结果 |
|---|---|---:|---|---|
| go2-arx5-hierarchical-vla-pi0241 | `071aa3fcfd71` | 702 | refs/对象/原字节/独立恢复 PASS | RETIRED_NATIVE_WORKTREE |
| go2-arx5-hierarchical-vla-pi0240 | `bebbe90eb59d` | 722 | refs/对象/原字节/独立恢复 PASS | RETIRED_NATIVE_WORKTREE |
| go2-arx5-hierarchical-vla-rtx4080s | `d080f4e0011e` | 1295 | refs/对象/原字节/独立恢复 PASS | RETIRED_NATIVE_WORKTREE |
| isaac-gr00t-rtx4080s | `51d4c89f72fd` | 431 | refs/对象/原字节/独立恢复 PASS | RETIRED_NATIVE_WORKTREE |
| go2-arx5-hierarchical-vla-rtx4080s-dynamic | `dffef989b2dd` | 855 | refs/对象/原字节/独立恢复 PASS | RETIRED_EXACT_MANIFEST |
| go2-arx5-hierarchical-vla | `dc246b6fa199` | 1333 | refs/对象/原字节/独立恢复 PASS | HELD_EXACT_REASON |
| isaac-gr00t | `51d4c89f72fd` | 3023 | refs/对象/原字节/独立恢复 PASS | PROTECTED_SOURCE_RETAINED |
| isaac-gr00t-rtx4080s-remote | `51d4c89f72fd` | 476 | refs/对象/原字节/独立恢复 PASS | PROTECTED_SOURCE_RETAINED |

上述 8 份 Git 工作树均已独立恢复并逐文件核对 hash、mode 和符号链接。所有捕获的 refs、HEAD/tree、reflog 和 unreachable 对象通过独立对象库核验；Git bundle、原始 common-dir、工作树原字节及精确 patch 保存在私有恢复区，没有借用待删 objects/alternates。本轮 Git 候选 HEAD 均由正常刷新后的 origin 覆盖；额外保存本地 refs、stash/reflog 和孤立对象，不把“remote HEAD 可达”扩大成“所有服务器内容都已上传”。

旧官方父仓库的 5 个初始化嵌套仓库（LIBERO、SimplerEnv、ManiSkill2_real2sim、robocasa、robocasa-gr1-tabletop-tasks）另行保存完整 Git 元数据、原工作树、portable bundle 并独立恢复。它们没有被当成空 Gitlink 丢弃。旧官方 CLAUDE.md 和 inference notebook 的 **原始字节及精确 patch** 均私有保存，原件 hash 未变，notebook 输出未上传。受保护 WBC、mesh、checkpoint/Trainer、原始数据及凭据未进入本次公开报告或新增 Git 文件。

`stage45e_n2_runtime_5080d53` 为无 Git 的 147 个小型源码、配置和文档，逐文件证明没有环境/数据树后完整迁入 archive，原路径实际退役。22 项前次 copy 的文件重新核对原 inode/mtime/hash，补齐空目录与权限并完整回读后退役；没有追溯改写前期 copy-only/delete0 收据。

## 两节点统一入口

沿用 `~/Projects/GR00T-WBC`。`code/isaac-gr00t` 已原子导航至该入口内部的独立官方 checkout，固定 `51d4c89f72fda44cbf77285c6a8114b52676b8a1`；新 .git/common-dir 和 alternates 不再依赖旧目录。`code/vla` 已正常 fetch，origin 为 `f9929001b32f442f54192205d8d043a1ede47d69`；PI0292 当前租约固定 `e3cdb22a1802fcae1d97e670487d5bc60b5e5c0b`，作业边界前保持这个 HEAD。

AMD 使用现成的 base312 管理环境（Python 3.12.9、NumPy 2.3.3）；4080S 仿真与 CPU 数据 profile 使用原来的绝对解释器；Hopper 只登记已有 Python 3.12.3/PBS 环境元数据。环境没有移动，新安装依赖 **0**，全局 PATH/SSH/Git 设置没有修改。

```bash
# 在 AMD 上
~/Projects/GR00T-WBC/infrastructure/enter.sh amd-management
# 在 4080S 上
~/Projects/GR00T-WBC/infrastructure/enter.sh 4080-simulation
~/Projects/GR00T-WBC/infrastructure/enter.sh 4080-cpu-data
```

两节点各实际运行 **18 项 CPU 工程测试＋4 项 resolver 测试，均 PASS**。实际解释器和 canonical 模块 __file__、固定官方 Git 身份、processor 源码字节、固定 Cosmos revision 的 9 个 tokenizer/config blob、原 PI0291 statistics/provenance/manifest、一个完整原始样本导出 hash、资产与数据的主机解析均验收通过。仿真环境只验收它已有的合法模块；官方 pose 的实际导入在 CPU 数据环境验收。错误主机实际返回 `HOST_REQUIRED`，没有自动 SSH 或启动工作负载。Hopper 入口未部署/执行，仅保留外部 PBS profile 元数据，不宣称 Hopper 完整入口 PASS。

四个旧 Gitdir 案例均在 AMD 的有效 common-dir 中通过 `git worktree remove` 非 force 移除。未先修复无须保留的 4080S 旧 Gitdir，没有 force、global prune、git clean、reset 或直接删除 .git/worktrees。

## 删除与空间

- 删除对象 allocated bytes 下界：**327,299,072 bytes（312.14 MiB）**。原生移除的 .git 指针文件和父库管理元数据没有计入，故为下界；已删除源对象的 hardlink 清单为 0 项。
- 其中冷态迁移源对象 **7,614,464 bytes**，在新 archive 完整保留，不计释放。
- 新私有恢复区：**5,630,922,752 allocated bytes（5.24 GiB）**；新官方 checkout **88,125,440 bytes**；新冷态 archive **7,614,464 bytes**。恢复区峰值低于 16 GiB；统计不跟随符号链接。
- AMD 共享文件系统 available bytes 的本次观测差值 **-5,926,895,616 bytes（-5.52 GiB）**，AMD 本地 SSD 差值 **-13,611,008 bytes**。4080S SSD 缺少本次 T0 容量采样，只有最终观测；本批没有删除任何节点本地 SSD 候选，不声称该 SSD 有本次回收。
- **物理净释放未证明**：Btrfs/reflink/快照共享、NFS quota 和持续的 PI0292 写入没有被独占测量。保留新增备份后，共享容量观测反而下降；没有把 copy、rename 或 symlink 当成空间释放。之前 BF16 删除 **44,017,266,688 bytes（40.994 GiB）** 单列排除，本次不重复计。

维护新增 model load/forward、physics、train、PBS、WBC、hardware **均 0**，终止健康作业 **0**。已记录 CPU 87.34s，连同早期未单独计时的消费者检查/辅助读回保守 allowance，上界 0.574 core-hours，小于 8。并发上限 4 CPU，后续维护及验收 worker 均限制为单 CPU；持续科学任务的 CPU 与写入不记作维护。

## 仍原位保留的 23 项及精确原因

- `GR00T-WholeBodyControl`：Protected historical WBC original; current science lease protects this repository. No retirement authorization for unique WBC.
- `_handoff`：Protected credentials and original handoff assets; deliberately unopened and unchanged.
- `go2-arx5`：Protected mixed asset root containing private WBC, PI0250/PI0276/PI0290/PI0291/PI0292 and host-local aliases.
- `go2-arx5-hierarchical-vla`：PARENT_HAS_SURVIVING_LINKED_WORKTREES
- `isaac-gr00t`：Initialized third-party submodules (5) and parent common-dir still required by protected PI0280 official worktree and PI0292 old shared official worktree. Dirty CLAUDE/notebook raw bytes and exact patches are privately restored.
- `isaac-gr00t-rtx4080s-remote`：PI0292 HELD_FOR_SCIENCE explicitly pins this direct source path; no switch while dispatcher is collecting.
- `Downloads`：User general-purpose Downloads directory, currently empty; outside cold project migration purpose.
- `bin`：Protected legacy Home environment; interpreter link targets unavailable unitree-rl environment.
- `include`：Protected legacy Home environment ABI/include directory.
- `lib`：Protected legacy Home environment package directory.
- `lib64`：Protected legacy Home environment symlink relationship.
- `pi0209_corelab_stage45i_20260918`：Unique historical model/checkpoint/Trainer lineage; about 54 GiB exceeds 16 GiB recovery cap and no complete independent restore exists.
- `pi0215_a049_runtime`：Retained linked runtime worktree and original environment dependency; also prevents old VLA parent retirement.
- `pi0220_v0_20260918`：Contains original native expert rows and anatomy evidence; not covered by the PI0291 fixed HF archive. This pass does not migrate original scientific datasets.
- `pi0223_goal_b_20260918`：Original environment/materialization tree; absolute interpreter and package provenance must remain intact.
- `pi0240_transfer_staging_20260920`：About 18 GiB historical transfer payload exceeds bounded recovery cap; full fixed external coverage/readback is not available.
- `pi0246_gate_pass_archive_20260924`：Checkpoint processor/statistics plus unique scheduler/RNG/Trainer metadata and partial transfer locks; preserve original lineage.
- `pi0246_gate_pass_campaign_20260924`：Mixed original runtime/environment and campaign evidence; preserve absolute environment and original science assets.
- `pi0246_gate_pass_screen_20260925`：Original condition/action/backbone feature evidence and model symlinks; independent full science recovery not qualified.
- `pyvenv.cfg`：Protected original legacy Home environment configuration.
- `stage13_report_staging`：Protected original private WBC delivery family, including empty report tree.
- `stage13b_staging`：Protected original private WBC delivery family.
- `stage13c_staging`：Protected original private WBC delivery family.

**唯一下一项动作**：在 PI0292 作业边界刷新科学 lease、切换已明确的旧源码绑定到独立 canonical 入口，并复核父库的 PI0250/旧环境 worktree 依赖，再判断剩余代码候选。该动作不重置科学时钟，也不自动扩大到唯一原始数据、checkpoint 或环境。

## 交付文件

- [代码备份覆盖表](CODE_BACKUP_AUDIT.json)
- [56 个候选的退役／迁移／保留清单](RETIREMENT_MANIFEST.json)
- [33 个实际退役源路径的删除收据](DELETION_RECEIPT.json)
- [跨节点 worktree 和父子关系收据](WORKTREE_RECEIPT.json)
- [AMD／4080S 统一入口和环境验收](ENVIRONMENT_ENTRY_CHECK.json)
- [Home／Projects 完整前后可见项、别名、归档、原位保护及容量观测](BEFORE_AFTER.json)

私有原路径、逐文件 hash/inode journal、FD、原 patch/notebook 和恢复 payload 保留在 owner 私有环境；此目录只含脱敏收据。ENTRY_IMPLEMENTATION 是实际部署入口的源字节副本；运行还需要 owner 私有 HOST_PATHS/ENVIRONMENTS/ENTRY_HASH_BASELINE，公开占位符不能当真实操作路径。
