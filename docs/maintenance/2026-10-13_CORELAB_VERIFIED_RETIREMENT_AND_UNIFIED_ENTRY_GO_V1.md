# Corelab — 已验证旧副本退役与统一入口执行 GO V1

日期：2026-10-13。类型：USER_REQUESTED_CONDITIONAL_MAINTENANCE_EXECUTION_GO。
执行者：已有 local-pc Codex，使用用户已经配置的 SSH；单一文件变更 owner。
审阅远端 HEAD：3ad70e75af8976149b37a061ba4a5cd9d1f060c0。
承接：Issue #47 comment6137134759，目录 reports/maintenance/2026-10-13-corelab-workspace-consolidation/。

## 1. 用户决定与范围

用户要求确认代码是否已推送，然后直接清理旧副本、整理 Corelab 环境并统一入口。本 GO 允许已有 owner 在逐候选通过备份、依赖及路径检查后，实际迁移/退役，不停留在再写一份建议，也不对同一已通过候选重复请求批准。此前 copy-only 或 retirement-deferred 是当时的结果，不是禁止本次有证据退役的永久限制。

这不是整机清空、Home白名单反选删除或唯一科研原件销毁授权。受保护 WBC、当前原始数据、唯一 checkpoint/Trainer、凭据和仍需运行的环境不因 GitHub 认证成功而变成冗余。遇到未知项，先在本次任务中补齐证据；确实不能闭合才按候选保留，不能由一个候选阻塞所有其他安全动作。

一次性有界执行：首次采用后最多4小时，最多8新增CPU core-hours、并发不超过4CPU；最后30分钟停止新增迁移并收尾。临时恢复区峰值不超过16GiB，空间不足优先处理较小候选，不删除其他源文件腾空间。此预算是上限，不是交付时间承诺。恢复沿用本次T0/计数；不重置PI0292科学时钟或资源。维护自身新增 physics/model forward/train/PBS/WBC/hardware均0，不终止健康作业，不新建云端owner或隧道。

本 ChatGPT 已读取 GitHub 收据，但没有直接 Corelab shell。发布本文仅代表执行指令已交接，实际SSH、删除数量和回收由owner收据证明。

## 2. 已知事实，不能再用一句“全部代码已上传”替代核验

- 现役GitHub分支已包含PI0291代码/结果、PI0292计划与最新工作区整合报告。新入口创建时code/vla是独立checkout，固定e3cdb22a；必须现场确认它是否已同步到当前远端，不把报告已push当服务器已pull。
- 新入口 ~/Projects/GR00T-WBC 已实际存在。code/isaac-gr00t仍指向旧共享clean worktree，是导航alias，不是独立可恢复源码副本。
- 共享isaac-gr00t仍有CLAUDE.md、getting_started/GR00T_inference.ipynb的本地修改。前次报告保护了它们，但未证明其内容已备份/推送。
- 四个linked worktree在4080S的.git引用AMD-only /mnt命名空间；在AMD能读取。这不是内容丢失证据，也不要求先修复所有主机再清理。
- 前次24个Projects观测是两节点观察，不是24份独立资产；另有44个Home review条目。ignored/untracked、stash、local-only对象及依赖未完成逐项恢复核验。
- PI0291固定HF revision c25eeadf9be0aa7d46bb6f7b65e260cbf3a305ea覆盖其归档成员，不覆盖所有Home文件、历史checkpoint或私有WBC。原56 TRAIN、33,656窗口和全部失败/原证据继续保护。

## 3. A0 — 与科学执行者锁定现役依赖

刷新#11/#47和真实进程/lease。复用四份原storage audit、完整私有候选表、ACTIVE_DEPENDENCIES和入口registry，不再次无目的扫描181万文件。私有原件需读取/hash；公开脱敏路径不能当真实操作路径。

记录每个候选的host、逻辑路径、realpath、对象类型、mount/filesystem身份、device/inode/link、所有权/权限、实际消费者及目标目录。共享NFS用共同export/相对路径等证据对应两节点；不能因st_dev在两节点不同就重复计算，也不能仅凭inode相同就合并不同文件系统。

建立maintenance lease与PI0292依赖排除表：运行中代码、Python解释器、editable install、数据、资产、工作目录不可迁移/删除；作业边界再切换。新主checkout或某个目录在使用时，仅延期该目录，其他独立候选继续。维护不让科学任务无限等待全部旧目录整理完。

## 4. A1 — 代码备份闭合，先救唯一内容

对每个不同Git common-dir及其worktree，使用其能正常读取的节点完成：

1. 刷新明确的合法origin，记录HEAD、全部本地branch/tag、detached HEAD、stash和worktree登记；检查local-only提交，必要时核查reflog-only提交。不得运行gc、prune或reset丢弃证据。
2. 核对tracked修改、staged与unstaged差异、全部untracked、ignored及嵌套仓库/子模块/LFS。普通git status为空不等于目录没有唯一文件。命令输出使用NUL安全处理，不能按空格拆路径。
3. GitHub可保存的项目源码在内容/凭据检查后推送用户已有私有仓库的独立recovery ref，或保存可独立恢复的Git bundle。不要把历史改动自动合入现役科学分支，不改变上游官方冻结源码。无须把已经由远端受保护ref覆盖的同一提交重复推送。
4. Git不覆盖的修改、ignored文件、LFS实体、submodule对象、未跟踪脚本分别处理。唯一小文件保留原始字节、mode、符号链接关系和hash到私有恢复区；大科研原件留在原位或合法持久存储，绝不能git add -A整仓上传。Git bundle不自动包含所有LFS实体、外部子模块、工作树差异或reflog-only对象。
5. 尤其保存CLAUDE.md及inference notebook原始修改和精确patch；notebook输出可能含私有信息。不能为方便GitHub推送而删除输出后丢弃原始字节。含凭据/私有数据的内容不进Git，私有恢复副本不改变账户权限。
6. 在独立新目录、不借用将被删除的objects/alternates的条件下，实际clone/restore候选代码并核对HEAD/tree/全部必须保存文件hash。bundle verify本身不替代内容恢复。小文件可完整回读；大文件以完整成员清单、远端固定对象hash及必要实际恢复证明，不以一个READY或一个样本覆盖整库。

输出逐候选CODE_BACKUP_AUDIT.json：remote-reachable、local-only、dirty/untracked/ignored、stash/LFS/submodule、保护位置、实际恢复验证、未覆盖项。最终可以声称“本批拟删除代码副本全部恢复闭合”，不能把它扩大为整台服务器所有内容都已在GitHub。

## 5. A2 — 统一入口必须脱离待删旧目录

保留现有 ~/Projects/GR00T-WBC，不再创建第二个项目顶层。目标：

```text
GR00T-WBC/
  README.md
  code/vla/                 # 独立主checkout，现役分支
  code/isaac-gr00t/         # 固定51d4c89f的独立源码，或明确仍受保护的依赖
  assets/ASSET_INDEX.json
  datasets/DATASET_INDEX.json
  experiments/
  infrastructure/
    HOST_PATHS.json
    ENVIRONMENTS.json
    enter.sh
    doctor.py
  archive/                  # 小型历史代码/报告的恢复索引；不盲复制大原始库
```

- 先检查code/vla dirty状态，正常fetch并fast-forward或保留并发提交后合并；禁止reset/force覆盖。运行实验绑定自己的源码快照，不跟随活动分支突然变化。
- 在删除官方旧common-dir前，code/isaac-gr00t必须真正独立：用验证过的官方固定源码建立独立checkout，核查.git/common-dir/alternates不再指向候选。先在新路径验证，再原子切换导航，旧alias本身仅unlink，不递归跟随。不能删掉它仍然依赖的parent对象库。
- 不合并/修饰师兄WBC；私有原件只登记受保护位置。不把SSD原始数据搬回NFS来满足目录外观。
- ENVIRONMENTS按角色登记AMD管理、4080S仿真/CPU数据、Hopper神经环境与绝对解释器、版本、源码位置。统一入口不等于强行共用一个venv。
- enter.sh显式检查host与profile，错误主机返回HOST_REQUIRED。doctor检查路径、模块__file__、Git SHA、固定processor字节、数据/统计hash；不加载神经模型、不启动physics。
- 保留可用环境在原处；venv可能写有绝对shebang、.pth、editable安装。不得直接mv整个venv然后假定可用。本轮不新装大型依赖，不改全局PATH/SSH/Git设置；不删除Home的bin/lib/include/pyvenv.cfg或隐藏凭据目录。
- 历史静态报告里写旧路径不必修改。需要闭合的是仍能执行的launcher、.pth、editable install、脚本、lease及访问入口；保存旧→新映射供历史复现。

## 6. A3 — 现在允许的真实退役

### 6.1 明确候选边界

仅从前次私有RETIRE_CANDIDATES/DIRECTORY_INVENTORY解析精确路径，按对象身份去重。优先处理下列旧源码/运行副本：

- go2-arx5-hierarchical-vla-pi0240
- go2-arx5-hierarchical-vla-pi0241
- go2-arx5-hierarchical-vla-rtx4080s
- isaac-gr00t-rtx4080s
- go2-arx5-hierarchical-vla-rtx4080s-dynamic
- isaac-gr00t-rtx4080s-remote
- go2-arx5-hierarchical-vla（旧主仓库，必须最后处理其common-dir关系）
- stage45e_n2_runtime_5080d53（须先分离源码/数据/环境；不能按名字整体判冗余）

Home仅处理既有44项review中经过同样资格验证的旧项目副本、小报告/脚本和明确可再生缓存；不是允许删除所有未列入新入口的Home对象。前期copy/verify过的22项可以复用其完整复制证据，但必须重新确认内容未变和消费者已切换，才能删除原副本。二期历史记录仍保持copy-only/delete0，不追溯改写。

### 6.2 每项通过即执行，不要求全机先PASS

退役条件是同时具备：精确path/type/inode/hash与预检一致；必要内容已独立恢复；不含受保护子树或未知嵌套mount；当前科学和环境不再引用；owner对两节点已知消费者完成核验且无写入者。审计只覆盖用户有权限的范围；用户私有明确受控目录无须检查其他用户全部进程，共享/跨用户使用不明确的目录则保留。

四个跨节点问题优先在AMD的有效Git命名空间处理。无消费者的旧linked worktree可从正确common-dir直接执行针对该单一路径的git worktree remove；不必先让4080S恢复旧路径。仍需保留的worktree才考虑有记录的move/repair，先备份相关管理元数据、确定可在两节点解析的路径并验证；不得全局字符串替换.git。

默认使用非force的native removal；被拒绝时先检查具体dirty/lock/submodule原因。禁止以--force/--force --force、全局prune --expire now、git clean -fdx或直接删除.git/worktrees绕过未过门。已备份的残留文件可通过精确manifest处理，tracked修改只有在保存并成功重建原内容后才能对精确文件恢复到已登记版本；再次确认clean，再正常remove。locked或真实消费者未解释时保留该项。

普通独立副本按封存的文件manifest逐项unlink、空目录rmdir，不使用Home或Projects顶层反选递归删除，不跟随symlink，不跨mount。Git主对象库最后退役：确认所有存续linked worktree、submodule和alternates已迁出或保留，且新checkout完全独立，否则不能删除旧parent仓库。

执行前再次lstat/检查目录内容没有新增、没有换成链接或挂载；若变化立即停止该候选。任何删除中途失败保存部分收据，重连从收据继续，不能按旧清单盲重放。不能删除当前脚本、cwd、恢复区或日志本身。

### 6.3 暂时不能删的历史资产也要归位

冷态、可迁移且没有运行依赖的小型历史代码/报告，可在hash验证与路径登记后迁到archive/legacy，源路径通过前述条件后实际退役。迁移仍保留数据，不计释放；不只是建立第二份重复副本就宣告整理完成。

原始科研数据、唯一Trainer/optimizer/RNG、PI0250/PI0276/PI0290/PI0291、当前PI0292数据与执行快照、私有WBC、原生mesh、_handoff凭据、Home环境继续保护。这里允许消除其经过完整验证的冗余副本，不授权销毁唯一原件；体积大或实验FAIL都不是充分删除理由。迁移这些活动/受保护根目录需要另有现成明确授权及任务边界，不包含在本批自动退役内。

## 7. A4 — 两节点验收与终局

实际运行现有18项CPU工程测试/4项resolver测试中仍适用于当前版本的集合，两端分别记录；验证canonical包导入、固定官方源码、原始样本/统计可读、数据和资产主机解析。必要的CPU loader读取可用已有合法环境，但不进入模型/physics。清理后重新核对新checkout、全部保留worktree与原基准hash；不能只看ls变整齐。

终局在reports/maintenance/2026-10-13-corelab-verified-retirement/<session>/记录：
- CODE_BACKUP_AUDIT.json：本批每份代码的远端/私有恢复覆盖与未覆盖项。
- RETIREMENT_MANIFEST.json、DELETION_RECEIPT.json：每个真实删除/迁移/保留对象、恢复引用和原因。
- WORKTREE_RECEIPT.json：四个跨节点案例和parent/child关系如何处理。
- ENVIRONMENT_ENTRY_CHECK.json：AMD/4080S统一入口及实际解释器/模块路径验收。
- BEFORE_AFTER.json：Home/Projects的准确可见项、别名/归档/原地保护分别列出。
- FINAL_REPORT.md：候选总数、已退役数、剩余阻断及唯一下一项动作。

空间分别报告删除对象的allocated bytes、仍有hardlink的对象、增加的备份/新checkout、实际文件系统空间变化。reflink/Btrfs快照共享或quota不可见时不伪称物理净回收已证明；rename、copy和symlink不是磁盘释放。之前40.994GiB单列，本次不重复计。

只有真实满足才返回LEGACY_COPIES_RETIRED_AND_ENTRY_VERIFIED。原位仍有必须保留项时返回PARTIAL_WITH_EXACT_HOLDS，逐项给具体原因；不能再次用笼统的global dependency UNKNOWN解释所有候选，也不能为了非零删除而删除不合格对象。“统一入口可用”与“Projects物理只剩一个目录”分别报告。

提交sanitized报告并在#47逐字回读，#11仅按需登记科学依赖握手；不修改PI0291 PARTIAL和PI0292原协议。私有精确路径、FD、恢复patch和凭据留owner私有环境。没有SSH执行收据之前，删除和释放仍为未执行，而不是本GO的结果。

## 8. 固定依据

- 工作区整合实际回报：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6137134759
- 整合原报告：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3ad70e75af8976149b37a061ba4a5cd9d1f060c0/reports/maintenance/2026-10-13-corelab-workspace-consolidation/CONSOLIDATION_REPORT.md
- 精确候选的脱敏索引：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3ad70e75af8976149b37a061ba4a5cd9d1f060c0/reports/maintenance/2026-10-13-corelab-workspace-consolidation/RETIRE_CANDIDATES.json
- 活动依赖/本地修改证据：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3ad70e75af8976149b37a061ba4a5cd9d1f060c0/reports/maintenance/2026-10-13-corelab-workspace-consolidation/ACTIVE_DEPENDENCIES.json
- Git worktree官方语义：https://git-scm.com/docs/git-worktree.html（执行者先核对实际Git版本，不假定新选项可用）
- Python venv非可直接搬迁目录：https://docs.python.org/3/library/venv.html
