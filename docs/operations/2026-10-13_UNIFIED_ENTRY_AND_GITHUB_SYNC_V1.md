# GR00T＋WBC 统一入口及 GitHub 同步规范 V1

**日期**：2026-10-13。**类型**：WORKSPACE_STANDARDIZATION_CONDITIONAL_GO。**执行者**：现有 local-pc Codex 单一维护协调 owner；不启动第二个云端 Codex。**状态**：规范已发布；本文件不是 CoreLab 执行收据，也不是新的科学／模型训练授权。

## 0. 决策

将 CoreLab AMD/4080S 共享 NFS 的 \`~/Projects/GR00T-WBC\` 固定为**唯一人类入口**。后续新代码与实验一律由这里的受检入口和不可变 run capsule 进入；在完成依赖切换前保留旧物理目录，**入口唯一**不等于**整机暂时只剩一个目录**。

本 GO 复用已经完成的入口：\`infrastructure/enter.sh\`、\`doctor.py\`、私有环境与 host-aware 路径 registry。不要再复制出第二套同名入口。对已有实现先做源码归属与自动测试再封装成可复用工程，不影响已运行实验。

**严禁**借此次统一改动修改 PI0291/PI0292 冻结物理、教师、数据、stats、训练目标、WBC、模型参数、安全阈值及科学质量判定；不擅自提交 PBS、模型加载、物理 rollout、新数据采集或新数据删除。PI0292 的现有授权／截止和科学 source capsule 继续生效。

## 1. 当前可据以交接的实情（刷新现场收据优先）

1. #47 清理终局 \`3bbbb9ca72637ee941aee73593fd66546e5fe756\`：33 个原源路径退役、Home 48→21、Projects 13→7、四个旧 linked worktree 已在 AMD 通过非 force 移除。状态 \`PARTIAL_WITH_EXACT_HOLDS\`，剩余 23 项有明确保护或依赖原因。本次不得重新删除、重复计数。
2. \`code/vla\` 是真正独立的现役仓库 checkout，但为保护运行中的 PI0292，当时 HEAD 保留 \`e3cdb22a1802fcae1d97e670487d5bc60b5e5c0b\`；远端分支继续前进。**不同步不代表丢失代码，更不能在进程使用时直接 git pull/checkout。**
3. \`code/isaac-gr00t\` 已经是新目录内独立的官方 pinned \`51d4c89f72fda44cbf77285c6a8114b52676b8a1\` checkout，并在 AMD 和 4080S 导入／hash 通过。仍有旧官方 parent/worktree 因科学依赖及第三方 submodule 保留。
4. PI0292 的 DEV16＋TRAIN8 已在新 successor namespace 首次尝试并取得 v2 bank \`64 TRAIN +16 DEV\` 准入与全窗口 CPU 数据资格。**截至本规范制定时新私有 HF/训练交付仍须以最新收据核查，不以 bank 资格代替 Trainer 模型运行。**旧 PI0291 自身的历史 PARTIAL 永不回写。
5. 原始 PI0291 RGB、物理、视频、模型资产在 CoreLab 4080S SSD；现役 Hopper 神经源／模型／PBS 由 Hopper 管理。AMD 上看不到 4080S SSD 路径不是丢失。
6. 并不是“服务器全部代码都已在 GitHub”：旧 \`isaac-gr00t\` 的 \`CLAUDE.md\`、inference notebook 本地修改及历史 Git refs/ignored/reflog 原字节已作私有独立恢复；这**不等同于已全部推送 GitHub**。不得把 notebook 输出、WBC 私有代码/mesh、keys/weights/raw 直接 git add。

## 2. 三类权威来源（single source of truth）

| 对象 | 权威来源 | 在 GR00T-WBC 中保存 |
|---|---|---|
| 自有 VLA 科学源码、入口代码、接口协议、脚本、测试 | GitHub 私有仓库 \`Leo-6-maker/go2-arx5-hierarchical-vla\` 的**固定 commit SHA**，活跃开发分支 \`pi0267/executor-decoupled-recovery\`，稳定历史 \`main\` | \`code/vla\` 可验证 checkout，执行时引用不可变 source capsule |
| NVIDIA GR00T 官方包 | 固定上游 commit \`51d4c89f72fda44cbf77285c6a8114b52676b8a1\`；如需第三方修改需单独 patch/adapter 及来源 | \`code/isaac-gr00t\` 独立 checkout；不能在官方树内混入未披露改动 |
| 自家采集/审计数据及大模型 | CoreLab SSD 工作原件＋私有 Hugging Face **固定 revision/prefix/file/member SHA**，Hopper 原模型/训练器保留副本 | \`datasets/DATASET_INDEX.json\`、\`assets/ASSET_INDEX.json\`、run manifest 和 host-aware 私有路径绑定 |
| WBC、机器人 mesh、历史唯一 Trainer、密钥、Home 环境 | 授权的原始私有存储及经真实回读的独立私有恢复来源；**不是 GitHub 备份** | 只读 role 及来源/hash 索引，按显式权限控制，不向 Git/HF 私自公开 |
| 实际运行状态／科学结果 | 活动 owner 的持久 ledger 和原始收据；GitHub #11 发布科学脱敏审计，#47 发布维护脱敏审计 | \`experiments/<PI>/<run_uuid>/\` 的路径登记、协议 SHA、报告链接 |

原则：GitHub commit 证明代码版本，HF revision 证明归档成员；两者都不自动证明远端进程运行、模型能力或全部节点数据已备份。没有实际 readback 不写 PASS。

## 3. 标准工作区（逻辑唯一，物理存储分角色）

\`\`\`text
~/Projects/GR00T-WBC/
  README.md                         # 人类唯一入口和现役科学/维护导航
  code/
    vla/                            # 自有VLA repo的独立Git目录
    isaac-gr00t/                    # 官方N1.7固定源码，独立对象库
  infrastructure/
    enter.sh                        # 已部署 profile wrapper，不重复造轮子
    doctor.py                       # 已部署只读身份+主机+hash验证
    ...                             # resolver/角色配置/入口的版本化实现
    private/                        # 私有、gitignored，仅host自己的真实路径和环境
    leases/                         # 活动PI科学/维护lease；不得因清理而删
  assets/ASSET_INDEX.json           # 路径+hash/来源，原WBC只读
  datasets/DATASET_INDEX.json       # 固定TRAIN/DEV版本+HF revision，非大文件库
  experiments/<PI>/<run_uuid>/      # 小型运行manifest导航；raw在所属SSD
  archive/                          # 经证明无活动消费者的小型历史代码/报告
\`\`\`

不移动 Home 整套 \`bin/lib/include/pyvenv.cfg\`；venv 与可执行脚本存在绝对路径和依赖，不可简单 mv。不要从 AMD 构造指向 4080S node-local SSD 的假跨节点软链接。保持 \`GR00T-WholeBodyControl\`、\`_handoff\`、\`go2-arx5\` 的受保护源权威，直到独立恢复/消费者切换的单项证明闭合。

### 唯一稳定用户命令（**已实际存在**）

\`\`\`bash
~/Projects/GR00T-WBC/infrastructure/enter.sh amd-management
# 在4080S执行：
~/Projects/GR00T-WBC/infrastructure/enter.sh 4080-simulation
~/Projects/GR00T-WBC/infrastructure/enter.sh 4080-cpu-data
\`\`\`

\`enter.sh <profile>\` 无额外命令执行 read-only doctor；可以在通过 profile 的本机用 \`enter.sh <profile> python -m <approved-module> ...\` 进入经验证解释器。不能把字符串命令当作无需 PI GO 的执行授权。错误主机必须 \`HOST_REQUIRED\`，不隐式 SSH。当前 Hopper 只有 profile 元数据，**没有**现场部署 PASS；合法 PBS allocation 才可开展模型神经计算。

### 下一步应由 Codex 版本化的稳定 API（**设计，尚未实施**）

新建经过测试的 \`scripts/workspace/\` 或 \`src/go2arx5_vla/workspace/\` 共享 CLI/library，复用上述实际 \`enter.sh/doctor.py\`，禁止引入第二套不一致的路径解析器：
- \`workspace doctor --profile ...\`：只读核查环境、代码与 processor、registry、stats和样本字节。
- \`workspace paths --role <ROLE>\`：通过 hostname＋registry 返回本机绝对路径或 \`HOST_REQUIRED\`；**不猜路径**。
- \`workspace status --run <RUN_UUID>\`：只读查看原持久ledger、job owner、引用和lease；状态快照不等于执行证据。
- \`workspace prepare-run --protocol <fixed-ref> --run-id <UUID>\`：仅封存新运行清单、权限和身份；任何物理／模型启动仍需单独协议 GO、资源保护、worker ownership 与 checkpoint。
- \`workspace validate-run --run <UUID>\`：核验 manifest、输出hash、科学/维护receipt与固定依赖的对应关系。

旧 \`scripts/piXXXX/\` 保持可追溯路径，不批量重命名；新实验统一调用共享库而不是复制新的裸 SSH/路径/锁实现。实现和命令实际存在前应明确显示 \`NOT_IMPLEMENTED\`，不得文档先写成 PASS。

## 4. GitHub 工作流：远端代码与服务器如何同步

### 4.1 分支职责

- \`main\`：已经正式接受、可复现的稳定导航／里程碑，不自动部署或用它覆盖活跃实验。
- \`pi0267/executor-decoupled-recovery\`：当前**实际活跃科学代码分支**，不是默认 \`main\`；读Issue #11及最新 receipt 决定允许的执行。
- 短期 \`feat/<PI>/<topic>\`、\`ops/<topic>\` 或 \`recovery/<topic>\`：维护/新功能的有限独立更改，完成验证后合并归档，不随每次模型训练无限新增持久头。
- 每个**已运行的科学实验**以 full commit SHA＋source seal＋run UUID固定；不能用 mutable branch HEAD 代替实际执行源码。

### 4.2 开发和 push（现有 local-pc 为唯一协调 owner）

在待修改的真正 Git repo 目录，先 \`git status --porcelain=v1 --untracked-files=all\` 和 \`git fetch origin\`；对差异、dirty、stash、local-only对象形成报告。改可复用源码时创建窄范围任务分支，做静态/CPU契约测试。**不要**把旧 WBC、私有mesh、checkpoint、raw视频、token、notebook私有输出 \`git add -A\`。

正常 \`git commit\` 后，普通 fast-forward push/PR，记录远端完整 commit SHA 与变更文件列表，再 \`git fetch\`／逐关键文件回读；合并冲突明确人工/owner解决，禁止 \`git push --force\`、\`reset --hard\`、\`git clean -fdx\` 或以 GitHub 能拉取 HEAD 代替本机忽略文件的恢复核查。

任何报告上传前按 GitHub 路径作脱敏；私有实际 host paths/token 保留在权限受控的 registry，而非仓库文件。

### 4.3 CoreLab 拉取更新的精确流程

**首先分两种工作树**：执行中的不可变 \`source capsule\` 永远不 pull；人类 \`code/vla\` checkout 只能在没有活动消费者、完成作业边界切换后更新。当前 PI0292 租约原绑定 e3cdb22a，不能因为 GitHub 新 commit 推送就硬切。

\`\`\`bash
cd ~/Projects/GR00T-WBC/code/vla
git status --porcelain=v1 --untracked-files=all
git rev-parse HEAD
git fetch origin
git rev-parse origin/pi0267/executor-decoupled-recovery
# 仅当工作树干净、科学owner确认进入job边界、当前分支正确且fast-forward可行：
git merge --ff-only origin/pi0267/executor-decoupled-recovery
\`\`\`

若 \`merge --ff-only\` 拒绝，则保存本地分叉和差异、停止该次更新；不用 reset/force 消除分叉。更新入口绑定及版本化 \`ENTRY_HASH_BASELINE\` 后，AMD 和 4080S 的 doctor／工程契约与实际模块 \`__file__\` 必须 PASS；若失败立即保留旧执行 snapshot并报告。

\`code/isaac-gr00t\` 则按 NVIDIA 固定 SHA 验证，**不**跟随 upstream 最新版本自动 pull。Hopper 同步的是已批准的固定 commit、官方 processor/权重完整 hash、DataSet/HF revision 和PBS环境，不从 CoreLab 本地文件夹盲复制。

### 4.4 保证“写了GitHub以后真能运行”

采用唯一\`RUN_MANIFEST.json\`规范，每个运行至少记录：
\`\`\`json
{
  "run_uuid": "<unique>",
  "pi": "<PI_number>",
  "authority": {"protocol_ref": "<immutable_git_sha>", "go_comment": "<url-or-id>"},
  "owner": "existing_local_pc",
  "code": {"vla_commit": "<full_sha>", "official_gr00t_commit": "<full_sha>", "source_seal_sha256": "<sha>"},
  "interfaces": {"action_schema": "EEF10_H40", "processor_sha256": "<sha>", "statistics_sha256": "<sha>"},
  "data": {"dataset_version": "<vN>", "hf_revision": "<full_sha>", "manifest_sha256": "<sha>"},
  "execution": {"profile": "<host-qualified-profile>", "private_path_role": "<ROLE>", "script": "<pinned_module_or_script>"},
  "limits": {"resource_ledger_ref": "<path>", "deadline_utc": "<immutable>"},
  "status": "PLANNED_NOT_EXECUTED"
}
\`\`\`

对 WBC-only 和维护任务，数据／模型不适用字段显式 \`NOT_APPLICABLE\`，不要虚构 SHA。状态只由运行收据从 \`PLANNED\` 跳为 \`RUNNING\`、\`PASS\`、\`FAIL\`、\`BLOCKED\` 等。run manifest 不含密钥或绝对私有路径。内部完整 manifest 可以引用 host-role 私有绑定。

## 5. 剩余 Projects 七项怎么处理

| 顶层对象 | 现阶段 |
|---|---|
| \`GR00T-WBC\` | **唯一工作入口，保留** |
| \`GR00T-WholeBodyControl\` | 师兄原始 WBC 历史仓库，**受保护**，通过资产索引只读引用 |
| \`_handoff\` | 私有交接／凭据，**原位保护** |
| \`go2-arx5\` | 混合资产、PI0250/0276/0290/0291/0292 原始数据，**不整树搬迁** |
| \`go2-arx5-hierarchical-vla\` | 旧VLA parent仍有存续linked worktrees。PI0292作业边界后检查旧依赖再有条件退役 |
| \`isaac-gr00t\` | 老官方parent含本地修改／五个历史子模块及仍保留消费者；不要以新独立checkout存在即删源 |
| \`isaac-gr00t-rtx4080s-remote\` | PI0292现役直接source绑定。必须在作业边界另做可复现consumer切换并逐字节比对才可退役 |

处理顺序：先保护旧源和所有运行 lease → 保存唯一文件与独立恢复验证 → 先迁消费者到新入口 → 两节点测试 → 若符合既有维护 GO 的单项精确删除门，再实际退役。保留不等于默认垃圾，**禁止为了让 \`ls Projects\` 只显示一个名字而移动受保护科学数据**。

## 6. Codex 具体执行批次和验收

**E0：观察**。刷新 #11/#47、活跃 PI0292 ledger/lease、HEAD/status、两端 \`infrastructure/doctor.py\`、剩余七目录、旧 parent consumers；实际记录精确私有路径/设备/refs，仅脱敏发布。若 PI0292仍写入，先在 source 变更上标 \`HOLD_ACTIVE_RUN\`。

**E1：统一入口代码的归属**。把已在 CoreLab 正常工作的 resolver、enter、doctor及小型配置 schema纳入可版本化自有 VLA repo，保留现有CLI语义；真实 PATH_REGISTRY/ENVIRONMENTS/ENTRY_HASH_BASELINE分 public schema 与 private host-bound文件，禁止把私有数据推Git。建立错主机拒绝、固定源码 SHA、数据registry、旧路径兼容、零物理/零模型验收。

**E2：后续所有新任务走入口**。建立模板和公共 \`RUN_MANIFEST\` schema，至少让一个零物理、零模型的诊断任务通过统一入口创建/验证。每个新PI启动前显式选择 profile+fixed code+data/model refs+科学GO+resource ledger；旧 sealed scientific source 保留原启动方法直到其作业边界。

**E3：安全切换，退役旧源码引用**。作业边界后 refresh active run，完成旧VLA/旧GR00T绑定向独立canonical的只读路径切换，在相同解释器和源版本下两端CPU测试PASS；若旧parent有其他 consumers，则精确保留并解释。只有独立恢复、子worktree/reference闭合后才在原有维护GO内删除，不强制减少剩余数。

**E4：GitHub同步闭环**。生成 \`WORKSPACE_CONTRACT_IMPLEMENTATION.json\`、\`GIT_SYNC_READBACK.json\`、\`RUN_ENTRY_CONTRACT_TEST.json\`、\`RETAINED_PATH_DECISIONS.json\`、\`ENVIRONMENT_ENTRY_CHECK.json\`、\`FINAL_REPORT.md\`，核对远端提交与files，#47发表实际维护终局，#11仅发科学依赖切换指针。最后显示两个独立状态：\`SINGLE_HUMAN_ENTRY_PASS/FAIL\` 与 \`PHYSICAL_TOP_LEVEL_ONE_DIR_TRUE/FALSE\`。

**验收条件**：唯一进入路径/固定身份/错误主机拒绝、AMD+4080S实际导入与数据hash PASS；新的只读测试由入口执行；所有活动PI source/数据原件不变；GitHub提交可按full SHA恢复；旧source按精确证据退役、每项有收据；未知引用不删除；无额外 WBC、physics、model-forward/train/PBS、健康作业终止。本轮维护与当前科学计算资源分别计账。

## 7. 责任及权限

本文件授权已有 local-pc Codex **实施统一入口规范及现有维护协议允许、单项恢复性已闭合的退役**；不赋予新的科学预算或跨越私人软件资产分发权限。当前运行不得被 Git 同步、缓存删除、路径迁移打断。维护和实验不得采用两个冲突 owner 写同一个 run root。

**报告按真实执行分层**：本 GitHub 文件为 \`DESIGN_AND_EXECUTION_HANDOFF\`；只有获得真实 AMD/4080S/private owner receipts 并回读后，才可宣称入口实施/目录删除/迁移完成。

固定历史依据：
- [#47 已实际完成的部分清理](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6137981521)
- [#11 PI0292 科学进展](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)
- [当前入口实施原字节副本](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/3bbbb9ca72637ee941aee73593fd66546e5fe756/reports/maintenance/2026-10-13-corelab-verified-retirement/20261013T031510Z-localpc-retirement/ENTRY_IMPLEMENTATION)
