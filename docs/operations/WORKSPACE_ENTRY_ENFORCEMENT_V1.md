# 统一入口强制接入与固定版本部署补充 V1

2026-10-13。**唯一主规范仍是 `docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md`**（已核对commit `79eb8825f97624825618c7d3fd31c4e16ab74b79`）。本补充承接其E0–E4及已有GO_TASK_CARD，不重新命名入口、不重做旧退役、不引入另一套gw命令或路径解析器。

用户要求不只是整理目录，还要使后续工作都接入标准流程。因此新增以下落地要求，由已有local-pc维护协调owner实施。只变更入口、发布/部署与路径管理；PI0292真实科学源、数据、统计、原有GO/截止/预算不变。本文件本身不是服务器部署收据。

## 1. 已完成与未完成，按最新证据区分

已知#47 comment6137981521确认33个源路径退役、Home48→21、Projects13→7，四个旧跨节点worktree已正常移除。旧“全部删除0/四个worktree仍未解决”是历史观察，不能继续当作当前事实。

已有enter.sh/doctor/三个CoreLab profiles实际通过；新的通用workspace CLI与所有后续launcher强制接入尚须实际实施。`code/isaac-gr00t`已是入口内独立固定官方源码，不再是早期指向外部旧对象库的唯一副本。`code/vla`此前因科学lease仍固定e3cdb22a，远端前进不等于本机丢代码。

剩余目录中包含现役消费者、历史parent对象库、原始数据/环境和私有WBC；它们的正确处理不是再无条件copy，也不是按ls反选rm。原恢复区/历史FAIL与已退役收据继续保留。

## 2. 入口源与部署副本分离

以已部署的enter.sh/doctor.py/registry为基准核对源码，复用其profile及已有验收。通用实现统一放在 `src/go2arx5_vla/workspace/`；必要的 `scripts/workspace/` 只做薄命令包装，不再复制路径解析/锁/同步逻辑。现有历史 `ENTRY_IMPLEMENTATION` 原件不覆盖。

部署后的 `~/Projects/GR00T-WBC/infrastructure/enter.sh` 与 doctor 保持现有接口，转调明确版本的入口实现。入口实现放 `infrastructure/releases/<entry-release>/`，以 `current` 或同等原子部署机制选择未来版本，并写DEPLOYMENT_RECEIPT。旧run绑定entry-release，不自动跟随current。

不要让新入口隐式依赖`code/vla`的可变HEAD。正在执行的科学源码不注入新AGENTS/入口文件来改变旧source seal；workspace根AGENTS可以指向当前规范，新run再采用。

私有HOST_PATHS、ENVIRONMENTS、ENTRY_HASH_BASELINE及registry继续本机受控存储。仓库只存schema/template/脱敏摘要；权限不能扩大。一个入口有多个角色环境，不直接mv venv。现有profile名为amd-management、4080-simulation、4080-cpu-data；hopper-neural仍需分别验证login/compute/PBS，不凭元数据登记声称已部署。

## 3. 后续任务的强制接入

沿用主规范规划的 `workspace doctor / paths / status / prepare-run / validate-run`，新增同步检查与部署语义，不改成第二套CLI。每个新collect/train/eval/audit/pack launcher在任何副作用之前：

1. prepare-run记录owner、PI/run UUID、protocol fixed SHA/GO、实际T0/deadline、code/source seal、entry release、registry revision、profile/host/解释器、数据/模型/统计/资产身份、argv、输出根、lease和预算。无关字段写NOT_APPLICABLE。
2. validate-run校验必要gate、权限、host、输入身份、输出路径和全局/任务锁。未知私有绑定或过期授权不猜测、不自动重建。
3. dry-run只输出计划。真正execute必须显式请求、通过对应科学GO，由既有approved launcher执行；任意shell字符串不是授权，不能绕过合法PBS和原安全门。
4. 新run输出必须在registry允许的节点runs根下，如`<verified-node-root>/runs/<PI>/<run_uuid>`；NFS的experiments只是小型索引。拒绝新输出到Home历史PI根和旧Projects副本，拒绝path/symlink escape、覆盖旧run和错误host。
5. resume只读取并接续原run/T0/ledger/消费cursor，不重开预算。未知进程状态标UNKNOWN，不能因SSH断开或旧锁文件存在/不存在就重复启动。

旧PI0292先登记LEGACY_PINNED_UNTIL_BOUNDARY，禁止重启/热切换。真正阶段/作业边界由原owner确认后，下一已获准阶段显式采用新入口；源数值、数据和统计不变，路径/入口差异记ADOPTION_DELTA。运行贯穿维护窗口则保留原绑定，不以清理为由取消作业。

规则通过启动器和测试落实，不只写在README中。具体命令未实现应报NOT_IMPLEMENTED，schema/fixture PASS不得宣传为真实模型/物理执行PASS。

## 4. 同步策略：GitHub发布不等于服务器部署

过渡期仍只有一个集成分支 `pi0267/executor-decoupled-recovery`。main维护路由/较早accepted状态，不自动作为当前runtime。通用实现之后经专门PR、测试和INTEGRATION_REF_CHANGE将main提升为唯一集成分支；本轮不强行合并整条历史科学分支、不改默认分支或删refs。

local-pc默认唯一集成发布者。子任务用窄范围feature/ops分支；现场修复保存patch/身份后经发布者正常提交，不让AMD/4080S分别维护两个“主仓库”。同一个NFS checkout跨主机只是一份对象，修改/同步必须互斥。

`workspace sync --check`：获得repository lock，fetch明确origin refs并比较管理checkout的HEAD/dirty/ahead/behind、entry release、run固定SHA和lease。不能checkout、merge、停止任务或自动安装依赖。source capsule不运行fetch。

`workspace sync --apply --release <full-SHA>`：只有显式指定发布版本、可从审核ref恢复、无活动消费者、工作树干净且可正常快进才更新管理checkout；否则返回DIRTY/DIVERGED/PINNED_FOR_ACTIVE_RUN。新run使用独立对象库或完整source archive的固定capsule，不通过alternates依赖将被删的父库。验证后才更新未来入口。

开发发布流程：status→fetch→明确文件修改/测试→commit→再次fetch并保留并发报告→普通push→远端SHA/关键文件回读。只在无活动依赖的开发checkout使用merge --ff-only；发生分叉由owner正常合并审核，不能reset/force消除。数据和私有配置不用scp/rsync覆盖源码树，不git-init工作区根，不git add整个数据/模型目录。

运行产物先进各run，脱敏小结果由发布者推Git；大数据/完整Trainer到已授权私有HF固定revision并实际回读。remote代码SHA、部署SHA、运行SHA和entry release分别展示，“有新提交”不等于需要更新正在跑的代码。

## 5. 余下目录的处理目标

本补充不扩大主规范/已有维护GO的删除范围。先撤销旧**可执行依赖**，再按原精确恢复/权限/文件身份门迁移或退役；历史报告里的旧路径保留，通过旧→新映射复现。按候选执行，不要求全机所有目录同时过门。

- 旧官方worktree：PI0292解除直接引用后采用入口内同SHA源码；两节点检查后正常退役。
- 旧VLA与官方parent：必须先处理存续linked worktree、环境、五个历史子模块、dirty恢复和对象库关系。parent不是普通linked worktree；禁止force/prune或删除.git来绕过。
- 冷态代码/报告：可完整归入archive/legacy，保留原字节/权限/索引；迁移不算释放。
- WBC、go2-arx5混合数据根、_handoff和Home科研/环境：先纳入资源索引与具体保留原因。受保护内容不为ls整齐而搬动，确需内容保留迁移须按现有明确授权及消费者切换验证；不能用本补充自动批准整树销毁。Home bin/lib/include/pyvenv.cfg和隐藏认证目录不动。

兼容旧路径可暂时保留但必须有owner、consumer和移除门；它们不能成为新任务默认输出，不能靠隐藏目录或保留旧symlink后宣称物理只剩一个目录。

## 6. 执行与验收边界

沿用主规范同一次入口实施任务的T0/ledger：总墙钟≤4小时、维护CPU≤8 core-hours、并发≤4CPU、新增源码/收据/暂存恢复峰值≤8GiB，最后30min只收尾。若owner已采用主规范，不因读取本补充重开时钟或叠加额度；旧维护和科学账独立保留。该额度只用于工程，不扩展原候选删除资格。大资源迁移无法在额度内验证则明确保留；不删除源或恢复区来腾空间。

零模型、零物理、零新增PBS、零WBC/硬件、零健康作业终止。验证新接口使用无负载fixture和现有允许的CPU导入/数据读取，不能启动一次新仿真来证明入口。

E0–E4按原任务卡继续，并执行其CPU-only CI契约测试；GitHub Actions不得启动仿真、模型、WBC或远端SSH。新增必交：ENTRY_RELEASE_MANIFEST、WORKSPACE_ROOT_AGENTS_READBACK、NEW_RUN_ADMISSION_TEST、SYNC_CHECK_AND_APPLY_READBACK、LEGACY_RUN_REGISTRATION、PATH_CUTOVER_RECEIPT。报告放原`reports/maintenance/2026-10-13-unified-entry-standard/<session>/`，避免再造平行维护目录。

至少完成：
- AMD/4080S现有适用18项工程+4项resolver测试及新负例；Hopper仅标真实验证覆盖范围。
- 同一无负载任务经入口prepare/validate实际通过，两个新任务不写旧顶层；错误host、未授权、重复run、超预算、path escape、dirty/分叉/活动lease同步拒绝。
- 一次固定SHA的真实获取/部署/回读；原活动run的source/data/stats、预算和进程保持不变。
- 所有现役/legacy-held资源都能从入口索引定位，剩余目录逐项有原因和下一条件。

四项分开判定：SINGLE_HUMAN_ENTRY、NEW_WORK_ADMISSION_ENFORCED、GIT_SYNC_VERIFIED、PHYSICAL_TOP_LEVEL_ONE_DIR。不能用文档存在替代部署，也不能因仍有历史物理目录就宣布统一逻辑入口无价值。

发布脱敏实际结果至#47并回读；#11只登记依赖/入口切换。该文档发布没有远端执行，也不更改正在进行的PI0292科学判定。
