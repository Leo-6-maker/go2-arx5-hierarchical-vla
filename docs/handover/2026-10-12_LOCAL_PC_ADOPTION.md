# GR00T＋WBC：新 local-pc Codex 接管与双服务器工作规范

日期：2026-10-12，Asia/Singapore。状态：ADOPTION_AND_REPOSITORY_MAINTENANCE_ONLY。你是用户本地PC上的唯一执行owner，不是新的科学审核PI，也不是GitHub cloud Codex。本轮先读懂项目、核实已有连接与资产、整理仓库并回报；只授权轻量接管检查、文档整理及有条件的分支维护，不授权新训练、GR00T forward、模拟物理、WBC运行、教师采集或真机动作。

## 1. 先理解最新研究方向，不自动续跑旧GO

仓库：Leo-6-maker/go2-arx5-hierarchical-vla。上层团队负责预训练GR00T的task-space manipulation：视觉＋语言＋实际TCP/夹爪反馈→未来EEF SE(3)轨迹＋夹爪命令。师兄WBC负责本体关节、底盘协同和稳定执行。基线UMI/UMI-on-Legs本身已有EEF/WBC分层；不把替换高层模型自动等同于新颖性或跨本体成功。

最新用户方向优先：我们不修改师兄WBC或继续替其调参；主线转到独立任务物理、专家数据、GR00T微调及独立闭环。师兄独立推进OCS2＋MuJoCo＋腿部RL，我们将来使用固定快照组合评价。借用机器人/场景资产不等于数据已经由WBC执行。独立工具、固定底盘、Panda、冻结WBC四类来源分别标注。

任务数据不必等待目标WBC达到所有跟踪门。旧target_mujoco validator硬依赖WBC资格，对它原来的数据语义有效；未来独立数据要建新的manifest/资格，不修改旧FAIL或把不合格WBC数据换标签放行。完整物体任务、首次到达、持续保持、参考正确和跟踪正确必须分开。

## 2. 本轮权限与停止条件

允许：读本机和GitHub、只读SSH连接/环境版本/项目进程/PBS/quota检查、读取小manifest并核对已有HF认证、编写接管报告；按BRANCH_RETIREMENT_PLAN.json窄allowlist完成备份后清理。

禁止：qsub新作业、GR00T权重加载/forward/canary/训练、启动MuJoCo/IsaacGym/ROS/WBC/teacher、修改WBC/gain/solver/模型/数据/评分、重启PI0285–PI0289、继承旧剩余额度、qdel/kill健康进程、删除模型/数据/原件、git reset --hard/clean -fd/强推保留分支、全量镜像HF、上传私有WBC或凭据。先读旧owner/launch/doctor再决定其是否有副作用，不因为文件叫doctor就执行。旧10/8–10/10截止不会自动顺延。

完成首轮回报后停在ADOPTION_READY_FOR_REVIEW或精确BLOCKED；提出下一份数据准备GO草案但不启动。用户已有两端HF key，不反复要求提供密钥；只对实际无法从配置解析的必要项一次列出缺项。

## 3. GitHub接管顺序与证据优先级

主ledger：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11 。工程/CI/分支整理：Issue #47。
现役分支：pi0267/executor-decoupled-recovery。
交接前审核HEAD：5c757eaea3b3078e3406b604e70267ca5d0c0130。它只是读档锚点，先刷新真实HEAD与#11最新评论并分页，再固定自己的read SHA。main是较早accepted代码＋导航，不等于最新运行源码；不要大合并、重命名现役分支或另开长期handover分支。PR77不是当前必写指针。

必读顺序：
1. START_HERE.md、本接管文件、分支维护清单。
2. docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md。
3. reports/PI0285/r3/FINAL_REPORT.md。
4. reports/PI0286/final/FINAL_SCIENTIFIC_REPORT.md和必要的模型/训练库存。
5. reports/PI0289/execution/REPORT.md、GATE_RECEIPT.json、终局#6093151230。
6. docs/coordination/2026-10-10_WBC_PACKAGE_COMPARISON_AND_TARGET_LEARNING_REVIEW.md。
7. docs/coordination/2026-10-12_TASKANCHOR_UNITAS_DATA_AND_FINETUNING_REVIEW.md。
8. 有限追踪scripts/pi0286_execution/、src/go2arx5_vla/target_mujoco/、现役imports；hpc/和旧owner只作路由/环境参考，不运行历史任务。
9. docs/PROJECT_STATUS.md、RESEARCH_INDEX.md、REPOSITORY_WORKFLOW.md在main的旧导航版本中存在，现役分支可能没有。按固定main历史SHA读取，不以404推断实验原件丢失；这些2026-10-01内容不是最新授权。

旧README与后续终局矛盾时，保留旧历史、以最新明确用户约束及固定协议/实际收据判断。GitHub写入或@codex不代表远端执行。此前@codex回复需创建cloud environment；本轮不用它，仍由你这个local-pc owner通过已有SSH执行。

## 4. 需要掌握的科学快照

- GR00T-N1.7-3B；EEF10=XYZ＋row-rot6d＋gripper；相对EEF/绝对夹爪，H40、20Hz、K4、前8点0.4s后新反馈。132D内部mask不等于132个关节。新embodiment route不是预训练即插即用能力保证。
- PI0285：Panda T1，PLACE4096＋DOSE2048共6144更新；TRAIN7/8、DEV13/16、CONFIRM_A20/32<26/32。T2教师存在但无共享policy资格。
- PI0286：两臂同一父状态各1024步；C0 TRAIN2/8、DEV未运行；C1 TRAIN8/8、DEV5/16；新确认未运行。27/32纠正示范准入不等于模型泛化成立。父模型与本轮噪声namespace不同，不能把分数差自动解释为严格配对收益。
- PI0289：安全启动10.000178504s PASS；首hold XYZ RMS0.011672954m PASS，SO3 RMS0.065199717rad>0.05 FAIL；其余五参考NOT_RUN；V4时钟残差0.047803157s>0.025s；目标教师/正式目标数据0，selected/deployment NONE。
- 最近科学ledger时间2026-10-10 11:08:34 SGT；10/12文献备忘录是设计稿，非新GO。接手仍要刷新是否有新结果。
- 新WBC为OCS2＋MuJoCo＋腿RL。默认MPC为简化浮动臂运动学优化；EEF-only 7D pose或EEF+BASE14D，原生JSON不是旧201D obs/15D actor/nine-point preview。低层质量问题现在交由师兄，不是本轮主线。
- TaskAnchor启发因果历史/预测task state，UNITAS启发三维标定/物体后果记录；未在项目验证。未来可比较B0原生/B1历史/B2预测状态，不能现在同时改所有模块或启动完整WAM训练。

## 5. 用户本次指定的资源分工与网络

local-pc：唯一调度与整合owner，GitHub提交/回读，分别SSH两端，保留小收据，不作为20GB权重默认中转盘。
Corelab：仿真环境、真实物理/渲染、教师采集、预处理、CPU评分、视频与报告。不能沿用旧Corelab GR00T eval server作为新默认。未来物理运行必需的冻结腿RL/MPC仍属环境控制，不是高层GR00T eval。
Hopper H200：GR00T训练微调、模型加载、推理和模型eval；全部GPU工作经合法PBS allocation。login节点只轻量检查/传输，禁止训练或长时模型服务。
GitHub：代码、protocol、budget、小收据、固定SHA、审核与状态交流。
HF：两服务器的数据、模型和大数值证据交接；各端已配置凭据，复用且不输出token。
硬约束：Hopper与Corelab不能直连。不得假设scp/rsync/ZMQ直通，不扫描或绕过网络政策；两端同一HF账号不等于共享文件系统。

### 5.1 已有连接/路径线索，现场状态仍需核验

Corelab历史实际路由在scripts/pi0287_target_sim/owner.py：local→corelab-amd-server→corelab-4080s；用户也用过直接corelab-4080s。先解析已有SSH Host/ProxyJump，不改ssh config或关闭host-key验证。
共享根：/home/liuyu-nfs/Projects/go2-arx5/。
私有交付：private/wbc_mujoco_minimal_20261009_6dd13098/wbc_mujoco_minimal。
历史SSD执行副本：/var/cache/users/liuyu-nfs/Projects/go2-arx5/executor-candidates/wbc-mujoco-6dd13098/wbc_mujoco_minimal。
PI0288/PI0289原件按各receipt查找，不原位改交付；4080S已用ROS2 Jazzy但现在需只读确认版本，AMD可达不代表4080S任务已运行。

Hopper SSH别名本次未可靠确认：从本机已有SSH配置、旧本地工具和private runmap识别，不虚构域名，不要求明文密码。公开只报别名/状态，IP/详细连接留本机私有配置。
历史项目根：/scratch/e1192548/Projects/go2-arx5/；旧PI0286根experiments/PI0286，不覆盖。
训练venv：/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv。
官方源码：/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t。
HF_HOME：/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home。
基础模型缓存线索：archive/stage_runs/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B。
依据scripts/pi0286_execution/environment.sh和C0_REPLAY1024.pbs。旧env设置HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1；网络交接应在单独受控进程按需在线，训练使用完整缓存，不把offline模式误判为key失效。
旧PBS示例用queue auto、1×H200、12CPU、225GB、2h，仅供阅读；本轮不提交。后续重新核对队列、实际allocation和配额，不锁定旧hopper-17或job ID。

## 6. 首轮接管只读检查

A. 本机cwd、repo origin、git status、branch、worktrees、未提交改动；不覆盖dirty工作树。查PowerShell/bash、Python、git/gh版本，复用认证。
B. GitHub全部分支、开放PR、#11/#47增量；分开记录执行授权/封存结果/计划。固定commit同步小代码，不默认写main运行代码。
C. 两端各一次有限SSH只读probe：hostname/date/id/pwd、已知目录存在性、环境包元数据、项目相关进程/lease；Hopper qstat只查自己，Corelab检查对应GPU占用。未知写UNKNOWN。ACCEPTED不等于RUNNING，GPU可见不等于使用授权。
D. df/inode、quota可查则查；区分scratch/NFS/SSD、缓存和输出。历史有ENOSPC；不删除模型、视频或唯一原件腾空间。
E. HF已有认证验证账号和指定repo读取及一个已发布小manifest；不打印token/headers/环境变量全集，不重新login覆盖现有设置。读权限不证明写权限；写canary如必要另列有限请求。
F. 模型/数据资产只读manifest/index/stats，区分官方基础模型、完整Trainer和推理视图；不全仓库下载或为接管重复读取所有大文件。
G. 一端暂不可达仍完成另一端和代码阅读，准确交partial，不反复等职责确认。提出下一数据准备GO而不启动。

## 7. HF交接协议：固定revision、按需下载、不覆盖身份

历史Hub：leoyuer/pi0246-gate-pass-campaign。repo_type以记录/Hub为准，不因prefix叫data猜成dataset。已有PUBLIC内容不意味着可公开私有WBC/mesh/策略/二进制或新受限数据；新数据先核对许可/可见性，必要时用用户指定private repo，不修改旧repo可见性。

流程：Corelab生成并校验数据→HF固定revision→Hopper按prefix下载→训练/模型eval→HF固定model/eval revision→Corelab读取所需数值结果/视频原件→GitHub小结。只在选定checkpoint交接，不每步上传，也不让完整HF历史归档阻断迭代。

TransferManifest至少包含experiment/task/split、producer/consumer、git执行SHA、repo_id/repo_type、完整revision、prefix、每文件path/size/sha256、data/teacher/executor/processor/stats/decode身份、STAGING/READY/VERIFIED。可分批上传，最后提交READY manifest；消费者固定包含READY的最终commit并核对完整集合。存在目录、脚本exit0、部分model文件不等于完整恢复点。

只用revision＋allow_patterns下载必要资产，不用HF main/latest作为模型身份。完整Trainer=权重/optimizer/scheduler/RNG/processor/stats/meta；model-only不能resume。inference bundle须绑定完整训练源。Corelab若只CPU评分不下载optimizer。同节点缓存复用；网络失败先查现有revision/文件，不重复上传整个仓库。

历史可定位但不能自动恢复训练的模型：
- arms/PI0285/REPAIR_A2048/checkpoint-2048 @ 75432eb91a2c602930a9e1e979aaebbeaebff02b，累计6144，未通过确认。
- arms/PI0286/C0_REPLAY1024/checkpoint-1024 @ b0b8c65de8d5bbdf0946ccdda9ecb23348cf16c9；旧报告Hopper仅metadata，完整源在HF。
- arms/PI0286/C1_CORRECT1024/checkpoint-1024 @ d35dcda18def3d0691b55957ce461fe6b8449437；当时Hopper有完整源，现在只读确认。
以上不是部署policy。未来从官方pretrained还是旧研究权重开始，由新科学协议决定。

## 8. H200 eval与Corelab物理如何在不互通时协作

离线eval：Corelab上传固定观测、合法历史、标签和必要噪声；H200计算预测/误差并上传；Corelab独立CPU复算。沿教师轨迹的离线预测不等于闭环成功。

真正闭环：每个chunk都必须先拿到由上一动作造成的新观测，再调用H200 policy。未来可由local-pc持有到Corelab和Hopper合法计算allocation的两条持久SSH通道，中继obs/action。服务器彼此不需直连。H200模型服务只绑定分配节点localhost及获准SSH转发，不在login GPU或公网运行。
消息含episode UUID、step、sim时间、ref_id、model revision、request/action hash、合法历史和timeout。独立任务仿真可按lockstep等待响应再推进，标NON_REALTIME；真实部署延迟另测。旧native WBC wall/ROS时钟不能只停一半组件，需独立时钟资格。
HF不是0.4s实时消息队列，不每个tick写HF/GitHub。local relay不可行时只做offline，或另行批准明确的异步暂停仿真/HF批量闭环；不能预生成整段轨迹后称reactive，也不能悄悄把GR00T eval移回Corelab。
本轮只交这套拓扑设计与缺项，不开server/tunnel/物理，不声称relay已实现。

## 9. 接管后的科学工作怎样与审核衔接

接管报告→小型数据准备protocol草案→用户/PI明确批准→固定protocol/budget/code SHA→#11 canonical并回读→真实运行→阶段收据→终局。换窗口、讨论稿、旧剩余额度均不是新GO。发布不等于实际执行。

未来主线：独立物理工具/目标几何场景与teacher→同步双RGB/actual EEF/gripper＋desired/committed动作＋完整历史事件＋三维标定/物体轨迹→原生GR00T baseline→历史/预测任务状态的等预算对照→两任务共享。WBC由师兄推进，未来Teacher/WBC与GR00T/WBC组合另行注册。
物体真值/teacher phase/成功标志只作监督或评分；不能泄漏给正式Student。动作从实际观测时刻解码；H40不跨reset/abort/接管；episode先split再切重叠窗口。真实噪声hash、数据顺序、dtype、processor和时钟都绑定。不同seed namespace不冒充配对对照。

## 10. 分支清理边界

另读BRANCH_RETIREMENT_PLAN.json。保留main、pi0267/executor-decoupled-recovery及三个含独立提交的分支。四个祖先候选只有在live SHA一致、祖先关系成立、无开放PR/保护/活跃lease使用时才清理。
先建立并回读同名compatibility tag（仓库已有惯例，保留旧ref链接），再用精确expected SHA的force-with-lease仅删除该refs/heads。tag冲突则SKIP，不覆盖。绝不强推main/active，不删除源文件/HF资产。
ChatGPT本轮尚未删除remote heads：当前连接没有删除分支的动作接口。你使用local-pc既有Git/gh认证完成并记实际before/tag/after；无法确认权限/保护/运行依赖则跳过并说明，不要求用户贴token。详细清理命令与readback见本次维护附件。

## 11. 首轮交付，然后停止

在reports/handover/2026-10-12/<session>/提交sanitized小文件，私有runmap留本机：
- ADOPTION_REPORT.md：阅读SHA/文件、最新规则/结果/未知、研究理解；
- RESOURCE_MAP.json：两端角色、SSH别名验证、实际路径/版本、PBS/进程/存储、时间与证据；
- ASSET_AND_HF_MAP.json：实际见到的基础/研究模型/完整恢复点/数据/固定revision；
- NETWORK_AND_EVAL_PLAN.md：HF大资产流、离线eval、未来local relay，已验证与仅设计分列；
- BRANCH_MAINTENANCE_RECEIPT.json：确实删/归档/跳过及前后SHA；
- NEXT_DATA_GOAL_DRAFT.md：最小数据环境计划、可复用与待实现代码、预算草案、状态NOT_AUTHORIZED。

#11发布单条ADOPTION_ONLY回报并回读，引用报告固定commit；#47记录仓库维护。不@codex、不创建第二云owner、不增加周期任务。当前对话总结实际完成和精确缺项，停在ADOPTION_READY_FOR_REVIEW/具体BLOCKED，等待下一科学GO。
本轮可做确认无下载/模型/仿真副作用的轻量CPU测试；新GPU allocation、GR00T forward、优化、teacher/physics均0；selected/deployment=NONE。

## 来源与阅读依据

项目固定审核基线5c757eaea3b3078e3406b604e70267ca5d0c0130；#11在2026-10-10T03:08:35Z之后本次未返回新评论。科学数据见PI0286/PI0289实际终局，不把这一查询当实时GPU状态。
路由见scripts/pi0287_target_sim/owner.py；Hopper环境/调度见scripts/pi0286_execution/environment.sh及C0_REPLAY1024.pbs。新的H200模型eval分工、两端无直连、HF凭据已配置来自2026-10-12用户明确要求，不由旧脚本推导。
HF固定revision/按prefix下载：https://huggingface.co/docs/huggingface_hub/en/guides/download 。分批上传最终manifest需以实际安装SDK核对，不升级共享环境：https://huggingface.co/docs/huggingface_hub/en/guides/upload 。精确lease与ref删除：https://git-scm.com/docs/git-push 。
