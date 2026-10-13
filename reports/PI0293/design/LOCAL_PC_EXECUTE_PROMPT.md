# 给现有 local-pc Codex：PI0293 夜间长程 GO

用户要继续任务空间物体操作主线，尽快得到一份在固定底座 ARX5 仿真中多任务可行的共享 GR00T policy，并行使用 CoreLab 采集/制作数据集和 Hopper 微调。本文是接手指令，不是执行收据。

## 采用顺序

采用本次 #11 交接评论所指固定 release，首先完整读取：
1. reports/PI0293/design/LATEST_ADOPTION_DELTA.md（发布前新状态优先：登录已恢复、smoke03因compute端amgr可执行文件检查在模型前失败）。
2. reports/PI0293/design/OVERNIGHT_GOAL.md（全部科学、并行、资源及验收要求）。
3. reports/PI0293/design/GO_TASK_CARD.json 和 SCENE_DATASET_SPEC.json。
4. reports/PI0293/design/generate_scene_cards.py（需owner实际运行/核验/封存，发布本身未执行它）。
再读取根AGENTS/START_HERE及唯一workspace标准。复用 ~/Projects/GR00T-WBC 的已验证enter/doctor/profile；所有新collect/train/eval/audit/pack先固定RUN_MANIFEST。现役集成分支为pi0267/executor-decoupled-recovery，不用main旧代码替代。

## 直接执行，不再重开一轮规划

- 记录新PI0293 run UUID/T0/账本，刷新PI0292进程、当前认证/project、本人quota、迁移后的data/model与同账号peer。PI0292没有健康/排队神经作业时，明确交接其尚未执行阶段，优先本次多任务训练，不要求先完成旧单任务4096步。健康任务不取消；原时钟/账本/失败不改；本团队PI0292/0293合计一个H200槽，账号最多两份独立GPU分配。
- Hopper先修最新明确故障：compute端amgr路径/权限/hash诊断必须分开；根据集群允许流程核实真实owner/project/allocation，不把login节点二进制必须同路径同hash当未经验证的普遍前提。不能绕过官方权限、借用peer或拷贝凭据。下一自己的小allocation通过工程探针后立即跑真实model smoke，不再只重复发布guard准备结果。
- CoreLab同步接续scripts/iteration：实现live worker预算adapter，2worker起、实测等价与吞吐通过后最多4。独立MjData/renderer/RNG/输出和CPU集合，中央单写者原子预留。8次诊断内完成真正串并行物理对照；synthetic PASS不算live资格。
- 构建两目标、四场景族：抓放到蓝区、抓放到黄区、沿桌面推到蓝区。每组初始状态/RGB相同，只换语言；蓝黄位置平衡。push用真实接触且不抬升，不能改名冒充pick-place。commissioning≤24、pilot最多两个版本合计24，失败保留，正式采集后teacher/scorer不追分。
- 在正式采集前运行固定scene generator、检查schema/静态边界/旧初态冲突并封存72组：TRAIN48、DEV12、CONFIRM12，同组三任务同split。优先交错采DEV与TRAIN-A前24组，边采边审计打包；复用原64TRAIN。Drop-A满足每任务20/24新TRAIN、10/12新DEV和各族覆盖后立即冻结送Hopper。TRAIN-B后24组在微调期间扩采，不热改当前DataLoader或stats。
- Hopper使用真实已合格数据做原生smoke/恢复；正式从pristineN1.7开始一个共享部分微调：冻结视觉语言骨干、训练projector与官方动作DiT/动作头，原生flow matching，lr1e-4、bf16、有效batch32、最多4096步。每步旧数据4/蓝抓放6/黄抓放6/push16，阶段均衡防止hold淹没事件；task_id/phase只供TRAIN采样，不能给模型。
- 尽早部署observation-only/ACK_CHAIN_V2端点，每任务两条真实teacher relay后，M2048就做开发闭环再同链继续。最终固定M4096在12新组×3任务上做Teacher/B0/M4096对照。一个checkpoint、三个任务逐项过门才报SIM_MULTITASK_POLICY_WORKING_V1，不用总均值或文件存在替代物体任务成功。
- 最后交完整POLICY_BUNDLE、可恢复checkpoint、逐任务成功矩阵、原成功/失败视频、Drop-A/B私有HF固定revision与回读、实测提速及资源账。正常fetch/merge/push保留并发提交，#11科学终局，#47仅基础设施/路径事实。

## 执行边界

T0后≤14小时且不晚于2026-10-14 12:00 SGT，最后60分钟收尾；本团队T0后神经allocated≤12 H200小时（≥3小时留评价）；新qsub≤4；CoreLab≤192 CPU core-hours/16并发核/4live workers，物理≤384次；正式训练≤4096步、smoke≤16实际updates。所有失败、等待和重连都计入，不重置旧PI历史。WBC/真机/新增删除/GPU仿真为0。

认证或模型工程受阻只暂停GPU线，CoreLab独立继续；需要用户交互登录时只提示必要动作，不索要凭据、不猜权限。并行无收益就退回更小并发。推物未资格只能标抓放语言任务或部分数据，不能拿reach冒充第二技能。先产出真实学习/闭环证据，不再围绕目录重构、无关小文件清理或WBC调参打转。

门槛通过即可连续推进，不必每阶段重复请示；预算内修确证工程错误，但不靠换seed、降安全阈值、临时挑CONFIRM最好checkpoint或自动新配方制造成功。
