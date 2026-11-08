# WM02 FUSION-NIGHT1 · 夜间自主融合入口

2026-11-06，Asia/Singapore。原始审阅基线：未合并PR #364固定 `ad72c0e1779f015f18780dd173eef9fe3f40125e`。本文件仅计划，没有启动任务。

**发布并发接续：** 计划发布时PR #364已推进到 `3b1ae512a390f756ba56c6ce71a40095534be2ce`。本分支通过保留双方父提交的非强制合并，完整继承该版本的执行代码与结果，再叠加四个计划/入口文件。详细方案和JSON的83次/旧Policy状态是原审阅快照，不代表最新现场。后续真实接触Policy TRAIN4/4及更新后DEV一致率75%→50%的负结果全部保留；不重复已完成学习。若父分支继续推进，在冷边界先核实际状态和差异，保留新实现与现役capsule，不按旧计划快照降级。

## 目标

**冻结GR00T任务先验 → 明确名义动作/期望实际结果 → 本体WM有界参考校正 → 冻结WBC → 真实反馈。**

先读 [详细执行计划](docs/handover/2026-11-06/WM02_FUSION_NIGHT1_EXECUTION_V1.md) 和 [机读配置](experiments/wm02/fusion_night1_v1.json)。不要重训已完成的PILOT2矩阵，不把VLA和WM当互斥路线。

## 接续顺序

1. 读取实际owner与正在运行的PILOT2任务；让健康批次自然封存，在冷边界独立worktree接续，不热改或新开第二个物理owner。
2. F0/F1确认可用GR00T checkpoint、真实输入、动作坐标/时间/夹爪语义和WM有效域。GR00T输出命令不自动等于期望实际运动；必要时用已验证源执行结果做行为对齐。
3. F2先实际运行已知意图的direct/静态校准/普通反馈/WM校正对照，目标是物理误差，不是模型自己的分数。先交一组真实结果，再按新父面板继续。
4. F3真实调用冻结GR00T并做shadow；F4在资格内比较同一VLA的有/无WM原生闭环。缺权重或语义不清就明确BLOCKED，继续F2和C接触学习，不用合成动作假冒GR00T。
5. F5用新TRAIN反馈训练小响应模型/参考adapter、恢复并重新原生评价。GR00T权重本夜不更新；原接触线可独立继续，不等所有自由空间Actor胜规则。

## 自主范围

不设项目内部累计工程预算，不逐200步或每个小修询问；每个实际作业仍需原机制下有效有限scope/lease、资源与原安全。旧CLOSED不复活，旧TEST不训练，当前capsule不改。默认1/2 CPU原生环境、单张可用本地GPU，新动作/接触先1环境。

**允许新版本外部高层参考校正，禁止修改WBC内部权重/增益/solver/安全或用局部模型外推大位移。** 小delta加在大命令上也可能OOD；检查完整最终参考。不能把原生ROS环境改成Torch训练环境。

H200排队/作业、GPU物理多世界构造/试跑、分布式或替代WBC批处理前停为 `AWAITING_H200_SCALE_REVIEW`，提交审核包，等待用户批准。无真机、主动行走、系统/他人任务修改和自动main合并。

实际启动后的首个当地09:00在安全批次边界写晨报；这是报告检查点，不是总时限或自动续期。真实site deadline优先，提前至少20分钟收尾；有合法后继可继续。未运行项目标NOT_RUN，独立重启不叫exact fork。

**最低返回：** 实际EEF校正组与负结果、真实VLA推理或具体资产阻断、nominal/desired/installed/achieved证据、模型/数据/source SHA、真实更新和原生分母、成本、活动/终态worker与下一条已实现命令。只有原生闭环支持时才声称控制收益或任务无示范迁移。

## 2026-11-07 有限续期窗口实际接续

原 owner 已实际登记新有限窗口，站点／P1 截止 23:34 SGT，FUSION 工作截止 23:14，保留20分钟收尾。当前11个原生 worker 已封存：100,174步、4,006个完整原子转移；7 PASS／4 FAIL，不是7个技能成功。本轮 GR00T／WM／Actor 更新均为0。详见 [续期实际回执](reports/WM02/fusion-night1-20261106/FINITE_RENEWAL_WINDOW_RESEARCH_ACTUAL_V1.json)。

C02完整Teacher初始化未进入源状态范围；有限反馈运行发生实际触碰。C03 home-only 静态候选在原生启动审计被拒绝：冻结 `PI0289ResetStartup` 将关节2、3覆盖为0.9 rad。没有 C03 物理 worker，不修改启动保护、WBC或旧失败。下一步需要原生支持且控制器状态一致的受控初始化入口；不能把home文件修改、静态FK或输入裁剪当作资格。见 [C03启动拒绝证据](reports/WM02/fusion-night1-20261106/C03_NATIVE_STARTUP_REJECTION_ACTUAL_V1.json)。
