# Codex 执行交接：WM02

日期2026-11-04。当前审阅main=`0a27753bbda8ed75ab6b74aeb64755e0427362c7`；本文件所在发布提交须固定完整SHA后使用。主协议：[WM02](../../protocols/WM02_GO2_ARX5_WORLD_MODEL_ACTOR_V1.md)，配置：[plan.json](../../../experiments/wm02/plan.json)。

## 要做的事

实现并验证“实际Go2＋ARX5冻结WBC采样→WM→短想象训练Actor→原生评价”。GR00T不再是先决条件；主Actor直接通过WM优化，PPO为匹配基线。WM01固定源域/PPO＋WM路线保留为历史研究设计，不能继续沿旧计划而忽略本轮用户方向。

本发布无训练器，不能把下面CLI描述当作已存在的命令。不启动新Codex会话或后台owner；由用户现有执行agent在真实权限内接管。不要改正在运行的旧任务，不盲重放未知副作用。

## 最新资源执行顺序：本地4080 SUPER 16GB先调通

用户已明确要求：**先在本地RTX 4080 SUPER 16GB调通完整可许可的模型/执行接口与短训练循环，后提交A40/H200排队**。执行前先读 [WM02本地GPU补充交接](WM02_4080S_LOCAL_FIRST.md)，它规定L0–L5、显存/吞吐profiling、何时分离sim与训练进程、集群PBS准入。**不能以较大集群GPU排队替代本地WBC/接触/梯度测试，不能将纯PyTorch smoke当作目标Go2 WBC动态资格**；本地没有目标私有后端或目标许可时，先做离线与合法源域实验并准确记录阻断。

## 先做P0和P1，不先提交百万步训练

先读根AGENTS、WM_START_HERE、协议、配置和实现映射。刷新相关#11/#47与实际owner状态，不把GitHub历史PID当现在真值。保护dirty/local-only工作，独立feature branch/source capsule，正常PR。

1. 从现役profile确认真正使用的目标后端/WBC版本、运行身份、资产、gripper、scorer与许可。仓库target_sim是Isaac Gym接口，上游是MuJoCo Warp＋OSC，不能混称。没有目标动态授权/解除隔离证据就先写代码和源域诊断，准确报告阻断，不绕过S1/T2。
2. 复用已有参考桥、控制器与安全。新增薄adapter、50ms记录、状态白名单、完整save/restore和8×7到EEF10/H40的compiler。特别验证首点时间、参考空间进度和尾部预览；不能假设执行0.4秒就等于消费前8行。
3. 验证未来控制状态可递推：真实起点可读特权信息，想象后续只能使用预测/解析更新，不能偷读replay的未来WBC历史。模型内Actor每8个原子步才重规划一次。
4. 运行实际允许的8个目标诊断和1/4/16环境吞吐profile。无目标许可时不执行；不会用OSC或固定底座替代目标证据。别为了提高并行改dt、solver、WBC或安全。
5. 输出一份RESOLVED_SPEC及P1结果。普通路径/序列化/数据型/CLI问题按V7自主修复，不为每个小修再索要GO，不建新预算系统。

## 然后按P2–P5连续推进

P2：按parent先划分TRAIN512/CALIB32/DEV32/TEST100。构造四类物理有效reset，每候选最多两次、成本全计；先64尝试验证记录，再累计256尝试，不只保留成功。没有抓持/释放覆盖应直接报告，不用空中轨迹数量冒充。WM200-step smoke做恢复；正式重新初始化后训练5000步。

P3：64个CALIB真实状态×5动作方案，最大0.8秒分叉。固定随机改动范数和计分方法，报告梯度方向/排名、接触误差和全部分母。模型无决策价值时不盲开BPTT长训；PPO对照和有证据的开发继续。

P4：A=PPO，B=WM-imagination Actor；同随机均值网络、task-state观察、compiler、reward、reset与任务。B的Actor不再混入PPO。每2048原子转移一轮数据/模型更新，WM每50ms递推，Actor每400ms决策，gamma按物理时间换算。先seed20261104各100k真实转移；到正式配置则原运行续到1M，再两个seed。不得换seed/最佳中间点/TEST选优。保存全部恢复状态与物理计数。

P5：正式末端100个新R0布局×3seed×2臂，预定600条原生完整评价。科学终点没达到就partial，安全停止保留未运行，不为凑数强跑。实际成功、恢复、拒绝、失败机制、成本及层级配对统计一起交付。无提升也是结果，不自动无限加步数。

## 视觉及后继

P6仅在状态策略有效后实施：200条教师尝试→双RGB学生10k更新→两轮各50条学生TRAIN交互、每轮5k更新。教师查询不等于实际执行示范，分别记录。单独100新布局对teacher/student做200条配对评价；不读取仿真真值进入student。目标输入先结构化，不必增加大VLA。

P7 DOOR/BOOK、多任务/移动需独立配方，不在首轮同时扩展。状态策略学到当前WBC响应是目标适配，不声称跨本体。

## 代码工作包和CLI要求

复用上游 `networks.py`、`agent.py`、`buffer.py` 的算法思想/可合法复用部分，保留许可与完整上游SHA。不要整体复制其OSC、reset bank、task thresholds或升级现役WBC环境。算法主体与task adapter分离。

拟实现：
```
scripts/wm02/resolve.py      # 只读绑定，形成唯一spec
scripts/wm02/qualify.py      # 合同/恢复/原生接触与吞吐
scripts/wm02/collect.py      # 真实转移与reset bank
scripts/wm02/fit.py          # WM单卡学习与恢复
scripts/wm02/diagnose.py     # CALIB物理分叉
scripts/wm02/train.py        # --arm ppo|wm_actor
scripts/wm02/evaluate.py     # --split DEV|TEST，原生scorer
scripts/wm02/distill.py      # 后继视觉学生
```
上述全部NOT_IMPLEMENTED，实施后再提供真实--help/运行命令，不宣称现在可执行。放在既有launcher/profile之下，禁止另造scheduler或跨站逐步RPC。

优先相关CPU测试：action时间与SO3/夹爪、尾部预览条件、终态与autoreset、parent split、recurrent模型多步不读未来、BPTT梯度非零而WBC不更新、400ms决策一致、reward一致、恢复和视觉白名单。重大共用修改才跑扩展回归。本发布没有可继承的CPU/CI PASS。

## 第一次返回必须具体

返回实际分支/full SHA、已确认的目标backend/WBC身份、P0/P1真实完成项、target许可是否具备、正在运行的旧工作是否保留、实际命令及退出码、真实转移/模型更新数量、首个确定阻断，以及下一条已实现可执行命令。不能只提交又一份计划或大量审计框架。

**开发调试与短循环优先本地RTX 4080 SUPER 16GB；实际profile支持后，正式规模training优先排合法A40/H200 PBS**；Core按实际角色采样/推理；H200不是自动获准节点。站点间只交换固定revision的完整批次/小checkpoint，数据仅走已授权私有路径。AMD不重算，私有WBC/资产/原始数据不进Git或未经批准的公开HF。

本次不执行任何服务器部署/训练/物理操作。接管后在有效条件下可以连续推进同一实验的工程与科学步骤，不要求每个工作包重新审批。
