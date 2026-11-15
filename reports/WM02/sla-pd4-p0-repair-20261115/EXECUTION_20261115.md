# PD4原8条件P0：依赖与原生路径绑定修复

当前预算为新增训练0、接续原8条UNSTARTED学生条件；完整用户计划见[长程计划原文](../../../docs/handover/2026-11-15/SLA_PD4_LONG_HORIZON_USER_PLAN.txt)。不是自动累加PD20、额外续训、旧DEV36或RL队列。[实际预检记录](PREFLIGHT_ACTUAL_20261115.json)。

## 已验证的变化接缝

原协调器在原生Python环境导入`pd4_training_v71.MANIFESTS`，经dataset／codec引入SciPy后失败。修复把不变的PD4身份常量移到纯配置模块，并把provider的数值依赖延迟到实际执行函数。协调器不安装SciPy，SLA worker继续使用既有隔离环境和SciPy 1.13.1；原生MuJoCo仍3.2.7。数值函数／类的非import AST及所有身份常量与旧版一致，相关46项本地回归通过，不重做87数组或RGB32恢复。

Core真实非交互受管预检CLOSED0，墙时79.388 s。它在禁止SciPy／JAX／Flax／Torch／MuJoCo进入协调器的条件下，实际核验两个完成500端点、保存包、统计、原8条顺序、执行配置和真实冷状态；另在原生与SLA两个实际解释器及CPU亲和下检查worker及延迟依赖、冻结资产、输出权限。模型、生成、训练、物理均0，没有创建新科学panel。解释器绝对路径与包METADATA hash保存在私有现场记录。

登记后的实际运行范围检查继续在任何模型／物理派发前执行。第二个协调器正确通过训练门，但其旧学生模板误携带state-snapshot原生二进制，严格门期望的是已封存N20及新统计8/8实际使用的native-ui二进制。两者不是相同字节。该检查真实CLOSED1，墙时7.463 s、子worker0、物理0；原8条件仍UNSTARTED，结果保留。

现在按原合同修正协调器的二进制路径及来源字段，绑定已通过新统计8/8的原冻结字节。solver、observer、outer、trace与恢复的原控制器字节未换；严格身份门没有放宽。此外移除协调器错误重复调用的旧source-pose审计；规范typed phase入口已经调用PD4封存生产者／准备前缀审计与live-pair准入，原生入口保持。只读复核8/8的修正recipe与PD4前缀证据通过，未伪造已缺失的旧源观测，也未恢复旧RP8。

原native binary的SHA256为`39e5136603f51c092b2f3e595245468dbb0c12dbca79818b0df08da8dd4af7dd`；原控制器仍`9e8ce46837d09bc39ae04d4b49f1ce4dd8c8c2d31bea39eac56c5806d108c1f3`。未运行的模板snapshot二进制是`82c135c1bf5bb093cac832554b953d0017ec33e35e584865d08ca14327f3aa29`，没有作为这8条件的实际物理执行结果。

## 原科学身份保持

两包500权重／adapter、统计、codec、raw H40／P10／60查询槽、场景、restart、顺序、种子、评分、安全阈值全部保留；旧PLAN hash仍`140b9087dc3200981c5a754b4acc819b2b304a63c675cb170c844d51af6bb3ec`。只另开合法资源窗口，不改旧协调器deadline或失败结果。登记后的完整live-pair检查通过前不派发。离线全部拟合false不阻止原门记录真实策略拒绝。

本报告尚无学生原生结果；待实际8条件后补充raw／installed／consumed链、证书、积分、首失败与每臂分母。专家运输8/8、S16与原16关节收敛诊断均沿既有证据复用，旧RP8、旧V6/DEV/TEST失败保留。
