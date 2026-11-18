# V6 P0 decision · source learning is a new contract

固定计划 `ae3d3c0dceade437b0104a150cd0402fef650052`，审阅执行父PR407 `1d78f2b758fb9b0dc5bbc05ecfc5a0aa3b1e63d3`。本实现worktree从该栈独立建立，仅接入五份V6设计文件；不假定main含runtime。

| 范围 | 当前决定 | 未取得的资格 |
|---|---|---|
| 原command＋SHIFT7 RealMix |仍BLOCKED；Real10/25不恢复 |真实命令生产者、frame/TCP/消费与单臂兼容 |
| SOURCE_NATIVE未来记录学习 |按FUTURE_LOGGED_STATE_REFERENCE进入S准备 |不声称state为已实现反馈、已发命令、硬件同步或真机SR |
| T可靠性对照 |B1同父、fresh Adam，仅FM500与完整生成H40约束500 |真实四轨迹梯度、500端点、恢复、新24闭环均待实际结果 |
| M共享HOLD/PLACE |可独立准备；按完整Teacher和新G7资格进入 |不因T通过或源离线预测而自动准入 |

PR407数值证明及其已固定converter/policy/loader/RGB/data-card追溯直接复用，本P0没有新增外部作者联络或全库追溯。未知生产者继续UNKNOWN；它不再阻断新source-only预测合同。输入仅anchor当前三RGB/记录state及完整原指令；标签action[i:i+50]按已验证offset对应未来state[i+1:i+51]，不二次shift，排除复制尾部；源局部坐标固定SOURCE_ARM_LOCAL_V1，双臂相对XYZ/SO3及原g保留。源不映射2–42mm、不套SHIFT7或Go2门、不直接控制Go2。

源60候选、HF revision1127bda94f172d757386945af3eb3a6754d1a4e6、36/12/12 episode槽位保持。首12全部是TRAIN，未用SOURCE_DEV/TEST作模型方法选择；没有session信息，仅作episode留出。TEST在S1000冻结前仅通用schema/文件完整性检查。新source-only统计只fit TRAIN，与C20分开。

V5.2完整24终态仍为B0 5/8、B1 6/8、B2 6/8，17成功、5拒绝、2STOP、0技术失败，无晋级者。旧B1/B2各500/16000和旧TEST人口不重训、不重跑、不回填。

现场只读核验：旧V5.2/source12均CLOSED；GPU0约31GiB空闲、主存约47GiB可用，node-local盘约580GiB可用，ROS198世界锁和神经锁空闲。两条旧collector虽遗留ACTIVE登记，所记录PID均已不存在；未关闭或重写其旧登记。现有user-service site绑定SHA核验，实际allocation截至2026-11-19 02:51:23.855488 SGT不延长。新V6分别登记有限服务，不能复用CLOSED角色或把旧E2服务期限转移过来。

T采样需固定十步可微循环：保留原cached_actual的vector field、Euler递推和forward精度语义，FM训练仍highest；静态scan只为实现完整十步反传，旧部署采样器不修改。原采样与新路径在真实模型预检中逐项比较，预登记容差：normalized最大1e-5、XYZ最大1e-6m、全40加速度差1e-4m/s²且原XYZ门结论一致；是否逐位一致如实记录，不由CPU玩具检查代替。aux四窗按4*(step%8)轮转当批槽位，独立noise按绝对cursor生成；两臂相同，T-FM只评估，T-H40才反传。实际remat及所有前向/反向另计；没有clean proxy、detach或输出加工。

首个源depot为独立CPU服务，2线程/4GiB/单worker，首12只复核复用，最多补48；测试集不调用数值profile。真实训练/恢复/任务成功只按后续ACTUAL更新，不由本决定文件授予通过。

## 源分层实现定义 · S step0以前固定

第一次CPU实现05b9ac02使用“完整未来H50任一点变化 / TRAIN action IQR”判动态，36 TRAIN的38106窗口全部动态，故原CPU角色以SOURCE_STRATUM_DATA_COVERAGE_NOT_READY封存，结果保持，不冒称通过。该定义并非协议指定的动态聚合方式，不能将实现选择造成的空stratum误当没有可学习源记录。

现在明确S采样对象为**锚点的相邻日志帧变化**：当前索引i相对i−1的XYZ/SO3/原g变化，阈值仍为对应TRAIN state维度IQR的1%。i=0没有真实前驱，分层UNKNOWN并从采样排除；不伪造上一命令或输入历史帧。完整未来H50的运动跨度另列，静态锚点不称整窗静止。TRAIN日志的初步相邻帧检查给出约8137个低变化锚点，36 episode均有两类；这是TRAIN-only方法定义，不看DEV/TEST模型结果。新定义、统计与访问银行在任何S模型生成/更新前封存；不改变源曲线、阈值系数、频率、窗口长度或原36/12/12人口。

后继CPU角色复用第一次已经验证并封存的数值数组，保留原覆盖结果，只新建分层/访问银行。任务和episode固定等量槽位，global16000动态/16000静态精确分配；源TEST依旧不读数值。此处是预训练前的reader/分层语义修正，不是重投模型科学失败、改变源数据来过旧SHIFT7门或重开旧角色。
