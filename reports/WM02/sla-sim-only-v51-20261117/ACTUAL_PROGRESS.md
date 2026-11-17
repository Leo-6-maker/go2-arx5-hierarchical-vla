# V5.1：实际原生接口与首个专家结果

本轮落实用户最新`SIM_ONLY_PROJECT_AUTHORIZED`授权，采用[新V5.1协议](../../../docs/goals/2026-11-17/MFREE_SIM_ONLY_PROJECT_AUTHORIZED_V51.md)。独立第三方签发前置已对新隔离MuJoCo scope取消；没有伪造旧READMISSION或修改旧失败。**任务仍在服务器运行，尚无SHIFT7学生闭环成绩。** 具体观测时刻、全部SHA和分母见[实际进度](ACTUAL_PROGRESS.json)。

| 端点 | 实际封存结果 | 原生分母 | 不可称作 |
|---|---|---|---|
| F1无接触接口 | 原M_FREE 7D安装、H40保持、完整P10、同后缀TTL续期及CANCEL/HOLD通过，无硬STOP；wall66.267s，调用进程CPU26.568s（不含C++子进程） | 1请求、5953tick；5703初始化＋250任务 | 抓持或学生策略成功 |
| F2无接触接口 | 同接口通过，无硬STOP；原trace检查通过 | 1请求、5953tick；5703初始化＋250任务 | 动态STOP最坏延迟认证 |
| 首个F1专家 | 原native grasp＋抬升20mm＋保持1s评分通过；最大抬升24.782mm、任务方向底座净位移102.698mm，综合移动任务成功 | 1请求、22953tick；全trace通过，无硬STOP | SHIFT7学生能力 |
| 其余专家 | 第一条之后的F1正在LIFT，已有实际native_grasp；其余事前固定条件排队 | 未终态，不提前计成功 | 已完成8/8 |
| SHIFT7 pilot | 尚未开始，模型服务尚未加载 | 0实际学生请求／8计划；0新生成 | 0/8任务失败或通过 |

以上封存原生分母共**3请求、34,859个2ms tick**；另有一个专家世界在运行。初始化、任务段和总积分独立计量；当前没有新优化更新、优化访问或真机请求。原生碰撞/STOP继续终止并计失败，不因仿真碰撞永久禁止其它固定仿真条件。

## 实际工程接续与旧结果保留

初始源码`0e6065ce`完成F1接口；随后F2的旧模板来自不同collector，world lock仍指向另一domain，继承锁检查在创建native输出或启动仿真前正确拒绝。该次是**启动前技术失败、0原生请求/积分/生成**；原日志、Journal INTENT和主服务失败保留。终态确认原主unit无PID/cgroup成员后，依据缺少native输出与锁拒绝证据将该未启动case独立核对关闭，未盲重放UNKNOWN。

源码`be742f62e744b9322dbd1a6c2b86a86af83b10f7`增加全panel domain/lock一致性预检，并只续接F2接口及其余16个未启动条件。F1成功接口按原spec/result SHA复用，**没有重复积分或替换旧失败**。新源修复仅在调度/所有权层；接口及原生程序、模型、WBC、原H40/夹爪/碰撞门不变。定向回归116项通过，随后接口与专家结果来自真实MuJoCo执行，不由CPU测试替代。

## 服务器托管事实

实际unit为`wm02-sla-sim-only-v51-20261117-02`，已从独立后续RPC读取到持续更新的进度和原生tick，`linger=yes`，无前台聊天依赖。新服务仅调度原有限面板；保留原绝对期限，未因技术接续延长。实际source seal：`cd7541499c14167bd64418e6847fac92cd7cc35dae0e4e6cf6b9bab2f8abcfa6`；controller spec：`ba483dd53c614f74142d261207cbd497758150334368e67d53ebbfa811c5275e`；[241源码成员](SOURCE_FILES.json)。PID/cgroup、真实路径、日志与STOP入口保留在私有现场回执，不公开站点或原始数据。

GPU0采用现有负责人授权下的共享站点绑定，启动前实际容量与owner检查通过；其它用户进程未改变。冻结SHIFT7单模型服务只在学生阶段加载，不重训、不借用旧R1/R3的已耗额度。新scope的项目授权是实际依据；旧spec继承的R3参考元数据不构成本轮授权。

源代码已接通普通body/wrist RGB＋可得状态＋上一实际命令→有限单模型IPC→原完整H40编译→实际安装ACK→原WBC/P10消费及续期→原评分。**这条SLA路径仍待本轮学生实际运行资格**。暂停仿真与连续350ms实时资格分开；动态STOP最坏延迟和controller-selected reference仍不伪填为已知。下一项科学交付为剩余专家及SHIFT7 pilot的真实抓持、抬升/保持、底座移动和首失败原因。
