# 自有暂停适配层同步修复

状态：工程修复设计与实现，尚未 build/CPU 通过或重新资格；不是 Teacher/物理 PASS。

## 原件证据

固定原 clock adapter 的 `RL_Sim::RunModel` 仅在 `native_mrt=true` 时获取 `sim->mtx`。此前自有 `task_pause_build.py::patch` 将该条件替换成无条件 `native_lock.lock()`，目的是保证暂停时 history 原子性。这意外把当前 N1 的正常推理也放进仿真/PD 同一把锁里。它是已确认的自有补丁偏差，不是已经证明的夹爪越界唯一原因。

04/05 机器人模型及真实零步 qpos 身份相同；实际正常控制力矩在首次策略运行附近分别保持 38/36ms。两次夹爪每步电机力与固定目标42mm下原400/3公式逐项重放误差0，不能再称漏发夹爪命令或重复映射。

## 最小工程修复

在独立 `task_pause_r1_build.py` 和独立 runtime 中构建自有暂停 overlay，旧文件/二进制保留：

1. **恢复原条件原句**：仅 native_mrt 模式获取 native sim mutex。原 RunModel 的完整计算主体、SetCommand 完整函数、采样周期、dt/decimation、权重、gain、physics、关节/夹爪范围不改。
2. 新增单独的暂停生命周期 mutex，只将 model/history 与暂停/恢复事务互斥；正常 N1 PD 不获取该 mutex，因此不因长 inference 被新增适配层阻塞。
3. PauseTaskClock 先获取原 sim mutex，再获取生命周期 mutex，等待在途推理完成后写实际暂停 ACK；原计数与 history 必须在 ACK 后冻结。MRT 路径保持同一 sim→lifecycle 顺序；原本已 paused 的 RobotControl 再次调用暂停使用可重入锁。
4. 原生 ENTER 暂停/恢复通过生命周期事务执行；非事务的正常 RobotControl 不串行到 inference。原 source 中控制数学与时钟算法保持，只修自有暂停同步范围。

依据为 R1 第3节允许的错误 pause/clock 绑定及自有 wrapper 生命周期修复，并沿本会话明确的连续 debug 授权。没有授权新的控制算法、增益或任务门槛。必须先完成代码主体等价断言、真实 C++ 并发 harness（正常PD可继续、暂停等待在途推理、嵌套暂停无死锁）、独立 build/hash/source seal 和受影响 CPU 入口收据，再进行有界启动。原真实安全事件继续累计，禁止原样重跑旧失败 module。

此修复若运行仍有真实安全 FAIL，完整保留并停止该 episode；不依次改变控制参数追成功。新旧 source 的比较不能把单次启动 PASS 当稳定性或任务完成。无新物理/策略 forward 的 CPU 收据先交付，重新资格后才允许进入 Teacher。
