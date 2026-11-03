# WM02 E0–E2实际环境矩阵

2026-11-04，Asia/Singapore。交接固定SHA `90c0507333d2a9c6b26ee9c22278b050a9249fad`；独立checkout／分支 `research/wm02-env-cuda-smoke-20261104`。没有切换旧checkout或活动capsule，没有假设main已包含交接。模型实际执行源码为 `21f2811f00aedb37b47e7b52d827d4ba1e7cb38e`。

| 环境 | 本轮实际版本／检查 | 结论与边界 |
|---|---|---|
| E-S，PC源CPU | Python3.10.12、NumPy1.24.4、MuJoCo3.2.7；无Torch。30项合同／预检／目标离线接口测试，普通与`-O`均通过 | OFFLINE_MATCH。没有重采旧872条转移；本轮新增源物理0。旧源资产／原生资格仍引用原不可变回执，不计作新成绩 |
| E-T，Core原生软件 | Python3.12.3，Ubuntu24.04.5，ROS2 Jazzy；实际MuJoCo＋ROS2／OCS2。9个核心＋14个初始化完整路径逐一核验 | 23/23 OFFLINE_MATCH；初始化重名run.py按不同完整路径核验，未用basename配对 |
| E-T，依赖绑定 | 初次裸shell及仅source ROS时缺库；应用原`NATIVE_INPUTS`的既有`runtime_environment`后，sim／OCS2／MPC三份冻结ELF的ldd均无缺失；numpy、mujoco、rclpy、rosidl_runtime_py、ocs2_msgs、std_msgs、nav_msgs、sensor_msgs导入成功 | OFFLINE_MATCH。问题是未绑定原环境；未安装／升级／重编译任何原生包、库、控制器或权重。未构造目标环境或启动ROS节点 |
| E-T，动态许可 | 原lease CLOSED、`native_safety_quarantine_preserved=true`，无当前S1/T2安全解除与动态准入 | TARGET_BLOCKED；预检exit0不授予动态许可。没有反复运行目标qualify碰运气 |
| E-G，独立Core神经环境 | Python3.12.3、Torch2.9.0+cu128、CUDA12.8、NumPy1.26.4、Triton3.5.0，driver580.173.02 | 已建立独立venv，28个已安装依赖发行版／49个顶层条目从可信旧神经环境复制；独立文件或文件系统CoW，无硬链接、不共享可写site-packages、不修改原环境。环境内torch／numpy实际导入路径均位于新venv |
| E-G，GPU0执行 | 5×2048 SiLU＋LayerNorm、GRU256，实际19,113,552参数；FP32；200主优化＋1次恢复续训探针，Actor/Critic各1更新 | 实际exit0、lease CLOSED。仅SYNTHETIC_ENGINEERING_SMOKE；原生数据、目标转移、完整任务均0 |

本轮E-G实际使用GPU0：NVML报告32760MiB，GPU1为16376MiB。运行时为Torch设置13GiB分配上限，未假设32GiB卡等于GPU1的16GiB实机验收。训练阶段Torch峰值allocated约503MiB／reserved672MiB；整个进程NVML采样峰值1192MiB。共享期间其他负载使全卡采样峰值达32085MiB，不能把该全卡数值算成本WM需求，也不能把启动时空闲量当长期reservation。详见 [LOCAL_GPU_PROFILE.json](LOCAL_GPU_PROFILE.json)。

旧Core派发队列、M13及各已知worker的关闭、Vanda旧作业状态和本地未启动W2的取消分开保留，不改写其原科学结果。E-G使用原registry、Journal／onceIO和原项目neural owner lock，无新调度平台。没有动他人的GPU0进程，也没有提交WM02 PBS。

9＋14身份、软件／模型／数据和成本见 [ENVIRONMENT_ACTUAL.json](ENVIRONMENT_ACTUAL.json) 与 [实际执行报告](EXECUTION_REPORT.md)。完整路径、原native软件、环境变量值和原生状态留在私有回执；公有Git不包含这些原件。
