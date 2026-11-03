# WM02本地准入：E0/E1完成，E2离线接口完成，目标动态待准入

2026-11-04，Asia/Singapore。实际交接90c0507333d2a9c6b26ee9c22278b050a9249fad；GPU执行源码21f2811f00aedb37b47e7b52d827d4ba1e7cb38e。完整结果见[环境优先回执](../environment/EXECUTION_REPORT.md)、[矩阵](../environment/ENVIRONMENT_MATRIX.md)和[机器结果](../environment/RESULTS.json)。历史P0/P1源872条记录及其并行成绩保留在[原回执](../wm02-20261104-p0-p1/FINAL_REPORT.md)，本轮未重采。

| 准入项 | 本轮实际结果 |
|---|---|
| L0／E0身份与依赖 | 原MuJoCo＋ROS2／OCS2，9核心＋14完整路径初始化文件匹配；原环境绑定后三份ELF无缺库，8模块导入通过；没有重装或修改WBC |
| 旧owner与队列 | 本项目Core旧派发与已知worker关闭、lease CLOSED；本地未启动WM01 W2关闭为UNRUN；Vanda旧W1自行完成5000并关闭，未新投PBS |
| L1／E2 CPU接口 | 30项合同／预检／容器／评分检查普通和-O通过；typed offline容器不是live exporter；完整目标控制状态恢复尚未绑定 |
| L2 目标原生8条件／并行 | NOT_RUN：S1/T2隔离没有当前解除／动态准入；目标转移0，不用S域47.97/s作目标WBC吞吐 |
| L3 WM200／重载 | COMPLETE_SYNTHETIC_ENGINEERING_ONLY：19,113,552参数、200主更新＋1恢复探针，effective sequence batch256；微批8/16/32均实测，normalizer／optimizer／RNG／cursor保存恢复；权重／optimizer最大差0 |
| L4 H16梯度 | COMPLETE_SYNTHETIC_ENGINEERING_ONLY：Actor／Critic各1次更新，starts64由micro16累积；Actor原子0/8重规划，WM冻结且物理输出对动作梯度非零；不是native CALIB/P3 |
| L5／E4真实闭环 | NOT_RUN：没有本轮真实采集→正式WM→native奖励Actor→新实际采集。真实TRAIN数值绑定与reward资格仍缺 |
| 5000正式WM／A40/H200 | NOT_RUN／NOT_SUBMITTED，未把合成烟测升级为正式预训练或百万步队列准入 |

本轮在GPU0完成，NVML报告32760MiB；GPU1是16376MiB。Torch设置13GiB分配上限，训练阶段allocated约503MiB／reserved672MiB，整个进程NVML采样峰值1192MiB。共享负载下全卡采样峰值32085MiB包含他人工作；单独模型低占用不证明原生sim＋WBC＋渲染能与训练同时驻留，也不是GPU1实机验收。[完整显存profile](../environment/LOCAL_GPU_PROFILE.json)保留逐阶段数据与边界。

最近已核对的owner/GPU时点在机器结果中；任何下一次GPU启动仍须实时查空闲量及原owner，不承诺长期reservation。目标动态首阻断依旧是独立有效的S1/T2准入；神经模型更新不解除它。
