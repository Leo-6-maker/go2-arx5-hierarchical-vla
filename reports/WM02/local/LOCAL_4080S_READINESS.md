# WM02 4080S 本地准入：第一轮部分完成

2026-11-04，Asia/Singapore。依据原交接 `ddc801f30e288365925c61c37f7ee445ddc29712`，以及正常合并的本地优先补充 `19f686c9c006bffce22d3819b2019e1e45ebbec2`。此表报告实际完成量，不是正式训练准入或目标动态授权。

GPU准备纠正：此前只关闭727048，仍运行的M13随后自动派发boundary1425203。现已停止管理器3892803及boundary、完成原native close；02:57:25 SGT复核三个PID均退出、两worker lease CLOSED、管理器USER_STOPPED、GPU1无计算进程且空闲15899MiB。下表02:45观测保留历史身份。详见 [后续停止纠正](../wm02-20261104-p0-p1/GPU1_FOLLOWON_STOP_20261104.md)。

| 项目 | 本轮实际状态 |
|---|---|
| L0 目标身份／权限 | 实际为MuJoCo＋ROS2／OCS2 Go2–ARX5；9个核心、14个初始化成员冻结SHA匹配；原目标lease CLOSED，S1/T2隔离保留。首个阻断为没有当前安全解除／动态准入。完整目标MPC／WBC恢复导出器与站立局部scorer绑定也尚缺。 |
| GPU1 准备 | 用户指定727048已通过原STOP＋一次SIGTERM终止，原owner关闭lease；1024／2048完整checkpoint保留。2026-11-04 02:45 SGT再次只读查询：总16376MiB、空闲15899MiB、无GPU1计算进程。共享许可不等于目标动态许可，不等于长期独占reservation。 |
| L1 CPU合同 | 20项普通和`-O`均exit0；相关原有接缝31通过、5跳过。实际冻结目标parser的41点／11错误包CPU检查通过；未做目标安装ACK或物理。实现字段角色和梯度历史接缝，GRU模型尚未实现。 |
| L2 源域诊断 | S域固定底座MuJoCo，8条件、16恢复分叉；672条真实50ms转移，完整源控制／积分／RNG／记录恢复最大误差0，硬安全0。释放仅覆盖0.4秒，尚未覆盖完整撤离。不能称目标资格通过。 |
| L2 源域并行 | 实际spawn 1／4进程，20.7486／47.9711转移每秒，2.312倍；另200条转移。16环境因绑定4 CPU及实际RAM余量不足未启动。不是目标WBC的GPU并行吞吐。 |
| L2 目标诊断／1、4、16环境 | NOT_RUN：S1/T2隔离未解除，无动态准入；目标恢复实现亦缺。目标转移、接触检查和任务均0。 |
| L3 5×2048 WM200／5000及重载 | NOT_IMPLEMENTED_NOT_RUN：WM02 fit CLI／模型尚未实现；更新0、模型加载0、checkpoint无。没有把随机输入反传或WM01模型测试计入本轮。 |
| L4 H1／H8／H16 Actor梯度／冻结 | NOT_IMPLEMENTED_NOT_RUN：仅历史输入梯度接缝检查通过，不是Actor或WM学习。真实反事实需要目标动态及恢复资格。 |
| L5 收集→拟合→Actor更新→再次采集 | NOT_IMPLEMENTED_NOT_RUN：没有可加载WM02 Actor、没有目标完整episode，不能声明端到端调通。 |
| A40／H200 PBS | NOT_SUBMITTED。尚不满足本地模型／H16／短回环准入，未提交长训。 |

已测源worker／接触／恢复审计CPU合计34.228685秒；父进程、测试、网络和准备成本未完整计时。WM02 GPU模型执行0，GPU模型峰值allocated／reserved／进程峰值均**未测量**。16GB独立WM／小批BPTT可行性仍需实现后实测；不能用空闲显存保证全套模拟器＋WM＋Actor能同时驻留。

机器数据、分档吞吐及数据身份见 [LOCAL_4080S_PROFILE.json](LOCAL_4080S_PROFILE.json)。实际命令／退出码、执行源码完整SHA、冻结WBC及数据摘要见 [首轮报告](../wm02-20261104-p0-p1/FINAL_REPORT.md) 和 [results.json](../wm02-20261104-p0-p1/results.json)。完整私有原件与绝对argv留在执行者私有目录，不上传资产、权重、原生状态或SSH路径。

下一条CPU复核命令（独立checkout内，`WM02_CPU_PYTHON`绑定至已核验的隔离解释器；不是新的物理运行或训练）：

```bash
PYTHONPATH=src "$WM02_CPU_PYTHON" -O -m unittest discover -s tests/world_model -v
```

后续合法代码工作是实现独立WM200-update／完整重载和H16神经梯度烟测；目标原生诊断仍须有效的原站点安全解除与动态准入，不能用源域或GPU空闲绕过。
