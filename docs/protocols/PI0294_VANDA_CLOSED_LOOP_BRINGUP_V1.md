# Vanda GR00T＋冻结 WBC 闭环建设 V1

2026-10-16。用户已在现有 local-PC 对话明确授权：可以按照计划开始推进 Vanda 端的实验闭环。此前已许可项目 GPU 额度并指定 WBC 交付包。本版本登记实际采用，后续同范围工程阶段不重复请求 routine GO。

## 固定输入与范围

沿用固定 M2048/v3.0/EEF10、官方源 `51d4c89f72fda44cbf77285c6a8114b52676b8a1`、原 processor/统计和四步 sampler。原有 Stage0/Stage1 作业保留，不重提。用户指定 WBC 包 SHA256：`6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`，实际归档 215194129 bytes；独立解包/编译，不修改 WBC 源、策略、增益、代价、安全阈值或 solver。新平台取得独立身份，不无条件合并 CoreLab 既有评价。

推进顺序：合法 PBS 上的 A40/CUDA资格和固定观测推理 → CPU 容器/原生编译 → 有界原生启动、传感/参考/停止、时钟一致性 → 同节点隔离进程的真实观测模型动作/SafetyGate/prefix8/实际执行 ACK → 完整 episode 与任务结果分别报告。普通路径、依赖、编译或通信问题自主修复；真实安全错误保留现场并停止受影响物理，不通过调 WBC 或安全阈值推进。

该 GO 覆盖 Vanda 平台闭环验证；本阶段没有训练、成功数据扩采、DAgger/FlowDAgger执行、真机或迁移/取消现役 Hopper E0。物理/模型工作只由分别绑定的该主机阶段 manifest/lease/固定源码执行；不调用已过期的 CoreLab R4/Teacher admission 来冒充新资格。历史失败保持原样。

## 本次先实际 dispatch 的 CPU 构建

新增 `vanda-native-build` bootstrap profile 与独立 run。单次 PBS：0 GPU、4 CPU、32GB、最长3h；编译并行2，所有进程继承最多4核亲和性。一次登记后24h内可获分配；过期拒绝开始。存储留在已验证项目根，节点临时根先确认至少24GiB可用。实际命令、源码校验、环境/镜像/二进制 SHA 和 CPU费用落私有运行根。

ROS Jazzy 基镜像按实际 Docker registry 的 linux/amd64 digest 固定，而非运行时 latest：`docker.io/library/ros@sha256:a5426de405f6f0a82b0f3def3bbd3bce2a71a91421fe980c17613b4e51ef38d9`。容器包含交付包原安装清单和隔离的 MuJoCo3.2.7 Python 工具；APT/Python实际版本清单封存在镜像内。构建不执行任何策略或物理。使用节点本地 Apptainer TMPDIR，项目空间保存最终 SIF；不在登录节点编译。

解包检查路径、链接/设备、成员和尺寸上限；原 SHA256SUMS 编译前后必须通过。仅构建输出/环境适配允许变化。成功只称容器及原生编译通过，不称 WBC质量、EGL、物体任务或闭环成功。后续物理阶段依据实际构建结果绑定同一交付包和原安全机制，不以历史保持 RMS 作为全局阻断，也不放过真正限位/非法碰撞。

## 后续阶段的验收

先验收可停止、可追踪的全机器人原生执行与实际双RGB/EEF/夹爪状态；冻结 WBC 下动作确实安装、消费与时钟对应，预测状态不能替代实际状态。若需要暂停/锁步，沿用已验证生命周期、时钟及消费契约，不能用 sleep 或放宽 watchdog 代替资格。M2048 仍是 EEF10，不能填零冒充 BASE 训练能力。

真实新观测→H40→安全检查→prefix8→实际 ACK→新观测形成闭环才算执行链成立。完成 episode、原生安全、物体成功分别判定；有效任务失败不反复换 seed 挑成功。结果按新 Vanda 环境注明，Hopper E0 的原12条件、source D0/S1、期限、资源与在途源码均保持。

来源：[Apptainer 1.4 容器构建](https://apptainer.org/docs/user/1.4/build_a_container.html)、[Apptainer 节点本地临时目录要求](https://apptainer.org/docs/admin/main/installation.html)、用户指定包内 README/build/install 清单、现有固定源码与平台实际资产收据。仓库仅存自研适配代码及脱敏小收据，WBC/vendor、原始观测、权重、凭据和私有路径不发布。
