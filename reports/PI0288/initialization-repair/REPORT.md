# PI0288 限定启动初始化修复收口

**终局：ENV_READY_INTERFACE_BLOCKED。初始化已真实应用；原生启动为有效安全 FAIL。**

原 Issue #11 #6087870253、REPORT、ADJUDICATION、默认模型审计及首步原件全部保留，未改旧终局。用户回复“你自行判断并分析”后，依据交付模型、控制配置及 CPU 结果判断已有 `home`，没有填写不存在的负责人初态确认，没有搜索其他姿态。

## 1. 初态与实际代码

交付 `home`：root=(0,0,0.34)m、WXYZ=(1,0,0,0)，18 个关节与 `base.yaml` 和 N1 策略名义值一致；速度、控制和时间均为零。18/18 原始关节限位合法、零步无接触或 MuJoCo 警告。四个 calf 的下/上裕度分别 1.22270/0.66224 rad；x5_joint2/3 的下裕度均 **0**。所有关节数值见 JOINT_MARGINS.csv。

四脚表面离地：FL/FR=16.4974mm，RL/RR=26.0793mm。它是瞬时合法、尚未落地的状态，不能据此宣称稳定。root 高度与 IMU 高度分开：home 的 IMU 高度由交付几何计算为 0.38232m，原 RL/MPC height target 为 IMU 处 0.30m；差 0.08232m。native bridge 已有 IMU 到 arm mount 的转换；没有修改这些设定，稳定高度未获验收。

私有交付副本未原位修改。隔离源码/输出通过共享 `PI0288ResetHome` 绑定启动创建、两条模型重载创建、键盘/手柄 reset、GUI reset 和 keyframe load，共六个实际调用位置。helper 写入真正的 mjData，随后 forward；安全 observer 的 snapshot 和阈值保持原样。每个物理 episode 使用 fresh whole stack，GUI reset 不冒充 reference/MPC/RL history 全生命周期重置。

只重编译两个目标对象并复用其余原生链接对象。原九项核心文件 hash 全部保持；新 source、build、config、缓存和收据身份见 BUILD_AND_CPU_RECEIPT 与 FINAL_PIN_AND_INVENTORY。私有完整 diff、模型缓存和二进制留 Corelab，不公开。

## 2. 编译自检与运行必须区分

一次缺失临时目录的启动尝试未创建 MuJoCo、积分=0；已保留日志并补启动目录检查。

随后发现原首步停止发生于 **mj_loadXML → mjCModel::TryCompile 的临时 mjData 自检**，在原生实际 mjData 创建之前。零积分截断探针保存了真实调用栈；MuJoCo 3.2.7 官方源码第3995–4000行同样显示该自检 mj_step。这解释了为何 home 初始化日志出现后，旧首步仍是 calf 接近零的临时默认姿态。原越界数值和安全停止没有撤销，也没有把 Feedback unavailable 当独立根因。

采用原生支持的 MJB loader 加载同一 XML 的编译缓存，避免启动时再次执行编译自检；完整 74,461,448 字节模型序列化往返逐字节一致，不改质量、惯量、碰撞、限位、solver 或 qpos0。CPU 编译自检的原始越界另存 XML_COMPILER_SELF_TEST.jsonl。原 CPU “零步”只覆盖模型加载后的显式操作；加载 XML 内部自检不能计成总零积分，新 phase erratum 已单独说明。

预执行审计和实际创建/reset 共用 helper、同一固定缓存。零积分实际原生边界探针确认 reset 与首次积分前为 **同一 mjData 地址、相同 home qpos**；真实运行第一步也保留了 home calf=-1.5，证明补丁确实进入执行路径。

## 3. 一次实际原生启动的结果

episode `53861d09823f4dbc8690e7cf24c4522d`；MuJoCo3.2.7，dt=0.002s；**42 步 / 0.084s 后原安全停止**。完整42个状态点已独立 CPU 复算。

| 关节 | 最大下越界 | 首次超过1e-4 | 终点连续超 epsilon | 停止规则 |
|---|---:|---:|---:|---|
| x5_joint2 | 0.000847217 rad | 0.082s | 0.004s | 未单独达到停止条件 |
| x5_joint3 | 0.000661625 rad | 0.066s | **0.020s** | 原持续越界规则触发 |

首个 foot-floor margin contact 的状态时间 0.056s，首个负距离 0.058s；峰值足地穿透 3.24766mm、峰值法向力106.309N，仅原允许的 foot-floor 接触。无非法 contact，所有原生 warning 计数为零。root Z 从0.34降至0.307802m；这段落地/GetUp 前缀中，六个 arm ctrl 均为零。MPC 尚未进入，不对它的跟踪能力或 RL 稳定能力作判定。零边界裕度、落地与原生 DAMPING 相伴，但这些原件尚不能唯一归因某个动力学机制。

真实收到17条 odom、17条 arm、1条 mode。registered SIGINT 先于原生退出139；退出时崩溃为另一个工程事实，不是此次停止的首因。原 RESULT 的 TECHNICAL_UNCLASSIFIED/Native process exit 保留；新裁决依据实际持续越界判为 **VALID_SAFETY_FAIL**。不以修退出崩溃为由重试这一有效失败。

## 4. 门与交付

| 项目 | 本轮状态 |
|---|---|
| home 瞬时 CPU / reset 一致性 | PASS |
| 实际 home 应用身份 | PASS |
| 有界原生启动安全 | **FAIL** |
| 六项原 G1 | **NOT_RUN，上游停止** |
| G2 实际部署夹爪规格 | **独立 BLOCKED** |
| 正式教师数据 / GR00T训练 | 0 / 0 |
| selected / deployment | NONE / NONE |

相机、命名映射、数据schema和只读 CPU 工程保留；现有 target_mujoco 窄回归13 PASS/1 private-source skip（16.46s）。真实私有 callback 使用原已通过收据，不为补报告重复执行。正式夹爪仍缺硬件型号/版本、实测速度和力限、pad接触几何/材料/摩擦；交付规划 URDF 声明不能替代部署规格。无新教师采集、RGB训练数据、神经模型查询、训练、G1质量重试或增益/姿态/seed 搜索。

31 个小原件202,810字节完成服务器到本地逐文件 SHA256 回读；补丁私有工作集约84.1MB逻辑占用。相关原生进程已停止。科学预算不重置：旧尝试、新技术尝试、三个零积分原生边界启动均保留并保守计账；早期 CPU XML 编译自检未直接计数的部分写 UNOBSERVED，不能声称全资源 cap 已观测 PASS。

**下一依赖仍阻断：现有唯一 home 在冻结初始化/控制时序下未通过启动安全；需要新的有依据决定，不继续调整或重试。**

[MuJoCo3.2.7 编译自检源码](https://github.com/google-deepmind/mujoco/blob/3.2.7/src/user/user_model.cc#L3995-L4000)
