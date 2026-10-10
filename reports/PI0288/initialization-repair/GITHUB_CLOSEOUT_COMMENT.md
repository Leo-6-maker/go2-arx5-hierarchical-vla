## PI0288 限定初始化修复：实际 home 已应用，启动安全 VALID FAIL

固定执行证据 commit：`4f90b59f4f32d0dbad4f61ad3b8cf9357f66d1c8`。原 #6087870253、旧 REPORT/ADJUDICATION/默认审计/首步原件和终局全部保留。

用户委托 owner 自行依据交付源码和 CPU 证据判断已有 home；没有制造 WBC 负责人对具体 keyframe 的确认。

- home 瞬时 CPU：18/18 限位合法、无接触/警告；calf 上裕度0.66224rad。x5_joint2/3 下裕度0；FL/FR离地16.4974mm，RL/RR26.0793mm。
- 隔离补丁覆盖实际创建、重载和reset，共六处；预执行审计与实际路径共用 helper。两目标对象编译成功，原九项核心文件和原安全 observer/阈值不变。完整私有 diff、模型及binary留Corelab。
- 查明旧首步为 `mj_loadXML → mjCModel::TryCompile` 临时默认 mjData 的编译自检。真实零积分截断调用栈和MuJoCo3.2.7固定源码均支持；原越界与SIGINT保留，Feedback unavailable仍是后果。此前CPU“零步”不包含XML内部自检的范围限制单独补erratum，没有覆盖旧记录。
- 同一模型的原生MJB缓存完整74,461,448字节序列化往返逐字节一致；使用原生binary loader避免重复编译自检，不改dynamics/限位/qpos0。首次积分前的实际mjData地址和home状态CPU核对通过。
- 唯一实际运行 episode `53861d09823f4dbc8690e7cf24c4522d`：第一步calf=-1.5，证明home真实应用；42步/0.084s触发原安全停止。x5_joint3自0.066s连续超过1e-4rad达0.020s，最大下越界0.000661625rad；x5_joint2最大0.000847217rad，持续0.004s。两者均未达1e-3单步规则，停止来自joint3原持续规则。
- 只出现允许的foot-floor接触（首margin contact状态0.056s、首负距离0.058s；峰值穿透3.24766mm/法向力106.309N），无illegal contact、native warnings全零。arm ctrl为零、GetUp前缀中停止，未进入MPC；不由此前缀推断MPC跟踪能力。SIGINT之后退出139也保留，不将退出崩溃当安全停止首因或质量重试理由。
- 六项G1 **NOT_RUN**；G2部署夹爪型号/版本、速度/力限和pad几何/材料/摩擦仍独立BLOCKED。相机/schema/CPU工程保留；窄回归13 PASS/1私有源skip，实际callback原CPU收据复用。教师/RGB数据/GR00T/optimizer/hardware均0。
- 3个新增owner启动尝试与3个零积分native边界启动均保留，含旧尝试保守计7启动；预算未重置。直接观测42个运行积分+3个编译自检积分，早期未直接计数的CPU XML自检记UNOBSERVED，不填全cap PASS。

31个小原件202,810bytes已服务器逐文件SHA256回读，并完成本固定GitHub提交的全部31文件回读。原件、初始化契约、全18关节裕度、接触/时间、source/build/config库存和独立CPU复算见：[完整报告](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4f90b59f4f32d0dbad4f61ad3b8cf9357f66d1c8/reports/PI0288/initialization-repair/REPORT.md)。

终局继续 `ENV_READY_INTERFACE_BLOCKED`；初始化CPU/应用身份PASS与实际启动VALID_SAFETY_FAIL分开。无进一步物理重试、G1、教师采集或训练；selected/deployment=NONE。无owned原生进程、merge、mail或monitor。
