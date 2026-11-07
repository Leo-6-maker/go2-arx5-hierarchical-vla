# FUSION：结构化响应与真实目标观察薄接口 V1

2026-11-06。继承固定 FUSION 计划 `f965a2c568ec26e9cecb8301fac0785f96a575a8`，不替换正在执行的四臂 DEV 面板、模型或源码 capsule。

## 动机与冻结比较

实际 48 窗口、768 条转移的平衡数据 SHA 为 `e5139d792d8d2f28e49ea28d7f57fbabbd5e79378e5358b747b2896c851ed795`。四个 TRAIN 父提供32窗口，两个 DEV 父提供16窗口。此前六个 MLP 均真实完成2000次更新并恢复，但三个动作条件模型都不及新的 TRAIN 平均响应规则。它们的结果、失败的200次更新和正在进行的192评价面板全部保留。

新假设只改变外部响应预测：当前 TCP 作锚，预测 H16 XYZ 位移；基线为 TRAIN HOLD 漂移加各动作相对 HOLD 的响应，再研究机器人前缀条件的 ridge 小残差。动作盲对照使用同一随机化动作分布的 TRAIN 边际均值，不能把未知动作强定为 HOLD。两臂移除物体状态、绝对世界平移、未来状态/时钟/参考和父身份特征；动作盲同时移除候选特征和基线中的动作条件。

唯一 ridge 网格为 `0.1,1,10,100,1000,INF`。四折整父 TRAIN LOPO，每折重新拟合静态曲线和 normalizer。选择指标是 H16 XYZ RMSE；有意义改善阈值固定为 **0.08124448952310876 mm**，由已冻结 TRAIN 重启噪声和原任务容差导出，CLI须复算来源。相等时优先较强正则；达不到阈值回退自己的静态基线。两臂全部保存、精确重新加载并封存 TRAIN 后才评价 DEV。记录实际模型拟合和线性求解次数，`optimizer_updates=0`，不将闭式拟合称神经更新。

DEV已用于此前研究，是开发数据，不能将这次复用变成独立确认 TEST。旧强动作静态规则始终是共同主比较；各臂自身静态对照另外报告。预测优于盲模型不自动晋级控制。任何新原生比较需新场景/冻结模型/原 owner admission；现役192评价不切换到本模型。

已实现入口：`scripts.wm02.train_fusion_structured --spec ... --output ...`。须通过原 registry/Journal、有效 FUSION scope、独立 job lease、固定 CPU 和非根输出盘；纯 NumPy CPU、CUDA不可见。不启动 GPU、多物理环境或新 Actor。

## 源动作时钟与机械夹爪

原源域40个命令的发出时刻为 `observation + [0..39]*0.05s`。前8个真正执行的命令各推进25个2ms步；第一个后果样本在 `observation+0.05s`，第8个在 `+0.4s`。32个未执行后缀不能作为实际后果标签，400ms前缀不能叫800ms的 H16 实测。decoder 新增显式 `OBSERVATION_INCLUSIVE_ISSUE`，保留旧默认合同，不给原数组整体加50ms。

冻结源 processor 原实现会在逆归一化中裁至 `[-1,1]`。新审计只披露原始 normalized 前10个物理通道的越界数量，不改权重、processor或其输出，也不增加融合 clipping。源输出的机械开度0..1不等于目标 Teacher 逻辑开度；原目标工作行程机械分数为 `[.002/.044,.042/.044]`。越界按原值报告，不能自动套 Teacher 变换掩盖不兼容。

## 真实目标观察的边界

`TargetShadowBinding` 接受原 `ActualState`、journal 原始 packet 字节、原 `FrameRecorder` 返回值、同一 packet 的双256方形RGB PNG和相机标定、独立已固定的 SHA。保留绝对原生步3503/3703/4103及2ms时钟，不伪装成旧 DEC1 的2500+200n网格。

相机保留真实 `body_rgb/wrist_rgb` 名称、父body、局部外参和FOV。源世界固定 body 相机55°与目标 base_link 相机75°不同。映射成 source-policy 输入槽只表示使用同一个冻结源策略评价目标图像，必须随附目标来源封套，明确视觉域与目标 embodiment 尚未资格。缺显式、独立固定的 source-from-target 刚体坐标关系时只保留目标观察，不产生可送入模型的 source-policy payload。

`scripts.wm02.capture_fusion_observation` 仅在新 owner 注册的 CPU EGL Mesa job 内，从已封存的原生 PASS 记录生成三对图像和绑定；禁止所有 `mj_step*`。只读预检发现现场没有OSMesa库，因此复用既有 EGL Mesa 软件渲染路径：固定Mesa vendor、`LIBGL_ALWAYS_SOFTWARE=1`、CUDA不可见、禁止GPU设备选择，并核实实际GL_VENDOR/llvmpipe；不安装系统依赖或退到GPU。原始 journal、场景、运行结果、加载的执行器身份必须匹配且旧 lease CLOSED。它是保存状态的真实几何渲染，**不是实时在线捕获、完整控制器恢复、视觉匹配验收或动态准入**。图片与原始状态仅留私有非根数据盘。绑定与渲染完成后，后续真实模型推理仍需各自新有效 lease 和资源检查。

## 当前确定限制

已审计的真实 E1 动作 SHA `0eb9040757d8fc4dc3922dc51ff2c839f043a69b754c159fa60301f2fe133a4e` 被原源域 SafetyGate 拒绝：最大线速度4.968442m/s（原阈值2）、角速度15.231471rad/s（原阈值4）、平移路径5.628812m（原阈值5）。14/40点的机械夹爪超出目标已合格工作行程；40点全部至少一个 normalized 物理通道超出单位区间，共136分量。这些是原输出的审计结果，没有新推理、积分或修正。

该动作还超过现有XY1mm、固定Z/旋转/开夹爪模型支持域。源有效命令到实际后果的语义与完整同观测状态仍需证明。不能以增大GPU、静默平滑、截短后缀、重采样挑成功或放宽安全阈值越过此限制。当前目标继续为合格局部参考校正和源资产核对；保持 WBC、安全、2ms、现役 capsule、旧实验与原截止不变。优先共享 CORELAB 4080S，本版本不申请 A40/H200。
