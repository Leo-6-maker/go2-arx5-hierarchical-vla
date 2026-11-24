# SEA-0 Source/Target Twin 的现场几何预检

基于 #492 计划提交 332e380d2530accd26f39f1b8ded346a75dfcab7。使用 Corelab 的隔离 CPU 渲染器。原 #491 工作区和 R5 冷验收保留。

## 源数据与固定面板

- 对两份官方水果实机 HDF5 回读验证文件哈希、30 Hz 时钟、三路 RGB、左右臂状态和旧 O0 回执。按每段原主窗的帧序，选右臂在随后 0.5 秒位移至少 5 mm、左侧实测与存档动作标签位移至多 1 mm、旋转至多 0.5°且夹爪信号跨度至多 0.005 的前两个窗口。选中两段的 108、447 和 90、276 帧。该筛选验证局部左臂静止，整条水果任务仍可能依赖双臂。四个冻结源动作相对原保持对照的位移 RMS 为 4.59–11.01 mm 对 25.61–203.65 mm，保留原定义和分母。源录像动作标签随后经逐行核验为下一帧记录状态。 [源数据与缺项](SEA0_SOURCE_CUSTODY.json) · [原 O0 归档](../../SLA_OFFICIAL_GO2_TRANSFER/O0_ACTUAL.json) · [源标签时钟](TOOL_AND_SOURCE_CLOCK.md)
- 重新核对 Go2 四个开发观测的真实 RGB、状态字节、历史索引和原闭合 owner。G00/G01 各使用 tick 6,203 和 8,203，保留 G00 后继无返回。两个固定种子的 64 个候选格已冻结，核对到 7 个保有原始采样值和相同观察/种子的 Go2 归档候选。O0 保存了已解码 EEF，四个源格都缺少完整原始采样证据。所有历史候选仍须在最终处理器和掩码锁定后确认是否可直接复用。[固定名册和归档候选](SEA0_PAIRED_ROSTER.json)

## 几何 Twin 与相机读数

使用官方模拟版 X5A URDF 和九个已验哈希的可视网格，以及 #483 的冻结 Go2 MJB。两份实机 HDF5 每个视角仅有 RGB 与尺寸，附录只有帧率。官方模拟资产提供几何外观，录制时的实机内参、随帧外参、物体位姿、真实指尖间距和法兰到共同工具的标定仍缺失。Source Twin 定级为近似。 [官方资产](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/Assets/Robots/x5)

八个当前状态在各自的几何模型内都通过了 1 mm／0.5° 的公共法兰配对检查。右臂 IK 只使用当前法兰和网格，不读取任务目标。源锚点用固定诊断夹爪行程，Go2 锚点读取原滑块，源侧指尖间距未标定。两模型共享克隆的 Go2 相机来检查可见范围，背景、对象与真实遮挡未重建。读数是数值几何对照。

Hugging Face 文件树已逐页回读。在固定 revision 和检查时的 main 上，这个实机水果任务的 ARX X5 目录各有 100 个 HDF5、300 个预览 MP4，缺少独立的标定文件。两份选定 HDF5 的 LFS 指纹在这两个版本均与封存文件相符。XPolicyLab 的几何接口要求 RGB、原分辨率内参和逐帧相机到世界外参来自同次采样，尤其是随臂运动的腕相机。恢复匹配输入仍需要这两份录像的同步标定及共同工具变换。 [固定源目录 API](https://huggingface.co/api/datasets/RoboDojo-Benchmark/RoboDojo/tree/1127bda94f172d757386945af3eb3a6754d1a4e6/data/RoboDojo_real/pack_and_pour_fruit/arx_x5) · [XPolicyLab 观测几何契约](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/Q0_Preview/docs/OBSERVATION_INTERFACE.md)

四个 Go2 开发状态使用冻结 MJB 的默认机身相机时均为 0 个虚拟源臂像素，诊断侧视四对均可见。后续核实原归档采集器对该相机施加了运行时位姿修正。按归档相机重放，四个当前画面均可见源臂，详见 [同视角桥接复核](../SEA1/TARGET_BRIDGE.md)。原腕相机每个状态有 9,286–9,288 个共同机器人掩码像素。两者只说明视野和轮廓范围，腕相机未找到可见的公共法兰投影点，真实场景遮挡和实机重投影误差尚无读数。保留默认机身视点的空掩码，诊断侧视单列。该作法借鉴 Cross-Painting 先分离本体外观的设计。 [几何与可见性统计](SEA0_ROBOT_GEOMETRY.json) · [Mirage 项目](https://robot-mirage.github.io/)

同一两份源录像的机身相机恢复试验已完成。以腕部自然标记作跨录像验证，所有相机候选均超出 2 px／5 px 门限。提前两帧读取状态降低误差但仍超限，原时钟和名册保持冻结。 [标定试验与完整分母](SOURCE_HEAD_CALIBRATION.md)

腕相机另用同侧静态台角和直边复核。四个跨录像拟合方向只有一个达到像素门限，拟合的腕相机安装位置随录像切换相差 333–448 mm。没有可重复的腕相机标定进入正式匹配。 [腕部标定试验](SOURCE_WRIST_CALIBRATION.md)

Go2 端已用 V2 的 160 个原生消费区间和 G00／G01 的四个封存观测区分单指滑块、双指内侧间距和执行 TCP。42 mm 单指指令对应仿真内侧间距 85.048 mm，源域物理开度仍待标定。 [工具端与夹爪读数](TOOL_AND_SOURCE_CLOCK.md)

## 分母与下一阶段门

最终封存 8 对当前法兰几何状态及 32 张机器人诊断 RGB，分别使用机身、腕和独立侧视。另有 16 和 24 张前序工程调试图，独立计入渲染成本。合格的新模型观测格、完整渲染历史、外部 Adapter 训练样本、真实 sampler 调用、SLA 更新、仿真积分和真机动作均为零。独立确认集未使用。

在同一画面切换机器人之前，先用真实相机的同帧内外参和工具标定核验源域一致性，并为腕视角校验遮挡和可见对应。S1 正式科学申请等匹配观测通过后登记。SEA-2 的 19 窗既有数值证据可独立复用，当前没有新的原生执行资格。 [SEA-0/1 预定资格](../../../docs/protocols/SLA_GO2_V19_TASK_AGNOSTIC_EMBODIMENT_ADAPTER_20261124.md)
