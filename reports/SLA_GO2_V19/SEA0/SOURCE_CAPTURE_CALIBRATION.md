# SEA-0 源录像标定资料核验

核对官方 `pack_and_pour_fruit/arx_x5` 的 Hugging Face 和 ModelScope 发布树。两者在核验版本的任务目录中均有同名的 **100 个 HDF5 和 300 个预览 MP4**。`episode_0000000.hdf5` 和 `episode_0000001.hdf5` 的两处 Git LFS SHA-256 与现场封存文件逐个一致。两处镜像均未提供该任务目录的独立标定文件。 [版本与哈希](SOURCE_RELEASE_CUSTODY.json) · [Hugging Face 固定目录](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/1127bda94f172d757386945af3eb3a6754d1a4e6/data/RoboDojo_real/pack_and_pour_fruit/arx_x5) · [ModelScope 官方发布](https://modelscope.cn/datasets/RoboDojo-Benchmark/RoboDojo)

两份现场 HDF5 的所有对象均无属性。每个相机只有 `colors` 和 `shape`，`additional_info` 只记录 30 Hz 频率。逐图扫描全部 **5,871 张原始 JPEG**，直至结束标记。每张只有 JFIF 与 `XPL-RGB1` 两种元数据段，结束标记后无非零字节。`XPL-RGB1` 由官方用于标记颜色解码约定。 [逐视角清单](SOURCE_CAPTURE_METADATA_ACTUAL.json) · [复现脚本](../../../scripts/sla_go2_v19/source_capture_metadata.py) · [官方颜色编码变更](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/discussions/58)

官方文档在仿真 HDF5 示例中给出 `intrinsic_matrix` 和逐帧 `extrinsic_matrix`。公开相机资产树包含 D435、Kinect 和 RealSense 模型，真实评测平台论文列出的相机是 Gemini 335L 和 Gemini 305。前者提供仿真资产来源，现有公开资料无法把设备标定关联到这两段真实录像。 [官方数据结构](https://robodojo-benchmark.com/doc/usage/install-and-download/#simulation-episode-structure) · [实机平台硬件](https://arxiv.org/html/2607.04434v1#A7.SS1)

## 尚缺的现场资料

- **三路 RGB 标定。** `cam_head`、`cam_left_wrist`、`cam_right_wrist` 的内参、畸变及导出到 640×480 时的裁切、缩放或镜像约定，附两段录像的对应关系。
- **共同坐标系。** 固定头相机外参、左右腕相机手眼变换或同步逐帧外参，以及左右臂基座到共同世界坐标系的变换和帧索引约定。
- **物理工具。** 实际 `link6` 到共同工具点的变换，夹爪归一化值到指垫内距的实测映射及其适用区间。

原四个源锚点和各自的历史索引保持固定。已尝试的头相机和腕相机录像内恢复均未取得可重复标定。本轮新合格匹配观测、模型前向、物理积分和优化更新均为 0。 [固定名册](SEA0_PAIRED_ROSTER.json) · [头相机试验](SOURCE_HEAD_CALIBRATION.md) · [腕相机试验](SOURCE_WRIST_CALIBRATION.md)
