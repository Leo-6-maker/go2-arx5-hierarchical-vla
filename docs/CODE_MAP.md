# 代码地图 · 2026-11-11

## 先区分两个版本

`main` 当前承担整合／发布入口。当前本轮参考实现固定在 `f0e19c8812896f98053e74b165d6c52e85327b69`；以下执行模块多为 feature-only。固定链接明确指向该 SHA，不能把文档存在当作 main 已可直接运行。

| 能力 | 固定源码 |
|---|---|
| 原始官方七文件身份 | [raw_parent_delivery.py（整理前 main）](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/scripts/pi0294/multitask/raw_parent_delivery.py) |
| 七类接口的目标类别切片 | [minimal_adapter.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/scripts/pi0294/generalization/minimal_adapter.py) |
| Core 训练 | [task_learning_core_local_v1.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_core_local_v1.py) |
| 原生 TRAIN100 数据 | [task_learning_native_dataset_v1.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_native_dataset_v1.py) |
| 冻结骨干特征 seam | [task_learning_features_v1.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_features_v1.py) |
| GPU 输入缓存 | [task_learning_gpu_rows_v3.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_gpu_rows_v3.py) |
| 原生暂停观测 | [fusion_paused_observation.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/fusion_paused_observation.py) |
| 模型与原生循环 | [target_gr00t_native_loop_v1.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/target_gr00t_native_loop_v1.py) |
| 匹配整机评价 | [task_learning_matched_native_v3_1.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_matched_native_v3_1.py) |
| 本轮基准解码表示 | [task_learning_endpoint_basis_v3.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_endpoint_basis_v3.py) |
| 表示到原生消费 | [task_learning_endpoint_policy_v3.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_endpoint_policy_v3.py) |
| 50 ms／40 区间程序 | [task_learning_grasp_v1.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_grasp_v1.py) |
| GPU 常驻历史配方 | [core_local_train100_last2_gpu_resident_20261110_v2.json](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/experiments/wm02/core_local_train100_last2_gpu_resident_20261110_v2.json) |

历史配方中的 SFT2000 父身份、旧日程、旧种子、固定1,450行假设等不是本轮新起点／源数据的执行配置。新协议需要适配入口与窄范围验证；不得修改旧 recipe 让旧审计结果看似自动适用。

`world_model/` 是历史目录名。可以复用其中训练／观测／消费代码而不启用 WM 预测或校正路径；本轮不为目录名称做整体重构。

## 导航

- [本轮协议](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/research/native-camera-arx5-transfer-v1-20261111/docs/protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md)
- [main／执行分支整合状态](maintenance/2026-11-11/REPOSITORY_INTEGRATION_STATUS.md)
- [整理前代码地图](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/docs/CODE_MAP.md)
- [仓库工作流](REPOSITORY_WORKFLOW.md)
