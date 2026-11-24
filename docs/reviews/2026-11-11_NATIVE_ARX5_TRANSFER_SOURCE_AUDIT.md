# 首轮方案来源审计 · 2026-11-11

## 结论

用户的A/B足以检验固定目标预算下公开ARX5预适配的增量价值。发布前核对了main／执行源码、官方模型与Table30固定revision；本次没有下载大模型／视频、运行模型或物理训练。具体新数据与新端点仍为PENDING。

[执行协议](../protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md)与[机器规格](../../experiments/native_arx5_transfer/native_camera_public_arx5_transfer_v1.json)落实下列修订。

## 与原提纲相比，明确了什么

| 原提纲中的简写 | 审计发现／明确决策 | 对结论的影响 |
|---|---|---|
| 完整H40、同一可行decoder | 现有basis使用当前状态及预测末点重建H40，完整消费重建40区间 | 结果属于GR00T+basis；raw包络独立报告 |
| H40、50ms | 既有knot offsets为0…1950ms，消费40区间共2s | 避免source标签整体移位50ms |
| 公开ARX5转换 | 固定Table30是6D RPY；示例不按timestamp转换，stride有索引错配风险 | 使用新time/frame审计，不复用v2 quaternion假设 |
| 未来实测EEF | 固定README仅有字段描述，实际feedback/setpoint仍须采集端证据 | SOURCE_ACHIEVED标签需先证明，不从文件名推断 |
| 源gripper width | m制实测量，与target nominal gripper不是同变量 | 保留source width(m)+自己的统计，不改消费者映射 |
| 同一官方起点 | 原七文件hash已登记；官方没有命名ARX5/ID10槽 | category10作为项目约定保留；新起点实测资格 |
| 冻结骨干cache | 只能缓存backbone输出，不能缓存依赖trainable接口的DiT前缀 | 保留接口梯度、flow时间与噪声 |
| 500cosine/25warmup | 旧trainer有8000/200以及0.1floor历史设定 | 新日程明确公式／LR序列，不隐式继承 |
| 3200/400窗口预算 | 固定源树约736GB，有大archive分片 | 窗口预算与获取／解码成本分开 |
| 8→40扩样 | 首筛结果决定是否扩样 | Stage1／新增Stage2／合并分别报，不作无选择显著性结论 |
| 新DEV网格 | 需要绝对scene身份和旧人口input-only交叉审计 | 不靠改名字避开旧TEST边界 |
| 整理main | 当前执行head相对旧main有730个commit／1522个blob差异 | 本次只合并入口整理，执行栈另行整合 |

## 固定来源

| 来源 | 身份／链接 | 已核实 | 未声称已完成 |
|---|---|---|---|
| 官方模型 | [2fc962…](https://huggingface.co/nvidia/GR00T-N1.7-3B/tree/2fc962b973bccdd5d8ce4f67cc63b264d6886495) | revision、H40、32层DiT、4次推理步、132dim | 新Core模型load／数值资格 |
| 官方embodiment | [固定映射](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/2fc962b973bccdd5d8ce4f67cc63b264d6886495/embodiment_id.json) | 无命名ARX5/Go2/ID10槽 | 官方专门ARX5微调能力 |
| 七文件原身份 | [原登记代码](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/scripts/pi0294/multitask/raw_parent_delivery.py) | 七文件共6,914,922,850bytes、完整SHA256 | 本次重新下载／二进制校验 |
| Table30 | [固定README](https://huggingface.co/datasets/RoboChallenge/Table30/blob/1eda76b8935b5c7a59cd635ca9e3fab882943e11/README.md) | ARX5字段、相机键、文档时钟单位 | 具体episode内容与实际时间配对 |
| 源示例 | [固定converter](https://huggingface.co/datasets/RoboChallenge/Table30/blob/1eda76b8935b5c7a59cd635ca9e3fab882943e11/convert_to_lerobot.py) | 索引读state／action与顺序read视频；不使用timestamp | 本轮合格的H40转换器 |
| 源归档规模 | [固定tree](https://huggingface.co/datasets/RoboChallenge/Table30/tree/1eda76b8935b5c7a59cd635ca9e3fab882943e11) | 页面显示约736GB及archive parts | 实际本轮传输量／可选任务数量 |
| basis | [精确实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_endpoint_basis_v3.py) | raw末点、currentEEF、consumed nominal grip重建quintic/SLERP | 全原始H40驱动控制／自动可行 |
| 原生时间 | [consumer](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_grasp_v1.py) | 25*i原生tick offsets、40区间 | 本轮新端点消费已通过 |
| cache | [feature seam](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_features_v1.py) | frozen backbone输出边界 | 新相机cache已生成 |
| Teacher24 | 用户本轮给定最终19/24；[旧first10](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/reports/WM02/task-learning-core-hopper-20261109/V3_1_TEACHER_FULL24_FIRST10_AND_PRE_E1_CONTACT_GATE_ACTUAL.json) | 用户结果保留；仓库旧部分回执9/10 | 完整19/24终态receipt已在该head找到 |

## 待补事实与最小处理

| 待补 | 应由什么证据关闭 | 影响 |
|---|---|---|
| body候选的FOV口径、全覆盖 | 原preview配置＋TRAIN100原图／Processor后mask统计 | P1前不能主训练 |
| 原数据完整性／新渲染 | 实际manifest、state/action/mask/stats/wrist对照 | 新target数据准入 |
| source实际EEF反馈语义 | 采集端定义或实际joint/FK/TCP与时间链 | 不可把setpoint标ACHIEVED |
| source RPY/TCP、width、时钟 | 具体episode、采集／标定、PTS或可信frame-ID映射 | B-source数据准入 |
| 原始获取与split | archive bytes/member hashes、采集组去重与固定list | 明确真正成本与数量 |
| 模型／参数／cache／恢复 | 本轮Core的最小受影响接口资格 | 三阶段训练与普通推理 |
| 新DEV独立性 | 原旧人口输入身份对照，禁止读旧TEST分数 | 正式新DEV注册 |
| Teacher24完整最终receipt | 19/24全条件终态、来源SHA与hash | 证据归档；不自动重跑Teacher |

明确区分“计划已冻结”“实际已执行”“证据未取得”。任何pending都不是要求用户重复批准常规修复；执行方应先用已有授权完成能做的工程工作。
