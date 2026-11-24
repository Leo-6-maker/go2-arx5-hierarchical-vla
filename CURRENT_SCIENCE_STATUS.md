# 科学状态与证据边界 · 2026-11-12

## SLA最新可核验快照与后继计划

证据：[PR #378](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/378)，HEAD `353bcdd1774d5f9fcbbc6d9a581d628428245bc0`，更新时间2026-11-12 21:44:53 SGT。本段来自仓库回执，未做实时服务器探针。

- S1已经完成500次真实更新、16,000访问和普通RGB部署重建。
- TRAIN01因raw夹爪越界被入口拒绝，TRAIN07初始化技术失败；候选H40实际消费为0。TRAIN13/25及DEV16已有有限后台队列，接手需核对真实效果后接续。
- S2更新、S1 DEV16和G DEV16在该快照均为0；教师/评价准备不计实际训练或任务成功。
- 旧SFT2000＋basis的TRAIN4有完整4/4回执。官方A500的4/4已由执行方报告，固定包已有绑定；同等详细的训练/逐条件终态仍需补齐，不能用旧SFT2000代替。

详见 [V2计划的事实与来源](docs/protocols/SLA_GO2_ARX5_SCALEUP_V2_20261112.md#1-已知事实未知项与现役衔接)。A40/H200迁移、并行采集、新动作合同和扩量矩阵均为后继设计，本次发布没有执行这些实验。

## 2026-11-11历史快照

以下为原相机/公开ARX5首轮入口记录；最新事实以带固定SHA的执行回执为准。

## 当前问题

在相同官方 GR00T 起点、原生 body+wrist 输入、目标 TRAIN100、目标 500 更新及相同解码／评分条件下，比较是否增加一次公开 ARX5 500 更新的预适配。

当前状态：**新协议准备与仓库入口整理；本次发布未运行新相机覆盖、训练或 A/B 整机评价。**

[详细协议](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/research/native-camera-arx5-transfer-v1-20261111/docs/protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md) · [整合状态](docs/maintenance/2026-11-11/REPOSITORY_INTEGRATION_STATUS.md)

## 已核验与待补证据

| 项目 | 当前状态 | 来源／边界 |
|---|---|---|
| 整理前 main | `0a27753bbda8ed75ab6b74aeb64755e0427362c7`，最后合并 #341，2026-11-03 | [固定快照](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/0a27753bbda8ed75ab6b74aeb64755e0427362c7) |
| 可复用执行实现 | `f0e19c8812896f98053e74b165d6c52e85327b69`，#371 的审计时 head | [固定源码](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/f0e19c8812896f98053e74b165d6c52e85327b69)；不是本轮已资格的执行端点 |
| 官方原始父权重 | revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`；仓库已有七文件 SHA256／大小清单 | [raw_parent_delivery.py](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/scripts/pi0294/multitask/raw_parent_delivery.py)；本次未重新下载或校验大权重 payload |
| TRAIN100 | 既有封存人口为 100 回合、1,450 个监督窗口；新相机版本尚未产生 | [既有执行报告目录](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/f0e19c8812896f98053e74b165d6c52e85327b69/reports/WM02/task-learning-core-hopper-20261109) |
| 既有 endpoint basis | 当前 EEF 与实际已消费名义夹爪值锚定，使用模型末点重建 40 点曲线 | [精确实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_endpoint_basis_v3.py)；不等同于使用全部原始预测点 |
| 新相机候选 | 用户给定的已预览安装；全量覆盖与状态重渲染待执行 | 不能以预览替代 95%／90% 全人口覆盖检查 |
| Table30 源 | 固定 revision 已核实，ARX5 为六维 RPY pose＋实测宽度 | [固定源](https://huggingface.co/datasets/RoboChallenge/Table30/tree/1eda76b8935b5c7a59cd635ca9e3fab882943e11)；具体 episode、时钟、RPY/TCP 约定、归档字节清单待核实 |
| Teacher24 最终 19/24 | **保留用户提供的最终负结果** | 当前核验的 GitHub head 只有较早 first10／first11 快照；补齐完整终态 receipt、逐条件分母、SHA 与哈希，不用旧部分结果覆盖用户结果 |
| 本轮 A/B 500 端点 | 尚未生成 | 250 端点只用于恢复／诊断；500 是预注册评价端点 |
| 新开发人口 | 提议 20 布局×2 重启／方法 | 身份重叠审计未完成，尚不能称为已证明独立的 DEV |

## 必须保留的历史结论

CORR2048、SFT2000、C500 是独立历史基线。原 BF16 跨硬件 REJECT、旧 DEV／TEST、原始预测拒绝、Teacher24 的 19/24 负结果均保留，不通过改相机、改父权重、重命名 decoder 或选择性重跑撤销。

旧 full-H40/raw 与 endpoint-basis 是不同组合策略。离线 FM、包络通过、公共数据保留集误差均不能替代整机闭环成绩；源任务也没有本轮闭环成绩。一个训练种子、四布局首筛及有条件扩样只支持 pilot 结论。

## 历史明细

- [整理前 main 的完整状态页](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/CURRENT_SCIENCE_STATUS.md)
- [执行快照完整状态页](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/CURRENT_SCIENCE_STATUS.md)
- [Teacher first10 原回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/reports/WM02/task-learning-core-hopper-20261109/V3_1_TEACHER_FULL24_FIRST10_AND_PRE_E1_CONTACT_GATE_ACTUAL.json)
- [旧 V7 总纲](docs/protocols/GR00T_WBC_SCIENCE_GOAL_V7_20261031.md)

本页只登记来源与结论边界；实时作业状态以执行端当前只读核验为准。
