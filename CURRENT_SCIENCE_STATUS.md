# 科学状态与证据边界 · 2026-11-17

PR403固定head dbf222b41cb666bff4dd5412b864d023aec69399：冻结SLA SHIFT7完成3次真实移动抓持，学生8尝试=3成功+4科学失败+1IPC技术中断；F1有效3/3、F2 0/4。18原生请求、319518tick、319生成；该轮新增训练0。原pilot未过，DEV/TEST未启动，证据限已知布局SIM_PAUSED_NATIVE。[不可变实测报告](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-sim-only-v51-20261117/FINAL_RESULT.md)。

本次新增 [V5.2后续计划](docs/protocols/SLA_MOBILE_F2_POSTTRAIN_V5_2_20261117.md)：SHIFT7冻结对照/普通FM500/边界loss500配对闭环，证据触发12次纠正示范及数据对照，再按门进入DEV32/TEST40。**本次仅发布计划，新训练/生成/仿真均0；未合并或部署执行栈。** [启动入口](SLA_MOBILE_V52_START.md)。

以下2026-11-11状态为历史快照，不作为当前SLA运行状态。

# 历史科学状态与证据边界 · 2026-11-11

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
