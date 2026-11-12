# 科学状态与证据边界 · 2026-11-13

## 当前问题

在冻结WBC、腿策略与原生安全条件下，建立Corelab多环境数据生产和GPU常驻训练／推理流程，取得Go2＋ARX5上的真实上层策略，并用数据规模、优化预算、纠正数据与独立评价解释收益。

[本轮V3计划](docs/protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.md) · [机读矩阵](docs/protocols/SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.json) · [交接](docs/handover/2026-11-13/SLA_CORELAB_PARALLEL_V3_CODEX_START.md)。

## 最新核验快照

以下来自 [aaf80b4的实际回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/reports/WM02/sla-scaleup-v2-20261112/CURRENT_STAGE_ACTUAL.json)，时间为 **2026-11-13 00:02:56 SGT**，不是本次登录机器测得的实时状态。

| 项目 | 已核验 | 尚未证明 |
|---|---|---|
| S1 | 500更新、16,000访问；原普通RGB重建通过 | 独立DEV任务能力 |
| S1 TRAIN4 | 2/4终态，保留策略拒绝与技术初始化失败 | 另外2个条件的最新终态 |
| S1 DEV16／S2／G DEV16 | 快照均未执行 | 最新现场进度需读回 |
| V2夹爪数据 | 1,450窗口／58,000点CPU派生 | epsilon端点偏差已通过原生夹持 |
| V2 batch | CPU micro1/8/16共同fixed32随机bank检查 | 真实SLA CUDA梯度和吞吐 |
| V2新更新／RGB生成／物理 | 0 | 新策略已训练或已执行 |
| EEF_STREAM_V2 | 未实现 | 门弧线／中间路径已被保留 |
| 新大卡allocation与site入口 | 未验证／未完成 | H200已开始训练 |

原TRAIN100为100个episode、1500 captures、1450专家窗口，50 intervention排除当前BC；该数量回执不是100/100无接管完整成功证明。[数据回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/353bcdd1774d5f9fcbbc6d9a581d628428245bc0/reports/WM02/sla-posttrain-20261112/SLA_TARGET_CODEC_DATA_ACTUAL.json)。

## 用户已授权的新工作

2026-11-13用户明确允许Corelab并行多环境采集并要求效率与避免CPU/GPU往返。授权覆盖独立世界与必要的项目内实现／验证，不再逐actor请求许可。旧Core全局串行解释改为每world独立owner，其他安全和资源边界保留。详见[AGENTS](AGENTS.md)。

本次文档发布没有运行新训练／仿真／数据采集，没有报告已实现的并行倍数或GPU吞吐。新计划的PLANNED与execution_ready=false不能改称RUNNING或PASS。

## 当前实现接缝与下一项结果

- body相机尚未进入已有持久session分支，采集仍有同步PNG／state写读、重复模型／renderer建立。
- settle_atom存在固定等待与已有事件就绪接缝；需要正确接通，不删除ACK。
- V2 learner尚需消除逐microbatch标量同步、完善更长LR／visit map与实际GPU验证。
- Corelab下一项结果应是持久采集完整episode和2／4 actor实测；GPU下一项是实际更新与可部署RGB包。
- 性能目标以合格完整示范/小时、覆盖和传输trace评价，不以GPU利用率或CPU测试数评价。

实现证据固定在 [#380](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/380)。原W0曾处于WAIT且有有限期限，接手按当前unit／注册组／终态处理，不能因旧快照再次提交。

## 实验与预算口径

继承[V2](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b07936899b10f196653f617f95c5cd8b124e5e99/docs/protocols/SLA_GO2_ARX5_SCALEUP_V2_20261112.md)的29k更新／928k访问主矩阵；V3只对选中配方增加条件性优化预算曲线。记U_probe为seed0实际探索最高步数、U_selected为最终选择步数，则完整分支最大工作量为：

29,000＋(U_probe−3,000)＋2×(U_selected−3,000)，最多80,000更新／2,560,000访问。

已探索但未选中的计算照实计费。另外两个seed使用同一固定LR日程到U_selected的前缀；不得不同预算混称同方法。长训练多次DEV用于选型，最终三seed TEST保持独立。

## 保留的历史结论

CORR2048、SFT2000、C500、BF16跨硬件REJECT、旧DEV/TEST、原始预测拒绝和Teacher24最终19/24负结果保留。源only未来实测EEF／宽度与目标实际消费专家命令仍是不同标签定义；fixed-base源成功不等于目标整机成功。

[11/11完整状态快照](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6843163d641eb24a29c3b1557f868765346ba66c/CURRENT_SCIENCE_STATUS.md) · [原生相机/公开ARX5计划](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/research/native-camera-arx5-transfer-v1-20261111/docs/protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md) · [历史V7](docs/protocols/GR00T_WBC_SCIENCE_GOAL_V7_20261031.md)。
