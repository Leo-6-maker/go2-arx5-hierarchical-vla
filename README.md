# Go2＋ARX5 Hierarchical Task-Space VLA

> **2026-10-12：新local-pc Codex先从 [START_HERE.md](START_HERE.md) 接管。** 本次更新只整理导航、交接和分支维护，不启动训练/仿真，不合并或重写历史运行代码。

## 项目与当前分工

视觉＋语言＋实际TCP/夹爪 → 预训练GR00T的EEF/夹爪操作参考 → 明确的frame/time/safety接口 → 师兄的冻结WBC → 实际反馈。

上层团队以独立任务数据、GR00T微调、物体闭环与多任务为主线；不替师兄继续调WBC。EEF表示降低本体耦合，但源Panda/独立工具成功不等于Go2跨本体或真机成功。

## 接管与证据入口

| 需要 | 去哪里 |
|---|---|
| 新窗口完整接管prompt、SSH/HF与双端分工 | [Local-pc adoption](docs/handover/2026-10-12_LOCAL_PC_ADOPTION.md) |
| 条件化分支清理；尚未删除任何head | [Audit allowlist](docs/handover/2026-10-12_BRANCH_RETIREMENT_PLAN.json) |
| 研究前提 | [任务空间研究章程](docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md) |
| 科学授权与实际进度 | [Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11) |
| 工程/CI/分支归档 | [Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47) |
| 最近上层学习终局 | [PI0286](reports/PI0286/final/FINAL_SCIENTIFIC_REPORT.md) |
| 最新部署路线执行器终局 | [PI0289](reports/PI0289/execution/REPORT.md) |
| WBC版本与接口比较 | [2026-10-10 review](docs/coordination/2026-10-10_WBC_PACKAGE_COMPARISON_AND_TARGET_LEARNING_REVIEW.md) |
| TaskAnchor/UNITAS启发，尚非GO | [2026-10-12 review](docs/coordination/2026-10-12_TASKANCHOR_UNITAS_DATA_AND_FINETUNING_REVIEW.md) |

本分支pi0267/executor-decoupled-recovery保存实际近期代码；main保留较早accepted代码与导航。每次执行绑定protocol/receipt所指完整SHA，不能从当前branch HEAD自动恢复旧任务。

用户最新资源要求：Corelab负责仿真/数据/CPU分析等非训练工作；Hopper H200负责GR00T训练、神经推理与模型eval。两服务器不能直连，数据和权重经HF固定revision；未来闭环obs/action中继由local-pc单独设计与验收。

## 最近结果与资格边界

PI0285源Panda独立确认20/32未过资格。PI0286 C0 TRAIN2/8，C1 TRAIN8/8、DEV5/16，新确认未运行。PI0289安全启动通过，首hold姿态质量与时钟资格仍未通过；其余运动、目标教师数据尚未完成。最近科学账本时间2026-10-10 11:08:34 SGT，需查询#11是否更新。这不是实时GPU/PBS状态。

`selected/deployment=NONE`。TaskAnchor、UNITAS和独立数据路线是准备/研究设计，不等于已实施。接管不会重置预算、放宽阈值、重新打开旧确认集或授权WBC/GR00T运行。

## 历史与资产保护

此前README含Stage45及PI0282/PI0285 deadline导航，完整保存在[交接前固定README](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/5c757eaea3b3078e3406b604e70267ca5d0c0130/README.md)。这些历史结果和代码均保留；本次只更换入口，不把历史内容改判为PASS。

大模型、完整Trainer、数据和视频留HF/已授权服务器，用manifest固定身份；私有WBC/vendor源码、mesh、二进制、密钥不得公开。权重+processor+statistics+mapping+decode+executor/clock共同构成可重现实验身份。

历史hpc/和owner脚本受各自协议约束，先读后用。本次首轮只做轻量接管与维护，没有新GPU、模型、teacher、physics或硬件授权。
