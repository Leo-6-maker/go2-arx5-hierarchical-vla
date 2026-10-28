# Project status — 2026-10-12 local-PC adoption snapshot

**ADOPTION_ONLY，非实时服务器状态、非新科学GO。** 新owner先读[完整接管规范](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3d3cecaa5a5d8b9c6a17158912b6f5816702b678/docs/handover/2026-10-12_LOCAL_PC_ADOPTION.md)，再刷新[Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)。

现役源码分支pi0267/executor-decoupled-recovery；main为较早accepted代码＋导航，未合并近期运行实现。2026-10-01状态完整保存在[原固定版本](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1f05bfc919b25d3146839529d2763ac847010d7d/docs/PROJECT_STATUS.md)。

## 最近实际结果

| 阶段 | 已确认 | 不得推断 |
|---|---|---|
| PI0285源Panda | TRAIN7/8、DEV13/16、独立确认20/32<26/32 | 无合格部署policy；不是Go2/WBC成功 |
| PI0286源Panda | C0 TRAIN2/8；C1 TRAIN8/8、DEV5/16 | C0 DEV未运行而非0/16；新确认未运行 |
| PI0289 OCS2/MuJoCo | 安全启动10s PASS；hold XYZ11.67mm PASS、SO3.0652rad FAIL；时钟47.8ms>25ms | 未完成其他五参考；目标教师数据0；不是完整任务执行器资格 |

最近已读取科学终局：[PI0289 #6093151230](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6093151230)，2026-10-10 11:08:34 SGT。10月12日文献设计稿不是实验执行结果。selected/deployment=NONE。

## 最新用户工作边界

上层团队优先独立任务仿真、可信专家数据、原生GR00T微调与物体闭环，不继续修改师兄WBC。WBC由师兄独立推进，后续按冻结快照组合评价。

Corelab负责仿真/采集/预处理/物理推进/CPU评分/视频；Hopper H200负责GR00T训练、模型推理与模型eval。local-pc唯一owner，分别SSH两端。两服务器不可直连；数据/权重经HF固定revision交接。未来真正闭环需local-pc obs/action中继或单独明确的可行拓扑，不能用离线回放替代。

首轮只做接管、只读连接/资产/HF验证、有限仓库维护、下一数据GO草案；不提交新PBS/GPU作业、不运行神经模型/物理/WBC、不重启旧授权、不恢复旧budget。

## 仓库维护状态

已审计9个heads、0开放PR。4个祖先head可在live recheck与兼容tag备份后清理；3个有独立提交的旧branch保留；main和现役branch保留。本次ChatGPT尚未执行远端head删除。[清理清单](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/3d3cecaa5a5d8b9c6a17158912b6f5816702b678/docs/handover/2026-10-12_BRANCH_RETIREMENT_PLAN.json)由local-pc执行并回读到[Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47)。历史代码/协议/证据/HF资产不删除。
