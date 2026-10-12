# 先读这里｜2026-10-12 新local-pc接管

**当前是接管与仓库维护，不是新的训练/仿真GO。**

## 唯一入口

- [完整local-pc接管prompt与双服务器/HF规范](docs/handover/2026-10-12_LOCAL_PC_ADOPTION.md)
- [分支清理allowlist及原SHA](docs/handover/2026-10-12_BRANCH_RETIREMENT_PLAN.json) — 尚未实际删除；local-pc先建立兼容tag并回读，再按精确lease删除四个祖先分支。
- [科学ledger #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11)
- [可复用代码/CI/仓库维护 #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47)

## 最新用户方向

我们负责独立任务仿真、可信专家数据与上层GR00T的EEF/夹爪行为学习。师兄WBC仅作冻结执行器，不再由上层团队调参修复。TaskAnchor/UNITAS只是设计参考，尚未实现。本轮只读接管、条件化分支维护与下一数据GO草案；不恢复旧实验、不启动新模型或物理。

| 位置 | 职责 |
|---|---|
| local-pc | 唯一owner；分别SSH两端、GitHub审核交互；未来可作闭环obs/action中继 |
| Corelab | 仿真/教师数据/预处理/物理推进/CPU评分/视频，非高层GR00T模型eval |
| Hopper H200 | GR00T训练微调、模型推理与模型eval，须合法PBS allocation |
| GitHub | 代码、协议、有限预算、小收据、审核 |
| HF | 数据/权重/大证据，固定revision、prefix和hash |

两服务器互不连通；不要尝试直连scp/ZMQ。HF用于大资产，不是每0.4秒的实时控制队列。闭环需要当前动作造成的新观测；local-pc双SSH中继尚需另行验证，离线回放不能冒称闭环。

## 最近完整科学状态

PI0285源Panda确认20/32未过26/32。PI0286 C0 TRAIN2/8；C1 TRAIN8/8、DEV5/16；未运行新确认。PI0289安全启动PASS，但首hold姿态RMS0.0652rad>0.05、时钟残差47.8ms>25ms；其余五参考NOT_RUN，目标教师数据0。最近ledger终局是2026-10-10 11:08:34 SGT；接手须刷新最新证据。selected/deployment=NONE。

现役代码分支pi0267/executor-decoupled-recovery。main仍是旧accepted代码＋导航，不能直接当训练runtime。旧README、hpc gate、10月8–10日deadline与PR77不构成当前执行授权。

第一轮交付：ADOPTION_REPORT、RESOURCE_MAP、ASSET_AND_HF_MAP、NETWORK_AND_EVAL_PLAN、实际BRANCH_MAINTENANCE_RECEIPT、NEXT_DATA_GOAL_DRAFT（未授权）；发#11回读后停住等新科学GO。
