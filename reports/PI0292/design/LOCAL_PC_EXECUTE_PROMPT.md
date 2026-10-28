# PI0292 — 交给现有 local-pc Codex 的执行指令

你继续作为唯一 local-pc owner。用户已经放行 Corelab 整理清理，并要求推进科学主线；现在接手新的有界 PI0292，不再只写研究草案。

从 Issue #11 本次 canonical 评论取得固定 release，完整读取：
- `docs/protocols/PI0292_ARX5_NATIVE_B0_TRAINING_AND_CLOSED_LOOP_V1.md`
- `reports/PI0292/design/GO_TASK_CARD.json`
- `reports/PI0292/design/CONFIRM_LAYOUTS.json`

完整协议控制一切预算与门槛。发布不是远端执行收据；接手先记录真实T0/run UUID/deadline=T0+24h，重连不重置。最后60分钟收尾。各阶段通过后在范围内自主继续，无须每步重等GO；质量失败不能换seed、改阈值/配方或自动续训。

1. 读取PI0291终局 #6136292691 与原 private HF固定revision `c25eeadf9be0aa7d46bb6f7b65e260cbf3a305ea`。原56 TRAIN/33656窗口/CPU验收/HF回读已完成，不重采、不覆写；原PARTIAL保留。
2. 与已放行的清理任务核对真实路径、环境、活动依赖/lease。科学任务不授权删除；只要依赖稳定即可继续，不必等全部历史清理。物理用4080S CPU，所有神经工作仅Hopper合法新PBS；原WBC/hardware均0。
3. 新建耐断连dispatcher和账本。旧PI0291 guard写死时钟，不得改原账本续跑。外层authority/路径/日志/守卫可新建；physics/teacher/camera/scorer/EEF语义保持冻结并出diff资格。先完成无物理断线/重连/重复启动等测试。
4. 两条环境连续性教师canary通过后，按16 DEV先、8 TRAIN后的顺序，在新attempt namespace执行原未尝试logical IDs。仅24个首次尝试，原card/精确seed关联。原56与新增合成新bank v2，全部分母明确且TRAIN准入≥56/64、DEV≥14/16才进入主训练。新TRAIN统计拟合一次，不覆盖旧统计、不让DEV参与。
5. 为现有已验证NPZ+PNG实现独立train-capable adapter，实际接官方processor/collator/Model/Trainer。不得调用会清CUDA和封禁forward的旧CPU guard，也不删除它；不在本轮重造LeRobot数据链。补冻结统计接口、processed-only collation、sampler/恢复及独立DEV evaluator。
6. 在Hopper验证官方 `nvidia/GR00T-N1.7-3B`完整revision/shards/processor与加载收据，然后隔离smoke验证真实forward/backward和4步连续对2+2恢复。smoke总optimizer更新≤16；主训练重新从pristine加载。
7. 只训练一个原生flow-matching基线：冻结VLM、训练官方projector+DiT，lr1e-4、有效batch32（micro2×accum16×1GPU；官方global_batch_size是累计前）、最多4096步。配置以协议为准。保留完整S2048/S4096，不加loss，不依DEV调参；B0=pristine同processor，B1=固定S4096而非挑最优。
8. 按冻结probe比较B0/S2048/S4096的TRAIN/DEV分阶段位姿、夹爪、事件、速度与Gate，先做expert阳性对照。再做dual-SSH pause-on-observation relay资格与两条B1 TRAIN布局canary；满足门后在8个新确认布局上跑Teacher/B0/B1同条件对照。必须是真实新观察→新模型查询→prefix8→物理推进，不是预录轨迹播放。
9. 所有54次物理start、864000积分、2624生成查询、8 H200 GPU-hours、最多3PBS、16 smoke+4096 main updates以及CPU/空间限额累计记账。任一真实安全违规停止对应臂；Gate拒绝保留为失败，不重采噪声。
10. 发布真实阶段/最终报告、原视频、全部attempt、数据/模型固定私有归档和回读、正确恢复命令。#11报科学，#47仅协调维护。selected/deployment NONE。局部阻断仍交已经成立的结果，不能用“等新GO”替代本协议已经允许且过门的下一步。
