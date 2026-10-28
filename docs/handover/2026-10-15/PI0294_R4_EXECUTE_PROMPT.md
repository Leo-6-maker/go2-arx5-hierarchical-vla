# PI0294 R4 — Codex 直接接续指令

读取同一固定release下：

1. `docs/protocols/PI0294_R4_EVAL_FIRST_PARALLEL_DATA_AND_MATCHED_TRAINING_PROPOSAL_V1.md`
2. `reports/PI0294/design/R4_PARALLEL_DATA_TRAINING_TASK_CARD.json`
3. 根 `AGENTS.md`、`START_HERE.md` 和 #11/#47 最新实际收据。

这是待采用提案，不是执行授权。审核证据基线 `5ae777a78495b043deb96b4d805e97d9cc46c4fc`，最后读取 #11 comment6173476155：R3已经从M512完整续训1536步到M2048，S1四Teacher资格通过；666401评价PBS在2026-10-15 00:46 SGT快照仍排队，16条模型面板未开始。先刷新真实状态，不照旧快照cancel、重复qsub或重跑已经完成的模型case。R3原03:16:40/04:16:40截止不自动延长。现有incoming local-pc唯一owner继续，不另起竞争控制者。

明确报告三件不同的事：父PI0294真实四worker采集约2.86x；R3 CoreLab接触/S1验证与Hopper训练确有重叠，但R3新TRAIN采集0；正式训练DataLoader为workers0，CPU测试过0/2/4不代表已取得神经训练加速。

用户明确采用R4完整包、新窗口/资源及公共checkpoint衍生权限后，按以下顺序连续推进，不每个小步骤重复申请同项GO：

- **先完成评价**：复用或接续R3 M512 TRAIN4、M2048 TRAIN4和DEV8，保留全部S1拒绝/失败/NOT_RUN。M2048真实面板未完成或输入契约错误时，不盲目再训一个M4096。
- **CoreLab数据线**：先把已合格但未用的v3.1-only TRAIN（122条）做成Hopper可读、去重的训练视图，保留原v3.0统计用于本轮数据对照。另按协议封存24组双任务48条新TRAIN、4条pilot；只做小幅接近初态和合格F1/F2组合，不改几何/Teacher任务/物理参数。pilot4/4后复用真实dispatcher开4worker；CPU不足降为2/1，评价优先。不得冒充新采全部295条，不替换失败seed。
- **训练加速**：第一正式更新前最多60分钟比较固定数据的loader0/2/4，保持消费顺序/原生张量/恢复一致，隔离更新总≤16。实际端到端收益至少5%才采用，且两模型臂用同一profile；否则沿workers0。保持micro2×accum16、数值模式、mask、SE3/夹爪编码。不要同时改attention、dtype或上多GPU。
- **Hopper学习线**：C0从完整M2048沿原v3.0续2048步到C0_V30_M4096；CoreLab同时制作/采集。新T0+3h固定AUG池，晚到新数据只归档，不热加入。若新48条包未合格，就只用既有v3.1-only TRAIN。C1有数据/codec/时间资格才从同一M2048重新分支，原任务/阶段权重下每任务50%原v3.0、50%novel，再训同样2048步。C1的数据sampler是显式新干预，不伪造旧cursor连续；不从C0继续训C1。
- **同条件评价**：C0/C1与已有M2048均用D0+实际合格S1、固定DEV8选型；S1只拒绝不改动作、拒绝计任务失败。满足最低DEV门后固定单候选，才跑原独立核心与旧抓放回归。完整成功只能叫`MULTITASK_SIM_WITH_S1_WORKING_V1`，不是无预检策略、WBC或真机能力。

总资源按任务卡：新12h/末1h收尾，3PBS/10H200allocated h且至少3h留评价，项目1GPU、账号3allocation须新明确采用；正式新增更新≤4096、Core≤96coreh/16核、160main starts/256万steps、shadow另计200万steps。48新TRAIN及4pilot在总物理账内，不加算。公共模型衍生最多两份60GiB，raw/视频/WBC仍私有。无历史删除、无控制器扫参、无M6144自动延期。

当前任何健康R3必须先在自己的合法范围/边界收齐；不因新计划发布修改其source或合并预算。可用CPU和GPU slot由实际native状态决定，不使用共享df冒充个人quota。完整本地checkpointREADY允许同节点评价，远端归档失败不得再次结束健康训练/评价。

最终必须给实际`PIPELINE_TIMELINE.jsonl`和`PARALLEL_EXECUTION_AUDIT.json`，计算数据采集/审计与实际optimizer区间的重叠，而不是只画两路架构；清楚列采集worker数、训练loader数和GPU数。报告每任务模型行为、真实视频、AUG数据来源、等预算两臂更新与样本数、固定版本和全部失败。某条线阻断不自动停另一条独立有效线；未知、未运行、假设不得写PASS。

源码和小报告普通推送GitHub并回读固定commit/评论；私有精确路径、凭据、模型、原始图像和资产不随Git上传。未来handoff继续由Codex直接读取GitHub，而不是依赖本地ZIP。
