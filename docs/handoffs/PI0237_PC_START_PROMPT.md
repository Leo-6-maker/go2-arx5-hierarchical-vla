# PI0237 PC/Codex 启动 prompt

把下面正文发送给有权访问仓库和 Hopper 的执行 agent。实际启动时，以 GPT 提供的 immutable commit 链接读取同 commit 的协议；不要只取不断移动的分支 HEAD。

---

你是本项目的执行 agent。现在明确授权你执行 PI0237，不是只阅读计划，也不是等待 `My request:` 后再出现另一句确认。

仓库：`Leo-6-maker/go2-arx5-hierarchical-vla`。
协议分支：`pi0237/failure-attribution-diagnostic-plan`。
协议：`docs/protocols/PI0237_FAILURE_ATTRIBUTION_DIAGNOSTICS_V1.md`。
代码导航基线：`7d3caca79f85dd9fc5e2faa4f81542920414a0a7`。
Canonical scientific ledger：Issue #11；PR #77 仅作 execution pointer，不 merge。

先完整读取指定 immutable commit 的协议及其引用的实际执行代码、PI0236 canonical comment 5746476491、本轮 PI0237 canonical comment 和最新相关状态。PI0236 终态为 NO_ADMISSIBLE_CHECKPOINT，5×1088 records、0 ACCEPT/7 WARN/5433 REJECT，原 selector 不改、selected checkpoint=NONE。不要把 baseline 或 fresh-7500 代选为 WBC 候选。

按 P0→P1→P2→P3→条件通过后的 P4→P5 执行：

P0：核验原 manifest/S3/S4/S5/profile hashes、真实 import 来源、TRAIN/CALIB split/allowlist、实际 fileset quota、重复 job；先按协议的元数据算法冻结 64 TRAIN+64 CALIB cohort，不根据 gate/loss/目标距离挑样。
P1：对旧 5440 records 完成 issue 分布/重叠、超阈值倍数、分层与配对趋势、all-row 与 non-REJECT 统计分离，缺失量明确标记；零模型 forward。
P2：完整 1088 CALIB+固定 64 TRAIN 专家 H40 与对应 StateV1 经原 provider/chunk/原 SafetyGate。先证明窗口/time/frame/mask 对齐，不能替换首点或补标签来制造 PASS；零模型 forward。
P3：真实训练 encoder/normalizer/mapping 与真实推理解码做 cross-path round-trip，比较最终生效的 TRAIN stats、embodiment、anchor、slots/mask、rot6d、时序；clipping、量化与语义 mismatch 分账；零网络权重加载、零模型 forward。P2 的负结果可继续纯 CPU 定位，但不得开启 P4。
P4：仅协议的全部 admission 条件通过，才授权 baseline-1000 与 623593 family 的 fresh-10000 进行无梯度诊断。固定 64 TRAIN+64 CALIB、两模型，最多 256 个 attempted get_action requests；复用可验证旧 tensors 不另跑，首条预定样本兼作 canary并计入预算，CALIB replay hashes 不一致立即封存停止，不重试到成功。固定顺序/seed=22301/原 processor/dtype/sampler/dt/ABI。一个 H200 job 上限，CPU 短作业按协议最多一个；禁止重复 qsub与自动重启。
P5：报告模型对专家的 XYZ/SO(3)/gripper/temporal error、原 gate、分层和 source-rollout 聚合，给有证据支持的归因及缺口；STOP_FOR_PI_ATTRIBUTION_REVIEW，不继续其他阶段。

本授权中“只读诊断”表示不改冻结数据、权重及科学契约，但允许写新的诊断脚本/测试/证据。P0–P3 严格零模型推理；P4 的有界新推理已在上述条件内明确授权，不需要再次询问“是否允许 forward”。本轮禁止任何训练/backward/optimizer/scheduler step、DAgger/记忆实验、WBC/仿真 step/receding/hardware、DEV-E/F或Final-F内容读取；禁止修改 selector/SafetyGate/阈值/时间/frame/StateV1/H40、stats/processor语义、checkpoint，禁止 smoothing/clipping补丁/retiming、选最不差模型和 PR merge。

只在独立新分支/worktree及新输出目录新增最小诊断代码，保留旧证据不动；离线运行、缓存定向到新目录，不下载新模型、不清理旧权重、不扫描受限数据树。若已有相同协议活动作业或封存结果，核验复用，不重提。若证据/身份/容量/语义/样本覆盖不满足要求，输出明确 BLOCKED/负结果及已完成产物，停在规定边界；不要靠修复科学配置或换样继续。

完成后将小体积代码、测试、报告和证据 inventory推送新诊断分支；Issue #11 写 PI0237 canonical closure/blocker，PR #77 写 pointer。回报：各阶段状态、实际 attempted/completed/reused queries、job IDs、关键 metrics、归因及限制、完整证据路径/hash、所有禁区计数和 selected_checkpoint=NONE。不要只回复“已读，等待确认”。
