# SCALE20K 目标数据与 Corelab GPU 接续

目标 TRAIN 已冻结为 **23 条示范、1,768 个窗口**。20 条 C20 HOLD 保持原统计，3 条历史完整 PLACE 贡献 392 个 RGB/H40 窗口及 3,920 个已核对的 P10 执行原子。三条 PLACE 均来自 F1 的两个场景、同一采集谱系，包含蓝左和红右两条指令类型。三条原终态均满足抓持、抬升 20 mm、入区、释放、撤离与原任务判定。

使用冻结的场景、几何和可追溯的采集谱系核对防泄漏。经核对的历史及新留出集合共有 184 个场景哈希、140 个带双物体位置身份的布局。新 DEV 和新确认各封存四个布局、24 条指令条件，冷态构建过程无积分。新留出布局的姿态与重置身份独立。F1/F2 仍复用各自的基础机器人和场景模板。旧 TEST 仅核对分割身份元数据，读取旧 TEST rollout 为 0。

| 训练域 | 示范 | 窗口 | 入口 |
|---|---:|---:|---|
| Go2+ARX5 | 23 | 1,768 | G7 H40，50 ms，真实上次命令 |
| RoboDojo-Real ARX X5 | 48 | 50,979 | 双臂原生 H50，名义 30 Hz |
| LIBERO | 1,261 | 209,146 | 原生 H10，10 Hz |

三域真实输入在 Corelab CPU 上各读回并完成分词 4 个。正式目标访问银行按 HOLD/PLACE 两类技能均衡，再按录制分层，混源臂和目标对照共享该银行。目标每步 16 个 HOLD、16 个 PLACE 访问。三个 PLACE 的覆盖仍窄，采样重复不会增加独立示范数。

## GPU 接续

Corelab 已独立回读后台服务存活、有效持有者和定时心跳。该服务等待既有 GPU0 所有者释放且满足现场显存、主存和存储边界。接续顺序为三域真实梯度、完整 step-zero 保存、在新进程恢复，再运行计入 MIX20K 的最多 200 步技术前缀。服务的执行上限为四小时，包含资源等待，使用原 GPU0 锁，自动重试为零。启动 SHA、期限、计数及现场服务身份见同目录 JSON。截止本快照，新 GPU 更新、优化访问、模型生成与原生积分均为零。

场景和采集审计的两次早期技术尝试分别缺失 SciPy 和受审源代码中的 G7 模块。第三次使用固定源码树和现有 SLA 环境，完成实际数据审计。前两次未产生新原生请求或参数更新。

## 官方配置与显存口径

| 官方示例 | 全局 batch | 更新 | 动作长度 | 策略 FSDP 分片 | OmniVGGT 配置 |
|---|---:|---:|---:|---:|---|
| Piper 真机 | 64 | 20,000 | 50 | 1 | 共享 8 个设备 |
| RoboTwin | 256 | 30,000 | 50 | 8 | 共享 8 个设备 |
| LIBERO-Plus | 512 | 30,000 | 10 | 8 | 共享 8 个设备 |

RoboTwin 使用 VLM LoRA 和完整动作专家。Piper、LIBERO-Plus 固定代码使用默认全参数范围。三个示例均从 SLA_base 开始，state dropout 0.25，EMA 0.999，且空间对齐权重 0.5。论文记录的是不同基准的全局 batch 与更新数，公开资料未给出逐卡实测后训练峰值。OpenPI 上游给了单卡纯 LoRA 大于 22.5 GB、全量微调大于 70 GB 的估计。该参考对应 OpenPI。SLA 全量训练还需考虑 OmniVGGT、并行方式和 batch。保留公开的八设备配置时，八张 80 GB 卡是保守的资源申请起点，峰值须实测。[Piper](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/external_config/train_real_world_handover_vegetable.py) · [RoboTwin](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/external_config/train_robotwin.py) · [LIBERO-Plus](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/18896e757647ed9c2a68816b35a7100afda6e287/SLA/src/openpi/external_config/train_libero_plus.py) · [论文 §4.2](https://arxiv.org/html/2611.08373v1#S4.SS2) · [OpenPI 单卡参考](https://github.com/Physical-Intelligence/openpi#requirements)

SCALE20K 采用已验证 SLA_base、冻结视觉、VLM LoRA、完整动作专家、各域独立 I/O 和统计。混源有效 batch 128，按 32/48/48 累积，Corelab 首次资格 microbatch=1。Go2 的原 G7 目标保留并加入物理夹爪宽度项。首版未开放 OmniVGGT 和全量 VLM。根据实际抽象参数清单，模型与 Adam 常驻下界约 **10.46 GiB**，尚未纳入激活、梯度、XLA 工作区和恢复瞬时开销。GPU0 实际报告 32,760 MiB。该服务配置的 JAX 百分比为 0.76，容量门另留 2 GiB。是否可完成完整前向、反向、恢复和 200 更新，取决于该服务的实际结果。
