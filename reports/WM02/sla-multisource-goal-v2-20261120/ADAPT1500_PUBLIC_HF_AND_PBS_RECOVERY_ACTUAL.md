# ADAPT 1,500 公开检查点与 VANDA 接续实绩

现场截面为 2026-11-21 00:04 SGT。VANDA 原 ADAPT 作业 `1594212.stdct-mgmt-02` 于 21:30 SGT 正常结束，封存精确 **ADAPT 1,500**。该段实际执行 250 更新、16,000／8,000／8,000 三域访问、1,000 条训练内 H40。ADAPT 阶段累计 1,500 更新及 **96,000／48,000／48,000** 访问，另有 MIX 父端点 500 更新。

## Hugging Face

VANDA CPU 发布作业 `1599361.stdct-mgmt-02` 退出码 **0**，墙钟 **7 分 19 秒**，使用站点已配置的 token。发布 **141 个文件、5,547,652,640 字节**，包括 45 个训练参数、94 个 Adam 状态、1 个 RNG 及封存清单。独立匿名核验 141/141 文件 SHA256，Corelab 另行匿名下载并验证了封存清单。恢复需要原冻结 SLA 底座与匹配的数据输入。 [ADAPT 1,500 固定版本](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints/tree/f39030655f2643e866253b041f0d21f61f703e97/ADAPT5K/step-01500-65e96b87175b)

同一仓库还保存 Corelab MIX 1,000／1,380 与 VANDA TGT 5,000。Corelab 现役 MIX 记账 **1,734**，三域访问 **55,488／83,232／83,232**。原发布服务正在等待现役段封存，公开 MIX 最新仍为 1,380。 [MIX 1,380](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints/tree/14db33bb467458490b4627cc843c108b2a5908dc/SCALE-MIX20K/step-01380-aefd0f0cd67c) · [TGT 5,000](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints/tree/924f34439976fd0022afc1e76309fd9b6cae6d5e/SCALE-TGT5K/step-05000-9eb71c0da7c0)

## VANDA PBS

原未启动后继和收尾在 PBS 中为 F，无启动或退出码，相关目录无训练／评价生效回执。撤下原因待核实。沿用原项目、配方、数据、模型与期限，使用新作业身份从已经封存的 1,500 续接。旧作业及其记录保留。

| 工作 | PBS JobID（后缀均为 `.stdct-mgmt-02`） | 读回 |
| --- | --- | --- |
| ADAPT 1,750 | `1599363` | R，已记账至 1,502 |
| ADAPT 2,000 | `1599364` | 依赖 1,750 精确封存后释放 |
| 独立 2K Eval | `1599365` | 依赖 2,000 封存 |
| 短链接续收尾 | `1599366` | W，原总截止 12 月 4 日 15:29 SGT |
| ADAPT 2,250→5,000 | `1599367–1599378` | 首段和独立 2K Eval 共同依赖 2,000，训练链不依赖 Eval 成绩 |
| 独立 5K Eval | `1599379` | 等精确 5K |
| CLOCK CPU 判定与激活 | `1599382`、`1599385` | 等独立 2K Eval 和原触发证据 |
| CLOCK 原 A/B 训练及 Eval | `1599386–1599391` | 等 CPU 激活成功，实际更新和评价 0 |
| 后继与条件链收尾 | `1599392` | W，原后续截止 11 月 24 日 09:35 SGT |

短训练和 2K Eval 的最晚执行界线为 11 月 22 日 08:33 SGT。两个截止保留原值。各新角色由各自作业目录中的 `STOP.json` 控制，训练及评价入口也检查共享 ADAPT lineage 的 `STOP.json`。日志随各 PBS 角色保存在该站运行目录。机读回执列出部分执行卡摘要及完整作业范围。 [机读实绩](ADAPT1500_PUBLIC_HF_AND_PBS_RECOVERY_ACTUAL.json)

## 数据与原生证据

两条已封存 HOLD 专家为 F1 红和 F2 蓝，成功 **2/2**，共 46,156 tick、139 条完整专家 H40 和 1,390 个已确认消费原子。两条共 139 窗已通过数据准入，D2 仍为 **26 条、2,031 窗**，等待精确 2K 边界的一次选择。 [原生与入库回执](ACTUAL.md)

新增学生原生请求和 tick 均为 0。ADAPT0 原六条结果仍为 **0/6**，396 份 H40 获安装 ACK，3,930 个动作原子确认消费，六条首次非法接触均是 `x5_link6` 与蓝块。剩余固定学生条件和 HOLD4 仍等待资源与原生 world 交接。 [六条原生轨迹审计](A0_FIRST6_STOP_CAUSAL_AUDIT.md)
