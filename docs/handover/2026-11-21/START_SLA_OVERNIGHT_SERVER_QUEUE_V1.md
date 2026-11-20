# Codex 夜间接续启动指令（可直接交给有现场服务器权限的Codex）

**用户要关闭本地电脑。目标是确认已有服务器后台依赖链可独立运行，并填补真正缺失的有限任务，而不是开启持续SSH或不断重新提交新作业。**

仓库 `Leo-6-maker/go2-arx5-hierarchical-vla`。

首先从**本次发布的固定Git提交**读取：
- `docs/protocols/SLA_OVERNIGHT_SERVER_QUEUE_V1_20261121.md`
- `docs/protocols/SLA_OVERNIGHT_SERVER_QUEUE_V1_20261121.json`
- `docs/protocols/SLA_MULTISOURCE_CLOSED_LOOP_GOAL_V2_20261120.md`
- `docs/protocols/SLA_SCHEDULER_FIRST_ASYNC_EVAL_V1_20261119.md`
- `AGENTS.md`及原运行与安全合同。

请直接执行，不要只回复新规划。当前用户现场时间2026-11-21 **01:59 SGT**（并非你执行时的实时状态）：Core MIX1908在跑，VANDA ADAPT1644在跑，F2第六原生已进行28,201tick未终态；F1前6加F2前5总共**11个已终结学生请求，0成功**。TGT5000完整；D2 26条2031窗准入。真实服务器状态必须以你登入后只读核验为准，不把用户快照当最新。

执行顺序：
1. **不打断任何健康作业**，只读盘点Core现役systemd、cgroup、lease/STOP/world、MIX完整检查点，VANDA PBS的`1599363`、`1599364`、`1599365`、`1599367..1599379`和`1599382`、`1599385`、`1599386..1599391`，核对真实状态、deadline、依赖、已存账本和用户级finalizer，严禁重投UNKNOWN。
2. **F2第六条**若仍在运行，交给原world owner继续；若已结束，读取2ms结果、首接触、已消耗P10及最后partial，按原序把F1六条／F2六条封存并核对累计分母，不重跑前11；再沿已经登记的ADAPT0余DEV/独立TGT5K RGB16/G8 HOLD4合法轮转，而不是三套world同时启动。现场资源不合格则保留BLOCKED，绝不杀别人的GPU进程或放宽门。
3. **Core MIX**保护健康训练，完整seal后从绝对s+1接续；达到**精确MIX2000**才按原固定面板释放独立288＋48 Eval；合法有限窗口截止历史为11/21 04:21:44 SGT，安全余量不足就封存完整前缀并等新allocation。只读核验已存在的MIX2K Eval observer和公开HF守候，旧身份还在就不另起同一job。
4. **VANDA ADAPT**延用当前新PBS DAG到精准2K，不重算原ADAPT1500，不引用旧撤下JobID作为新前驱。2K封存后**一次D2选择**（合格则Go2 12新52旧＋R32 L32；否则原D1 fallback），后继训练与独立2K Eval并行。2K Eval再释放CLOCK CPU判定；只有原24到期联合0/24且所有科学和资源门满足，才使用现役条件性A/B各500链；不能因为闸门结果未出而提前训练或热改主线。保持2K→5K队列原deadline与父状态。
5. HF端点和Core跨站包输送只在完整seal及已登记的独立评价成功后执行。RGB16没通过不能派原生；TGT5K不重做训练或原Eval。Core后续native仍沿原固定人口，不把0/11解释成模型能力的终局或用旧TEST调参。
6. 在断开当前SSH以后，从**新的独立连接**回读实际PBS JobID/H/Q/R/F、Core service PID/cgroup、stop、有限站点租约、原world single-owner；仅当短链真实缺失且确认无重复效果时在当前真实权限内补交最短有界后继、失败及最终到期收尾。站点不支持或无新分配立即明确BLOCKED，不假装排队。
7. 发表`OVERNIGHT_SERVER_MOUNT_ACTUAL.md/json`到执行PR #418，写当前实际观察时间和每站已安装／待安装区别，最新训练步、F2真实终态、正式2K/D2/Eval状态、运行中/已封存分母、当前deadline、stop入口，独立读回证据。GitHub发布失败不影响健康训练继续。完成后台配置之后无需实时轮询或持续Codex会话。

保持原固定H40/50ms、P10、2ms世界积分、冻结WBC、碰撞STOP和评分；STOP后0tick；不改变动作/状态统计、原科学预算、原TRAIN/DEV/TEST、原任务人口、官方runtime/Adam/RNG。**不得把GitHub文档已发布写成已在Core/VANDA执行。**
