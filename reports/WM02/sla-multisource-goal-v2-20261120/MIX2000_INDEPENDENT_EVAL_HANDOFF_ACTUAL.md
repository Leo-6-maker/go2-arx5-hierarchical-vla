# MIX2k 独立评价守候已登记

Corelab 上的 `sla-observer-mix2000-eval-20261120-01.service` 已启动，用户级 systemd 独立回读为 active，PID **781205**。源为 `943d070f91e9c7c28cc62c281e767af8268e57d4`，固定执行卡 SHA 为 `b0632a5ba2fed52d9ee5d9524a7f651941641741df18f8b1801dfe6eebca14f0`。 [机读实绩](MIX2000_INDEPENDENT_EVAL_HANDOFF_ACTUAL.json)

守候先等待现役 MIX 段退出并核对封存。若只保存早期前缀，读取既有 ADAPT0 第二块后的 MIX 轮转真实接续，再等待该接续终态。只有原训练代码、精确 2,000 步、退出 0、完整 140 数组和连续账本均成立，才检查 GPU1 的真实用户占用、锁、显存、主存与磁盘。原固定面板总计 **288 条普通预测、48 条独立恢复**，GPU 评价运行上限 90 分钟。CPU 守候不占 GPU，评价成绩不进入训练的释放条件。

守候依赖现役 `wm02-sla-scale20k-mix-core-after1000-20261120-02.service`，以及现有 `sla-observer-mix-after-adapt0-block2-20261120-01.service` 和它登记的训练后继。当前 GPU 评价 `wm02-sla-scale20k-mix2000-core-offline-20261120-01.service` 的 PID 仍为 0。站点截止 **11 月 21 日 04:21:44 SGT**，最晚派发 **03:51:44 SGT**。原计划的 14 天终点未变。守候的日志是 `watch.log`，评价日志是 `supervisor.log`，各自根目录的 `STOP.json` 可停止该角色。已将两角色纳入原 systemd 到期与 STOP 收尾，独立回读定时器和路径触发器均 active。

截至 **18:38 SGT**，MIX 已记账 **1,210 步**，目标／RoboDojo／LIBERO 访问 **38,720／58,080／58,080**。VANDA ADAPT `1594212` 为 R，已记账 **1,291 步**，访问 **82,624／41,312／41,312**。较 18:32 实绩增加 **19 次优化更新、2,432 次样本访问、76 条训练内 H40**。新增守候的训练更新、访问、模型生成、原生请求和积分均为 **0**。TGT5k 仍等 GPU RGB16，ADAPT2k／5k 尚未封存。若当前站点窗只产生早期 MIX 前缀，则先保留该前缀并用下一段已核准资源补齐，之后才能释放精确 2k 评价。
