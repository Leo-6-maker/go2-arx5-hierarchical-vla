# 发布前增量回读 · 2026-11-18

PR405在审阅期间从`01b4a11c80f95956742379f73404be8fbdf909e1`前进到`170560fe112e3e0a35d8c679206df3bfd20c2202`。compare确认只有1个提交、5份报告更新，运行源码/训练器没有变化。协议固定的初始00:38快照保留；本文件给出发布前更近的**已提交回执**，不是新SSH现场检查。

最新回执时间：**2026-11-18 00:44:41.143756 SGT**。

- E2：10/24已启动，9终态，1在途，14未启动；6个综合成功，1个接触STOP，2个H40拒绝。
- 完成F1/T0–T2：B0为1/3，B1为3/3，B2为2/3。F1/T3-B0在途；尚无本轮F2终态，不作正式每模型8条件结论。
- 闭合积分：`5323*25+13+9*5703=184415`；在途至少13703 tick。worker生成568=535终态对应返回+33在途返回；旧2个接口只复用，新增请求不重复计数。
- B2新内部k34的0.261575m/s²拒绝仍保留，旧端点loss没有被热改。B0碰撞后积分0。
- B1/B2各500/16000、792离线生成、模型与统计身份不变；活动native source仍为`2797d0c2c458667e241aa0d24d6d0b005f3caf1a`。
- 没有候选选择，确认/E3/DEV/TEST请求仍0。条件评价代码存在，不等于已部署。

这项增量不改变本计划的顺序或预算：现役24面板继续；数据审计可在实际余量下独立准备；新GPU/原生在健康消费者冷态边界之后。不能把B1当前3/3与B2 2/3说成B1已获胜，也不因为新报告覆盖旧首快照而删除历史失败。

来源：[最新E2逐条件回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/170560fe112e3e0a35d8c679206df3bfd20c2202/reports/WM02/sla-mobile-f2-v52-20261117/E2_PREFIX_ACTUAL.md)、[最新ACTUAL](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/170560fe112e3e0a35d8c679206df3bfd20c2202/reports/WM02/sla-mobile-f2-v52-20261117/ACTUAL.md)、[仅报告的增量compare](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/compare/01b4a11c80f95956742379f73404be8fbdf909e1...170560fe112e3e0a35d8c679206df3bfd20c2202)。
