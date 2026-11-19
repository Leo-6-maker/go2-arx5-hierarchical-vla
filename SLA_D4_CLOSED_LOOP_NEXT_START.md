# SLA D4 后继：闭环抓持、物理时序监督与多源联合后训练

2026-11-19（新加坡时间）。本入口发布计划，当前新增训练、模型生成、原生请求和积分均为 0。

**下一项应交付 D4 首个布局的 12 条真实 MuJoCo 对照结果。** 先补齐 D4 专用原生路由，沿用已封存 A/B 和 48 条人口。夹爪 W/T 两个训练臂与 RoboDojo 联合训练支线可以在独立授权资源上推进，不要求原 Teacher 全格通过。

- [固定事实、问题归因与官方配置边界](reports/WM02/sla-d4-next-plan-20261119/ANALYSIS.md)
- [完整实验计划](docs/protocols/SLA_D4_CLOSED_LOOP_AND_GRASP_20261119.md)
- [机读规格与所有条件预算](docs/protocols/SLA_D4_CLOSED_LOOP_AND_GRASP_20261119.json)
- [Codex 自主执行启动 prompt](docs/handover/2026-11-19/START_SLA_D4_CLOSED_LOOP_NEXT.md)
- [事实清单](reports/WM02/sla-d4-next-plan-20261119/AUDIT_FACTS.json)
- [文档核验记录](reports/WM02/sla-d4-next-plan-20261119/VALIDATION.md)

执行栈从 PR #415 固定提交 6d0ff9f8d8d33c47e5dde308203c57bb5b2adc64 独立接续；本计划 PR 以 main 为文档集成基线。不能假定 main 已包含现役 runtime，不能把本次文档发布当作部署或站点授权。

首个有界工作单元：Core 部署路径 RGB32；F1/F2 两条接口共 11,906 tick；D4 原次序第一波 12 科学请求，上限 548,436 tick、1,920 次模型生成。所有任务必须现场绑定真实 owner、可用资源和有限截止。旧 11 月 19 日凌晨的 allocation 已不能作为新任务的依据。

全条件上界为 154 原生请求、6,398,762 tick、新训练 2,000 更新、64,000 目标访问与 4,000 源访问；包含尚未准入的 D5 确认与 J 支线，不是要求一次全部启动。WBC、2ms/H40/P10/50ms、TTL、接触 STOP、原评分及旧 TEST 保持。禁止真机通信；暂停时钟仿真成功不构成连续实时或真机资格。
