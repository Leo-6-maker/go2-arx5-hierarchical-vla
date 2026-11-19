# 最新接续：SCALE20K连续优化与Go2+ARX5适配

2026-11-19（新加坡时间）。本页最新依据为17:35发布的PR418固定55d0111c与PR417固定05dfe6d4；现场状态仍需执行者读取。SCALE已完成独立step-zero恢复和7次真实优化更新，访问224目标+336RoboDojo+336LIBERO，训练H40共28次；此前4次资格生成单列。公共源1309条/36任务，目标23条/1768窗。当前任务是持续推进20k并封存非零完整端点，补齐新checkpoint评价/Provider出口，同步扩充目标语义与接触覆盖。

- [最新官方配置、D5证据和本体适配计划](docs/protocols/SLA_SCALE20K_OFFICIAL_REVIEW_AND_TARGET_ADAPTATION_20261119.md)
- [Codex最新接续prompt](docs/handover/2026-11-19/START_SLA_SCALE20K_NEXT_ADAPTATION.md)
- [现役SCALE固定研究合同](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/926226914a0c08ee1ebb143e0edea1434728107b/docs/protocols/SLA_SCALE20K_DIRECTION_20261119.md)

现役PR418仍绑定92622691；不热改其数据、loss、学习率、参数范围与预算。D5-W/T各500及Core RGB32归一化输出/原门复验已完成，GRASP24执行器已启动、发布科学分母仍0/24；J0已保存500端点，J1源优化仍0。旧D4按用户指令暂停28/48，余20不自动恢复。本次计划增补没有启动新训练、生成或原生请求。

## 历史计划入口

以下保留此前两版发布时的文字、分母和计划，不能将其中“0/48”“先完成D4首12”或旧优先级当作当前调度指令；当前接续以上方链接和现役固定合同为准。

---

# 最新优先级：SCALE20K多源后训练

2026-11-19用户追加决策：规模化联合后训练作为下一主线，现役D4/D5/J500步按原固定身份完成，作为并行诊断；不再要求小模型先达到高抓取成功率才允许大训练。

- [SCALE20K主线、官方规模依据与数据/参数计划](docs/protocols/SLA_SCALE20K_DIRECTION_20261119.md)
- [规模机读设计](docs/protocols/SLA_SCALE20K_DIRECTION_20261119.json)
- [SCALE20K Codex启动prompt](docs/handover/2026-11-19/START_SLA_SCALE20K.md)

本增补仅改变未来研究优先级；PR417正在执行的固定计划cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb不被热改。当前规模计划新训练/下载/生成/原生执行为0。

---

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

