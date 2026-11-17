# Codex 启动 prompt：SLA 移动底座后训练 V5.2

将下面完整内容交给具有项目服务器访问权限的 Codex 执行。计划发布本身没有启动作业。启动时先固定所读取的计划 commit SHA；若实际执行分支已前进，核对新事实，保留并接续健康作业，不回退源码。

```text
继续 Leo-6-maker/go2-arx5-hierarchical-vla 的 SLA 移动底座后训练主线，执行 V5.2 有界计划。我要实际训练、真实MuJoCo闭环及明确结论，不能只交付工程清单或离线合法率。

先读取本计划分支中的 SLA_MOBILE_V52_START.md、docs/protocols/SLA_MOBILE_F2_POSTTRAIN_V5_2_20261117.md 和同名JSON，固定计划commit。再读取PR #403固定head dbf222b41cb666bff4dd5412b864d023aec69399的FINAL_RESULT.md/json、执行版本AGENTS.md及最新合法后继。文档main不含完整执行栈；从复核后的执行版本建立独立worktree/普通PR，不盲合、不force、不热改在途胶囊。

事实基线：冻结SHIFT7已有3次真实移动抓持，学生3/8尝试、3/7有效，F1有效3/3、F2 0/4；三次原H40首尾加速度拒绝及一次腕部碰方块。pilot_GO=false。当前是SIM_PAUSED_NATIVE，旧350ms实时失败仍保留。不要重做已结束的缓存×精度四路线，也不要把SHIFT7理解成物理延迟350ms。

我的执行范围是该计划中有界的SIM_ONLY_PROJECT_AUTHORIZED研究及其必要应用层修复。复用V5.1负责人直接授权机制，不再要求第三方READMISSION，也不为普通小修、每个episode反复询问GO。先现场核对真实owner、旧服务终态、共享GPU容量、冻结身份与有效许可；禁止真机。旧allocation截止记录为2026-11-19 02:51:23.855488 SGT，不能自行续期，实际更早期限优先。

按依赖推进：

1. D0：读取已存320输入/319动作及8匹配专家，零新生成/积分量化H40端点vs内部余量和接触前5秒几何，核对C20全部1376标签。不同状态专家不能当作失败学生状态的纠正标签；无法完整恢复WBC历史时不冒称反事实。
2. E1：从相同真实SHIFT7包+匹配fresh Adam，B1普通FM500与B2仅加endpoint hinge500，各16000访问。原C20、stats、28active数组、SHIFT7 codec、micro4/B32、noise/time/dropout/LR、cached_actual/10步不变。clean=x_tau-tau*field，物理端点margin .20、scale .25、lambda .1、tau<=.5；执行门仍.25。用新窄recipe处理原训练器硬加载C20 parent1000的问题，不能改旧manifest绕guard。
3. E2：按已封存开发人口让B0/B1/B2各做同样8条件，共24次真实闭环。按身份/恢复/codec/原保护准入，不因B0/B1仍有已知科学失败删对照。报告逐条件结果和B2对B1、B1对B0差异；按计划选唯一合格候选，再做8次确认pilot。使用未执行过的预注册条件身份，不回填PR403。
4. 只有真实结果支持接触/学生状态覆盖缺口时才进入E3：最多12次Teacher纠正episode，F2八/F1四，前两个F2包含在12内。P的开发选择不要求先满足部署门。接触前、夹爪仍开、合法P10边界handoff，保留实际reference/width时钟/WBC状态、mask、失败及部分tick；不能采到成功为止。数据门过后，C0旧数据/C1一半旧一半新各500、同父同loss，16次配对screen及8次确认。新reader保持C20统计fit身份，不伪造五阶段、tick或1376窗；随机银行按共同绝对访问槽绑定。
5. 新训练/模型选择前冻结按scene/layout与初态簇隔离的DEV32/TEST40。候选过确认门后，DEV至少24/32且每族12/16，再一次TEST至少32/40且每族16/20，均0hard STOP/技术失败。换seed不是新环境；TEST不参与选模型/修数据，未过DEV不开TEST。

保持完整H40、50ms/P10、2ms、TTL、WBC/腿/MPC/solver/臂PD、碰撞/STOP和原评分；不裁剪、平滑或删后缀，不新增BASE/阶段oracle、RL或复杂任务。碰撞立即终止该episode并计失败，可继续其它预登记仿真条件；STOP后继续积分、隔离/冻结/owner异常则停受影响campaign。技术失败核实效果后仅续接未启动后缀，科学失败不换seed重投。

最多新增2000更新/64000优化访问；全条件分支最多140科学/数据episode+2接口，5010326tick，原生高层生成保守16800、离线1696。上限不要求耗尽；每服务最多12h、整计划最多24h且受实际许可更早截止约束。期限前保存完整状态并关闭自有服务。复用现有Journal/owner/调度，不新建多层审批或重复账本。

先交付D0、两个真实500端点、24次配对闭环及明确选择/否定，再按条件接续。每阶段推送ACTUAL.md/json、逐条件分母、源码/模型/统计身份、真实成本、首失败原因和下一动作。最终给普通RGB可加载的冻结候选与原生测试结果；未达到门就如实报告，不把loss下降、软件检查或发布PR算policy成功。
```
