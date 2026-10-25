# 匹配 C/D 单因素阶段卡：50% 对 25% 新库替换

**状态：DRAFT / NOT_NATIVE_AUTHORIZATION / NOT_RUN。** 本卡是V5两份核验成果收敛后的唯一候选，不是新训练或时钟授权。先完成[后验分析](AB_POSTMORTEM_AND_NEXT_HYPOTHESIS.md)指出的state超界最小核验，若发现系统标签/表示缺陷则撤销本卡，择一窄修复，不同时改比例。

| 项目 | 预先限定 |
|---|---|
| 唯一C/D差别 | C普通专家lane新库50%；D25%，不扫比例 |
| 共同父权重 | 原CORR2048完整manifest `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`；不从较优A4096继续 |
| 共同初始化 | 各自fresh、完全相同optimizer/scheduler；同train/data seed20261019，原flow loss、VLM冻结和可训练模块不变 |
| 有限训练候选量 | 各1024新增更新，effective batch32=32768访问/臂；仅一个终点。此量是待验证有限方案，不称已足够或最优，不自动4096 |
| 共同数值 | 原BF16/TF32训练、microbatch2×accum16、workers0；LR1e−5→1e−6同cosine，64 warmup，scheduler horizon1024。原optimizer超参从同一sealed recipe逐字绑定，不新增调参 |
| 数据与任务 | 同126旧+139新+C1八导出固定manifest，任务1:1；C1占全部25%，原介入起点/专家接续各半，原parent/window选择规则不变 |
| 执行与测量 | 双方LEGACY_MATCHED、原双RGB/物理/Teacher标签/R0/EEF10/H40/Prefix8/夹爪/参考门/S1/评分相同；同推理精度、4steps及N294101，不混COMMON_HOME或GEN数据 |
| 最低闭环交付 | C、D各完整原首DEV面板24条=48；12几何×T1/T2，实际raw/完整H40/安装ACK/原评分独立审计，全部有效失败入分母，不挑中间checkpoint |
| 主要判读 | D−C的T2配对差，T1保持；报告12几何block联合重采样2000次(seed20261024)区间、7类式逐条件机制、S1/参考/物体失败和实际安全故障。已打开DEV只作开发研究 |
| 有限继续判据 | T2至少净增2/12、T1不减少、工程缺失0且native实际安全故障0，才考虑单独确认价值；这是后继研究筛选草案，不改原18/24发布门，不承诺统计显著 |
| 停止条件 | NaN/身份时钟/实际安全/unknown/全资源或截止不满足即原owner STOP并真实结算；任一臂缺完整端点或最低48评估资源则不启动配对。D未过有限判据则不自动加步数、扩数据、第二noise或winner-only重训 |

拟定访问数是新配方算术，尚无真实采样：C的C1/T1旧/T1新/T2旧/T2新=8192/4096/4096/8192/8192；D=8192/6144/2048/12288/4096。两臂总32768、T1/T2各16384、C1各8192。两臂C1应复用完全相同实际序列；ordinary lane仅bank选择阈值改变，记录真实parent/anchor和forward消费。25%不是去重、不是重采样事件、不是新R1。

**最小工程阻断：** 当前成熟PV1 recipe/Trainer严格绑定4096和旧arm枚举，不能直接把本卡传进去或伪装旧A/B。后继有效范围内，复用原trainer/sampler/export/eval，仅作版本化冷态recipe/arm/1024停止接缝并CPU证明匹配；封存实际共同optimizer超参、数据及源码hash，再fresh资源准入。当前没有实现/部署/注册C/D，也未热改旧源码。

只核本阶段两训练端点及配套48评估和收尾。参考旧A末2048更新约12345 GPU秒、完整A/B Eval总29183 GPU分配秒(约8.11h)，仅供初算，不是新卡吞吐保证或资源许可；须测当前站点吞吐、计资格/控制/模型IO/CPU/live+shadow/输出/失败上界，并保证两臂及最低评估同时可支付。原停新/停训已过，须有明确后继有效时限；不动旧预算账本、matrix或最终保护，不预订12周工作。

若D在这个有限训练量下无改善，反驳的是“本阶段25%比50%更有效”的操作性预测，不否定所有覆盖/比例/训练量。若改善，单seed且已打开DEV仍不能证明泛化；新测试及双方法seed复核另据证据决定。原A/B/CORR全部保留，旧CORR2048/R0目标core64不替换，完整长程goal未完成。
