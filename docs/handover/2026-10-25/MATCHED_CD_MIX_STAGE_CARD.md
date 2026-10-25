# 匹配 C/D 单因素阶段卡：50% 对 25% 新库替换

**状态：DRAFT / NOT_NATIVE_AUTHORIZATION / NOT_RUN。** 本卡是V5两份核验成果收敛后的唯一候选，不是新训练或时钟授权。[state范围与消费核验](STATE_RANGE_AND_CONSUMPTION_AUDIT.md)已得到REPRESENTATION_VALID_FOR_MIX_STUDY；保留R0并结束P0，不自动扩大诊断。仍须后继有效范围和原生冷接线。

| 项目 | 预先限定 |
|---|---|
| 唯一C/D差别 | C普通专家lane新库50%；D25%，不扫比例 |
| 共同父权重 | 原CORR2048完整manifest `94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`；不从较优A4096继续 |
| 共同初始化 | 各自fresh、完全相同optimizer/scheduler；同train/data seed20261019，原flow loss、VLM冻结和可训练模块不变 |
| 有限训练候选量 | 各1024新增更新，effective batch32=32768访问/臂；仅一个终点。此量是待验证有限方案，不称已足够或最优，不自动4096 |
| 共同数值 | 原BF16/TF32训练、microbatch2×accum16、workers0；LR1e−5→1e−6同cosine，64 warmup，scheduler horizon1024。原optimizer超参从同一sealed recipe逐字绑定，不新增调参 |
| 数据与任务 | 同126旧+139新+C1八导出固定manifest，任务1:1；C1占全部25%，沿原算法逐cursor的episode/anchor及实际介入/接续直方图完全匹配，不额外强制各半，原parent/window规则不变 |
| 执行与测量 | 双方LEGACY_MATCHED、原双RGB/物理/Teacher标签/R0/EEF10/H40/Prefix8/夹爪/参考门/S1/评分相同；同推理精度、4steps及N294101，不混COMMON_HOME或GEN数据 |
| 最低闭环交付 | C、D各完整原首DEV面板24条=48；12几何×T1/T2，实际raw/完整H40/安装ACK/原评分独立审计，全部有效失败入分母，不挑中间checkpoint |
| 主要判读 | D−C的T2配对差，T1保持；报告12几何block联合重采样2000次(seed20261024)区间、7类式逐条件机制、S1/参考/物体失败和实际安全故障。已打开DEV只作开发研究 |
| 有限继续判据 | T2至少净增2/12、T1不减少、工程缺失0且native实际安全故障0，才考虑单独确认价值；这是后继研究筛选草案，不改原18/24发布门，不承诺统计显著 |
| 停止条件 | NaN/身份时钟/实际安全/unknown/全资源或截止不满足即原owner STOP并真实结算；任一臂缺完整端点或最低48评估资源则不启动配对。D未过有限判据则不自动加步数、扩数据、第二noise或winner-only重训 |

以下是实际CPU sampler准备序列的计数，尚无训练消费：C的C1/T1旧/T1新/T2旧/T2新=8192/4096/4096/8192/8192；D=8192/6144/2048/12288/4096。两臂总32768、T1/T2各16384、C1各8192。两臂C1逐cursor复用相同episode/anchor；普通专家D新库位置是C预定子集，共享lane使用同parent/anchor，记录实际forward消费。CPU真实支持准备中的C1为4069介入/4123接续，两臂完全一致；不是未来已发生的训练。50%/25%的分母是普通专家lane，占整臂新库37.5%/18.75%。25%不是去重、不是重采样事件、不是新R1。新的C50使用1024步LR时程，不等于旧B4096或旧B@1024，不能以旧checkpoint替代C。

**已实现的CPU冷接缝：** `pv1/mix1024.py`提供独立C50/D25 recipe、复用成熟bank选择的精确采样、1024步LR、恢复元数据/停止边界及成熟Trainer的全状态保存适配；旧4096白名单未改。真实冻结支持的全部32768位置/1024batch、1025个LR值、多个恢复后缀通过核验。中断于256也要求完整optimizer/RNG/scheduler，不能继承旧small-run的inference-only保存。CPU fixture不是原生恢复或模型资格。

**尚未连接：** 后继有效authority/标准manifest、当前平台owner/实际update-once账本及完整训练＋配套48评估的fresh资源准入。没有新dispatcher或预算体系，没有注册/提交。详见[代码与精确阻断](V5_CD_CPU_IMPLEMENTATION_AND_REMAINING_NATIVE_BINDING.md)。

只核本阶段两训练端点及配套48评估和收尾。参考旧A末2048更新约12345 GPU秒、完整A/B Eval总29183 GPU分配秒(约8.11h)，仅供初算，不是新卡吞吐保证或资源许可；须测当前站点吞吐、计资格/控制/模型IO/CPU/live+shadow/输出/失败上界，并保证两臂及最低评估同时可支付。原停新/停训已过，须有明确后继有效时限；不动旧预算账本、matrix或最终保护，不预订12周工作。

若D在这个有限训练量下无改善，反驳的是“本阶段25%比50%更有效”的操作性预测，不否定所有覆盖/比例/训练量。若改善，单seed且已打开DEV仍不能证明泛化；新测试及双方法seed复核另据证据决定。原A/B/CORR全部保留，旧CORR2048/R0目标core64不替换，完整长程goal未完成。

结果表并列历史A：T1 7/12、T2 4/12，仅工程参照，不重跑、不当新1024控制。D优于C但低于A时，只能说比例相对更好而未恢复已有水平；不据有限DEV差值认定遗忘或泛化。
