# C50/D25：真实CPU实现、验证及剩余原生接线

2026-10-25；状态 **CPU_COLD_PREPARED / NATIVE_NOT_CONNECTED / TRAINING_NOT_RUN**。P0已闭合为[state表示可用于混合研究](STATE_RANGE_AND_CONSUMPTION_AUDIT.md)，此后不再扩大诊断。代码复用成熟数据reader、bank/parent/window规则、MP1/E1 Trainer及LR函数；没有另建调度器/账本，没有修改旧4096源码或validator。

## 实际新增实现

[`scripts/pi0294/pv1/mix1024.py`](../../../scripts/pi0294/pv1/mix1024.py)使用新的schema/route/C50-D25身份，旧`validate_primary`仍拒绝这些recipe。共同CORR父、原R0、原数值/optimizer超参、LEGACY_MATCHED、两任务和C1位置保持；只改变ordinary lane的替换分配。底层全银行reader仍复用原B银行读取角色；它不是新训练的请求或run身份，新sampler状态明确为V5_MIX1024/C50或D25。

- recipe与Trainer参数为各1024更新、batch32、同64 warmup和1024 horizon；新C50不是旧B@1024复现。
- D的新库位置为C的固定子集；相同lane使用成熟的absolute-cursor/lane命名子流选择同parent/anchor。
- C1逐cursor一致，不强行平衡介入/接续；本次真实支持上实际为4069/4123。
- LR复用成熟E1函数，全部1025值一致；resume核对完整成员SHA、recipe、数据、R0及step×32 cursor，不改seed或从prefetch计进度。
- 1024末边界请求停止并保存完整状态；版本化Trainer保存适配在中断step256也保存optimizer/scheduler/RNG，避免继承旧small-run的256 inference-only分支。

## 真实CPU证据及限制

[准备回执](../../../reports/PI0294/design/parallel-science-20261022/V5_C50_D25_REAL_SUPPORT_CPU_PREPARATION_20261025.json)使用原126旧、139新和8个C1导出的真实manifest/合法anchor，构建实际新sampler序列，检查全部32768位置、1024个batch。C50与同一CPU环境下原B算法的1024采样前缀一致；不是旧GPU消费重放，也不包括旧4096 LR。共享lane相同的26624位置逐项一致，D全部新位置属于C；任务与C1逐项匹配，原TRAIN-only guard有效。完整序列私有封存，SHA和source/support hash、五lane总量、全部LR值公开。[CPU序列复算程序](../../../reports/PI0294/design/parallel-science-20261022/V5_REPRODUCE_MATCHED_STREAM_20261025.py)从既有私有source-materials包生成同一完整序列，不读RGB、raw或权重；公开程序已实际执行并匹配原序列SHA。

C50计数C1/旧T1/新T1/旧T2/新T2=8192/4096/4096/8192/8192；D25=8192/6144/2048/12288/4096。resume后缀检查step1/256/511/1023，连续cursor无重复。原算法未为C1新1024前缀强凑4096/4096。

[CPU回归](../../../scripts/pi0294/test_v5_mix1024.py)加原recipe/sampler/dataset共25项通过：完整序列配额/耦合、LR、末步、pending/unknown拒绝、恢复身份损坏拒绝、旧schema保持等。256全保存用临时框架/文件fixture测试，明确不是实际CUDA、真实optimizer恢复或15成员native审计。真实支持的初版CPU检查脚本误在每个cursor重建sampler，已仅停止该本地CPU检查并改为每段构建一次；未影响服务器或任何科学作业。pytest旧配置有一条pythonpath提示，已显式提供模块路径；不升级环境。

## 下一最小阻断

没有可以直接派发的C/D原生owner：新request schema、真实authority/时钟、标准入口、现有update-once会计和平台资格仍需在有效后继范围内作冷接线。新代码未注册/提交/部署；成熟训练factory不等于完整资源准入。必须一次覆盖两端点及48个配套Eval和收尾，不能只付训练或借旧A/B CLOSED lease。原停新02:29:56、停训14:29:56 SGT已过，本审阅没有延长。当前只报告精确缺口，不预订12周上限或自行增加训练预算。

GEN1 V2增量按原8,830条历史索引检查：原36设计不改，加12预览variant时3个原FAR/V0完全相同，唯一设计45；无跨split连接，DEV/TEST原hash保持。见[增量回执](../../../reports/PI0294/design/parallel-science-20261022/V5_GEN1_V2_HISTORY_DELTA_CPU_ACTUAL_20261025.json)。24张真实图仍NOT_RUN，缺独立有效构建/渲染范围；COMMON_HOME未合格，不影响本C/D。不执行Teacher pilot或向C/D添加GEN数据。

下一真实行为交付仍严格是单对C50/D25各1024＋各24原DEV，原筛选只是开发投入判据。历史A7/12、4/12单列工程参照，D相对更好不自动叫恢复或泛化。全部原有效失败、费用、模型和旧CORR/R0 core64身份保持，完整goal未完成。
