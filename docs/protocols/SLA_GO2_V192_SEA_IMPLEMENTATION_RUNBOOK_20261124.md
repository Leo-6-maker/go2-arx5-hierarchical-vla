# V1.9.2 执行交接：SEA1B → SEA-TASK / SEA-E2 / SEA-GRIP → 一次SEA3 → Pick & Hold

本文件是 implementation runbook，**不是现场运行或科学分配授权**。先读 [正式协议](SLA_GO2_V192_SEA_SPATIAL_AND_EMBODIMENT_ADAPTATION_20261124.md) 与 [机读候选登记](SLA_GO2_V192_SEA_REGISTRY_20261124.json)；以 #497 固定 `470aa1ce40d0a5d75a9f06b394d7324c58e61881` 为证据父端点。保留 #492/#495 的旧名册和所有结果，禁止改正在运行的源码/权重/owner。

## 1. 第一批可直接准备的工作（互不阻塞）

- **SEA1B**：从 SEA0_PAIRED_ROSTER、A1历史和#497确认源HDF5/输入/实际sampler/noise、固定虚拟左臂物理状态的来源。建立32格名册；现有 source adapter/sea1a_panel.py/sea1a_score.py、reach_forward/feedback_gpu、官方real processor可复用，单独版本化mask/state干预。真实前向必须先取得独立科学分配和GPU lease；没有分配先完成CPU回放/冷验收。
- **SEA-TASK**：盘点公开真实源数据实际目录/episode、语言、后继时钟、单右臂局部能力及左手支撑；锁定候选技能及T0/T1/T2/T3，优先历史返回。源实机标定请求保持独立；不能因缺失标定阻塞其余P0。真实source-sim要先做同real PT阳性；无原生支持不得假设任务学会。
- **SEA-E2**：回读#494真实2ms/50ms消费、ACK、实际TCP与意图/安装证据，以及SEA1A和19窗历史；复用V1.9 sea_reference/world_intent_clock/compiler/standing_reach和原接收器。CPU测试连续消费、protected prefix和实际进展触发，明确已消费/ACK/selected/achieved。禁止改变原门并继承旧资格。
- **SEA-GRIP**：回读 TOOL_AND_SOURCE_CLOCK、GO2_TOOL_GEOMETRY_ACTUAL、SOURCE_TOOL_CLOCK；核实两种宽度语义、源真实工具/开度标定。并行准备接触允许对和H0具体任务的独立资格，不放宽原非接触约束。

四项P0以真实产物完成，不等待完整source twin或所有P0“PASS”才做独立合法工作；进入接触和训练需各自依赖齐备。用户要求计划发布不等于批准实际GPU或积分。

## 2. 建议代码工作包（未来实现，不声称已存在）

| 新版本工作包 | 复用点 | 验收 |
|---|---|---|
| SEA1B source factorial | SEA0 source roster, sea1a_panel/score, official processor | 32唯一格；mask为唯一视觉改变；固定virtual state；真实state token变化合法；噪声/历史逐项同源 |
| SEA-TASK skill registry/scorer | O0/A1/V4、source custody、原source codec | 原始任务与改写，源后继与hold、阶段/开闭事件，独立相似性诊断 |
| SEA-E2 consumer-aware clock | native owner、world_intent_clock、compiler、sea2_native_score | 真正消费接缝、实际source进展、TTL、极值与STOP；原门保留；新profile另核验 |
| SEA-GRIP tool manifest | sea_tool_geometry、官方源工具 | 每指/总间距、单位、link6/TCP、单调映射、未知/不可部署 |
| SEA3 external adapter | sea_feature/sea_fit构想，冻结PT source bypass、#491 cold preflight | 真实梯度仅外部参数、source bypass一致、独立分组、配对全50物理指标 |
| SEA4 native contact runner | 原冻结WBC、真实接触owner、H0、原评分 | 匹配H0/A/B、安全拒绝与物体结果分开、STOP后积分/安装0 |

新测试不得在CI自动触发GPU/仿真。统一沿用现有Journal/owner/lease/证据保存机制，不创建第二调度器或热改现役capsule。严格分离source_mode、target_deployable、oracle_only；拒绝oracle发布，测试GT/目标标签/未来帧污染、跨episode history、左右手/坐标/时钟、旧回执、TTL、重复和未知请求。CPU mock和历史readback不是新真实资格。

## 3. 运行前固定与独立科学分配

以真实代码、checkpoint/processor/normalization/tokenizer、源输入/HDF5、原WBC/leg/PD/native二进制、场景/owner spec SHA封存具体批次，核对旧#494/#496已关闭。SEA1B固定32格、任务诊断和所有其余候选按协议实际绑定；未能精确绑定标BLOCKED，不伪造输入、结果或资源。对训练先选A或B，声明数据分组、计算图与参数名/shape、loss单位、固定训练预算/验证方案和源权重不变验收；100步计入1000。GPU资源必须核查实际UUID、同用户/项目owner、显存/主存/磁盘、有效期限，已有作业不动。#497的12,671MiB和23.78s只是旧批次记录，不是新训练显存/时间保证。只在科学分配与站点许可均有效的profile内执行；不使用未授权主机、不抢占H200/其他GPU；无真机连接。

## 4. 按阶段报告和选择

报告目录新建 `reports/SLA_GO2_V192/<stage>/`（未来实际执行才生成ACTUAL）。每阶段提供 REGISTRATION、ACTUAL.md/json、输入/模型/环境/结果manifest、资源与终态；只提交安全的哈希与统计，不公开私有原始HDF5、RGB、模型或机器人资产。登记数、entered/returned/exact_reuse、screened/rejected/technical/unstarted、原始输出及完整物理意图、实际source进展、训练样本/优化更新、owner/lease/STOP后tick和install、实际运行成本分列。技术UNKNOWN先对账，不盲重试；科学失败保持。

- SEA1B结束后结合SEA-TASK：mask或状态无学习候选恢复则直接冻结确认；观测退化可恢复才首选A；空间对应可信才备选B。没有证据则不训练。
- SEA-E2无明确有效requery收益则保留已资格执行器，不因Z单次增益修改WBC。接触前须独立GRIP和对应owner/H0，不能借无接触证据。
- 仅一条SEA3主要训练臂，有限更新后按独立确认及原始物理意图判断；源改善而目标无改善单列。停止无界调参。
- SEA4先H0再同布局A/B；相同臂去重。若匹配技能实现，申请独立Pick&Place/第二技能和新对象；若无法迁移，明确R5有监督对照的触发条件，保持原57数据和8/12静态划分，不将其作为严格SEA数据。
- 新行动必须依据真实终态与单独分配，不以计划、入队、CPU或渲染报告充当科研结果。计划发布阶段新增效果全0。
