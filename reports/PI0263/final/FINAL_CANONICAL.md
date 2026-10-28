# PI0263 final canonical — execution COMPLETE, both endpoint qualifications FAIL

Authority: `PI0263_TRAIN_TEMPLATE_COUNTERBALANCING_GO_V1`, release `1c574b6e6e9d6a7851e64ca57e89c468a4c9eb28`, original canonical5913659202 / pointer5913663574. Evidence commit: **`539c175ff9f955a8b49bcaf7eedb29c606d538eb`**.

**`PI0263_TEMPLATE_COUNTERBALANCING_PILOT_COMPLETE — STOP_FOR_INDEPENDENT_GENERALIZATION_OR_PHYSICAL_ADMISSION`**

两臂均完成原定2000更新，全部3456逻辑行评价、独立计算路径CPU复算、完整Trainer及结果HF公开核验、固定数组视频已交付。**F2000、X2000均为有效质量FAIL；synthetic_research_candidate、selected_checkpoint、deployment_selected_checkpoint全部NONE。** PI0261完整资格FAIL与PI0262 COMPLETE/MIXED原样保留。

## 质量结论与可解释范围

| Final point | 原四数值组 | 原八语言组通过数 | 新八KIN-VAL鲁棒性组通过数 | 原完整资格 | 新完整鲁棒性 |
|---|---:|---:|---:|---|---|
| F2000 |4/4|5/8|2/8|FAIL|FAIL|
| X2000 |4/4|7/8|5/8|FAIL|FAIL|

这些通过组数只是分组摘要；协议要求每组分别通过，不能按平均值或多数表决晋级。

X的TRAIN原绑定T0/T1=60/60、60/60；other-T=60/60、58/60，明显好于F的45/60、45/60。新组合已进入X监督，只称counterbalanced training support。X仍失败于：

- 原M1 SEMANTIC/B1：**8/12**，c0=3/6、c1=5/6（门≥10/12）。
- 新KIN-VAL T1/B0：**16/20**；V0/B1：**14/20**；V1/B0：**16/20**（各组门≥17/20）。
- V0/B1 KIN-VAL c0=4/10、c1=10/10；V1/B0 c0=10/10、c1=6/10。TRAIN V1/B0更有30/30对10/30的命令差异，不能用合并分数掩盖。

两臂M3原门保留PASS，X的M3 PARAPHRASE/B1由P/F的12/12降至11/12，保留该退化。O64仍大幅FAIL（median XYZ约183–186mm，median SO3约1.09–1.10rad）。F也有共同改善，X并非所有验证组优于F。额外训练及共同采样设计不能纯归因于模板平衡；模板边际频率也改变，未单独识别固定边际下的去相关效应。

V0/V1未入本项目监督/统计拟合，但属于已研究的DEVELOPMENT VALIDATION；不是盲TEST、未见基础预训练语言或物理泛化证据。30TRAIN/10KIN-VAL家族内测量相关，不作数千独立样本的二项推断。g/b仅描述输出空间方向/偏置，不宣称唯一神经机制。

## 实际完整性与预算

两臂同固定P2000 `875abdde2cc7c9e6858fac008e53f2f4b3375255` 初始化。冻结1000源unit双访问，8000实际微批颜色/输入/标签/mask检查、7项负控、相同native epsilon/tau/free-S4实际包全通过。各自真实8 vs4/save/fresh-restore4 canary中模型、optimizer、scheduler差均0；四Euler梯度和模式恢复、原loss/精度/可训练范围通过。559冻结tensor byte-identical；471可训练tensor均变化。F640396、X640622终止exit0；640356的CRLF pre-Python失败为0计算，日志保留。

| Accounting | Actual | Hard budget |
|---|---:|---:|
| Main updates / microbatches |4000 /8000|4000 /8000|
| Canary / completed technical reserve updates |32 /0|32 /256|
| All completed updates |4032|4288|
| Raw training heads |72624|100000|
| Unique logical rows |3456|3456|
| P donor reused / new donor queries |1152 /0|1152 /0|
| New queries / inference heads |2232 /8928|2432 /9728|
| Exact F/X forward reuse |36 /36|labels分别重算|
| Invalid attempts / quality retries |0 /0|无新增科学查询|

每点680 legacy＋640 crossed−168重叠=1152 unique；两原面板索引和分母完整。首个有效输出保留。CPU由同一负责人以独立计算路径完成，**不冒称外部签字**。

两项只读收口更正都保留原件：训练loss reader改为重放实际FP32运算次序；CPU原收据因循环cursor把实际3456行误标为X1152，保留原收据/原代码后核对全部3456原件哈希及真实表达式，修正范围标签/1152 donor复用数。预测、指标、门、失败结论、随机包和计算预算均未改变；未因此重训或重查询。

## 完整模型与公开结果

HF PUBLIC repo `leoyuer/pi0246-gate-pass-campaign`，以下每点为**16文件完整Trainer state**，固定revision逐文件全文readback PASS。Corelab只物化model-only推理视图，不冒充resume。

| Point | Fixed HF revision | Full Trainer tree SHA256 |
|---|---|---|
| F1000 |768cc6b74700d6e3a849064e3c91694f645bdf0a|cdf5155478c885ab77a93e8ff3c823c032c66afb45162ed5e766d69d2fdb1194|
| F2000 |4e5f40a4f6fe516003e5ee322a8c92eb7b03a6a4|ac62469c76725ed28fd6f7eba99da159b940d89865cdc1ac2f9844d68ac6cd0b|
| X1000 |fb9b9f84b5fcb1b67cebb72aea831c68e9be2287|3f274001d714345fac898f9c5446add04716f708fd8b0984c185f84ee6031d6c|
| X2000 |a2f9f767a11e4af269377603f3da432f4931d623|4ec3b3a44ad40a057cfdcd7ad992a86b68889b4efba98be70ed109e571497694|

Prefixes: `arms/PI0263/{F|X}/checkpoint-{1000|2000}`。1000未参与质量筛选。

结果公开revision **`8c378ad358c7415b587d82758f621a965a1c1a06`**，prefix `evidence/PI0263/counterbalanced-template-fixed-20261001`；tree **`9fb131a1ba425934b0b7b898ae2f95d3dc3b0f38a7b2c995b3320db0cb81b367`**。报告、视频、manifest及509900800-byte原始tar全部匿名全文下载验证，**9379 members /503118864 source bytes逐成员核验PASS**，包括全部3456 record/tensor pairs与训练证据。

- [完整质量报告与所有逐模板/颜色/家族/噪声/方向/回归表](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/539c175ff9f955a8b49bcaf7eedb29c606d538eb/reports/PI0263/final/MODEL_QUALITY_REPORT.md)
- [最终逐要求完成性核验](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/539c175ff9f955a8b49bcaf7eedb29c606d538eb/reports/PI0263/final/FINAL_COMPLETION_AUDIT.json)
- [HF公开readback收据](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/539c175ff9f955a8b49bcaf7eedb29c606d538eb/reports/PI0263/final/PUBLIC_RECEIPT.json)
- [固定版本公开原始证据](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/8c378ad358c7415b587d82758f621a965a1c1a06/evidence/PI0263/counterbalanced-template-fixed-20261001)
- [P/F/X固定数组对照视频](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/8c378ad358c7415b587d82758f621a965a1c1a06/evidence/PI0263/counterbalanced-template-fixed-20261001/PI0263_P_F_X_fixed_arrays.mp4)

视频18个预冻结chapter，720帧/20fps/36s/1280×720，全帧解码检查和18章节代表帧可视检查；明确KINEMATIC RENDERED REFERENCE / OFFLINE PREDICTION / NO PHYSICS OR WBC水印，仅已有数组，0新camera render/模型/物理查询。

**科学范围已关闭：不启动新loss/seed/adapter/scene/IK或验证追分。selected/deployment均NONE，停在独立泛化或物理准入。** 本次physics/WBC/TEST/DEV-F/真机、新统计拟合/场景为0；没有新monitor、mail、大清理或merge。执行完成与质量失败分别落账，下一科学执行需新授权。
