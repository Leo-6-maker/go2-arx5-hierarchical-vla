# PILOT2 实际进展：先修采样，再增加物理父

2026-11-06，Asia/Singapore。执行计划固定于 `6d82fa23c892f167001abe743dd5eadfca6fd777`（PR #363，未合并）。本报告是中期封存，**PILOT2 尚未完成**。

已得到的主要结论是：原顺序采样确实妨碍了这组小 Actor 的教师拟合；无需增大网络，打乱采样即可在三个 seed 上显著改善。首批四个新物理 TRAIN 父也已完成真实采集、并驱动新一轮三种子训练。尚未进行新策略原生收益评价，不能把教师一致率当作控制成功。

## A0/A1：匹配数据、采样顺序与 seed

保持原 ARX、任务、两层 128 网络、batch128、AdamW lr0.001、纯软教师 CE 和 TRAIN-only normalizer。12 模型设计为 24/53 TRAIN 历史 × 原 grouped cyclic/global shuffle × seed36001/36002/36003。每历史的256目标是合成教师标签，不是256个物理父。DEV仍为来自两个物理父的8条历史，没有读取旧 TEST。

| 8k 主更新端点 | TRAIN 教师动作一致率，三个 seed | DEV 教师动作一致率，三个 seed |
| --- | --- | --- |
| 24 / grouped cyclic | 83.69%, 64.52%, 87.08% | 91.02%, 62.06%, 87.35% |
| 24 / global shuffle | 99.07%, 98.75%, 99.35% | 95.65%, 96.24%, 97.36% |
| 53 / grouped cyclic | 92.55%, 89.45%, 73.38% | 94.63%, 95.75%, 80.22% |
| 53 / global shuffle | 98.71%, 97.63%, 98.81% | 97.66%, 97.17%, 96.39% |

![三个 seed 的实际学习曲线](A1_MATCHED_CURVES.png)

曲线为全部三个 seed 的中位数和 min–max。右列是 **ARX 模型评分的 regret**，不是本体物理误差。自然 CE 含教师熵，不要求接近零。2k/8k 是等更新主对照；20/40完整标签 epoch 为单列辅助对照：24窗口在960/1920更新，53窗口在2120/4240更新。二者不能混称等 compute。

完整 sampler 保存 permutation、PCG64 generator、epoch、offset、draws，恢复探针同时核对模型、优化器、normalizer、Python/NumPy/Torch/CUDA RNG 和 sampler。12个模型均恢复误差0；原24/53 cyclic seed36001 的2k端点与既有 checkpoint 的模型和优化器完全一致。两个数据版本各自六个模型的 normalizer 数组哈希分别完全相同；不同数据版本仍使用各自 TRAIN 拟合，未声称只隔离了覆盖的单一因果因素。

按事前冻结的 TRAIN 高 margin 错误、2k→8k KL下降和 DEV 非劣化规则，arm0/1/5/8获得有限20k后继。每个从8k完整状态新增12k更新，其余八个未为凑步数继续训练。四个后继恢复误差均0；端点 TRAIN/DEV一致率依次为95.25/95.46%、92.60/96.83%、94.83/97.27%、97.17/97.90%。这些结果支持继续使用 shuffle 基线，不支持无限增加 cyclic 更新。

实际累计：12个不同模型，96,000主更新；4个延续新增48,000；合计144,000 Actor 主更新＋16个单次恢复探针。WM更新0，原生积分0。各 checkpoint、数据和逐父/目标/类别指标见 [匹配曲线数据](A1_MATCHED_CURVES.json) 与 [条件延续及normalizer审计](A1_CONDITIONAL_CONTINUATIONS.json)。

## B0：新物理父和真实动作覆盖

采集前固定18个新父及分区（12 TRAIN、6 DEV），仅改变允许的初始 root/object XY；四个原动作和原时钟不变。面板 SHA256：`8e502a719431b43db807ddec0994fea32080a0b91f41e0123b15011a8731b944`。已完成首批 P2T01–P2T04：

- 4次单场景资格＋16次双槽动作批次（32个动作 episode），36/36原生PASS，147,708个2ms tick，worker退出0且lease CLOSED。
- 每父 HOLD/+X/−X/+Y各两个独立重启，动作顺序和repeat/slot在采集前随机化；没有按教师喜好挑动作。
- 新增32个可用 TRAIN 窗口、768条晚决策50ms转移；原始动作采集记录2048个atom。完整prefix及raw记录不冒充768个独立初态。
- 合并后85 TRAIN/8 DEV，既有DEV保留，TEST未读。Replay SHA256：`89aa357c5bf90746ac4869fe720bdcfc21375061e3c0affa17193319df11ffa0`。
- 三个fresh seed使用同一shuffle配方各8k更新，24,000新增主更新＋3恢复探针，全部退出0/CLOSED、完整恢复误差0。TRAIN一致率98.13/98.57/98.52%，DEV97.02/97.90/98.14%。尚未选择唯一最佳 seed，也未执行新原生策略评价。

| 动作 | 8个实际episode平均TCP响应 XY，mm | 同动作独立重启平均/最大差，mm |
| --- | --- | --- |
| HOLD | (0.4566, −0.2401) | 0.0544 / 0.1356 |
| +X 1mm | (1.0954, −0.2508) | 0.0625 / 0.1700 |
| −X 1mm | (−0.1408, −0.2862) | 0.0915 / 0.2089 |
| +Y 1mm | (0.5002, 0.2026) | 0.0410 / 0.0761 |

这是独立重启，**没有完整 MPC/WBC 中点恢复，不能称 exact fork 或精确 oracle regret**。HOLD自然运动及最大0.209mm重启差异说明，后续B1必须保留重复、按父分析并预登记实用改善阈值。采集阶段目标只是固定合同字段，任务成功率不作为策略收益证据。逐episode、pair、checkpoint与模型身份见 [首批实际数据](B0_FIRST_BATCH_ACTUAL.json)。

## C0/C1：真实接触与新训练入口

SC07B同一原生进程中，三个不同的有限新任务均已原生PASS（各6703 tick/168 atom，退出0/CLOSED）：KEEP64持续3.6s受支撑夹持；RELEASE56先保持1.2s、末端连续释放2.236s；OPEN72始终打开，夹持目标失败，保留为有效负例。累计20,109 tick/504 raw atom，均为脚本行为，不能算Policy学习或完整lift/carry/place。

随后固定了8 TRAIN＋4 DEV新支持布局（object/support XY各在原布局±0.5mm内同时平移）。每个布局检查229个完整初始化轨迹离线状态，12/12通过静态筛查、积分0；该结果不是动态PASS。新布局先单环境原生检查，当前有限接触试批正在执行，终态另封存。

已实现 `build_pilot2_contact` 与 `train_pilot2_contact_wm`：真实历史条件直接预测1/4/8/16/32/64个50ms步的TCP/物体位移及原生接触/双侧grasp/支撑/物理打开事件。未来输入仅由预登记程序生成，过去信息在action-conditioned和no-future-program消融中相同；实际未执行或跨安全回退的预测后缀被mask。按自然、事件转换、稳定区间分别报告Brier/precision/recall和位移误差，不将部分状态头当自由rollout模拟器。完整模型/优化器/RNG/sampler恢复和数据扩大时明确冻结旧TRAIN normalizer的接口已有软件测试；**该新接触WM尚未在真实数据上训练**，非恒定Policy原生闭环亦未完成。

## 源码、命令与验证

- 采样/诊断/矩阵：`8219ef1ba32f5a3408df282462640e5a19ac475a`。
- 可变进度写入修复：`b75cac1b393bc53d3ff9cb53034500de80a123ff`；模型trainer字节未改。
- 支撑接触程序及20k后继：`c01ad06c82a07e8515eb049363c3e71cfdfeed5e`。
- B0真实随机动作、分区安全Replay与85窗口训练：`1506dc223ba8b5851c18c442d4eb5b648bba2b14`。
- 接触多视野builder/trainer：`49ba05dde242ec5dfcfc43afe9b271682fe603c4`。

实际模块包括 `scripts.wm02.train_pilot2_matrix`、`scripts.wm02.train_pilot1_actor`、`scripts.wm02.pilot1_native_worker`、`scripts.wm02.pilot1_parallel_worker`、`scripts.wm02.build_pilot2_coverage`；使用登记的不可变spec与实际Python。完整实际argv/退出码在各私有worker/build回执中保留，不把宿主机绑定信息或私有数据发布进Git。

本地验证命令：`PYTHONPATH=src <model-native-env>/bin/python -m unittest discover -s tests/world_model -q`，218 tests通过，退出0。新增多视野4项测试在`-O`下也通过，含真实训练函数的软件夹具恢复；这些夹具不计科研模型更新。GitHub CI尚未得到执行回执。

## 失败、成本和接续

第一矩阵worker在完成一个有效8k模型后因对可变进度调用exclusive writer而退出1；旧失败和完整模型保留。修复后fresh worker核对结果/spec/checkpoint/旧lease CLOSED及trainer字节，直接复用完整arm，未重训。初次B0传输曾遇SSH握手reset；独立只读核对无源目录/无登记/无dispatch后，才经原入口用已核验不可变父源码＋Git delta传输；没有修改网络或重放不明写入。

A1两矩阵worker合计144.650s wall/134.686s子CPU；四个20k延续worker合计78.004s wall/70.747s子CPU。16个训练阶段CUDA事件累计187.920s，峰值Torch reserved556MiB；该事件计时不等于全进程GPU成本。B0采集coordinator实际892.866s wall（含14.199s Replay build），20个独立qualification/batch worker合计55.374s自身CPU＋2416.569s子CPU，批次子树只计一次；不能用coordinator自己的2.363s CPU冒充全成本。85窗口三种子worker合计43.668s wall/41.350s子CPU。所有大数据、模型和日志均写入已验收的非根本地数据盘。

累计已封存PILOT2学习为168,000 Actor主更新＋19探针、真实新WM更新0；这是多模型累计。尚缺余下B0父、B1带重复DEV及冻结新TEST、C0布局动态验收和C1真实学习闭环。下一步实际入口为新接触试批的原worker及builder，封存后按原owner登记有限神经任务；其它READY批次继续接力。冻结WBC、安全、2ms dt和唯一owner均保持；未构造GPU物理多世界、未使用H200或分布式。
