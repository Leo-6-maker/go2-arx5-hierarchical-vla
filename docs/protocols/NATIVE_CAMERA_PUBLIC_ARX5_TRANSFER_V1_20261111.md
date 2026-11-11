# 原生机身＋腕部输入与公开 ARX5 迁移 · 首轮执行协议 V1

**日期：2026-11-11。协议 ID：NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1。**

状态：**已完成设计审查，待实现与实际准入。本文及声明性 JSON 不是运行成功回执，也不是可直接传入旧 launcher 的配置。** main 入口整理通过 [PR #374](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/374)；本协议在独立研究分支发布。当前复用代码固定于 `f0e19c8812896f98053e74b165d6c52e85327b69`，不表示整条执行栈已集成 main。

配套文件：

- [机器可读实验规格](../../experiments/native_arx5_transfer/native_camera_public_arx5_transfer_v1.json)
- [20 布局／80 请求预注册顺序](../../experiments/native_arx5_transfer/dev_panel_v1.json)
- [实现与执行交接](../handover/2026-11-11/NATIVE_ARX5_TRANSFER_EXECUTION_HANDOFF.md)
- [结果报告模板](../templates/NATIVE_ARX5_TRANSFER_REPORT_TEMPLATE.md)
- [来源与未决证据](../reviews/2026-11-11_NATIVE_ARX5_TRANSFER_SOURCE_AUDIT.md)
- [本次协议静态验证记录](../reviews/2026-11-11_NATIVE_ARX5_TRANSFER_PLAN_VALIDATION.json)

## 1. 问题、判断和结论边界

### 1.1 本轮要回答的唯一主问题

在官方 GR00T-N1.7 同一起点、原生机身＋腕部输入、冻结 WBC、相同目标 TRAIN100 和相同目标 500 更新条件下，**额外增加公开 ARX5 500 更新的预适配，是否提高 Go2＋ARX5 站立整机抓持、抬升与保持的成功率？**

判断：用户方案方向合理，能够隔离“公开预适配”这一增量因素。新相机、官方起点、参数范围、目标采样、目标优化器、解码器和评分在 A/B 中全部相同。先固定相机再训练、源与目标分阶段、保留全部拒绝和负结果，是这一对照成立的关键。

这不是旧外部视角与新机身视角的因果对照。若本轮成功，可说“在新观测版本下 B 比 A 改善”；不能把收益分别归因于换相机、换父模型或公开预适配以外的因素。

### 1.2 不从本轮推导的结论

本轮不检验公开数据替代目标数据、WM 学会接触规律、行走中的移动操作、动态基座的普遍鲁棒性、接触物理理解或跨任务零样本泛化。它也不是等总训练算力比较：B 比 A 多500个源更新。

站立场景保留完整腿、机身和控制链，可检查基座耦合与随体相机的影响；它不提供行走／大幅机身运动的训练覆盖。公开 global 图像也不能被描述为随 BASE 的 body 图像。

单训练 seed 和有条件扩样的开发集只支持 pilot 结论。首轮没有收益时，不据此宣布公开迁移无效，也不自动追加同分布训练。

## 2. 固定范围与历史保留

| 项目 | 本轮固定值／约束 |
|---|---|
| 官方父模型 | `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| 原始七文件身份 | `0e493187af087d156f6352d46ce14e049ff18e7c1cb1eabc8671357ab4b96e71`；逐文件哈希见配套 JSON |
| 参考执行源码 | `f0e19c8812896f98053e74b165d6c52e85327b69`；实际新实现另记完整 SHA 与变更文件 |
| 站点 | Core 4080S，单一物理控制 owner；不增加大卡作业 |
| 范围 | 完整 Go2＋ARX5 的站立整机仿真 |
| 观测 | body RGB、wrist RGB、原 EEF9＋实测 gripper 状态、语言 |
| 训练范围 | `INTERFACES_LAST2_DIT`；七类接口的 category10 切片＋DiT30/31 |
| 冻结 | 视觉／语言、其它参数／类别、WBC、腿策略、动力学、2 ms 步长、原生安全 |
| 目标标签 | 原消费证明支持的 Teacher 名义 H40；动作、干预 mask、统计不变 |
| 源标签 | source-only 实测 EEF／width 的 H40 参考，不冒充控制命令 |
| 目标阶段 | 纯目标数据，无公开回放、WM、额外损失或 contact 重采样 |
| 主评价 | 500 更新端点；同一 endpoint basis＋原消费链；原始 H40 另列 |

保留 CORR2048、SFT2000、C500、原 BF16 跨硬件 REJECT、旧 DEV／TEST 和 Teacher24 最终19/24负结果。用户明确给出19/24；核验的 GitHub 执行 head 尚只有 first10／first11 部分回执，应补完整终态、逐条件分母、来源 SHA、哈希，不用较早部分结果覆盖最终负结论。这一证据补齐不自动要求重跑 Teacher24，也不让旧负结果因新训练失效。

## 3. ABI、可训练范围与解码器

### 3.1 保留原接口；角色变化必须写在 manifest

原 Processor 的视频键仍是 `external_rgb,wrist_rgb`。本轮目标 body 图像映射到原 `external_rgb` 键，数据 manifest 明确标记 role=BODY、安装版本和真实外参；源 global 也使用该键，但 role=SOURCE_GLOBAL。同一个键名不表示同一种安装。

状态为 `eef_9d+gripper_position`。EEF 使用原 RELATIVE／XYZ_ROT6D action 处理；rot6d 是矩阵前两行展平；夹爪 action 为 ABSOLUTE。padding 与 mask 保持最大 state/action dim132，H40。保持原 world↔policy 固定转换、一次 Processor decode、一次 policy-frame 逆变换，以及原参考传输。不得把新 body 相机坐标直接当 EEF/action 坐标。

目标状态中的实测开度与目标动作中的 nominal gripper 是不同变量。消费者的原映射不变：

`working_closed = .04 * .05`，`working_open = .044 - working_closed`；
`mechanical_fraction = (working_closed + (working_open-working_closed)*nominal)/.044`。

保留原计算表达式、边界和浮点语义；不把近似常量或源 width 替换进去。标签、归一化和执行映射分别记录。

### 3.2 精确参数白名单

七类接口：

1. `action_head.state_encoder.layer1`
2. `action_head.state_encoder.layer2`
3. `action_head.action_encoder.W1`
4. `action_head.action_encoder.W2`
5. `action_head.action_encoder.W3`
6. `action_head.action_decoder.layer1`
7. `action_head.action_decoder.layer2`

每类仅 category10 的 `target_W,target_b`，共14个接口张量；另训练匹配 `^action_head\.model\.transformer_blocks\.(30|31)\.` 的参数。其它类别表、前30个 DiT block、视觉／语言、vlln 与 vl_self_attention 不训练。

`new_embodiment→10` 是项目约定，官方 embodiment 映射没有命名的 ARX5/Go2/ID10 专槽。source、A-target、B-target 必须使用同一个适配槽，source→target 不重置这些接口。

新官方起点必须重新枚举参数名、shape、numel、dtype、hash、共享／tied 参数处理。历史实现的77,847,684可训练参数只作核对线索，不能替代新模型实际审计。固定原官方配置中开启的 tune_* 标志不得覆盖白名单。完整模型普通推理重建必须保留没有更新的官方父参数及正确 tied-weight 关系。

### 3.3 统一解码：完整消费重建 H40

冻结当前：

`H40_CURRENT_STATE_PREDICTED_GOAL_QUINTIC_REPRESENTATION_V3_01`

实际实现读取原始 H40，使用**当前实测 EEF＋上一个实际已消费的 nominal gripper＋原始预测末点**，重建 quintic 平移／夹爪及 SLERP 旋转的40点轨迹。完整消费重建后的40个50 ms区间；raw H40完整保留并独立报告包络。

因此：

- A/B比较的是“GR00T＋固定 endpoint basis＋原控制链”；
- 不称原始40点全部参与几何规划或已经直接执行；
- 此表示不保证所有预测末点可行，原资格与安全拒绝仍然有效；
- 无逐点 clipping、retiming、Teacher fallback 或物体真值输入；
- 不在本轮引入另一种“拟合全部原始40点”的新 decoder；
- 原始 H40 与重建 H40 的位移／旋转／夹爪差异、包络和拒绝原因分别存档。

代码： [basis](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_endpoint_basis_v3.py)、[policy compiler](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_endpoint_policy_v3.py)、[50 ms consumer](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_grasp_v1.py)。

## 4. 新相机版本：先覆盖、再数据、后训练

### 4.1 安装定义

相机版本：`BODY_BASE_FIXED_V1_20261111`。

- parent：`base_link`。
- position BASE：`[0.25,-0.24,0.45]` m。
- look-at BASE：`[0.38,0,0.34]` m。
- up BASE：`[0,0,1]`。
- FOV：75°，实现为 MuJoCo vertical `fovy`。先与原内存预览参数核对；若预览采用不同口径，登记差异与新候选版本。
- 原始输入沿用256×256 RGB；腕部相机安装、内参、分辨率、时钟保持原配置。

position、look-at 和 up 每帧均用同一实际 BASE SE(3) 转到世界坐标。朝向由固定安装决定，禁止查询物体真值追踪目标。记录 camera-to-base 与 camera-to-world、内参、畸变／渲染约定、模型与标定哈希；用基座平移、roll/pitch/yaw 的离线姿态样本验证刚性跟随。

### 4.2 覆盖率的可计算定义

主分母是1,450个唯一有效监督窗口，不能先过滤看不见的样本再算通过率。全100回合的所有封存 capture 另表保留；若原完整记录为1,500个 capture，则50个干预窗口也进入附表，但不重新加入 BC。

| 检查 | 定义 | 通过条件 |
|---|---|---|
| body 投影边距 | 目标完整几何的 amodal 投影 bbox 在相机有效前方，四边距图像边界≥8 px | 至少 ceil(0.95×1450)=1,378 窗口 |
| 分阶段实际可见性 | 遮挡感知实例 mask／depth 中目标可见像素数；body/wrist 任一路≥64 px | 每阶段至少 ceil(0.90×n_phase) 窗口 |
| Processor 后视野 | 将 mask 随真实 resize/crop 变换，检查最终模型输入 | 每阶段必须有≥ceil(0.90×n_phase)窗口在至少一路最终输入中目标可见≥64 px，否则P1不通过；另报告裁切量 |
| 空阶段／缺数据 | 原有阶段映射和预期 count 对照 | 缺阶段、缺图、缺状态不能静默 PASS |

阶段使用原审计字段，覆盖 PREGRASP/DESCEND/CLOSE/LIFT/HOLD 的既有含义；若原记录使用别名，先冻结明确映射。记录各阶段整数分子、分母、阈值、最差回合与可复核截图。bbox 是不考虑遮挡的投影；64 px 是遮挡后的真实可见 mask，不能相互替代。mask 与阶段／接触真值仅用于审计，不进入模型。

候选未过时，只在 TRAIN100 上改安装／FOV／预处理并重新版本化；不能用新 DEV、旧 TEST 成败选相机。全部通过并封存后，才进行主训练。若改变安装，所有依赖的新图像、manifest、cache 都使用新身份。

### 4.3 实际状态重渲染与不变量

使用原完整场景及其封存状态、时钟、资产和渲染参数，在独立渲染副本恢复实际 qpos/qvel、相关 actuator/mocap／对象状态和时间；只做必要 FK／render，不前进物理。body 图像重新生成，wrist 原件逐文件复用。

数据名称必须包含“实际状态重渲染的新相机版本”，不能叫新遥操作或新增100回合。原始状态与 action 数组、干预 mask、监督窗口顺序、语言及统计文件逐项保持；图像、安装 metadata 和总 manifest hash 合理变化。

验收至少覆盖：

- 原100 episode ID与1,450监督 window ID／顺序完全一致；
- 非视觉数组的 dtype、shape、逐元素及必要字节身份；
- 所有原 wrist SHA256完全一致；
- 原体状态、对象状态、时钟与图像配对一致；
- 目标统计沿用原封存内容，若重算只用于证明一致，不替换成另一统计；
- 缺文件整批记录／修复后再导出，不删样本、补末帧或伪造状态。

在新 observation manifest 中为每个窗口记录 camera role/key、安装／标定版本、图像与状态 SHA256、真实时间、原窗口 ID、label_semantics、mask来源和消费证据引用。旧外部视角数据及接口封存不变。

## 5. 公开 Table30 ARX5 的准入与转换

### 5.1 数据身份及获取成本

固定为 `RoboChallenge/Table30@1eda76b8935b5c7a59cd635ca9e3fab882943e11`，不是 Table30v2。只选 metadata 确认的 ARX5 episode；保留原 global＋wrist RGB，忽略第三路不使用的侧视角。

固定源树约736 GB，部分任务以大归档分片提供。最多3,200／400窗口不等于有限下载量。先建立：

- 任务、robot_id、episode／采集组候选清单；
- 所需 archive parts 的顺序、远端预期 size/hash、成员范围；
- 实际需读取／传输的字节、磁盘／CPU解码／cache成本；
- 可用存储及站点限额下有限获取计划，优先复用可信已在本地的同身份文件；
- 实际抽取成员 SHA256和使用条款来源；不因公开可见而虚构某个许可证。

先看 metadata 和可验证成员清单，再确定可负担的任务覆盖；不启动全库下载。若归档结构要求取齐整组分片，应报告真实成本，不假装可以按窗口随机取到。因获取受限未读到的数据与经审计确实不足的合格样本分别计数。

### 5.2 已核实的字段与尚需验证的约定

固定 README 中的 ARX5：

- `end_effector_pose`：6维 `xyz,roll,pitch,yaw`，相对 arm base；
- `gripper_width`：实测米制宽度；
- `timestamp`：文档声明每帧浮点毫秒；
- `arm_realsense_rgb.mp4`：wrist；
- `global_realsense_rgb.mp4`：global；
- `meta/task_info.json`、`meta/episode_meta.json`：任务／机器人／视频／回合元数据。

执行前用具体 episode、采集端定义／标定核实平移单位、RPY轴序与内外旋、角度单位、TCP/tool frame、width是总间距还是单指量、时间基准与offset/drift。另核验end_effector_pose究竟是实际反馈／实际关节反馈经FK的姿态，还是期望pose／setpoint：登记反馈来源、FK模型与时钟；仅凭states.jsonl文件名不足以赋予ACHIEVED标签。若为setpoint，不能直接作为本轮监督；只有具备可证实的实际关节反馈与固定FK/TCP时，才可另记转换得到actual EEF，否则B-source不准入。README 中的 episode Unix 时间示例与逐帧毫秒字段不能无条件视作同单位。不得猜测后靠 loss 下降判定转换正确。

源 EEF 保留可核实的 arm-base 语义，必要的常量坐标／TCP变换明确列出并验证逆变换。使用原相对 EEF Processor 合同；不把 source arm-base 和目标固定 policy frame 冒充同一物理坐标系，更不做依赖目标物体真值的重定位。

**本轮源夹爪物理字段保留原始 measured width（m），由 source 自己的统计归一化。** 不把它除以目标44 mm后冒充 nominal command，不用样本min/max作为硬件标定。源输出仅用于离线源误差；不能直接送入目标消费者。目标阶段使用原 target 实测状态／nominal标签及原映射。

### 5.3 时钟合同：不直接使用上游示例转换器

固定版本的示例脚本按 `state[idx-stride]`／`action[idx]` 取值，每次循环视频只读一帧，且不使用真实timestamp；stride>1会产生索引对齐风险，也未实现H40／50 ms合同。它只作schema参考。

本轮用可审计转换：

1. 统一状态、episode元数据和各视频PTS／已证实采集时钟到秒；记录offset、drift、丢帧／重复时间。
2. anchor处图像采用不晚于anchor的最近已采集帧；首轮设置每路最大age50 ms、双路skew≤25 ms，逐窗口记录实际值。阈值是本轮预注册工程选择，不是源数据已满足的事实。
3. 原生 knot时刻固定 `t_i=t_anchor+0.05*i, i=0…39`，即0…1950 ms。完整执行为40区间共2 s；不能错移成50…2000 ms。
4. 平移线性插值；已核实RPY先转SO(3)/quaternion，处理符号连续性后最短弧SLERP，再转原矩阵前两行rot6d。夹爪实测宽度线性插值，保留边界和原始值。
5. 状态插值只使用同一episode内支持时间；首轮最大连续状态缺口100 ms。重复时间同值可按固定规则去重，矛盾重复必须拒绝并记原因。
6. H40所有点必须在真实支持范围内，不跨episode、不外推、不重复末帧补齐。源观测不使用未来图像。
7. PTS与状态时钟关系没有证据时，先补最小采集时钟证据；可验证的同步采集行索引／frame-to-state ID映射也是证据，不要求每路视频直接携带UTC。禁止静默用frame_index/fps冒充真实时间配对。若仍不可确认，B-source不准入，A的数据／工程准备可继续。

时钟准入有明确有限失败结果，不通过时不无期限反复解码。时间容差若确需改变，应在看A/B结果之前形成新数据版本，并报告原规则排除量。

### 5.4 先 episode split，再固定窗口

预注册确定性规则：

- 去重按原episode／采集组及内容身份，近重复或同次采集的切片作为一组，整组划分；
- 以task为层，以seed20261111的稳定hash排序组；每task至少2组时，保留组数为 `min(n-1,max(1,ceil(0.10*n)))`；单组task仅入TRAIN并报告该task无保留集覆盖；
- 分组完成后才生成各自的合格窗口；保留组没有合格H40时不迁入TRAIN补数；
- task轮转、episode轮转、episode内按同一固定hash排序选anchor；TRAIN最多3200、保留最多400，唯一窗口不重复填充；
- 任务／episode窗口数量及物理时长、相邻窗口重叠率、独立采集组数分别报告；最多窗口数不是独立样本量；
- 总数不足用实际值；无法形成非空source TRAIN与保留集则不能宣称完整B比较已就绪。

源split清单、候选/选中/拒绝原因、具体窗口list与顺序在任何源梯度更新前冻结。源任务／窗口选择不得依据A/B闭环结果调整。

### 5.5 两种标签和两套统计

| 域 | 观测／标签 | 统计与使用 |
|---|---|---|
| source | 原global+wrist、实测EEF/width、原语言；label=`SOURCE_ACHIEVED_H40_T0_INCLUSIVE` | 只用source TRAIN；保留集不参与；source loss与物理误差分别报 |
| target | 新body+原wrist、原EEF状态／语言；label=`TARGET_TEACHER_NOMINAL_H40_CONSUMPTION_PROVEN` | 保留原TRAIN100统计、mask、action与消费证明；A/B完全相同 |

官方statistics.json保留为父模型身份，不直接冒充新source或target统计。source统计只拟合最终选中的唯一TRAIN窗口：每窗口一次anchor state及40个H40值，统计方法沿原Processor类型并登记；不按16,000重复训练访问加权，保留集不参与。源、目标Processor独立实例及hash。B从source转target只承接白名单权重，不带source统计、优化器、日程、采样游标或RNG；不重置学到的category10接口。

公开保留集在official、source500、A-target500、B-target500各阶段做固定离线探针；切回source Processor只用于评估副本，不修改目标部署包。它衡量源域误差／遗忘，不是公开任务闭环成功。不同域归一化loss不直接作绝对数值难度比较。

## 6. 冻结骨干缓存与 GPU 运行资格

### 6.1 缓存边界与身份

重新执行原Processor和冻结视觉／语言骨干，缓存只包含：

`backbone_features, backbone_attention_mask, image_mask`。

保留原state、embodiment_id、action、action_mask。缓存身份至少包含官方权重、upstream代码、实际Processor/tokenizer依赖、相机role/key/顺序/安装版本、图像SHA、语言SHA、所有图像变换与dtype／数值运行模式。新目标视角与source使用独立cache，旧SFT2000缓存不复用为官方起点缓存。

前30个DiT block虽然权重冻结，其输入仍依赖可训练接口、flow时间及噪声。**不能缓存它们的输出，也不能对这一路用no_grad切断传向state/action接口的梯度。** 完整action head仍参与训练图。首轮不新增随机视觉增强；若后续引入增强，需独立定义每种增强身份，不把固定cache称为在线增强。

### 6.2 最小充分的新资格

复用已有资格的未变部分，只检查新官方起点、新观测、source接口、cache／恢复受影响的边界：

1. 七文件身份、实际代码／Processor依赖和白名单；无随机重置或旧CORR/SFT权重混入；
2. 少量跨阶段TRAIN输入的完整原始路径与cache路径：输出、FM loss及全部白名单梯度／一次完整更新对照；
3. source真实短批的state/action/time/frame/mask审计，不用合成夹爪或全零padding掩盖错误；
4. microbatch1／accum32的GPU峰值、完整更新耗时、输入解码与cache读取耗时；
5. fresh-process保存／恢复：参数、Adam、scheduler、CPU/CUDA RNG、采样cursor、下一输入／噪声及后续一个更新一致；
6. 250和500端点在普通推理入口重建，核对归一化H40／物理H40／Processor身份；普通部署输入仍是真实RGB，不依赖训练cache；
7. 同一Core卡上的重复性及旧BF16 REJECT的隔离；不把当前同卡新资格宣布为旧跨硬件REJECT已解除。

匹配更新的数值准则沿用项目现有严格比较；如出现不一致，定位具体变更，不按A/B分数调容差。资格更新用独立临时模型，不能混入候选预算。已有可信资格可按身份复用，不重跑完整历史面板。

### 6.3 内存与成本约束

可训练参数、梯度、优化器状态和训练激活留在GPU，CPU只做解码、存储／checkpoint序列化和审计；不沿用CPU训练激活offload。冻结视觉骨干可在离线编码阶段工作，正式训练只保留所需head与冻结feature；按原资格重建普通完整推理。

不能假定旧仅1450行的ResidentRows可直接承载source3200行；新reader以manifest实际长度和字节身份为准，target仍强校验1450。GPU行缓存可按已核验容量一次驻留或有限搬入输入，训练激活不搬CPU；这属于实现选择，须对A/B target使用同一方式。

OOM时先定位常驻项、缓存与生命周期，在不改科学合同下修复；仍无法容纳时报告RESOURCE_BLOCKED，不自动换大卡、减H40、换精度、减有效batch或解冻／冻结其它层。资格产出每阶段预测wall/GPU时间及实际存储需求，不在本协议虚构小时数。

## 7. 配方、配对与预算

### 7.1 三个有限训练阶段

| 阶段 | 起点 | 数据 | optimizer calls | 训练访问 |
|---|---|---|---:|---:|
| A_TARGET | 官方共同父模型 | target TRAIN100 | 500 | 16,000 |
| B_SOURCE | 同一官方共同父模型 | 固定source TRAIN | 500 | 16,000 |
| B_TARGET | B_SOURCE500白名单权重 | 同一target TRAIN100 | 500 | 16,000 |
| 总计 | 两条方法路线 | 三个阶段 | 1,500 | 48,000 |

如果target1450均匀访问，16,000相当于平均约11.03次／窗口；source达到3200时平均约5次。实际重复／独立访问分布按采样trace报告，不能把48,000访问称为48,000新样本。

### 7.2 优化设置与明确日程

每阶段：microbatch1、gradient accumulation32、AdamW、lr1e-4、weight_decay1e-5、betas(0.9,0.999)、eps1e-8、原decay/no-decay分组、grad_norm clip1.0、原FM mask／loss、不增加auxiliary/execution loss。

新日程为500更新、25步warmup、cosine至0；**不沿用旧8000/200或旧helper隐含0.1最低比例**。为避免第一步零LR含混，固定用于实际第 `u+1` 次更新的倍率（u=0…499）：

- u<25：`(u+1)/25`；
- u≥25：`0.5*(1+cos(pi*(u-25)/475))`；
- 第500次更新完成后的scheduler位置500对应LR0。

500次实际更新使用的LR均为正；预生成500项LR序列并封存hash，保存used_lr与next_lr。这里是本新配方明确的边界约定，不借用旧配方资格。每阶段独立重启此日程；250只保存／恢复／诊断，主评价固定500，不选最优checkpoint。

精度沿Core当前已资格的FP32可训练参数／原BF16 autocast模式，先完成新起点同卡资格。实际kernel、dtype、TF32／确定性选项、GPU标识与软件依赖记录，A/B一致。

### 7.3 目标阶段配对

A_TARGET和B_TARGET的seed均为20261111，重置Python／NumPy／Torch CPU/CUDA／dataloader RNG、optimizer、scheduler、step和cursor。沿原ORIGINAL exposure采样规则，不用CONTACT_EVENTS重采样；在目标训练开始前封存两者共同的16,000项window索引／cursor映射与FM噪声规则。

同一个 `(target_update,microbatch_offset)` 对应同一个window、mask、flow时间与噪声。评估／保存不要消耗后续训练RNG。B仅承接source白名单权重；冻结父参数与target统计／Processor和A相同。源阶段使用独立随机流，结束后不能把源RNG状态带入B_TARGET。

B_SOURCE对固定manifest顺序的N个窗口使用独立NumPy PCG64(seed=20261111+cycle)逐epoch permutation；cycle从0开始，cursor连续到16,000，最后一轮只取所需前缀。采样RNG独立于FM噪声，不再按task二次加权。目标接口的旧100/1450、parent_global_step2000与SFT2000安装限制需要新profile包装，不能只改旧JSON的seed／路径即执行。

### 7.4 保存与恢复

每阶段250、500保存完整状态；checkpoint manifest包含父身份、阶段、已完成optimizer calls／非零LR calls／sample visits、参数／optimizer／scheduler／RNG／cursor／cache／Processor／统计的hash。

正常恢复从同一阶段完整状态继续到原500终点；不得把恢复次数当新实验、重复计算已封存更新或换采样顺序。B_SOURCE→B_TARGET是新的优化阶段，仅load白名单权重，其余状态重新初始化。

资格／probe／编码／下载成本与候选48,000训练访问分开记录。推理／保留集查询不算SGD访问；失败启动和恢复已经花费的资源也保留。

## 8. 尽早得到整机反馈：四个 TRAIN 诊断

相机覆盖通过后，先使用现有SFT2000在四个已知TRAIN布局各做一次完整整机诊断。布局在查看新结果前固定为TRAIN01、TRAIN07、TRAIN13、TRAIN25（执行前对原25布局manifest核实ID与物理参数；若命名不同，用原ID映射并封存）。

保持SFT2000自己的权重、Processor／统计身份，只切换已版本化的body+wrist观测与同一固定endpoint basis，不混用官方起点统计。四项全部报告，不用它替代A/B或新的官方起点资格。

它的目的：尽早观察新视野、遮挡、原消费链及同卡推理是否可用，定位工程问题。任务失败不等于相机数据无效；只有对应明确观测／时序／接口缺陷才修改数据并重新封存，不能根据四项分数追加训练或选相机。源数据CPU准备可并行；Core物理控制与GPU作业按实际资源和单owner约束串行。

## 9. 新开发人口与预注册顺序

### 9.1 布局与身份

固定20布局：

`X={-6,-4.5,0,4.5,6} mm`，
`Y={-4.5,-2.5,2.5,4.5} mm`。

每布局两次独立重启；每方法40请求。完整拟定顺序在配套JSON，X/Y以整数微米另存避免浮点身份混乱。绝对位置原点／场景base identity、物体姿态、初始机械状态、资产／物理参数、reset实现必须在运行前固定；仅有offset还不是完整布局身份。

注册前比较旧TRAIN/DEV/TEST的**输入身份manifest**：既比较完整场景身份，也比较物体相对基座／EEF的物理相等性，避免换seed／名字制造“新布局”。不读取旧TEST结果／奖励／成败日志。若重叠或缺少证明：

- 列出原20布局中的具体冲突或未审计项；
- 不静默删项、补点或读取旧TEST成绩；
- 在查看A/B新DEV输出之前完成独立版本的修订／理由；原拟定网格保留；
- 未完成身份审计时，JSON标记PROPOSED_NOT_REGISTERED，不称新DEV已独立。

旧TEST只有可隔离的输入manifest不可得时，先做输入导出，不打开含成绩的原始评估报告。

### 9.2 首筛与扩样

首筛四布局为X±4.5、Y±2.5 mm，各两重启，A/B各8请求，共16。每个布局两次重启交替A-first/B-first，匹配layout、restart、reset seed与query noise key；A/B target500同时固定并完成资格后才打开本轮新DEV。

首筛固定8对全部完成且有可核验终态，若：

`success_B - success_A >= 2`

且无B新增硬安全或技术问题、任一方法均无未解决技术问题，则接续其余16布局×2重启×2方法，共64新请求。已完成的8／方法直接计入，不重跑。未达到条件，封存首筛，不自动扩样或追加训练。

该门不作显著性判断。首筛A8/8、B8/8也不会触发扩样；这意味着本首筛没有可观察增量，不是B无能力。

### 9.3 随机性、初态与异常

配对噪声按 `protocol|layout|restart|query_index` 生成稳定hash种子，不让某方法早停改变另一个方法后续随机序列。两个方法执行顺序不共享可变RNG游标。每次冷重启按原reset合同，报告实际base SE(3)、qpos/qvel、EEF、object、gripper初态差异；匹配不声称bit-identical。

任务运行保持原时间／查询上限，实际值从原评价配置提取并在看新DEV之前写入run manifest；不根据候选表现延长。原生停止条件优先于完成H40，任何实际中断的未消费后缀都不得记作已消费。

故障分类：

- 参考包络／预测夹爪范围等模型输出引起的原生拒绝：策略失败，保留在分母；
- 数据损坏、解析／shape错误、消费者审计断链：技术终态，保留请求与已花费资源；
- 原安全停机：硬安全终态及原评分；
- 尚未启动：NOT_STARTED，不伪装为失败或成功。

正式请求不以成功重试替换失败。技术修复只做独立诊断请求；原请求保留终态，未解决问题不扩样。若尚未触发物理但已发生模型预测拒绝，也不能把请求删成“没跑”。

### 9.4 推理暂停的现实边界

既有接口在观测／推理时暂停仿真。本轮保持它以隔离数据迁移，并同时报告推理wall latency、暂停wall时间和物理sim时间。不要把这轮结果解读成满足真实机器人连续运动中的在线低延迟控制；也不据此声称已验证WM可补偿机身运动延迟。

## 10. 评分、完整分母与统计解释

主要任务成功要求沿原评分同时满足真实抓持、相对起点抬升≥20 mm、保持≥1 s以及原站立／安全条件。复用原证据谓词与代码，不将“名义夹爪闭合”或“目标参考抬升”替代真实抓持／实际物体抬升。

每请求输出：method／checkpoint／layout／restart／seed、实际初态、首次失败阶段、原始／重建H40包络、参考接受／拒绝、夹爪事件、抓持／接触／抬升／保持证据、提交／消费tick及比例、技术／安全终态、wall/sim耗时、模型查询数。

总表同时提供。planned包含最多80个条件槽；requested仅指已正式激活的一批请求。首筛未触发扩样时，Stage2列为CONDITIONAL_NOT_REQUESTED（planned32／方法、requested0），不算已跑失败：

- planned/requested、attempted、reset_ready、technical_valid、success；
- success/requested 及 success/technical_valid（后者不能替代前者）；
- NOT_STARTED、技术失败、策略拒绝、任务失败、安全停机的数量与身份；
- 配对共同成功、共同失败、A-only、B-only（n10/n01）；
- 每布局的两次restart结果，不把80请求当80个独立场景；
- Stage1 8／方法、Stage2 32／方法及合并40／方法三张表。

若首筛触发扩样，合并40为有条件选择后的描述结果，不能作为未经筛选的确认性显著检验。新增16布局的Stage2是观察增益是否延续的重要证据。样本少时使用完整计数和配对差异，任何区间／bootstrap须说明按layout聚类，且不能消除单训练seed不确定性。

源保留集离线误差、训练FM下降、原始／重建包络通过、四TRAIN诊断均分开报告；不相互充当任务成功率。

## 11. 执行依赖、停止点与实现工作

| 顺序 | 工作 | 通过后允许 | 未通过时 |
|---|---|---|---|
| P0 | 版本／原文件／现场资源与owner核对 | 数据准备／窄范围实现 | 补缺口，保留事实；不碰活动胶囊 |
| P1 | 相机覆盖和实际状态新版本 | 新target cache、四TRAIN诊断 | TRAIN内改相机版本；无主训练 |
| P2 | 公开源获取／schema／时间／split | B_SOURCE及source holdout | 报源准入未完成；不伪造或换revision |
| P3 | 新父权重／白名单／cache／GPU／恢复资格 | 三阶段训练 | 仅修具体工程问题，不增变量 |
| P4 | A_TARGET500、B_SOURCE500、B_TARGET500 | 对应保存、恢复及普通推理核验 | 原有限阶段恢复或明确失败 |
| P5 | 新DEV身份、完整endpoint资格与评分封存 | 首筛16请求 | 不读取新DEV输出选配方 |
| P6 | 首筛完成与预注册扩样条件 | 最多64接续请求 | 封存，不自动扩大 |
| P7 | 失败谱系、成本、证据与决策 | 独立后继提案 | 保留负结果 |

P1与P2可并行；P1完成后的四TRAIN诊断优先获得反馈。默认GPU阶段顺序A_TARGET→B_SOURCE→B_TARGET，A的DEV结果不提前查看；如资源／源准备顺序不同，只改变作业排程，不改变样本、训练或评价身份。

实现应复用已有训练、观测和消费者模块，新增本轮profile与source reader／camera导出适配；旧profile仍保持原父身份与站点约束。不得通过删除旧检查来让新实验冒充旧实验。新实现需要的具体CLI在实现PR中登记；本计划不编造当前不存在的launcher命令。

旧 `world_model/` 目录中的训练／消费函数可复用，WM预测／校正调用必须保持禁用；不做无关整体重构。main与执行代码相差大，运行实现使用独立checkout与明确依赖，后续集成通过普通PR保留ancestry。

## 12. 交付文件与验收

所有运行产物先保存到既有项目允许的artifact位置，再提交去敏摘要和manifest；不把原始私有图像、模型权重、vendor资产、主机凭据或host-bound私有路径推到公开GitHub。

| 交付 | 必需内容 |
|---|---|
| CAMERA_COVERAGE_ACTUAL | 版本／模型／实际状态hash、主与全capture分母、各阶段通过数、原图／Processor后检查、失败样本与固定安装验证 |
| TARGET_OBSERVATION_MANIFEST | 原100/1450身份，新增body图、复用wrist、状态／动作／mask／统计一致证明 |
| SOURCE_ADMISSION_ACTUAL | pin／archive获取成本／member身份、schema／坐标／时间与skew、episode split／窗口清单／排除分母、源统计 |
| MODEL_AND_CACHE_ADMISSION | 七文件、actual code、参数白名单／冻结指纹、完整路径与cache对照、内存／吞吐／保存恢复 |
| TRAINING_ACTUAL | 三阶段真实calls／nonzero-LR／visits／unique counts、sampling/noise trace、各250/500完整端点身份 |
| DEV_REGISTRATION | 新旧人口输入身份对照，不含旧TEST成绩；绝对场景及随机性／顺序／原超时／评分hash |
| NATIVE_EVALUATION_ACTUAL | 全部正式请求与终态、H40／消费／评分／初态／耗时，分Stage1/Stage2/合并 |
| FINAL_REPORT | QUESTION / CHANGED / ACTUAL / NEGATIVE / DECISION / NEXT；完整分母、真实更新与资源成本 |

所有未生成hash用null和PENDING状态，不能填假hash或写PASS。只要实际文件身份未变化且原验证可信，不要求重复全量hash大文件；变更边界才重新验证。保持最少充分审计，不引入新分布式审批／重复队列框架。

## 13. 首轮结束后的判读

| 观察到的模式 | 能支持的判断 | 合理后继（另立实验，不自动执行） |
|---|---|---|
| B首筛及新增布局持续更好 | 固定目标预算下公开预适配有增量价值 | 第二训练seed、更多独立布局／任务、总算力匹配对照 |
| B源误差下降、目标无增益 | 本次源学习没有转化为目标闭环收益 | 检查视觉角色、动作／夹爪语义和目标覆盖，不立即堆源步数 |
| A/B大量同类参考拒绝 | 当前组合策略与参考可行域存在问题 | 分析原始／重建包络及目标末点，不改WBC掩盖原因 |
| A/B接近／预抓好，接触失败 | 目标接触／恢复覆盖或观测不足的线索 | 定向新增目标接触／恢复数据；先证据后考虑辅助模型 |
| A/B在抬升后因body视野丢失失败 | 相机／视觉状态估计链可能是瓶颈 | 在新TRAIN版本补相应视角／状态覆盖，独立比较 |
| B较差且源/目标标签表现冲突 | 本配方可能出现负迁移 | 先定位frame／clock／grip／normalization，再独立测试回放或参数范围 |
| 技术故障使对照不完整 | 尚不能判数据迁移优劣 | 保存正式失败，修具体接口并提出完整新版本 |

WM只有在失败谱系明确显示“观测历史／短时状态预测／接触不确定性”是瓶颈时才成为后继候选。本轮不为假设性需求增加WM，也不让它承担相机标定或时钟错误的补救角色。

## 14. 来源

- [官方固定模型树](https://huggingface.co/nvidia/GR00T-N1.7-3B/tree/2fc962b973bccdd5d8ce4f67cc63b264d6886495)
- [官方配置](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/2fc962b973bccdd5d8ce4f67cc63b264d6886495/config.json)
- [官方embodiment映射](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/2fc962b973bccdd5d8ce4f67cc63b264d6886495/embodiment_id.json)
- [Table30固定README](https://huggingface.co/datasets/RoboChallenge/Table30/blob/1eda76b8935b5c7a59cd635ca9e3fab882943e11/README.md)
- [Table30固定转换示例](https://huggingface.co/datasets/RoboChallenge/Table30/blob/1eda76b8935b5c7a59cd635ca9e3fab882943e11/convert_to_lerobot.py)
- [原生Processor ABI](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/scripts/pi0291_arx5_data/native_dataset.py)
- [接口切片实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/scripts/pi0294/generalization/minimal_adapter.py)
- [Core训练实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_core_local_v1.py)
- [冻结feature seam](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_features_v1.py)
- [GPU输入缓存](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_gpu_rows_v3.py)
- [原夹爪与轨迹适配](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/fusion_trajectory_adapter.py)
- [匹配原生评价](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/f0e19c8812896f98053e74b165d6c52e85327b69/src/go2arx5_vla/world_model/task_learning_matched_native_v3_1.py)

上述代码与来源说明已有事实；新相机／源数据／训练／评价是否通过，仍以本轮实际产物为准。
