# Codex 执行计划：GEN1 多场景环境准备与可验证泛化

日期：2026-10-25，Asia/Singapore。类型：IMPLEMENTATION_HANDOFF / DESIGN_ONLY。
源码审阅快照：`d352563dc2bba47d90c1f645ef1a86ae35dfb851`。
承接：[后续多场景泛化设计](POST_EVAL_MULTISCENE_GENERALIZATION_PLAN.md)。

**目标：在已有固定基座ARX5源域上建立可参数化、可复现、可隔离训练与测试、可接现有Teacher/GR00T接口的场景配置与验证层；不是另建仿真平台，也不是现在启动一轮大规模采集或训练。** 当前V4A/B24的不可变源码、模型、人口、R0、评分、安全与owner不改变。本文件不提供新的GPU/物理/采集授权，不重置原时钟，不把小时监督任务变成执行owner。

## 0. 工作边界及已核实基础

最新已读12:03 SGT回执为B13/24，3成功、10有效失败；同13格A成功7。它是顺序中途结果，不作最终B−A结论，也不据此挑选新场景。现役B24应沿原队列独立完成。场景准备可先做纯配置/源码/CPU单元测试；任何原生模型构建、OpenGL渲染、Teacher物理或模型查询须使用真实可用的隔离执行范围。不能因为“零积分”便占用当前Eval GPU或另开未登记仿真进程。[E0]

源码中已经存在：

| 组件 | 实际可复用内容 | 注意事项 |
|---|---|---|
| `scripts/pi0294/scene_cards.py` | T1/T2固定指令、确定性几何、visual字段、2mm/2度近重复检查 | F1联动空间变化，F2联动视觉，F3再加相机扰动；不可把这些旧family直接当作独立因子设计 |
| `scripts/pi0294/scene.py` | 固定基座ARX5原生引擎；table/floor颜色、light_scale、相机平移/旋转；静态IK/碰撞/视野检查 | 已有颜色和光照支持；尚不能据此声称已有纹理/杂物/新物体支持 |
| `scripts/pi0291_arx5_data/scene.py`、`config.py` | 真实关节/有限力矩、活动夹爪、红色4cm立方体、蓝目标框、双相机及冻结物理 | 不修改原WBC资产、动力学、摩擦、增益、控制频率或任务评分 |
| `scripts/pi0294/pv1/plan.py`、`population.py` | 有限候选池、预先排重、静态验证及原DEV/LOCKBOX完整人口 | 原128组和原DEV48是冻结实验，不能改成新GEN人口；只能复用通用函数和验证语义 |
| `scripts/pi0294/pv1/export.py` | `committed_packet`、`codec_packet`、`export_committed_case`等共享数据/编解码审计 | 外层原PV1组白名单保持；新GEN数据需要窄范围显式接入，不能伪装成旧TRAIN ID |
| 已有24个Teacher候选 | 12个原TRAIN几何，每个T1/T2；来源和静态证据可追溯 | 只有候选/历史静态资格，不是24条新成功示范；不得改名放进新TEST |

先遵守根AGENTS和现有统一入口/源码发布约定。单一集成发布者；不得在有活动消费者的checkout或run capsule上pull/修改。不要新建另一套资源账本、监控服务、collector/trainer或通用任务框架。必要缺口用小模块/适配器补齐。[E1–E6]

## 1. 首版场景变化范围：少量独立因素，而非复杂世界

### 1.1 GEN1纳入与排除

**第一优先：布局。** 改变物体XY、目标XY、相对方向/距离和小幅yaw；任务仍是红方块抓放/推送到蓝框。物体尺寸/质量、桌高、接触和执行器不变。

**第二优先：一个纯视觉因素。** 先只改变桌面颜色。背景颜色和光照虽然已有接口，第一项训练对照先保持固定；后续单独启用。场景外观在整个episode内保持不变，不逐帧随机闪烁。

**同场景双任务。** 每个场景都生成T1/T2卡，初始物理和相机完全相同，仅原任务指令与对应评分分支不同；同一模型/adapter/stats，不按task-id切换控制器。

**暂不纳入首版：** 新物体几何/尺寸、多可交互物体、遮挡物/可碰撞杂物、相机位姿随机化、桌高/摩擦/质量/执行器变化、BASE、第三任务、目标WBC数据。先把这些标为后继可独立研究的维度，不创建占位成功数据。

研究重点是受控的布局/外观/组合泛化，不是生成更多装饰不同的房间。LIBERO的分维度评价和MimicGen的示范适配思路可供参考，不要求迁移整个benchmark或替换现有Teacher。[R1][R2]

### 1.2 让几何、视觉、任务的随机性彼此独立

使用显式、版本化seed子流：geometry_seed、visual_seed、reset_seed；推理noise和训练seed另存。增加一套visual不能改变原geometry/任务顺序。改变执行顺序、进程数或字典遍历顺序不能改变场景名单；禁止使用进程随机化的Python `hash()`充当固定seed。

旧F0/F1/F2保持原义，不重定义。GEN1使用独立的layout_stratum与visual_profile字段，并保存旧family/来源；不能让“反向=深色桌面”或“T2=另一背景”成为预测任务的捷径。

## 2. 可达范围：从已验证配置派生，不凭空放大

### 2.1 原legacy生成范围与硬边界分开

审阅源码中的legacy空间采样为：source x约0.435–0.465m、Y主方向gap约0.16–0.20m、center_y在±0.012m、目标x相对source最多约±0.006m、cube yaw在±0.08rad。legacy粗检查接受source/goal x在0.42–0.48m、|y|≤0.14m、水平距离0.15–0.22m。**这是配置/粗检查范围，不是任意组合均可达或均成功的证明。** 先复用精确来源版本，再做原生静态/可见性检查。[E1]

重要：已准备的DATA_COV1候选有独立来源和范围。例如公开候选有约0.1466m的源—目标距离、0.1492rad的yaw；不能用上述legacy范围一刀切重判旧候选，也不能偷偷扩宽legacy validator。Codex先产出每个population的范围、来源和适用验证器表。旧候选按原版本验证；新GEN1从一个明确选定的范围生成，不能混称相同分布。[E6]

### 2.2 先做四个布局层

在所选范围内划分：`+Y/NEAR`、`+Y/FAR`、`-Y/NEAR`、`-Y/FAR`。near/far分界写入配置并在看新结果前冻结；若使用legacy的0.16–0.20m gap，可把0.18m作为设计分界，端点归属必须确定。对导入的COV1则保留其原distance_bin，不按新分界改历史标签。

新增候选先在原已验证小范围做连续XY/yaw扰动。不要直接把物体移到桌边、扩大到任意方向或增大yaw，除非作为另一个版本并完成相应任务可行性验证。

每个候选记录静态拒绝原因。冻结有限候选池和替代规则后，允许仅按预先定义的几何、IK、碰撞、可见性条件过滤；不根据Teacher/学生成功率反复挑选容易场景。动态测试失败保留，不重抽成功seed。

## 3. 初始状态的两个关键陷阱

### 3.1 现有机械臂起点跟随物体位置

`make_engine`默认把初始TCP放在`[source_x, source_y, table_top+0.10]`。这意味着合法EEF状态本身与物体位置高度相关。它不是已证明的数据泄漏或模型不看图像，但单看换物体位置后的成功，不能宣称模型学会视觉定位。[E2]

实现两种明确reset profile：

- `LEGACY_ABOVE_SOURCE`：严格复现原实验，仅用于回归/与旧结果对接，不能改旧AB。
- `GEN_COMMON_HOME`：在独立TRAIN calibration中，从有限预登记pose候选选一个共同可达/可见的home；不随source/goal/task ID变化，使用已有`initial_tcp`接口。选择依据只允许TRAIN静态几何/IK/FOV，不读取TEST表现。不能为每个TEST场景临时选最有利home。

共同home还要通过从home到原Teacher第一阶段的路径检查和后续TRAIN动态pilot。若无法找到适用home，报告`COMMON_HOME_UNQUALIFIED`，暂保留legacy并明确视觉证据限制，不降低原视野门或做隐藏瞬移。新reset分布是一个版本化干预，后续control和multiscene两臂必须共享它；不能直接把旧A与新reset策略归为纯数据效应。

### 3.2 配置hash不等于真实状态相同

现有`task_card`中的`initial_state_sha256`来自scene group字典摘要，不等于对实际qpos/qvel/相机图像的测量。保留旧字段原义；新记录另加`design_hash`、实际`reset_state_digest`、`camera_calibration_digest`和实际初始RGB哈希。T1/T2初态相等要用新鲜真实状态验证，不能仅比较两个配置卡。[E1][E2]

状态复现应使用当前安装MuJoCo版本的完整适用integration-state接口，加上实际控制器/reset相关状态、seed、精确模型与资产版本。仅存qpos/qvel不足以声明接触仿真的逐步复现；不升级MuJoCo来获得新API，也不要求跨GPU/渲染驱动逐像素bit一致。相同固定软件栈内验证确定性，跨栈只给预声明容差和实际测量。[R3]

## 4. 视觉变化：先用已有支持，守住指令和可见性

原任务明确指定`red cube`和`blue square`，原视野检查也统计红/蓝像素。因此首版固定方块红色、目标框蓝色；不改其颜色，不增加红蓝干扰物，不以同色背景伪造可见性通过。[E1][E2]

建议视觉配置（只是待验证值，不是已渲染结果）：

| profile | table RGBA | floor/light/camera | 用途 |
|---|---|---|---|
| V0 | 原基准[0.69,0.70,0.66,1] | 原值 | 回归和布局基线 |
| V1 | [0.72,0.65,0.58,1] | 不变 | 暖色桌面 |
| V2 | [0.55,0.62,0.70,1] | 不变 | 冷色桌面 |

以上颜色都在既有F2通道范围附近；是否保留足够的红方块/蓝框对比仍要实际渲染。第一版不同时改光照；以后光照可单独比较原1.0与已有配置范围内的0.8/1.2，不能把颜色和光照的联合变化宣称为单因素效果。[E1][E2]

白名单只允许预声明的visual字段变化。验证编译前XML和编译后物理字段：机器人/物体尺寸、质量、惯量、关节、力限、摩擦、solver/dt等不变。改变source/goal属于layout，改变home属于reset，不应混进“纯视觉不变性”比较。使用现有摘要函数，不重建一套资产管理系统。

继续运行原像素/投影检查；额外用geom-ID分割或投影-遮挡诊断，确认检测的确是cube/goal而不是背景。分割/真值只用于离线验证，不能成为策略输入。若与原检查不一致，保留拒绝与原因，不放宽旧门追求通过。

纯视觉兄弟应有相同初始物理状态和机械参数。若未来做轨迹重渲染，必须有完整状态、相机和模型记录；没有就不合成同步RGB标签。视觉兄弟仍是同一几何/轨迹支持，不计成新的独立物理示范。

## 5. 场景分区与组合留出

### 5.1 数据单位

至少区分：

- `geometry_id`：忽略外观和任务的实际源/目标/yaw布局。
- `lineage_id`：同原始父场景、近重复/派生布局及恢复分支的谱系。
- `scene_variant_id`：geometry＋visual＋reset＋camera profile。
- `task_condition_id`：scene variant＋T1/T2＋推理noise（仅评价）。
- `episode_id`：一次真正的物理尝试；另有状态窗口/重渲染版本，不能冒充新的episode。

按lineage/geometry先分区，再派生外观、两任务、H40窗口与恢复。精确hash和原2mm/2°近重复检查都要做；外观变化不能绕过几何排重。近重复可传递，跨分区的连通谱系整体归组，不只贪心比较相邻文件。2mm/2°是最低排重检查，不是独立性或OOD保证；还须报告TRAIN到DEV/TEST的最近邻距离分布和各分层占比。

### 5.2 最小准备人口，不是采集/评估订单

首轮元数据提案：12个已有TRAIN候选几何保留来源/TRAIN身份，另外准备12个全新G_DEV几何和12个全新G_TEST几何，各自四个布局层均衡。合计36个几何只是设计规模，不能宣称已采36条示范；每几何有两任务，外观按需要派生，不自动做全笛卡尔积。12个TEST组仅适合作为小型研究验证，不能承诺充分统计功效。

G_DEV/G_TEST必须排除全部已知训练、C1父/分支、PV1人口、原DEV/旧N1确认、现有LOCKBOX及所有派生外观/相机版本。旧候选只可TRAIN，不因改名变成未知测试。不可读取的私有排除清单报告`EXCLUSION_INDEX_INCOMPLETE`，不声称零泄漏；不要重拷贝整库，只用现有私有索引和必要hash/几何元数据。

原始PV1的DEV48/LOCKBOX80完全保留；新GEN分区不覆盖它们。新命名空间如`GEN1/G_TRAIN/...`只是一项设计，原native validator不认识时必须增加窄范围适配，不修改旧白名单、伪装旧ID或把TEST映射为TRAIN来过export。

### 5.3 组合留出的例子

可以在新GEN设计中预先留出：`+Y/FAR×V2`和`-Y/NEAR×V1`。TRAIN和G_DEV都不包含这两种组合，但各方向、距离档和V1/V2单独在其他组合出现。G_TEST使用全新几何，在同一测试几何上配对V0与指定未见组合。

报告分开写：新布局上V0的结果、同新布局上视觉扰动的变化、未见组合结果。这样不会把测试新几何的困难全归因于外观。所有任务/噪声/外观兄弟在统计中按几何组聚类，不能按渲染图片数增加独立样本量。

如果已有replay/COV1训练已经覆盖这些组合，必须重新依据**真实训练暴露全集**选未见组合，或将主张改为“新布局下指定外观鲁棒性”。不能只检查新GEN目录就宣称组合从未见过。

## 6. SceneSpec与最小代码结构

### 6.1 SceneSpec示意（设计，不是已有CLI或实际资格）

```yaml
schema: GEN1_SCENE_SPEC_V1
study_split: G_TRAIN
geometry_id: GEN1/GEOM/000
lineage_id: <original-parent-or-new-lineage>
source_population_sha256: <actual-source-hash>
layout:
  generator_version: <verified-profile>
  source_xy_m: [0.45, -0.08]
  goal_xy_m: [0.45, 0.08]
  cube_yaw_rad: 0.0
  direction: +Y
  distance_bin: <derived-by-frozen-rule>
visual_profile: V0
reset_profile: LEGACY_ABOVE_SOURCE
camera_profile: ORIGINAL_DUAL_RGB
tasks: [T1, T2]
physics_profile_sha256: <frozen-current-profile>
codec: R0_EEF10_H40_PREFIX8
qualification:
  metadata: NOT_CHECKED
  static_ik_collision: NOT_RUN
  actual_dual_rgb: NOT_RUN
  dynamic_teacher: NOT_RUN
  raw_and_codec_audit: NOT_RUN
```

所有数值有限、单位明确、alpha固定、四元数/旋转表示检查齐全。将study metadata保持在sidecar/审计层；模型只看到原双RGB、原任务文字及合法实际EEF/夹爪状态。不能加入scene ID、split/family、目标真值、Teacher stage或未来动作信息。

### 6.2 建议实现位置

如已有等价模块则复用，不照目录重新复制。以下新路径/命令均为待实施设计：

```text
scripts/pi0294/generalization/
  spec.py          # 纯schema、单位、白名单与原card转换
  population.py    # seed子流、有限候选池、split/lineage与组合留出
  validation.py    # 参数不变性、已有静态/FOV/codec适配
  preview.py       # 只读取固定scene，调用已有双相机并生成gallery
  cli.py           # 接现有标准入口的薄包装，无新调度器
configs/pi0294/generalization/gen1.yaml
scripts/pi0294/test_gen1_*.py
```

复用`task_card`/指令常量、`xml_scene`、`qualify_engine`/`rendered_field_of_view`、原Teacher和collector、`committed_packet`/`codec_packet`及现有独立raw audit。不要调用带旧PV1总量、旧deadline或128组硬绑定的run后直接改计数；抽取/调用共享纯函数，新scope通过既有入口窄绑定。

模型与相机环境、MuJoCo、CUDA等锁定当前可复现版本，不以新场景准备为由升级依赖。第三方私有mesh、完整数据、模型权重和密钥不进入Git。

## 7. Codex阶段任务与验收

### M0 — 复用地图与冻结输入（纯读取）

交付`REUSE_MAP.md`：列出已有函数、真实输入/输出、各population范围、已有候选状态、相机与指令限制、活跃run保护。核当前GitHub/Issue11/47和本地owner，不从过期状态推断任务已停。先识别实际阻断，不能开始通用环境重写。

验收：能指出每项功能复用哪里，哪些仅为设计；明确COV1与legacy不同范围、现有home依赖source、当前红/蓝语义。发现已存在模块直接引用。

### M1 — 纯元数据生成、分区与回归（无MuJoCo/渲染/模型）

实现SceneSpec、有限候选生成、seed子流、显式split/lineage、全历史排重和组合表。只生成清单及静态数值报告；任何engine import应在此模式被拒绝。可用synthetic fixtures验证规则，但必须标为fixture，不能冒充私有全排除库资格。

交付`SCENE_SPEC.schema.json`或等价类型定义、`GEN1_SCENE_MANIFEST.jsonl`、`SPLIT_REPORT.json`、`FACTOR_COVERAGE.csv`、单元测试结果。文件名为目标输出，不代表本计划发布时已经存在。

验收：相同seed重复输出相同；打乱枚举/并发序不改名单；换visual不改geometry；T1/T2同场景；全部历史排除有可核对来源；G_TEST和holdout combinations未入TRAIN/G_DEV。

### M2 — 隔离静态构建与真实双相机预览（零积分，但仍须实际可用资源）

先在TRAIN中固定6个几何，覆盖四方向/距离层，比例可用+Y近2、+Y远1、-Y近1、-Y远2，并在任何预览结果前固定具体ID。先V0与V1：6几何×2外观×2相机=24张初态图，外加T1/T2状态一致性回读。构造次数和内存按实际记录，不等于24个物理episode。

复用原静态IK/碰撞sweep与FOV规则，验证模型time和积分计数为0。preview只调用固定记录，不触发Teacher rollout、GR00T或背景自动下一阶段。动态可行性明确`NOT_RUN`。

交付轻量`SCENE_GALLERY.html`、必要缩略图、每场景静态报告与layout/visual/reset物理差异审计。每张图注明ID/seed/profile/相机、static状态和动态未执行。完整TEST清单保持封存；测试静态筛选只按冻结规则自动运行，不能人工看TEST图后调视觉/范围。

验收：红cube和蓝框真实可见，投影与分割诊断一致；T1/T2物理/双相机初态相同；纯视觉变化不改变机械参数/初始物理；原基准卡通过旧回归；失败不靠调评分/安全门消除。

M2不能使用正在运行Eval的GPU进程、改其环境或抢占其CPU/IO。渲染需要GPU上下文时按既有隔离入口处理；无合法空闲范围则止于M1，不把“现在没采集”写成工程失败。

### M3 — COMMON_HOME资格（新分布的独立准备，不改旧Eval）

用TRAIN有限home候选做共同静态验证，选定后冻结；验证位置不从source/goal/task派生，初态两任务一致。已有source-relative home作为显式回归profile保留。

交付`RESET_PROFILE_REPORT.json`，结论必须是`COMMON_HOME_STATIC_QUALIFIED_PENDING_DYNAMIC`或明确阻断。不能仅有静态PASS就声称模型已能从home完成任务。后续任何训练对照两臂使用相同reset设计。

### M4 — 小规模Teacher动态pilot（A/B结果审阅及有效执行范围之后）

首批建议只对M2预定6个TRAIN几何的V0各执行T1/T2一次，共12次首次物理尝试；不是12次必须成功。此规模是后继建议，不是自动授权或必须预付的预算。当前B24运行中不并行争用资源，也不沿用已过期的采集窗口。

保留原有限力矩/接触/S1/32秒评分。所有首次尝试计入分母。Teacher可用真值生成命令，学生输入不可。一个场景失败后不自动重抽直到成功；若某层系统失败，定位任务生成器/范围/重置问题，再决定是否修一个版本。

对纯视觉变体先证明物理相同，不为每种颜色重复执行整条Teacher；只有具备完整状态记录、同步重渲染与标签一致性验证时，才可派生相关视觉数据。否则按真实新episode记成本，不能伪造。

交付12条件attempt表、成功/真实拒绝/工程缺失、真实视频与raw索引、事件时间线、实际成本和下一步结论。不在看到结果后规定“必须≥某个成功率”作为追分目标。某任务/层无合法成功支持则不得宣称对应数据可用。

### M5 — 数据导出与本轮准备终局（不自动训练）

成功示范必须同时满足完整任务、安全、观测/已提交命令/时钟一致、独立审计和split有效。失败保存为failure bank，不混入成功BC。

沿原实际state10和640条已提交命令构建H40，完整32秒时为601个合法窗口；601窗口不是601条独立示范。记录事件在Prefix8与H40中的实际覆盖、按parent的有效支持，避免再把大量重复窗口当规模。不能用未来achieved状态代替Teacher命令，也不能把未执行的学生尾部当作已执行专家监督。[E5]

R0/EEF10/相对SE3/rot6d沿现有codec逐窗口检查。状态超范围、反归一化裁剪及物理误差均显式报告；不能静默refit统计或换R1来让GEN数据通过。确需R1时列为独立表示实验，不改变当前A/B。

最终交付数据卡、manifest、全部attempt和失败、raw/codec报告与实际支持数量。只有此后、结合完整A/B结果，才选择一个新的匹配微调变量。首批数据没有形成可信支持时停止扩采，不自动生成几百条或重训4096步。

## 8. 最小测试清单

Codex至少实现并明确区分纯CPU与原生测试：

1. 同seed/配置稳定，进程/枚举顺序无关；视觉seed不影响几何。
2. 非法单位、NaN/Inf、未知字段和非白名单physics变化拒绝。
3. 老卡默认构建保持原XML/物理语义，现有回归不退化；不更新旧冻结源码hash以掩盖变化。
4. 几何近重复忽略外观/任务，跨split谱系拒绝；缺历史排除索引不能PASS。
5. T1/T2配对初态与RGB一致，任务文字保持原常量；policy不接收真值/ID。
6. COMMON_HOME不从source/goal推导；无可行共同home明确阻断。
7. red/blue语义不变，背景不得冒充目标通过可见性检查。
8. render/static零积分，构造/关闭资源可审计；无自动rollout或模型加载。
9. 原生重置可复现；渲染可视差异、物理不变分别测试，不把跨驱动RGB差当动力学差。
10. TEST/DEV不会进入export/train，恢复/重渲染兄弟沿parent继承split。
11. 真实命令标签、601窗口、Prefix8/H40事件统计和原codec误差检查一致。
12. 新配置不改变活跃B24/原DEV48/LOCKBOX，运行中capsule只读；所有失败和未知效果保留。

不以“新增多少单元测试”为成果指标；完成指标是每一项准备状态有真实证据、能被现有collector读取、且没有污染当前研究。

## 9. 后续评估如何接入（本次不启动）

G-layout：全新几何、V0、同一冻结reset；G-visual：同一全新测试几何配对V0/V1或V2；G-compose：预封存未见组合；G-language：同场景双任务均须满足各自原谓词。先按任务/层给完整成功和拒绝，按几何聚类，不报告单一平均分掩盖弱项。

纯视觉鲁棒性与新布局能力分开；训练见过全部因素不代表见过所有组合。COMMON_HOME若改变，需要两臂共享；旧AB只作历史参考，不能跨reset/相机/codec变更直接归因数据。新场景准备通过不等于模型泛化通过。原PV1发布标准与DEC1固定CORR64格另行保持。

## 10. 可直接转给Codex的任务指令

> 读取本固定提交计划和根AGENTS、START_HERE、当前科学/维护Issue及真实owner状态。先实施M0/M1：复用现有ARX5源域、scene_cards/scene/Teacher/export，建立GEN1独立场景配置、可复现seed子流、严格谱系分区及纯CPU回归。不要改当前B24、旧PV1人口、R0、安全评分、WBC或时钟；不要新建调度器/账本。新场景先只变布局与桌面颜色，红cube/蓝目标和相机物理不变。先核COV1与legacy范围差异，保护所有历史排除人口，明确源码配置hash与真实reset状态的差别。用已有initial_tcp准备独立COMMON_HOME，不能默认source-relative起点已证明视觉泛化。M2按隔离合法范围只生成6个TRAIN几何的双外观双相机初态gallery与零积分静态报告；不能抢现役Eval资源。M3只做TRAIN共同reset资格。当前不采集、不训练；M4的12次Teacher pilot与M5导出须在完整A/B审阅及有效执行范围后分阶段进行。所有新CLI/API在实际实现前标NOT_IMPLEMENTED，fixture不冒充原生PASS。每阶段提交精确改动/测试/失败/下一项最小阻断，普通fast-forward发布、回读验证，私有资产/raw/权重不入Git；不得热改运行capsule。

## 证据与研究参考

所有仓库源码事实固定到本文审阅SHA；后续owner须按实际部署版本复核，不自动替换正在运行版本。

- [E0] `START_HERE.md`、`docs/handover/2026-10-25/V4A_A24_B13_PAIRED_INTERIM_RESULTS.md`。
- [E1] `scripts/pi0294/scene_cards.py`、`scripts/pi0294/pv1/plan.py`。
- [E2] `scripts/pi0294/scene.py`、`scripts/pi0291_arx5_data/scene.py`、`scripts/pi0291_arx5_data/config.py`。
- [E3] `scripts/pi0294/pv1/population.py`。
- [E4] `docs/handover/2026-10-25/POST_EVAL_MULTISCENE_GENERALIZATION_PLAN.md`及`V4A_FAILURE_ATLAS_TRAIN_COVERAGE_AND_CANDIDATES.md`。
- [E5] `scripts/pi0294/pv1/export.py`。
- [E6] `reports/PI0294/design/parallel-science-20261022/V4A_FAILURE_ATLAS_AND_TRAIN_AUDIT_20261025_TEACHER_CANDIDATES.json`。
- [R1] LIBERO，NeurIPS 2023，https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html 。只参考任务/知识转移拆分，不导入整个平台。
- [R2] MimicGen，CoRL/PMLR 229，https://proceedings.mlr.press/v229/mandlekar23a.html 。只参考示范适配和实际执行验证，不承诺论文收益。
- [R3] MuJoCo官方状态/复现文档，https://mujoco.readthedocs.io/en/3.2.0/computation/index.html ；当前文档 https://mujoco.readthedocs.io/en/latest/programming/simulation.html 。实现以项目锁定版本的真实API为准，不因文档更新升级引擎。

本次发布只交付设计，不声称已实现模块、已渲染场景、已取得Teacher数据或已证明泛化。

## 后续实际接续：M0

[M0复用地图与原候选范围核对](gen1/REUSE_MAP.md)已完成纯源码／元数据检查。12个旧COV1 V1候选几何均通过原几何validator，但其中6个NEAR会被现有原评分前置距离门拒绝；历史静态资格保留，不直接动态派发、不静默换成修订版。GEN全历史排除索引尚未完成；M1未实现，M2–M5未运行，现役B24不变。
