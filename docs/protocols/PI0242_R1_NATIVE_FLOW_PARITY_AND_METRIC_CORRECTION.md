# PI0242-R1 — direct-flow 特征单次处理、native parity 与指标更正

日期：2026-09-21（Asia/Singapore）  
PI decision：`GO_SCOPED_DIAGNOSTIC_CORRECTION; TRAINING_NOT_AUTHORIZED`  
发布状态：`AUTHORIZED_DIAGNOSTICS_NOT_STARTED`  
授权 token：`PI0242_R1_NATIVE_PARITY_GO`  
审核基线：`b3a6c92012ef6949bb08dc4a4ed41dd9760a907e`  
上游 GR00T：`51d4c89f72fda44cbf77285c6a8114b52676b8a1`  
Issue #11 是科学账本；PR #77 只放 pointer、不 merge。

## 1. PI 裁定：保留负结果，暂缓 local-flow 机制归因

读取依据：PI0242 canonical `5755139669`、pointer `5755140350`、上述基线的 `reports/PI0242_TRAIN_INFER_BRIDGE_CLOSURE.md`、`scripts/pi0242_train_infer_bridge.py`，以及同基线的 `scripts/stage26_h1_c1_fixed_noise_local_flow.py`；交叉核对上游 pinned `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` 的实际 forward/get_action/feature-processing 调用链。

保留：PI0241 fixed16 自由生成失败；PI0242 的 16 个原 stock outputs、原 SafetyGate 16/16 REJECT、旧 checkpoint/噪声/输入身份与真实 H100 job `625549` 记录。未发现这次重复特征处理发生在 stock 原始输出产生之前，因此不能借本问题撤销所有既有模型负结果。

暂缓：把 PI0242 direct-flow 的 0.25–0.45m oracle XYZ 重构误差用于批准下一训练配方，以及基于这些量排除 local-to-free-running gap。标记为 `DIRECT_FLOW_ATTRIBUTION_PROVISIONAL_PENDING_NATIVE_PARITY`。这不等于已证明修后 local flow 会好；需要量化实现偏差影响。

原 stock 的 C_T/C_I 差异可继续作为所测四窗上的描述性证据：final1000 XYZ median 0.213020m vs 0.209237m，所有轨迹仍拒绝。它既不证明预处理是主要根因，也不保证其他样本无此问题。

本轮只授权修复诊断实现和重测受影响的冻结矩阵，不授权新的优化器更新、训练、loss/LR/精度/模型/数据干预。selected_checkpoint=NONE。

## 2. 可定位的源码问题

### 2.1 hook 保存了可变对象，direct-flow 再处理一次

当前 `_stock_query()` 注册：

```python
model.backbone.register_forward_hook(
    lambda _module, _inputs, output: backbone_outputs.append(output)
)
```

这里保存的是同一个 output 对象，不是 raw tensor snapshot。上游 `Gr00tN1d7.get_action()` 将该对象传给 action head；`_encode_features()` 调用 `process_backbone_output()`，后者执行：

```python
features = self.vlln(backbone_output['backbone_features'])
features = self.vl_self_attention(features)
backbone_output['backbone_features'] = features
return backbone_output
```

其后 PI0242 `p1()` 再执行：

```python
processed_backbone = model.action_head.process_backbone_output(
    stock['backbone_output']
)
```

然后送入需要已处理 features 的 Stage26 `_teacher_forced_local_flow()`。因此在已审核源码执行路径中，stock 使用 F(h)，direct helper 接收 F(F(h))。这不是等价的 native training conditioning 合约。F 若在实际权重/配置上近似幂等或 identity，数值影响可能很小；不能单凭控制流认定它解释全部 20cm 误差。必须记录实际 VLLN/self-attention 配置及 h、F(h)、F(F(h)) 的差异。

修复只能在诊断捕获/数据生命周期层做：raw snapshot 与 processed-once snapshot 分开，不能修改上游模块使其变 identity，不能直接更换权重/processor。

### 2.2 angular-speed sidecar 公式不对

当前 `_physical_profile()` 先算 pred 对 expert 的各时刻姿态误差，然后：

```python
angular_speed = rotation_error[1:] / DT
```

这不是预测轨迹角速度。应为：

`omega_t = geodesic(R_pred[t], R_pred[t+1]) / DT`。

专家角速度另算。原 pred-vs-expert SO(3) RMSE 仍为误差指标，不变。原 SafetyGate 使用独立正式实现，不按这个 sidecar 判定；不因此改写原 gate 决定。

### 2.3 审核范围与实际做过的测试

GPT 对 GitHub 与上游 pinned 源码做了静态交叉检查；没有 SSH、真实 checkpoint load、项目 forward、Hopper 作业或原 NPZ 数值重算。

本地执行了两个合成 CPU 机制 fixture（不是项目实测）：(1) 可变输出 mapping 经非幂等 F(x)=2x+1，stock 为 [3,5]、旧 direct 为 [7,11]，独立 snapshot 的 direct 与 stock 相同；(2) 固定0.2rad姿态偏差、轨迹本身不转动，旧 sidecar 报4rad/s，正确相邻姿态速度为0。两项通过，只证明故障机制与测量公式，不能声称项目 pytest 或真实 native parity 已通过。

## 3. 本轮范围、数据和计数

沿 PI0242 原 bridge4，不重新选样：A/episode1013、B/2485、C/5442、D/8247；source IDs 以原 sealed run card 为准。两个原权重 initializer 与 PI0241 final1000；两个条件 C_T/C_I；同一原全132D噪声及 hash、原 t=0/q10/q50/q90、seed22301、H40/StateV1/dt/frame/TCP、原 active mask、精度、stock 四步 sampler 不变。

科学矩阵仍为 4×2×2：

- 最多16个 fixed stock capture queries，用来获得正确 raw/processed-once 特征与原生 trace；如已有足够且身份完整的快照可复用，不重新调用。
- 64个更正后的 direct-flow measurement cells，使用原 PI0242 的 frozen x_t 和离散 t 定义，不新增 probe。
- 为防止新 helper 再次偏离 native，在上述每个 model/window/condition 的 t0、q50 各做一个 native-forward/helper paired control：32个 control pairs，最多32次官方 action-head.forward 和32次 helper 前向。它们是实现一致性对照，不加入64-cell机制统计。
- 因此 head 调用规划上限：stock 内部64次 + corrected measurement64次 + parity controls64次 =192次；backbone 默认最多16次，复用不可变 raw features 不重复编码。记录实际 calls，包括技术失败；这些是唯一合法 cells/control pairs 的规划，不是无条件允许扩大测试集合。
- h→F(h)→F(F(h)) 的特征诊断另计为 feature-transform calls，不冒充零计算或 head calls；不改变模型参数。

同设备、同环境完成本轮配对。旧 H100 与新 Corelab 的 raw SHA 不同不自动视为失败；新结果明确标新 backend，不把它冒充旧 tensor。必须验证当次 pairs 的条件、noise、dtype、mask、time 与权重身份。

工程修复无任意重试/job/token数上限，但不得因此增添新seed/样本/权重/采样器步数或效果驱动重跑。已完成的 first-valid correction cells 固定；同一阶段只有一个 active 执行。

## 4. R0 — CPU 修复与回归，先不加载项目模型

继续修改现有 `scripts/pi0242_train_infer_bridge.py` 和已有对应 tests，不另造框架；保持旧 protocol 和 closure 原文不动。

1. hook 当场保存独立 raw container 与每个 tensor 的 `detach().clone()`/等价独立快照，包括 feature、attention mask、image mask及实际其他字段；保留 dtype/device，必要时无损移到 CPU。不能只 detach 或保存外层引用。hook 返回 None，不修改 stock 正在使用的 output。
2. 建立 raw 与 processed-once 两个清晰字段。每个 direct/native branch 从 raw snapshot 的独立副本开始，F 只执行一次；或复用经证明是一次处理的 immutable snapshot。不要调用 F 已处理的 stock output。任何测量 double-F 影响的对象必须是独立诊断副本，不能被后续模型路径再使用。
3. feature provenance 记录 raw/once/twice 的 shape/dtype/hash、object/存储别名检查与 transform count；容器身份与内容身份分开。将“共享全局临时hook”约束在单进程串行作用域，finally恢复。
4. 修正角速度侧报，调用项目已验证的 row-rot6d→matrix/geodesic；用原库存 NPZ 零forward重算侧报，更正文件独立保存，引用被更正 artifact SHA。
5. 不改 gate、SO(3)误差定义、目标距离、XYZ RMSE或统计分母。输入mask/非法旋转/NaN仍严格记录，不裁剪成PASS。

至少覆盖：非幂等变换复现原bug；原对象后续赋值与tensor原地修改均不污染快照；raw/once不会串用；hook观察不改变toy stock输出；identity/idempotent条件不谎报数值影响；静止但相对expert有姿态偏差时角速度为0；已知恒定角增量/dt与expert速度分开；mask/shape/noise配对；native/helper缺失不得自动PASS。

只复用原 P0 证据与run card，不重算10880专家、旧256 G4或重新训练。没有相关 dependency/input 改动的检查引用旧hash。提交前运行完整CLI小fixture，不能只compile。

## 5. R1 — 固定输入的原生路径验收与更正矩阵

### 5.1 先记录真实运行身份

核验原权重完整shard inventory、实际上游 `__file__`/SHA 与 pinned 源码一致；如果远端代码不同，先封存真实差异，不把 GitHub 静态调用链强加给另一个runtime。所有观察器/noise注入/条件构造以同一revision运行。

记录实际 `use_vlln`、`vl_self_attention`配置和各模块mode，不用dataclass默认值推断checkpoint配置。测出 F(h) 与 F(F(h)) 实際tensor差距。即使差距为0，也保留控制流修复并记 `DOUBLE_PASS_NUMERICALLY_INERT_ON_MEASURED_INPUTS`；不得自动宣称原direct结果因此已经与native一致。

### 5.2 两种 parity，不混为一谈

**A. helper 对官方 training forward 图的实现一致性（必须闭合）。**

在 model.eval()、无梯度条件下调用真正的官方 `action_head.forward`，输入 raw features 独立副本、原 state/action/mask/embodiment。仅以受控、可恢复的方式提供固定noise/time；保持native其余计算不变。捕获它实际 action_encoder 收到的 x_t、整数t，以及 action_decoder 返回的预测velocity；记录native真正生效的mask、attention arguments、dtype和feature快照。

再让 Stage26 helper 使用**这些实际捕获的相同 x_t、整数t、一次处理features、state/embodiment及所有attention参数**，比较raw velocity，不只比较最终gate。先检查所有输入相同，再比较输出。优先逐bit相同；浮点差异必须逐层定位并报告 max_abs/RMSE/ULP，不能看结果后扩大容限。未闭合则 `IMPLEMENTATION_PARITY_UNRESOLVED`，它是需自主修复的诊断问题，不是新训练失败。

native noise/t在BF16下的构造和旧PI0242先CPU-float32构造x再cast可能不同；因此native/helper control使用native真实捕获输入，而64-cell correction保留旧probe定义。两者分别记账，不能把各自不同的x/t导致的差异伪称图不一致，也不能把control标量loss混入旧fixed-probe MSE。

**B. stock第一步对局部图的关系（测量，不设错误的强制等价）。**

记录同noise条件下stock t0的实际x、features、state、t、attention参数/flags与direct图的差异。上游native training图可能传 `encoder_attention_mask`、`return_all_hidden_states=True`，stock调用未必相同。不能未经核对就要求两者velocity必等；如果签名或图有差异，保留并解释，不擅改stock mask/flags来追parity。已测native/helper一致性与实际train/sampler结构差异分开。

### 5.3 执行与发布更正值

对应cell的native/helper control通过后完成其原四个t的更正direct measurements；不因结果好坏中止剩余固定矩阵。新stock capture仅用于一致的本轮特征与配对参照，不替换旧16/16负结果，不变成重复选模。

每个cell保存：raw与once features身份、原noise/x/t/mask、predicted velocity、各action组误差、implied endpoint、decoded物理误差、对应original cell的locator/hash，另有native/helper control对照。保持旧epsilon+v_pred oracle指标的命名和含义：不是policy，不执行WBC。不得对任意off-trajectory sampler点伪造oracle target。

修正角速度独立写 `metric_correction` sidecar；SafetyGate原始结果不变。新sampler结果仍保留所有REJECT，不重新挑样。输出原direct→更正direct的逐cell差值、q分层和condition/model配对，而不仅新的median。

## 6. 最终判定和下一训练决策边界

该goal终点为 `PI0242_R1_NATIVE_PARITY_REVIEW_COMPLETE — STOP_FOR_PI_SINGLE_RECIPE_DECISION`。

- 若native-parity成立且更正后的local-flow仍明显偏离专家尺度：可恢复 `LOCAL_OBJECTIVE_FIT_DEFICIT_SUPPORTED`，仅限已测支持。此时提交一个真正明确的single-intervention训练run card（指出唯一变化、假设、initializer/data/schedule/criterion），不要用“继续action-head适配”这种没有说明相对PI0241变化的措辞代替配方。
- 若更正后的local重构显著改善而stock仍差：原机制结论需要追加更正，报告local→stock关系与剩余mask/图差异；不自动扫采样器步数。
- 若double-F数值影响微小：如实保留该结果；不能为了让bug解释失败而放大影响。仍以native/helper验收和更正值决定归因。
- 若parity未闭合：停止扩大计算范围，修复已定位工程原因；确实需要更改科学图或未知来源才回PI，不把不明差异硬贴为模型欠拟合。

无论出现哪个结果，本文件不授权下一训练/backward/optimizer、loss/LR/精度/模型/采样器变更、CALIB/DEV-E/F、WBC/sim/DAgger、硬件或PR merge。旧PI0241负结果与selected_checkpoint=NONE继续有效。科学有效失败不能作为工程理由反复重训。

## 7. 资源、存储和交付

本地PC做最小代码/pytest；Corelab优先做CPU重算与无梯度真实集成。输入只需要原两份推理权重、bridge4条件与必要旧NPZ，不搬完整optimizer档案。

Hopper仅在Corelab实际显存/可用性有记录的不足时使用；同一原子paired campaign在同一GPU运行，H100或H200的选择以所需容量和既有站点规则为准，不因目录叫p1-h200就假定设备。保留一次合法job、不并发重复跑；普通路径/import/测试问题回本地/Corelab处理。

不新建monitor；不删除旧sealed权重/数据/evidence；新attempt输出独立，实际host/fileset容量只为当前新增证据检查。持久化cell/control ledger，因IO失败只恢复缺失结果；不再把所有结果只放内存等到最后写。

交付保持紧凑：

- 现有runner最小patch与CPU tests；
- 一份R1 run card/输入身份；
- feature single-pass provenance、native/helper parity与原stock关系报告；
- 原矩阵更正records/trace NPZ、角速度sidecar与旧新指标差值；
- `reports/PI0242_R1_NATIVE_FLOW_PARITY_CLOSURE.md`，列已保留结论、暂缓结论、纠正后结论、剩余未知及一个待授权的配方建议。

小报告/代码/inventory写GitHub；Issue #11先canonical更正，PR #77只pointer，不编辑/覆盖原PI0242 closure。记清raw/stock/direct/control/backbone实际计数、零backward/optimizer/WBC/受限数据访问。不得把本次文档发布说成项目修复或复测已经完成。

## 8. PC agent启动prompt

执行PI0242-R1，token PI0242_R1_NATIVE_PARITY_GO。完整读取本固定协议、PI0242原closure5755139669和审核基线b3a6c92012ef6949bb08dc4a4ed41dd9760a907e；不重开全库审计。

先在现有pi0242_train_infer_bridge.py修复可变backbone output引用导致的F(F(h))，用独立raw/processed-once snapshot绑定实际features；修正angular-speed sidecar为相邻预测姿态的geodesic/dt。CPU回归后复用原P0、两权重、四窗/两条件/noise/time，进入本R1固定native/helper parity和受影响矩阵更正。控制流bug的存在不等于已知真实数值影响，必须测量F(h) vs F(F(h))。

最多16 stock captures、64 corrected direct cells、32 native/helper control pairs。native/helper对照使用native真正捕获的x/t/features，不把cast或attention参数差异混成实现错误；不未经核对要求stock和training图逐bit等价。保留所有旧结果和first-valid更正值，零训练/更新、零WBC/sim/CALIB/DEV/F。

Corelab优先，工程问题自主修复、不设任意次数上限；不增加科学样本/seed/模型/采样步数，不在H200迭代普通pytest。完成后Issue#11发布精确更正与归因、PR#77放pointer，STOP_FOR_PI_SINGLE_RECIPE_DECISION。selected_checkpoint=NONE，下一训练仍NOT_AUTHORIZED。

## 参考

项目源文件：上述commit下scripts/pi0242_train_infer_bridge.py、scripts/stage26_h1_c1_fixed_noise_local_flow.py。上游模型源码：NVIDIA/Isaac-GR00T @51d4c89f72fda44cbf77285c6a8114b52676b8a1 的gr00t/model/gr00t_n1d7/gr00t_n1d7.py。

PyTorch官方forward-hook与Tensor storage文档只用于解释观察器语义，不升级项目pinned runtime：https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook 。
