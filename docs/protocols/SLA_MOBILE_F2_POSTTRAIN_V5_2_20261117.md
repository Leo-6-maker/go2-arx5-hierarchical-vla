# SLA 移动抓持后训练 V5.2：先辨识边界，再补接触状态

日期：2026-11-17（Asia/Singapore）。状态：**PLAN_ONLY / NOT_EXECUTED**。本轮用户要求审阅 PR #403、发布后续实验计划及 Codex 启动 prompt；本次发布不启动服务器作业。用户将配套 prompt 交给执行代理后，按本计划有界推进，复用 V5.1 的负责人直接授权纯仿真机制；不恢复第三方 READMISSION 签发前置，不把旧 allocation 自动延期。

配套：[机读设计](SLA_MOBILE_F2_POSTTRAIN_V5_2_20261117.json) · [Codex 启动 prompt](../handover/2026-11-17/SLA_MOBILE_F2_POSTTRAIN_V52_START.md) · [根入口](../../SLA_MOBILE_V52_START.md)。JSON 是声明性规格，不是现成的 runtime 配置。

## 1. 本轮要交付的 policy

当前已有可复现的 **SLA SHIFT7 → EEF/夹爪 H40 → 冻结 WBC** 移动抓持原型。下一目标是让同一个高层模型在 F1/F2 两族都可靠完成原任务，而不是继续提高已看过的离线合法率。

最终候选保持 M_FREE、普通 body/wrist RGB、自身 EEF 状态、语言、固定任务 frame、上一实际发出的命令与原生时钟；不引入 BASE 预测头、教师阶段、物体真值、自动 CLOSE/LIFT 或接管。最终交付包括冻结可加载模型包、普通 RGB 推理入口、原执行链、独立测试表和失败轨迹。

原成功定义保持：native grasp，物体抬升至少 20 mm，连续保持至少 1 s，任务方向底座净位移至少 80 mm。四项及综合结果分别报告。DEV32 要求至少 24/32 且每族至少 12/16；TEST40 至少 32/40 且每族至少 16/20，两者均 0 hard STOP。上述数字是**预注册验收目标，不是预测成绩**。

交付标签只写 `SIM_PAUSED_NATIVE_MOBILE_PICK_HOLD`。350 ms 连续实时、真机、导航、开门和整理书架仍需各自证据。

## 2. 固定审阅版本与已经取得的进展

- 文档发布基线 main：`6843163d641eb24a29c3b1557f868765346ba66c`。
- 审阅执行 PR #403：`dbf222b41cb666bff4dd5412b864d023aec69399`。本轮实际修复运行源码为 `c2b245cb6da251833cfc9cf9813f4e01e9da2d85`，不能把报告提交 SHA 当运行 SHA。
- SHIFT7 模型包：`ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb`。
- C20 statistics：`f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e`。
- Core 原终态结果：`ac62fcb13cea6b704e54ce2af7c0a41728e67e8b02eaae4ec963164408018f50`。

本次审阅核对公开源码及已提交结果/哈希清单；没有访问 Core 私有原始 RGB/动作文件、重放物理或独立重算全部 319 条轨迹。下述 D0 是执行端后续工作，不能写成此次已经完成。

| 阶段 | 已有真实证据 | 对下一步的含义 |
|---|---|---|
| C20 | 20 episode、1,376 窗，1,000 更新/32,000 访问；旧学生 0/13，12 次拒绝均夹爪速度 | 终末开度正确不等于机器人时间曲线可执行 |
| 数值路线 | cached/full × actual/highest 完整采样对照，四路线 CLOSE 均 0/16 | 不重做精度四路线作为主线 |
| 夹爪时序监督 | 匹配普通 FM500 CLOSE 0/20，辅助项 7/20 | 部分改善，未解决 |
| TIME500 | CLOSE 20/20，但 LIFT 12/20；内部 k26 曲率失败 | 夹爪表示与共享空间预测发生耦合 |
| SHIFT7 | 同 C20 父、fresh Adam、500/16,000；只移动参数槽；TRAIN96/96 | 不是继续 TIME500，也不是物理动作延迟 350 ms |
| SHIFT7 单模型 CN38 | 专家观测状态上 180/180，全为已知近邻布局 | 不代表学生滚动闭环或盲测 |
| PR #403 | 8 学生尝试：3 成功、4 科学失败、1 IPC 技术中断；F1 3/3 有效、F2 0/4 | 首次真实移动抓持；当前瓶颈转到边界余量与接触状态 |

项目名称为 GR00T+WBC；当前成功的上层父模型是 `haofuly/SLA_base`（Spatial-Language-Action，pi05 配置），不能写成原生 GR00T-N1.7 权重已获得同样能力。

### 2.1 分母及证据边界

18 原生请求 = 2 接口 + 8 专家 + 8 学生；319,518 tick = 102,654 初始化 + 216,864 任务。专家抓持/抬升/保持 8/8，综合移动 7/8；F2 一条 79.519 mm 未过 80 mm，保留失败。

学生 320 个输入/IPC 请求，319 次生成，316 完整 H40 通过，631 安装 ACK，315 完整 P10，3,154 确认原子，另 14 partial tick。`3154×25+14+8×5703=124488` 与学生总 tick 一致。H40 query 合法率 99.06%，而综合成功仅 3/8 全尝试或 3/7 科学有效条件，二者不能互相替代。

F1 实际覆盖 T0 三次、T1 一次；F2 为 T0–T3 各一次，不能把两族成功率差直接解释为控制了布局后的方向因果。C20/CN38、旧 DEV16 已查看部分、PR #403 及旧 RP8/JOIN10 全部按训练或开发材料处理。CN38−C20 的 18 episode 虽未参与梯度，已经用于模型选择，也不是盲 holdout。

## 3. 失败机制与本轮假说

### 3.1 H40：问题是首尾差分，不是删去后缀的理由

原 `_geometry` 使用 `v=diff(p)/0.05`、`a=diff([0,v,0])/0.05`。首尾检查等价于：

\[
\|p_1-p_0\|,\ \|p_{39}-p_{38}\|\le 0.25(0.05)^2=0.625\ \mathrm{mm}.
\]

| 原失败条件 | 最大加速度 m/s² | 首次位置 | 对应端点位移 mm | 内部最大 m/s² |
|---|---:|---|---:|---:|
| F2 T1 / query2 | 0.254252 | 尾端 k39 | 0.635631 | 0.118005 |
| F2 T2 / query12 | 0.251407 | 首端 k0 | 0.628517 | 0.124430 |
| F2 T3 / query21 | 0.294609 | 首端 k0 | 0.736523 | 0.117964 |

这是原 EEF 几何边界项，不能混写成 joint3、actual TCP 到第一点跳变，或旧 TIME500 的内部 k26 故障。reference transport 对 H40 平移加同一个常向量，不能放大这三条轨迹的相邻差分；desired/reference 重算一致符合代码。

专家 `TRANSLATION_SPEED=0.0075 m/s` 对应补零端点上界约 0.15 m/s²，理论上有余量。因此优先假说 H_edge 是：模型位置拟合没有充分保住端点导数余量，学生滚动状态使其暴露。D0 仍须核对真实标签分布，不能只凭理论上界宣布根因。

### 3.2 接触：EEF 合法不等于整机接触可行

F2 T0 在任务 22.228 s、tick 16,817 发生 `x5_link6` 腕部–目标方块接触，尚未双垫抓持；STOP 后积分 0。首次小穿透是保护触发时的记录，不能因此忽略碰撞，也不是最坏停止墙钟延迟证明。

移动模式的固定 root IK 证书原本就是 `shadow_only`；完整 H40 通过不等于动态全身碰撞预测通过。F2 Teacher 有 +8 mm 抓取目标修正，学生没有 family/phase/goal 真值。H_contact 是学生接近/下降时的姿态、净空或实际跟踪/状态覆盖不足；目前不能归结为某一项。

第三个风险是 F2 移动裕量：通过的匹配专家仅约 82.9–83.8 mm。边界修复后仍可能出现“抓持成功但未到 80 mm”的新首失败，须按原指标记录，不改门或 WBC。

## 4. D0：复用存量证据，零新生成/积分归因

在一个有界 CPU 工作包内读取全部 320 输入身份、319 已保存动作、逐 query 消费、8 匹配专家及碰撞前最后 5 s 的可用状态。先核缺件和哈希，不因材料不足反复启动原生世界。

输出 `D0_BOUNDARY_CONTACT.md/json`，至少包括：

1. 每 query 的首/尾/内部加速度、距原门余量、desired/reference 一致性、夹爪速度/上一实际命令接缝、底座已移动距离。绘制按 family/query 的分布；316 条已通过轨迹也纳入，不能只挑三条失败。
2. 全 C20 1,376 条专家标签首尾余量。若标签本身与本假说冲突，先查合同而不是直接训练 B2；旧标签不覆盖。
3. 腕部–物体/桌、pad–物体最小距离，TCP 位置/旋转、底座与臂贡献、desired/reference/actual 三者误差。匹配专家按几何接近程度对齐，不能仅按 query 编号比较。
4. 普通 RGB 中目标/腕部的可见性、阶段转换与夹爪事件；真值仅用于分析，不进入学生输入。
5. 三个边界拒绝、一次接触及 IPC 分别列因果证据、尚缺证据、对应可证伪预测。未知 controller-selected reference 保持 UNKNOWN。

全状态反事实需要 solver/controller 历史及完整 WBC 状态可验证；只恢复 qpos/qvel 或仅做 FK，必须标为静态几何诊断。不同状态的历史专家 H40 不得直接作为学生失败观测的 BC 标签。

若 D0 证实的是坐标、时间或读取工程错误，先做最小修复和受影响检查，形成新身份；不为维持计划形式强行做无关 loss。若没有支持新干预的证据，交付具体否定结论，不无限增加候选。

## 5. E1：只改边界目标的匹配后训练

### 5.1 三个模型和共同条件

| 模型 | 起点 | 新训练 | 唯一用途 |
|---|---|---:|---|
| B0 | 冻结原 SHIFT7 | 0 | 同期冻结对照 |
| B1 | 同一 SHIFT7 + fresh Adam | 普通 FM32，500/16,000 | 控制新增训练量 |
| B2 | 同一 SHIFT7 + 相同 fresh Adam | FM32 + endpoint loss，500/16,000 | 辨识边界目标效果 |

两臂 C20 20 EP/1,376 窗、原统计、SHIFT7 codec、28 active 数组/52,153,376 参数、SigLIP/非 LoRA 冻结、batch32/micro4、state-token dropout .25、精度/采样/访问/noise/time 全匹配。保持 `cached_actual` 和 10 步采样，不加 CN38、不换统计、不改夹爪布局、不解冻更多参数。

精确学习率复用 `mfree_training_v1.learning_rates()[:500]` 并固定数组 SHA：它是 1,000 步日程的前 500 步、50 warmup，不误用另一个 25 warmup/500 日程。AdamW β=(.9,.95)、eps=1e-8、weight_decay=1e-5、clip_norm=1，沿用已实测实现。只选固定500端点；250可保存恢复点，不按成绩挑 checkpoint。

原 SHIFT7 模型 lineage 累计 1,500/48,000；每个后继新增 500/16,000 后为 2,000/64,000。两个分支的更新不得相加成一个模型的训练经历。

### 5.2 唯一新增损失：物理端点余量

现有 flow 定义为 `Xτ=τ·noise+(1−τ)·actions`，`v≈noise−actions`，故 clean estimate 为 **`Â=Xτ−τvθ`**。先按原 q01/q99 与 `+1e−6` span 逆归一化 XYZ，再计算两端的相邻位移/.05² 的 L2 范数 `a_start,a_end`。

\[
L_{edge}=\frac1B\sum_b\mathbf1_{\tau_b\le.5}\frac12\sum_{j\in\{start,end\}}\left[\max\left(0,\frac{a_{b,j}-.20}{.25}\right)\right]^2,
\qquad L=L_{FM32}+\lambda L_{edge}.
\]

B1 λ=0，B2 **λ=.1**；.20 m/s² 是固定训练余量目标，等价 0.50 mm/10 mm/s 端点差分。执行端依旧原 .25 门。常数是本轮设计选择，不是假装经过调参的最优值。不能看 DEV 调 λ；不得对模型生成动作做裁剪、平滑、时间拉伸或末端重建。

辅助项只直接依赖 XYZ，但共享参数仍可能损伤夹爪。普通 FM 及动作进度回归必须保持，避免不动获胜。clean proxy 不是完整生成轨迹，最终结论只用完整 10 步自由生成和真实闭环。

必要的实现验证只覆盖：oracle-field clean 恒等式；物理单位与坐标不变性；零轨迹 loss/gradient 有限；超限梯度方向；λ0 复现普通 FM；tau mask 按全 batch 平均且 micro4 累积等价；step0 两臂权重/新 Adam/RNG/访问/LR 匹配；冻结指纹及独立恢复。零范数处采用明确的数值安全导数，原硬 evaluator 不改。

### 5.3 固定离线诊断银行

- 复用原 TRAIN96 和 CN38 180，均是开发诊断。
- 从 #403 已生成输入选 `ROLL96`：每族48，强制纳入三条拒绝及接触前最后10个可用 query；其余按 case 均衡、时间等距、ID 排序填充，去重，不复制窗口凑数。在 B1/B2 生成前封存名单与 noise。
- 每个新模型固定 372 条完整预测（96+180+96），另16条独立普通 RGB 恢复重放。B0 的相同输入/噪声旧预测可复用，新增16条恢复重放单列。
- 报告原完整 H40、端点/内部 margin、CLOSE 20与36事件、上一命令接缝、EEF 偏差与推进量；不得把 ROLL96 上模型新预测当作实际执行。

进入**开发性配对 screen**只要求身份、恢复、输入、codec 和原执行保护正确；不因 B0/B1 未修好已知失败就淘汰对照。明显格式/非有限输出是工程阻断；合法模型被原 H40 拒绝是实验结果。进入确认 pilot 则需原 TRAIN96 全通过、CN38 180全通过、3个已存拒绝输入完整生成均过原门、CLOSE无遗漏且原100 ms时机标准不退化。

## 6. E2：先完成三臂闭环，再选一个候选

### 6.1 配对开发面板：24 次学生请求

F1/F2 × T0–T3 各1，共8条件；每条件运行 B0/B1/B2，合计24。布局取已知开发支持域，使用**未执行过的预注册 reset/noise 身份**，不重新提交 #403 的原条件，不替换其失败。换 seed 仍属于同布局开发验证。

三臂共享相同 scene、初态生成规则、noise 银行、评分、初始化和任务预算。按条件交错模型次序，预先固定循环次序避免总是某臂先跑；一个物理 owner、重模型按实际容量串行。独立 world 的 post-init 状态差异需记录，不把相同 seed 自动当完整控制历史一致。旧专家在 scene/控制合同确实匹配时直接复用，不能匹配则明确不是同状态对照，不机械再跑8条。

主比较是 B2 对 B1 的 **8组配对综合成功差、F2差、hard STOP和原H40拒绝差**；B1 对 B0 说明额外训练的作用。输出8行三列逐条件表，不只输出总平均。n=8只作开发选择；不用显著性措辞，也不把 query 当独立样本。

### 6.2 预注册选择与确认 pilot8

候选须同时满足：完整可解释8终态、技术失败0、hard STOP0、screen至少6/8且每族至少3/4，以及上一节离线晋级条件。排序依次为综合成功数、较弱family成功数、较少H40拒绝、较低成功episode任务时间中位数；仍相同按 B0→B1→B2 的较简单方案优先。不逐 episode 切换模型。B0 原已有已存拒绝，因此不会因门设计被删去screen，但除非取得相应新证据，不满足离线晋级条件。

冻结唯一胜者后，做另8条件确认 pilot：F1/F2×T0–T3，各1，新的事前固定 reset/noise。保留 V5.1 原门（两族各至少一次综合成功、0 hard STOP）；V5.2额外要求8终态可解释、技术失败0、至少6/8且每族至少3/4。新增条件单独标 `V52_PROMOTION`，不能回改 V5.1 的 `pilot_GO=false`。

该 pilot 是冻结后的确认，不是独立布局泛化。无合格者进入 E3 的证据分支或结束有界试验，不按不同 seed 重投到过门。

## 7. E3：仅在证据支持时采集12次纠正示范

触发条件：完成 E1/E2 后，边界合法性有改善但接触/抓持/移动仍不足，或 D0/E2 明确支持学生状态覆盖缺口；必须给出具体状态与预测。若主要仍是未解释的边界失败，不自动借扩数据掩盖失效假说。

先从身份正确、结果完整的开发模型中，按上一节的排序键选择一个**开发父模型 P**并冻结；此处不套用部署/晋级资格门，避免无晋级者就无从纠正。P只是采集/比较起点，不代表获得部署资格。预登记12个请求：F2四布局×2 reset=8，F1两布局×2 reset=4。所有布局留在训练/开发支持域。首2个F2作为该12内的可行性检查；若二者均无有效纠正轨迹，先结束并交付具体高层 Teacher/几何障碍，不继续盲采。

优先从完整 fresh world 在 P 下滚入事前指定的安全接近状态，再在接触前切到合法 Teacher，记录真实 handoff、实际状态/参考与时间。也可在完整控制历史可验证时重放已存实际前缀。无法完整重现时，标明是新的实际纠正状态；禁止声称与旧失败状态完全相同。发生 STOP 的旧 world 不重开。

handoff限定在合法P10边界、夹爪仍开且接触前；验证accepted EEF reference、上一实际发出的width及其50ms时钟、当前native tick连续，不重置WBC/物理/episode时钟。学生期间没有Teacher滤波历史：允许显式声明的新Teacher初始化（包括零滤波初态），如实记录，不伪造继承。现有Teacher构造会清空滤波/close_start且默认开爪，须新增窄handoff入口并检查首H40/命令接缝；原安装、P10、TTL保持。首段真实专家P10消费后才接收该段BC，无法合法交接计入12请求的可行性失败。

这些是**教师辅助纠正 episode**，不是学生自主成功。只把实际检查并提交的专家最终 H40 配对到当时真实 RGB/状态；学生前缀及拒绝动作不作 BC。`supervision_mask` 与 `consumed_mask` 分开，合法当前H40计划可监督40点，未消费后缀不计执行/转移。失败/partial/hand-off片段身份不伪装成原完整五阶段episode。

最多12次，不“采到12成功为止”。进入数据比较至少取得8个安全完成的辅助episode（F2≥6且覆盖≥3布局；F1≥2且覆盖两布局），全部所用窗口通过真实输入/标签/codec核验。否则交付不足原因及已发生结果，不自动 C60/C100。

### 数据变量对照

| 模型 | 同一父模型 P / 同一 loss | 新训练访问 |
|---|---|---|
| C0 | 旧 C20 续训，500步 | 16,000旧访问 |
| C1 | C20 + 新纠正，500步 | 8,000共同旧访问 + 8,000新访问 |

两臂相同 fresh Adam、500/B32、noise/time/dropout/LR；共同旧8000逐项一致，另一8000替换槽按family×可用阶段预先分层。复用现有按absolute cursor/替换槽ordinal产生noise/time/dropout的机制，不用含episode/anchor的Catalog标识重新取随机数。替换槽只能称随机张量和分层匹配，输入观测本来就不同；只有共同旧8000可称逐项同输入。新片段不存在的阶段不得伪造或复制补齐，对照抽对应旧阶段。P的loss系数保持，不同时再改 λ、统计、codec或参数范围。保留至少50%旧访问保护F1与闭合技能。

每臂除372+16外，再固定64个新纠正状态完整预测，全部是训练/开发诊断；必须实际存在至少64个合法不同窗口，不足记NOT_READY，不复制凑数。新阶段若无真实旧阶段对应，只作单列诊断，不伪造阶段匹配。之后 C0/C1 各8次配对screen（共16），用与 E2 相同的规则选一个，最终完全同分优先C0，再做8次确认pilot。每轮采用新的条件身份，旧失败保留。最多这一对数据候选，无自动第三对或任意加步。

## 8. DEV32 / TEST40：冻结后证明有限泛化

在**新训练或新模型选择前**，建立并哈希封存 TRAIN/开发、DEV32、TEST40 的 episode/布局人口；TEST结果在冻结候选前不读取。不得窗口随机划分，不得仅改seed把旧布局称盲测。

DEV采用每族8个新布局×2固定噪声=16，共32；TEST每族10个另行新布局×2=20，共40。只考察预注册的同物体、同相机、同M_FREE任务的局部布局推广，不捆绑材质/光照/新物体/任务语义改变。基于已有安全可达场景生成器选择有限邻域；静态几何检查在看模型结果前完成，不用模型成绩筛选布局。实际坐标、scene SHA、初态身份、noise和语言全部封存在 split manifest；未填完整不能派发。

必须与 C20、CN38、旧DEV16、#403、E2/E3及旧RP8/JOIN10/TEST按实际scene/布局/初态簇去重；人工旧lineage标签不代替实际几何。若现生成器无法构造这样的独立集，报告能力边界并先修数据划分；不能把同布局新seed改名为泛化。

只运行唯一冻结候选：DEV≥24/32、每族≥12/16、0 hard STOP、技术失败0，才进入一次TEST；TEST≥32/40、每族≥16/20、0 hard STOP、技术失败0。技术失败计入全部尝试分母，不替换。合法科学失败可继续预登记后缀；触发全局工程停止则未启动条件仍列出，未完成面板不能过门。DEV若未过，不打开TEST；TEST一旦用于修改模型或采集，整个已看过集合退为开发集，未来另立TEST，不重复到通过。

同时给episode级成功率和按布局聚合结果；如报告置信区间，应说明同布局两seed相关，不能当成完全独立的40个新环境。门限是工程验收，32/40不等于已证明总体成功率下界80%。

## 9. 必须实现的小范围代码工作

所有位置以下均指 #403 固定代码，main不含完整执行栈。新实现基于复核后的执行后继建立独立worktree，文档PR不盲合整条运行栈。

| 工作 | 现有位置 | 必要修改边界 |
|---|---|---|
| 真SHIFT7续训 | `close_temporal_v1.py`、`time_training_v1.py`、`time_shift7_training_v1.py` | 原构造硬加载C20 step1000；新增小型successor recipe，显式加载SHIFT7 active数组/30包成员，再建共同新Adam；不改旧manifest骗过guard |
| 物理边界loss | `TimeEngine.objective`、`codec_v1.py` | 按本计划唯一可微项，固定span/单位/tau平均；hard evaluator原样 |
| 导出/恢复 | `scripts/sla/train_close_time_v1.py`、`provider_time_shift7_v1.py` | 原parent1000硬合同不适用；新增明确lineage/version，独立普通RGB恢复，原包不覆盖 |
| 扩展纠正数据（条件性） | `mfree_training_v1.MobileDataset`、`TimeDataset`、SHIFT7 sidecar | 原reader限≤20EP、fit集合等于全manifest、完整五阶段/固定tick；新增窄schema，明确stats仅C20拟合，保留真实handoff/绝对tick/mask与实际新窗数 |
| 夹爪标签 | `gripper_time_v1.label_parameters` | 维持单调单事件/原2s开闭和上一实际命令50ms；不支持的恢复曲线明确拒绝，不静默拟合 |
| 有限比较面板 | `sim_only_campaign_v51.py`、`sim_only_scope_v51.py` | 原controller硬18case/固定SHIFT7；新增V52声明性case表与包绑定，复用现有Journal/owner/STOP，不关闭旧validate或把新面板冒充V51 |
| 真实执行 | `sim_only_mobile_v51.py`、`sla_dense_policy_v5.py`、`mobile_wbc_reference_v1.py` | 保持capture→真实推理→完整H40→ACK→P10/TTL消费；只做版本化模型接入与记录，不引入动作加工 |

不新建大型调度器、多层签发框架或重复账本。必要变更做相关回归后尽快进入预登记的真实训练和闭环，不把文档/CPU检查数当成果。

## 10. 预算、故障与自主执行

### 10.1 最大工作量（分支有条件，不自动全部排满）

| 阶段 | 新更新/优化访问 | 新原生请求上限 | 说明 |
|---|---:|---:|---|
| D0存量分析 | 0/0 | 0 | 0新生成/积分 |
| B1/B2 | 1,000/32,000 | 0 | 两个500端点 |
| B0/B1/B2 screen + 确认 | 0/0 | 24+8 | 未过门不派发确认 |
| 纠正采集（条件性） | 0/0 | 12 | 包含前2可行性检查 |
| C0/C1（条件性） | 1,000/32,000 | 0 | 两个500端点 |
| C0/C1 screen + 确认 | 0/0 | 16+8 | 原失败不替换 |
| 新DEV/TEST（条件性） | 0/0 | 32+40 | 唯一冻结候选 |
| 改变接缝时的最小接口回归 | 0/0 | 最多2 | 否则复用V5.1资格 |
| **全路径保守上界** | **2,000/64,000** | **142** | 140科学/数据episode＋2接口 |

每科学/数据episode初始化5703tick、任务≤30000tick，总≤35703tick；每接口≤5953tick。全路径原生积分上界 **5,010,326 tick**。高层原生滚入/学生生成保守上界 **16,800**（140×120；纯Teacher不产生SLA生成，按真实角色计数），离线完整生成上界 **1,696**：B0恢复16，B1/B2各388，C0/C1各452。技术已发生生成计入额度，不能靠失败从账本扣除；不暗加采样重复、checkpoint sweep或新训练资格更新。所有实际forward/backward尝试另列，不把0优化器更新写成0数据读取。

这些上限不是让代理耗尽预算。首个成果是D0＋两组500端点＋24次配对闭环及明确选择/否定；无需先等后续所有数据和测试准备完才报告。沿用Core实际可用单卡/共享容量，单物理owner；不抢占健康作业。单个受管服务≤12h，本轮计划最大24h且不超过真实allocation截止中较早者。旧记录的截止是 **2026-11-19 02:51:23.855488 SGT**，启动时核对实际有效许可；计划不能自行续期。容量不足正常有界等待，期限前保存完整状态并关闭。

### 10.2 失败之后怎样继续

- 原完整H40拒绝：不安装，该episode科学失败，继续事前固定的其它条件。
- 非法接触/原保护STOP：立即终止该episode，保留trace、部分tick、成本；其模型不得据此晋级，但其它预登记纯仿真条件可继续。不得重开失败world或换seed挑成功。
- IPC等技术错误：核清是否已生成/已积分，保留原尝试；修复后仅续接核实未启动后缀。每个阶段最多2次有新证据的小型技术修复接续；UNKNOWN先核对、不重复派发。预算和绝对期限不因修复扩张。
- STOP后积分>0、隔离破坏、冻结SHA变化、未知owner/launch或tick超界：停止受影响campaign，先解决具体工程问题；不可伪造CLOSED/许可或修改历史收据。

普通应用层路径、类型、IPC、导出、计量和有依据的Teacher几何修复由Codex自主完成，不逐补丁询问GO。真机通信、RL、WBC/腿/MPC/solver/臂PD修改、原安全门放宽、BASE oracle、旧C60/C100恢复和无限候选均不在本方案内。

## 11. 交付与后续任务主线

每阶段交付紧凑的 `ACTUAL.md/json`：实际模型父/端点/源码，更新/访问/完整生成，科学/技术/未启动分母，原生tick/部分消费，抓持/抬升/保持/位移/综合，首失败及真实资源成本。同时附逐条件表、模型包/统计身份和可回溯证据，原数据/权重/站点私有路径不公开。

核心决策表必须回答：B2是否优于B1；B1是否优于B0；改进来自边界减少还是任务真的更多；F1是否退化；F2是否转成接触或位移失败；条件C1是否优于等预算C0。即使没有晋级，否定了哪一假说也要写清。

本轮通过后，主线按以下顺序另立下一有界协议：

1. **PICK_HOLD→PICK_PLACE**：在同一移动本体加入运输、放置、释放和退出，先验证完整Teacher，再做等预算学习与任务闭环。不能拿当前保持1秒代表已会放置。
2. **接触任务**：门先拆成接近/抓把手/带约束拉动/通过或释放；书架先拆成抓取/运输/插入/释放。以真实完整成功示范和新闭环失败决定数据量；不从3次抓持推导已具备这些技能。
3. **连续实时**：在稳定冻结policy后另做端到端相机/推理/编译/安装测量、controller-selected reference观测和动态STOP测试。复用旧p95≈484ms失败；不让实时工程阻塞本轮paused科学比较，也不把paused结果标实时。

这里没有给复杂任务虚构成功率或固定样本复杂度；当前可量化交付就是上述两族DEV/TEST和失败机制。

## 12. 固定来源

下列均为已审阅的不可变GitHub来源，资料中的历史“下一步/授权”只适用于其原版本；本计划不回改原结果。

- [R1 PR403最终实测](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-sim-only-v51-20261117/FINAL_RESULT.md)；[机读结果](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-sim-only-v51-20261117/FINAL_RESULT.json)。
- [R2 V5.1范围/原pilot门](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/docs/goals/2026-11-17/MFREE_SIM_ONLY_PROJECT_AUTHORIZED_V51.md)；[campaign实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/scripts/wm02/sim_only_campaign_v51.py)。
- [R3 H40几何](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/src/go2arx5_vla/world_model/fusion_trajectory_adapter.py)；[FiftyMsProgram](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/src/go2arx5_vla/world_model/task_learning_grasp_v1.py)。
- [R4 移动reference/IK shadow](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/src/go2arx5_vla/world_model/mobile_wbc_reference_v1.py)；[专家速度](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/src/go2arx5_vla/world_model/mobile_continuous_teacher_v2.py)；[F2专家几何](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/src/go2arx5_vla/world_model/mobile_grasp_budget_v3.py)。
- [R5 C20/CN38实际](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/mfree-c20-recovery-diagnostics-20261116/ACTUAL_RESULTS.md)；[精度完整采样](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-close-safe-mobile-20261116/G1_S_ACTUAL_AND_G2_STATUS.md)。
- [R6 匹配两臂500](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-close-safe-mobile-20261116/G2_MATCHED500_ACTUAL.md)；[TIME500失败](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-close-safe-mobile-20261116/TIME500_BOUNDED_FAILURE_ACTUAL.md)。
- [R7 SHIFT7与codec诊断](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-eef-grip-decoupling-20261117/ACTUAL_MATRIX.md)；[SHIFT7解码](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/src/go2arx5_vla/sla_adapter/gripper_time_shift7_v1.py)。
- [R8 单模型CN38](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-shift7-r1-r3-v4-20261117/R1_ACTUAL_RESULT.md)；[R3实时失败](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-shift7-r1-r3-v4-20261117/R3_ACTUAL_RESULT.md)。
- [R9 loss/父加载](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/src/go2arx5_vla/sla_adapter/close_temporal_v1.py)；[TIME训练](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/src/go2arx5_vla/sla_adapter/time_training_v1.py)；[原数据reader](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/src/go2arx5_vla/sla_adapter/mfree_training_v1.py)。
- [R10 SLA初始协议/模型定义](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/docs/protocols/SLA_GO2_ARX5_POSTTRAIN_V1_20261112.md)；[参数实存清单](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-posttrain-20261112/SLA_PARAMETER_INVENTORY_ACTUAL.json)。
