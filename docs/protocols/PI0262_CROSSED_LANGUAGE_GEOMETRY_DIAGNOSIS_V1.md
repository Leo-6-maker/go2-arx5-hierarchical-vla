PI0262 — 分离 M1 的句式迁移与场景迁移
状态：建议执行协议，尚未写入 GitHub 授权，尚未执行。
建议 goal ID：PI0262_CROSSED_LANGUAGE_GEOMETRY_DIAGNOSIS_V1
终止状态：PI0262_CROSSED_GENERALIZATION_DIAGNOSIS_COMPLETE — STOP_FOR_SINGLE_INTERVENTION_ADMISSION
1. 唯一科学目标
在不更新模型、不生成新场景的前提下，对冻结的 Donor / C2000 / P2000 补齐 M1 的“训练/验证场景 × 训练/保留句式”交叉评价，辨别 PI0261 的改善及残余失败主要出现在哪一维，并据此给出下一项单变量干预建议。
本轮交付是实际完成的交叉推理实验，不是新的训练方案或工程预检。补齐全部固定评分行、完成复算和归档后才 COMPLETE；不要求强行找到唯一神经机制。不自动进入训练、物理迁移或真机。
2. 冻结的历史与审核依据
仓库：Leo-6-maker/go2-arx5-hierarchical-vla
PI0261 终局 evidence ref：3e3983c。执行前将此短 ref 解析为完整 SHA 并封存，不以 default branch 替代。
主要来源：
- Issue #11 canonical comment 5909947990。
- reports/PI0261/MODEL_QUALITY_REPORT.md。
- reports/PI0261/SEMANTIC_BY_FAMILY.csv。
- reports/PI0261/MAIN_TRAINING_AUDIT.json。
- reports/PI0261/S0_REPORT.md、S0_EXPOSURE.json。
- scripts/pi0261_train.py。
- reports/PI0260/R3/motion2_eval_cases.json。
- 原 PI0261 三份 CPU replay、原始配对计划、完整源数据及来源回执。
已知终局：C/P 均完成 2000 更新，数值门均 4/4 PASS，语言门分别 4/8、5/8 PASS，完整资格均 FAIL。M1 TRAIN：Donor 41/60、41/60；C 50/60、51/60；P 59/60、60/60。P 的 M1 原指令为 8/12、8/12，改写为 8/12、10/12。历史 verdict 和所有原分母不变。
P 不只是改善 TRAIN：原 M1 开发验证的四组描述性合计从 Donor 28/48、C 25/48 增至 P 34/48。但这不是独立试验成功率，也不替代每组 10/12 门；同一家族的措辞、两个 command、两个噪声不是独立样本。
关键已核实混杂：R3 M1 TRAIN 只有 Approach/Bring 两种句式；旧验证的 SEMANTIC/PARAPHRASE 来自 Move/Reach 两种保留句式。PI0261 实现仍使用每条原序列固定的 instruction。因此 TRAIN 高、验证低同时改变了场景和句式，不能直接推定为视觉过拟合。
同一原 SEMANTIC 指令对也可能同时改变目标颜色和句式。例如 family-120 的 command 0 使用 Move/red，command 1 使用 Reach/blue。原语义门仍有效，但原 instruction-swap 的输出差异不是只改目标词的纯对照。
审核边界：本协议依据已读取的仓库报告、代码和逐行记录制定；制定时未下载并独立复算 HF 原始完整 tar、未逐字节检查模型权重、未观看原视频、未运行新模型。执行者应复用既有可信来源回执并验证实际使用的字节，不将本协议当成新的完整原始数组验证证书。
3. 固定三个模型；不存在新训练臂
HF repo：leoyuer/pi0246-gate-pass-campaign。
点	固定 revision	prefix
Donor = R3 Base6000	52abc8d151b67bb667e43fcf874dfe2cfa253364	arms/PI0260/R3/Base/checkpoint-6000
C2000	ce5122232b3577005ecabb4cdd95f252cde5b2a6	arms/PI0261/C/checkpoint-2000
P2000	875abdde2cc7c9e6858fac008e53f2f4b3375255	arms/PI0261/P/checkpoint-2000


C/P 的报告树哈希分别为 c476ded1519fe792372c5de50ca19bed1e71e60a1eb484efa5b1a5da71626f33、882931d754cb25653105a9213fe0dbeb4d2b14e463efb27e49c6101fd5fc5e4b。
复用相同原 processor、statistics、embodiment、camera preprocessing、S4 四步采样、action mask、H40/.05s 和物理解码。不加载 1000-step 点、不比较新 seed、不换模型、不拟合小分类器、不修改提示词之后重选成绩。
4. S0 — 有限 CPU 来源核对及已有数组重组
先刷新实际源版本、现存输出、dispatcher、进程与 GPU 所有权；只接管本 goal 的工作，不干扰其他健康任务。复用 PI0261 的配对输入、恢复和模型归档证据，不重复整个历史审核链。
需要完成：
1. 从完整原始 split manifest 确定全部 30 TRAIN、10 KIN-VAL 家族。家族编号不连续，不得从 ID 大小推断 split。
2. 绑定每个家族 M1 的两个原始 command、共同起点 index 0、原始 body/wrist RGB、state 与 H40 标签。验证实际非语言观测相同；没有有效 index 0 时报告具体来源问题，不挑另一个更好索引、不补渲染、不借用其他 split。
3. 从原场景/序列语义映射确认每个 command 对应的目标颜色。用原元数据与原指令互相核对，不把 c0 固定解释为某一种颜色。
4. 核对原 strict own/opposite XYZ margin 的参考匹配、符号与 tie 规则。用专家作为预测及交换参考的 CPU 正负对照；无新 GPU 调用。
5. 将旧六个验证家族的已存 SEMANTIC/PARAPHRASE 输出按实际模板重新配对。形成同一模板下的 c0/c1 对照，以及同一颜色/command 下的换句式对照。只改变二级报告的 join key，不改变旧门、旧分母或历史得分。
6. 对 R3 与 PI0261 分别统计 template × color × command × family 的真实暴露；R3 访问计数不得冒充 PI0261 访问计数。仅作描述与来源核对，不据此删改本轮面板。
发现真实错配应版本化报告。旧报告中的小样本、已知同指令时间别名、不能唯一解释网络机制，都不是无限追加审核的理由。
5. S1 — 冻结完整交叉卡
仅使用以下四个原有 M1 模板。这里的“保留”指未进入 R3/PI0261 的项目监督，不代表基础模型预训练从未见过相关词语。
T0: Approach the {color} target and align the tool with its axes.
T1: Bring the tool to the {color} marker with the pose shown by its axes.
V0: Move to the {color} target, matching its displayed orientation.
V1: Reach the {color} marker and match the marked tool frame.
所有四个模板都用于全部 40 个家族的两个原 command，颜色只绑定真实选中目标，原 XYZ/SO3 标签不变。每个点使用原 B0/B1 实际噪声数组。不得只保存 seed 名称。
每个模型点：
40 families × 2 commands × 4 templates × 2 noises = 640 logical rows。
三个模型点：1920 logical rows。
四格：
场景	T0/T1 项目训练句式	V0/V1 项目保留句式
TRAIN 30 家族	检查原支持与句式重新组合	在熟悉场景隔离句式迁移
KIN-VAL 10 家族	固定熟悉句式检查场景迁移	重现两种变化同时存在的条件


这四格均为开发诊断。新增四个验证家族的入口评价也不标为盲 TEST。V0/V1 在 TRAIN 家族上仅用于无梯度评价；不能进入优化、归一化、checkpoint 选择或 prompt 搜索。
行键至少包括：模型 revision/文件身份、processor/统计身份、family、原 command、颜色语义映射、split、index/time、模板 ID/完整文本、实际条件哈希、噪声数组哈希、解码身份、原参考身份。缓存必须包含实际语言条件。
在第一次新模型质量查询之前冻结完整卡及来源映射。面板不得根据某个点的结果修改。
6. 缓存、查询与资源预算
每个点预计可直接复用：
- 原 M1 TRAIN 入口：30 × 2 × 1 条原训练句式 × 2 噪声 = 120 行。
- 原六个验证家族：6 × 2 × 2 保留句式 × 2 噪声 = 48 行。
合计每点 168 行，三个点 504 行。复用须满足模型、实际条件、噪声、processor/解码等身份完全一致。一个 forward 可支持多个原标签评分，但评分行不可丢失。
项目	预算
交叉面板逻辑行	1920
预计复用既有行	504
计划新增质量查询（进一步精确缓存前）	≤1416
无有效结果的工程重试余量	96
新质量 query attempt 硬上限	1512
新 S4 action-head forward 硬上限	6048
optimizer update / backward /训练 canary	0
新专家轨迹/相机渲染/IK/统计拟合	0
新 O64、M3、1000-step 质量查询	0
WBC/physics/TEST/DEV-F/hardware	0


新预算不延长旧 PI0261 预算。每次无效/失败前向均记账；质量差不是重试理由。缓存节约不能换取新场景、噪声或提示词。若既有证据暂不可读，应恢复正确来源，不用备用次数随意重算已有有效结果。
Corelab 或现有获准且兼容的资源承担 no-grad 推理和 CPU 复算，不需要新 Hopper 训练作业。一次只加载一个点，保留精确 checkpoint 来源；无需重新下载 optimizer 或上传已有大模型。不得启动新常驻服务、monitor、邮件、无人值守轮询或宽泛清理。
7. S2 — 固定指标，解释不同的失败方式
7.1 原门不变
原四数值组、八语言组、全部历史分母照旧引用。新交叉结果不能替换原 SEMANTIC/PARAPHRASE 分组，不能将原 FAIL 改成 PASS。全程：
synthetic_research_candidate=NONE
selected_checkpoint=NONE
deployment_selected_checkpoint=NONE
新行仍使用原 H40 XYZ 语义规则：
m = d(prediction, opposite_reference) - d(prediction, own_reference)。
严格 m > 0 才正确；ties、不合法结果和 REJECT 保留在分母并单独报告。保留原 XYZ/SO3 数值和 gate 结果，不能仅报告分类准确率。不得改 endpoint 规则、DTW、选最近专家、平滑输出或进行 floor 扣除。
7.2 必须报告的拆分
输出每点 × split × 实际模板 × 噪声的正确数、分母、原 command/颜色分解、正/负/零 margin、双指令同时正确的家族数。另报告两种模板的合计，但不能由合计掩盖某条模板失败。
对于同一模型、家族、command、噪声，统计只改句式后的正确→错误、错误→正确、保持正确、保持错误。这是更干净的句式干预。
在相同实际模板下比较 TRAIN/KIN-VAL，并检查共同颜色/command 支持。家族 split 不是随机化的“纯几何”实验；颜色分布、位置、外观等均可能不同。完整面板分母不变，可另加预先定义的共同支持描述，不能通过删除难样本提高主分数。
7.3 同模板的目标选择：方向错误还是公共偏置
仅对同一 family/template/noise 下两个 command 的预测进行以下二级分析。
设参考为 a0/a1，D=d(a0,a1)>0；令 e_ck=d(pred_c,a_k)。从原同一 XYZ RMSE 定义计算：
s_c = (e_c0^2 - e_c1^2) / D^2。
正确配对应满足 s_0 < 0 < s_1。定义：
g = (s_1 - s_0)/2，b = (s_0 + s_1)/2。
在相同度量下，双指令正确等价于 g > abs(b)。据此区分：
- g 不正：没有正确的指令排序，或发生反向响应。
- g 为正但公共偏置过大：有正确方向的响应，但其中一个选择仍错。
- 与噪声变化相关的边界穿越。
此分析不新增 loss、不新设资格门、不改变旧 margin；也不把输出差异范数较大当作正确理解。
7.4 不确定性
家族是分析的主要聚类单位。同一家族的两个 command、多个模板和两个噪声是相关重复测量，不按 1920 个独立样本计算显著性。B0/B1 分开列出；不声称两噪声覆盖了模型条件分布。结果是入口参考判别，不是 M1 完整到达或 M3 270 度运动完成，更不是物理成功率。
8. S3 — 独立复算与终局
新预测保存原始 action、解码值、参考、状态、实际噪声及模型来源。CPU 复算只读保存数组，独立重建评分和上述 join；复用原输出需能回到其原归档。确认 1920 逻辑行都有明确结果和来源，缓存与独立评分的分母正确。
必交：
- CROSSED_EVAL_CARD.json 及 source/template/split manifests。
- 全部原始/解码数组及 CROSSED_RESULTS.csv。
- FOUR_CELL_REPORT.md：三模型、两噪声的四格表。
- WORDING_FLIPS.csv、TARGET_CHOICE_DECOMPOSITION.csv。
- 独立 CPU replay 回执、query/head 账本。
- NEXT_INTERVENTION_DECISION.md：LANGUAGE、VISUAL_GROUNDING、MIXED 或 UNRESOLVED 的证据与反证，以及下一项唯一优先实验建议。
- 固定 revision 的新证据归档及回读。只上传本轮允许公开的新证据，不复制旧大模型、私有机器人资产或凭据。
可用原存 RGB/预测制作紧凑对照图或短视频；不得为美化增加模型/渲染调用。视频不是完成条件。
实际新推理、复算、账本与归档闭合后才 COMPLETE。仅生成卡、拿到队列号、完成单点都属于 PARTIAL。质量仍 FAIL 或机制 MIXED/UNRESOLVED 不妨碍一个完整诊断实验正常结题。
9. 结果之后的下一项干预方向（本轮不执行）
- TRAIN+T 好、VAL+T 也好，但 TRAIN+V 明显弱：优先做 TRAIN-only、同目标多句式和模板×颜色×command 解耦的受控训练；不把 V0/V1 直接塞回训练后宣称通过原保留测试。
- TRAIN+T 好、VAL+T 明显弱：优先场景/视觉条件研究，检查颜色—目标位置绑定；不是仅靠添加同义句。
- 两种变化都造成损失：明确报告交互/混合问题，下一步只能选一项有对照的干预，不能一并改数据、loss、骨干和相机。
- 配对改善主要改变公共偏置而非正确排序：后续才考虑有明确假设的边界/条件特征干预。
任何语言增强、adapter、骨干解冻、数据生成、额外推理、物理迁移都需新的独立 goal。PI0262 不预先授权“跑到通过为止”。
10. 主 agent 启动摘要
执行建议协议 PI0262_CROSSED_LANGUAGE_GEOMETRY_DIAGNOSIS_V1。
先读取本文及 PI0261 canonical 5909947990 / evidence ref 3e3983c。
单一主负责人；不用等待另一个 agent 签字。
保留 C/P 完整资格 FAIL 和所有 NONE。

本轮零训练。冻结 Donor、C2000、P2000 和全部原推理配置。
仅 M1：30 TRAIN +10 KIN-VAL 家族，原共同起点、双 command、
T0/T1/V0/V1 四个原模板、原 B0/B1，三点共1920逻辑行。
按真实颜色填模板；同模板比较目标选择，同目标比较措辞。
先重组旧数组，预计复用504行，新增≤1416次，
含无有效结果工程余量硬cap1512 attempts /6048 heads。

补齐四格而非继续猜测。独立CPU复算，报告逐模板/颜色/家族、
margin、双指令正确、方向与偏置，及有反证约束的唯一优先干预建议。
不以1920行当作独立样本，不修改旧门，不选新checkpoint。
不训练、不加新句式、不重渲染、不跑O64/M3新查询、
不做WBC/physics/TEST/DEV-F/真机，不monitor、不merge。
完成实际全卡推理、CPU复算及证据归档后，停在下一单项干预准入点。