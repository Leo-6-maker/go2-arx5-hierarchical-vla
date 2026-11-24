# V1.9.4c · World-frame reanchoring 实施清单

对应[主协议](SLA_GO2_V194C_WORLD_REANCHOR_PLAN_20261125.md)与[机读登记](SLA_GO2_V194C_WORLD_REANCHOR_REGISTRY_20261125.json)。实施前核实现场绿色版真正使用的代码和报告，本清单不是现成可启动CLI。2026-11-25，NOT_RUN。

## W0 现场绑定（复用原流程）

1. 文档从 main@a718abb5a18a299e116326af59ead3ccb1db4e9a 发布；前版协议 #511@98f1e3c297d2c06ad16cf5125da4500b8b986951；可复用远端执行快照 #510@719882e9fd154ffbe995484e7a766ada3b3ecd02。最新绿色实际工作区尚未可读，固定其实际实现SHA/dirty差异hash、场景/复位、原接触receiver及WBC/腿/动力学、三相机、虚拟左臂、官方模型/统计、代码/资产和两批原生归档。确认H0同配置可复用。只有实际一致才作成对结论。
2. 用现场现有 owner/lease/输出路径与权限；复用V1.9.4b已修的日志序列化、EEF时间戳滞后修复及终态核验；不把旧6ms容差当新准入；不另建运行框架。
3. 原现场报告（ACTUAL.md/json、ARCHIVE_AND_LANGUAGE.md、PAD_CLEARANCE.png）、原始逐2ms journal及参考/ACK保存于现场，计算实际hash并以允许公开的结果索引发布。用户摘要只作当前计划来源。所有技术中断分母保留。

## W1 E0 最小审计

按主协议同tick消费证据重建8条有效学生和两条H0的原数据。输出 E0/ACTUAL.md/json；关键列：layout/attempt/query、native_tick、phase、source_index、ref_id/solver_target/consumption_witness、actual与consumed TCP/姿态、原始意图、左右实测与命令slide、基座/足位、静态b_z、动态残差、左右pad signed clearance、首次接触与终态。未消费suffix只作为预测另列。对2^3组合使用实际pad几何：箱体最低点高度为中心z减各半轴在世界z投影的绝对值加权和；只在实际桌面投影覆盖处使用平面公式，桌边用真实有限几何/原contact规则。核对原审计口径与几何误差。原生不积分回放即可，无法证明时标为缺测。E0不是重新跑完整旧审计，新增0推理0物理。

## W2 实现一个纯函数和最小曲线包装

在绿色现役名义参考模块中新增明确 profile 参数（建议 REFERENCE_WORLD_REANCHOR_V1；原 DIRECT 保留）。复用 `world_intent_clock.world_offset`（blob 0571e09625a8ad071a8d78fb275c7d28cfd44ab1），不要重新实现已核验的矩阵顺序。原SourceCurve用于物理意图，包装适配 curve.at(index) 返回修正参考，curve.grip(index)/arm/events/protected保持原值；编译器所需 knots 按同一修正重新生成。若使用源矩阵right multiply的参数化方法，必须证明与对完整世界TCP曲线应用旧公式等价，不先在源局部轴做世界平移。

```python
# pseudocode: actual green runtime names must be bound first
I = decoded_nominal_curve(query_payload, official_decoded50x16)
A, C = same_tick_measured_tcp, last_exactly_consumed_eef
assert query_tick == measured_tick == consumed_tick
ref_at = lambda s: world_offset(I.at(s), A, C)
g_at = I.grip
# build full-50 reference using same compiler, seam & P10 as green
# read/retain all raw intent, not only ref_at
```

保留原 `Compiler` 参数 bounded=False/protect_all=False、last consumed EEF锚和incoming由最后连续两已消费参考求得；原MODEL_FIRST_KNOT jaw接续、2–42mm工作行程、40mm/s单指速率。保留实际反馈用实测（不回填重锚命令），原16D状态/官方逆变换一次、右臂选择和左臂存档。每个新query固定A,C直到新提案；新提案必须从I重新变换，严禁用上一已补偿轨迹再补偿。若实际返回晚于查询边界，沿原暂停同步规则拒绝失配，而非任意重锚旧输出到新时刻。

仅复验改变接缝：恒等A=C、I=A→C、非交换旋转、相对世界位移/旋转保真、无累加、全50行及夹爪不变、原编译速度/加速度/可达、时钟和ACK/STOP负例。实际预览/消费合法性与物理意图净空分别报告；不能将修正参考较高误作真实净空。原安全门不放宽。CPU fixture非原生成功；不要求额外模型前向。

## W3 E1 固定已知意图 D/R

两布局独立同reset原稳定入口，使用实测TCP位置 p0、旋转 R0 定义固定已知物理路径。无目标/教师/GT进入校准动作生成，桌面几何仅供离线安全预检。建议事前固定同一序列：

| 段 | 相对稳定入口的目标 | 目的 |
| --- | --- | --- |
| L | p0 + 20mm world Z；R0 | 安全向上工作点，公共物理意图起点 |
| H | L，保持≥1s | 静态偏差 |
| Z | L→L+4mm Z→L，末端各保持≥1s | Z响应与残差 |
| SO3 | L，绕世界X轴+2°→R0，末端各保持≥1s | 腕姿贡献 |
| MIXED | L→L+(4mm X,0,4mm Z) 且绕世界X轴+2°→L，末端各保持≥1s | 耦合响应 |

这里 L 的进入段可能因D/R不同而具有不同接续时间，必须记录lead及其净空；D/R使用同一确定性时间参数化规则，校准主体比较以同一意图相位及明确的dwell窗口为准，并报告绝对时间、实际动态响应。编译前锁定路径/姿态/时序、两布局及评分窗；已知意图不是SLA预测，不得计成学生成功。以上幅度是新建议预注册值，不是既有实测可达。CPU预检确认原速度/加速度/可达/站立及两侧指垫保守净空，包含历史入口约24mm偏差、实际观察到的动态和腕姿不确定性，采用原安全裕量，记录计算依据。整个进入/接续及未执行预览也必须审核；若任一路径无充分净空，拒绝该卡，不为了得到配对而触桌，不事后调参数。若两布局均无法在既定序列安全完成，报告BLOCKED，另立安全已知意图，不自动新开扫描。

D00→R00→D01→R01，每卡原生20,000tick含初始化，墙钟≤1,800s。原固定BASE、42mm开爪、无接触、P10 50ms atom/25tick、atom0/7引用刷新、TTL400ms。使用已验证的实际消费边界，单次已知意图锚在稳定入口，整段不逐tick重锚；记录原始物理意图、直接/修正参考、安装、消费、实际TCP和pad。原H0与SEA2不重跑。评分窗口按各段实际source progress与末端dwell固定，报告全段峰值/RMS、窗口峰值/RMS、轴有符号增益、实际净空和终态。若R两卡未达主协议4mm/2°局部工程标准或有原安全异常，E2不启动；直接分支若被拒保留未启动及理由，不能宣传完整配对。

## W4 E2 原绿色四格，只改参考转换

绑定E1合格及当前真实绿色driver、原receiver和站立H0条件，不把前版旧固定开爪脚本当学生driver。原指令green、内部角色red、G00/G01、seed0/1、实际RGB和历史、完整模型50×16、名义夹爪及P10不变。原始源时间i/30末49/30s；物理50ms atom与源索引分开。保持原每次查询消费10个正SOURCE atom或实际到达source end后在P10边界刷新，不在ALIGN重复查询。第120次已接纳动作正常消费至下一查询边界；tick+25≤50000才开始完整atom。保留400ms TTL、原实际STOP/安全门及末端保持语义，禁止保握clamp。

四格 C00,C01,C10,C11；每格≤120前向、50,000tick、1,800s墙钟。模型在合法无接触或接触边界的新预测都按照同一重锚规则，不在段内追偏或篡改jaw/动作事件。接触风险仍由现役门判定；无接触E1只提供有限资格，接触闭环是本轮真正检验。硬STOP先终止owner、判断共享安全缺陷再决定其他格；不得把有效失败当技术重试。无独立授权不启动。

## W5 最小报告

E1四格和E2四格分开，固定计划/有效/拒绝/技术/未启动分母。E1报已知物理意图误差、D/R配对、实际动态增益、静态偏差与净空；E2报完整11类信息：查询次数、原50行动作与夹爪裕量、同tick A/C 与offset、编译lead/source、已安装与实际消费、实际位姿/双指、两垫净空、抓持/物体抬升/保持、原安全/STOP、失败最早层、实际资源与终态。尤其比较4b 3/4 ALIGN触桌是否下降及有无新增风险，不能只看安装参考较高或更长运行。保留原4b语言消融，若另修接口须分版本解释。

报告可在 `reports/SLA_GO2_V194C/{E0,E1,E2}/` 各写REGISTRATION、ACTUAL.md/json和既有终态回执索引；这是建议目录而非当前已生成文件。全部NOT_RUN，0新增模型/积分/训练/渲染/科学分配。
