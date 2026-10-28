# 更新长程 Goal：PV1继续，CR1先查清动作范围，再决定恢复数据扩采

日期：2026-10-20，Asia/Singapore。Goal ID：`CR1_CODEC_RANGE_AND_PV1_V1`。
审核基线：`ef6f2e5ae37b5ec62f70c52f194bde05ae52efd7`。
本文件是唯一自包含新入口，承接 `reports/PI0294/DATA_COV1/DATA_COV1_NEXT_CODEC_RANGE_DRAFT.md`，不另建一套调度体系。

**权限状态分开：PV1沿既有GO继续；DATA-COV1保持BLOCKED/CLOSED；CR1为已发布、待一次明确科学GO的新CPU研究。用户要求更新计划，不虚构其已批准下面新额度。文件发布、CI通过、原生采用、实验完成是四个不同状态。**

批准CR1时只需引用本固定版本和文末的有限范围；获得有效批准、站点权限和新范围绑定后，R0→R4可自动推进，不再逐文件/逐parent请求GO。没有CR1批准时，现有owner记录新计划，继续PV1；不得借原数据剩余额度或关闭租约自行拟合新统计量。本次不启动远端进程。

## 1. 已证实什么，尚未证实什么

最新科学/工程收据为#11/6288068789、#47/6288069231（17:41 SGT）；以下不是当前实时qstat：
- PV1最后已证B0=14/48，9成功、5有效失败，剩34；A/B实际优化更新仍0。现役 `policy-b0-11 / 678230.hopper-m-02` 最后为Q，1H200/12CPU/225GB/8h，运行源 `d7dc38ef842263e605be210401fbfa6bb5ef2052`，精确CI39393487629成功。原owner继续；不取消、重投、改source或重评有效条件。
- DATA-COV1首四条：1条旧NEAR几何与评分输入域冲突、task_success未知；1条完整物理成功/raw PASS但codec拒绝；2条全准入且同实际几何的T1/T2对。实际2/4低于3/4，原FAIL、44个NOT_RUN、P72/P144未启动全部保留。
- 收尾已补齐两条独立parent的原官方CPU跨episode混batch与确定重复，PASS。它不能倒改首四条门、不能证明新stage银行READY或HF交付；不要再重复该审计。
- DATA原采集与收尾审计均CLOSED、预留0、unknown=[]；累计Core 5242.09774367 CPU秒、local保守3000秒；原4次、64000 live+64000 shadow不变。当前PV1 139条银行未被修改。
- AP0/-Y/修订NEAR/T2的具体失败为anchor209/action31、参考12s，SO3往返误差0.0001381733rad > 原0.0001rad，旋转通道4/6有inverse clip。同窗共现不是已证明的唯一因果。
- D0未来原参考资格并集168条/93组，仍不是已封存新训练库；历史12mm变体单独保存，本CR1主比较不混入。

**结论：这次是数据/流程准入失败，不是VLA学习失败，也不是“Teacher只能成功一半”。** 3条已知物理成功中2条通过codec；另一条工程未知不能被重新归为任务失败。四条/两个初始设计组不足以估计全24组的成功率。

### 1.1 必须承认并修复的计划问题

原NEAR=0.12–0.15m与冻结评分器source-goal至少0.15m的输入合同矛盾，是此前设计/准入遗漏。已有批准将新NEAR改为0.15–0.18m，原不可评分attempt不重写。未来任何新场景首先检查**全部上下游输入域交集**：几何、原评分前置、任务成功谓词、Teacher可用范围、相机、动作编码及运行资源，不只检查静态IK。

首四条3/4是原组合式运行准入门，不是总体Teacher成功率检验。后继新pilot应先独立完成零积分工程合同检查，再评数据产出；仍保留所有实际attempt分母。不能事后删除工程未知让旧门变3/3，也不能把旧2/4改成PASS。

## 2. 新Goal的完成标准：不要再以“扩采更多”作为唯一出口

**主线成果A**：PV1完整B0、同父A_CONTROL/B_COVERAGE各4096更新、同DEV48评价和可解释的共享policy结果，合格才进入原LOCKBOX80。

**数据研究成果B（CR1）**：仅用已保存TRAIN数据，给出可复现结论：主要损失发生在相对变换、归一化近常量通道、float32量化、反向裁剪、rot6d还原还是误差度量；用一个预登记、仅TRAIN-fit生成的新范围候选检验是否能改善未参与该拟合的几何组，同时不损害旧合格数据。

成功交付不要求一定得到“新统计量更好”。允许结论为 `RANGE_SUPPORTED`、`RANGE_NOT_SUPPORTED`、`NUMERIC_OR_INTERFACE_DISCREPANCY` 或 `INSUFFICIENT_EVIDENCE`。所有结论都须有原数组和控制实验依据。**codec成功不是policy成功；新统计量候选不是可直接部署的原模型处理器。**

## 3. PV1即时工作：原授权内执行，与CR1不互为前置

现有单owner从既有入口读取AGENTS、START_HERE、Issue #11/#47、原PV1协议/资源卡，刷新678230实时Q/R/F、权限/nonce、两端lease、已安装动作/ACK、实际有效cell、预算与unknown。当前快照失效时采用真实原生证据，不再重复提交。

保持当前139条银行、CORR2048父、v3.0统计量、原sampler、WBC/plant/夹爪PD/S1/参考门/评分不变。当前B0全48在A/B新optimizer之前的准入仍有效。获卡后先完成本allocation原资格，仅续未完成34格和已明确允许的工程替补；过去有效失败不得重试。

排队时仅做必要冷态准备与经批准的CR1：最终binder之后验证完整RGB/numeric而非metadata、模型alias、0400/owner/hash/schema、quota稳定读与实际文件描述符情况；不要再重建已完成P5或反复发布准备卡。控制端ENFILE是工程可用性问题，先核实自己进程的FD增长与关闭行为；事后全局file table回落不能排除自身泄漏，也不能归罪他人。

主A/B、各DEV48与最终交付优先；P3默认跳过、second-seed仅在可支付且不挤占最终验证时执行。现役B0门的移动、减少分母、增加并发/预算/窗口都不由CR1批准。若资源/日历证明主线不可完成，一次报告实测缺口及具体方案，不悄悄删guard或无限等待。

原PV1资源96H200h、24PBS、同时1卡及至少20h final保护保持；最后已证20275H200秒结算、10PBS、1054queries不当实时余额。原T0=2026-10-19 02:29:56 SGT；10-25 02:29:56停新job/候选，14:29:56停新训练，18:29:56停新物理；10-26 02:29:56全部收尾。CR1不重置PV1时钟。

## 4. CR1：一个有界的保存数据CPU研究，不是新采集

### R0：证据、实现和候选公式预登记

只使用注册的、已关闭、原件hash完整的TRAIN档案。原PV1数据、旧FAIL、DATA-COV1终局及stat文件始终只读；新输出独立放在registry批准的CR1根，不复活CLOSED DATA lease、不新建第二位协调owner。

固定并回读：项目源码；原官方GR00T提交 `51d4c89f72fda44cbf77285c6a8114b52676b8a1`；NumPy/依赖和实际dtype；原stats `fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`；模态/相对坐标/rot6d顺序；所有实际使用的norm_params。CR1真实执行源码需独立CI和冷绑定，不把本markdown commit当runtime。

已读源码事实：
- `scripts/pi0292/train_adapter.py` 使用use_percentiles=False、clip_outliers=False、relative XYZ_ROT6D。
- 固定上游 `gr00t/data/utils.py` 的normalize_values_minmax对 `np.isclose(max,min)` 通道输出0；inverse始终先将输入裁剪到[-1,1]。因此encode不裁剪不等于roundtrip不裁剪。
- 上游state_action_processor将配置为RELATIVE的action规范覆盖为relative_action统计。必须证明读到的是实际生效范围，不仅修改一个未使用JSON键；已有statistics覆盖与缓存也需实测。
- 项目pose10/matrix_from10用两行rot6d与Gram–Schmidt、float64；误差函数使用trace/arccos。要记录近共线性、正交性和误差函数dtype，不能把float32近常量丢失全归于inverse clip。

先封存R1诊断方法、R2分组与R3唯一范围规则，再看候选相对结果。不得试很多margin挑最好的却仍称单变量预注册。

### R1：原失败最小重建与数值定位

先准确复现原AP0/T2的601窗口结果、first failure anchor209/action31/12s及原gates：XYZ 1e-5m、SO3 1e-4rad、夹爪1e-5、relative_SE3 1e-8（以固定函数逐项回读为准）。数组、有效范围和原dtype必须相符；误差阈值临界不以打印四舍五入判断。不一致则先报告身份/数值差异，暂停候选拟合。

记录该窗口逐级值：raw实际state和committed target → 原相对变换 → active min/max与isclose mask → normalized → float32 packed → inverse clip mask → unnormalized relative → rotation解码 → absolute action →原误差。

只在隔离诊断数组上做预先定义的2×2解释性比较：原float32传输/数值参考float64传输 × 原inverse clamp/诊断性取消clamp。后者只测量信息损失，不进入正式reader、训练、动作发布或原评分；原官方路径的PASS/FAIL是唯一基线。同时检查近常量通道被归零、rows/columns顺序、quat/rot6d正交性、基坐标和度量数值条件。

新增最多32个“诊断窗口”按固定first-failure/邻接/首中末规则从已存数组取，包含至少一个原合格对照。float64或替代角度计算只作交叉核查，不得用更有利的误差公式重判旧FAIL。若无裁剪的同数组仍超门，不能宣布范围是唯一机制。

### R2：只按独立几何分组的TRAIN-fit / TRAIN诊断 / 已打开挑战

主拟合档案：D0最终清单中原4mm、完整安全任务成功且独立raw合格的PV1 TRAIN parents，**不以是否codec通过筛选**。12mm、raw审计失败、任务未知/不成功及非TRAIN全部排除并列理由。实际人数由封存清单核算，不能把192个各参考合格parent直接当独立几何。

同一几何的T1/T2、外观兄弟、参考变体和fork归一个group lineage。在task/direction/legacy family元数据分层内按固定SHA256排序，以group为单位约75% fit、25% diagnostic；至少2组的层强制有诊断组，小层不可拆窗口凑样本，缺少支持如实报告。把准确名单及所有例外在拟合前固定。

**已打开的DATA-COV1三个raw成功parent全部放入KNOWN_CHALLENGE，不参与fit、不用于选padding；原不可评分T1只做合同故障证据。** AP0问题已被用于提出本假设，即使全部修好，也只能叫已知案例解释，不叫盲测泛化。

原旧TRAIN已合格行为可形成独立RETENTION列表，不能偷偷拿DEV/CONFIRM/LOCKBOX来检查后挑候选。若某旧参考stat本来就拟合过保留组，披露这种历史暴露，不能声称新未见测试。所有切分先group、后窗口；不同anchor下的relative target语义不同，不能只按target时刻误去重。

总读取上限400个唯一parent；名单超额时按冻结确定性规则取子集并报告范围，不按成功或改善筛选。主档案优先完整保留原4mm候选，额外retention在剩余容量内选，已合格原139尽量覆盖；无法全覆盖时不宣称全库无回归。

### R3：只拟合一个候选范围R1，与原R0同数组逐项比较

原R0全路径完全不变。唯一候选为 `R1_TRAIN_ENVELOPE_PAD5`：**仅改变new_embodiment实际生效的relative EEF9 action min/max**；state、gripper、其余本体、modality、相对变换、clamp算法、cast路径、rot6d重建、原误差gates都不改。不拟合12mm，不修Teacher，不改采样。

对每个原统计单元j（保留原per-channel或per-offset形状，不擅自flatten），流式提取TRAIN-fit所有有效窗口在官方相对转换之后、normalization之前的值。设a0,b0为实际原界限，m,M为该fit集合极值：
`a=min(a0,m); b=max(b0,M); s=max(1,abs(a),abs(b));`
`p=max(0.05*(b-a), 2*(atol+rtol*s), 16*eps_float32*s);`
`a1=a-p; b1=b+p`。
rtol/atol是本固定NumPy实际np.isclose默认参数，先记录；eps是该float32 dtype机器精度。只要非有限、形状不合、空fit、a0>b0、或candidate仍被isclose判同，停止该候选，不通过再调padding救结果。此规则是拟议的单一研究设计，不是已经证明最优。保留旧区间用于避免主动缩窄旧表示范围；额外padding为5%跨度加确定的数值下限。

inactive mean/std/quantile字段和其它键保持原字节并明确标为继承而非重新估计；候选manifest标注只研究active min/max。必须从新实例的norm_params读取回实际候选，而非修改旧实例/污染cache。候选文件不可硬链接原stats后原地写，不能作为任何现役BANK/READY或checkpoint的替代。

R0/R1分别在相同fit、held-out-from-fit诊断、retention及known challenge上运行全部601窗口（较短合法旧片段按其原窗口清单，不padding到601）。最终拟议准入还要六锚点真实图像reader/processor、至少两个不同parent混batch、确定重复；不得用纯数值检查冒充完整链。relative EEF9以外失败单独报告，不能为改善总通过率追加state/gripper改动。

报告四种配对状态：原PASS→新PASS、原FAIL→新PASS、原PASS→新FAIL、原FAIL→新FAIL。按task/方向/几何组及独立事件统计，不仅报告总体窗口率。记录first/max/分位误差、clip/near-constant数量、边界外幅度、range扩大倍数、normalized信号压缩程度。旧跨度近0时ratio标N/A并报绝对变化，不能除0。

数据可表达性候选的支持条件：复现合格；fit all-window满足原误差门且无未解释非finite/退化；诊断至少有一个此前FAIL的独立group被恢复；诊断与已检查retention中没有原PASS变FAIL；官方reader/collator通过；candidate未用诊断数据重新拟合。若诊断原本无FAIL，只能说明兼容性，不能验证改进。只有已知挑战恢复而诊断无证据，结论是KNOWN_CASE_ONLY。

这些是离线证据门，不授权physical rollout、训练或合格bank发布。不要为了达到门而换分组、删parent、扫多组padding或放宽SO3阈值。若R1不支持，保留负结果并解释哪一级误差占主导。

### R4：原模型兼容性风险与正式收口

统计量变动不是无害的数据清洗。对同一个归一化输出z，inverse unclipped区间中`x=a+(z+1)*(b-a)/2`；换a/b会改变实际动作。报告这一映射差异与边界斜率变化，不将旧CORR权重＋新decoder当原baseline，也不能说几何可表达性提升已解决S1或任务泛化。

CR1不加载神经模型。若需要模型级兼容性/低精度训练检验，列为后继GO，不用CPU前向偷做。当前float32 codec往返PASS也不意味着BF16 flow loss或policy可行性已验证。

最终给出一种主结论和唯一下一建议：
- RANGE_SUPPORTED：原clip/near-constant解释及R1诊断支持充分；提出独立新BANK/任务pilot和匹配训练方案，仍待GO。
- RANGE_NOT_SUPPORTED：无需或不足以换范围；停止扩采，按证据讨论数值/接口而不是继续扩大动作界限。
- NUMERIC_OR_INTERFACE_DISCREPANCY：复现或基坐标/表示/误差计算存在明确偏差；提交最小复现，不在原协议中改FAIL。
- INSUFFICIENT_EVIDENCE：缺必要原数据或独立支持；列最小下一测量，不发明结果。

交付7类固定小工件：`CR1_ADOPTION_AND_INPUTS.json`、`CR1_SPLIT_AND_RULE_SEAL.json`、`CR1_REPRODUCTION_AND_CAUSAL_TRACE.json`、`CR1_CANDIDATE_STATISTICS_AND_EFFECTIVE_PARAMS.json`、`CR1_PAIRED_RESULTS_AND_READER.json`、`CR1_FINAL_REVIEW.md`、`CR1_RESOURCE_FINAL.json`。原逐级数组留私有注册存储，只把脱敏数值摘要/hash/代码索引放Git。关账后新scope/lease预留归0；发布结论，不自动进入采集。

## 5. CR1一次性审批对象与有限预算（当前仅提议）

`CR1_SAVED_TRAIN_CPU_ONLY`：CoreLab最多12 CPU核时（含检查、所有数值/reader和打包），local控制最多2核时；最多2个互斥单核worker；24h日历，从首次合法采用记录独立T0，T0+20h停止新大批次，+22h停止计算，+24h必须关账。队列等待、主线让资源与会话重启不重置。若原共享控制权限更早到期以较早者为准。

最多400个唯一parent、R0与一个R1全扫描、32个局部诊断窗口；新存储峰值8GiB、新网络传输2GiB（优先本地只读，不复制RGB库）。不新增MuJoCo model构造、live/shadow/reconstruction tick、任务episode、GPU/EGL上下文、PBS、神经forward/query、optimizer或模型checkpoint。所有这些计数上限均0。仅CPU图像解码和官方processor/collator允许；若依赖偷偷初始化CUDA/神经前向即拒绝。

新科学批准须明确允许“拟合隔离候选action范围”，不能冒用纯读取GO。预算只在批准并完成native scope/lease/site容量绑定后可支出；不得自动从DATA-COV1旧残余额度扣出新研究，也不增加PV1原预算。单次批准覆盖R0–R4和同范围必要CPU修复，不要求每步再请示。

原唯一owner继续协调；CR1与B0 CPU/内存/磁盘互斥并采用低优先级，PV1-R资源压力出现就不派发新CR1批次，保存纯CPU进度后安全让资源。不得因CR1取消健康PBS、延长心跳或恢复已停止AI子代理。现有监控只报告，不充当第二dispatcher，不新增永久cron。

## 6. CR1之后如何真正恢复数据与policy（路线，不是本轮自动授权）

只有可表达性与官方reader得到证据，才提出新的独立数据版本和小型采集验收。先在CPU测真实source-goal下界（含边界两侧）、task/scorer域、codec构造与final bind，再做新GO下的有限物理任务。原COV1四条和STOP完整保留，不从剩44条直接恢复已关闭scope。新阶段应将工程合同资格、数据通过率和policy评估分开；所有planned/attempted/unknown都披露，不能把工程未知藏在所谓成功率之外。

为避免资源分散，先做最小数据pilot，不把24→72→144全部预设为必跑。通过才封存真实新数据版本，未通过交机制结论。不要在同一轮同时改normalizer、12mmTeacher、事件权重、任务历史、DAgger或world model。

后续训练区分两个问题：
1. **统计量干预**：同一批common-support原始数据、同更新等条件比较不同处理器，明确权重warm-start/新optimizer与数值坐标变化，评价需新GO。
2. **扩大覆盖的收益**：两个臂共同使用同一已资格的新processor/统计版本、同父初始化与训练量，只替换数据支持，防止把处理器差异算作数据收益。

不必一口气做四臂factorial；依据CR1和当前PV1真实结果只选下一项最有价值比较。离线通过不自动要求更大backbone，也不允许把已打开DEV/CONFIRM/LOCKBOX回流拟合。世界模型和on-policy DAgger继续留长期战略，不作为获得第一份共享policy的前置。

## 7. 当前首次回报与发布边界

现owner先报 `UPDATED_GOAL_RECORDED`，当前678230实际状态、B0 n/48、A/B实际step、单owner和unknown/资源。若找到可核验的本CR1明确新科学批准，绑定后报 `CR1_NATIVE_ADOPTED`；否则报一次 `CR1_AWAITING_EXPLICIT_SCIENCE_GO`，继续原PV1，不全局停止。若权限缺失是原生访问而不是科学GO，准确写对应原因，不要求用户再次批准已有范围。

可一次明确批准的文本：**“批准CR1_SAVED_TRAIN_CPU_ONLY，按此固定Goal完成R0–R4；上限12 CoreLab核时、2本地核时、24小时、一个隔离范围候选；无新物理/GPU/训练，不修改PV1或WBC。”** 原生容量不足仍不得执行。

本次发布只更新后续策略，不做远端SSH、PBS、owner交接、拟合、采集、训练或删除。#11登记科学结论与下一决策，#47登记真正的权限/部署/资源/效应证据。Git固定SHA回读，不force/reset，运行capsule与管理HEAD分离。当前139银行、原S1与任务score、WBC和DATA-COV1历史结果全部不变。
