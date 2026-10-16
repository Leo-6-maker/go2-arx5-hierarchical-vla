# Codex接手：E2追加预算，连续推进GR00T学习→DAgger→冻结WBC目标迁移

**用户已经明确授权追加预算和更新长程Goal**。现役唯一local-PC Codex owner使用既有统一入口直接接管；本hand-off以及GitHub提交不是远端任务已经运行的证据。对本任务卡范围内的常规科研执行和同语义工程修复，不逐阶段再请求 routine human GO；超额、越界科学因素/新物理安全风险与站点账户不具备权限时不得擅自处理为已获准。

从同一固定release读取：
1. docs/protocols/PI0294_E2_EXPANDED_SCIENCE_AUTONOMOUS_GO_V1.md（严格科学scope、预算与安全停止）
2. reports/PI0294/design/E2_EXPANDED_SCIENCE_AUTONOMOUS_TASK_CARD_V1.json（机器执行上限、条件分流）
3. docs/roadmap/2026-10-17_GR00T_WBC_SCIENCE_LONG_HORIZON_ROADMAP_V2.md（完整长程研究问题与证据出口）

首先读取START_HERE.md、AGENTS.md、已部署workspace统一入口、活跃branch与#11/#47最新原生审计；使用唯一owner和mutating checkout，不force push/reset，不热改active E1源capsule或抢现有PBS/lease。所有站点GPU/CPU硬限、权限和调度政策高于本卡。不要在login节点训练或执行长期仿真。

## 首先完成S0；不能假定已经没有在途任务

- 当前已发布HEAD仅到8fdb2a179f5db88c294fea4182c2d177a2e2801a，审计通过的M1@1024 F0/000 T1完整32s自主成功成立：首次真实任务物体成功/641双RGB/41次hold/原S1和native安全0。独立审计PASS；**不是双任务或完整Go2+WBC成功**。T1原source Oracle从2独立E0学生episode成功恢复，C0真实833个去重窗口来自2episode，只能叫E0_STATE_AGGREGATION_ONLY；T2恢复0/2。不得再写成“所有GR00T仍完全0成功”，也不能将833写成独立示范。
- E1原GO/预算/T0与第二job670729最后GitHub快照为R。直接在合法站点入口**只读**查实际job、pending intents、租约、M1@1024其余TRAIN4、C1/两M2状态和E1闭包。保留M1@256有效负例、工程未完成及一次scope deviation，不重跑有效负例洗白，不趁新E2追加E1额度。
- Vanda旧三个实际PBS jobs（1462438/1462451/1462595）继续自己的运行/资源身份；其最新发表Q/H/Q不是此刻实时结论，先只读查、不重复提交。完整Go2/ARX5原TF-R3本体安全STOP不得直接恢复。
- 一次登记E2_ADOPTION_AND_RESOURCE_BINDING：新的run UUID、合法站点项目、唯一新T0/相对期限、原件与权限、各端核心资产/lease、实际有效预算和容量（比本GO小就使用更小）。E1对应GPU/source lease尚活跃时不得在同资源上开新E2 GPU/新source物理；允许隔离CPU规划/fixture但不覆盖使用中资源。
- 记录原E1未用余额不转E2；E2为新增池：**H200累计24 allocated GPUh/8 PBS（每<=4h）＋A40累计12 allocated GPUh/6 PBS（每<=3h）＋CoreLab176coreh/Vanda96coreh/Hopper80coreh/localPC16coreh**；新增训练<=6000更新，source starts<=64、target<=96；其他物理/存储/时钟上限以卡为准，T0+10天停止排队接纳/T0+21天停止新科学启动/T0+28天硬收尾。未重新采用不准消费新的site resources。

## 自动推进计划：已出现T1成功，不再把每段研究退回准备

**S1 Source learning**：先复用真正合格M1@1024和已取得E1结果，固定v3.0 TRAIN/stats/EEF10/S1。对T1的第二不同TRAIN布局、T2的真实失败阶段逐条形成行为/接触/S1结果；按一个因素一个新分支改进原源成功示范的训练覆盖/阶段采样或小型几何loss，必要时v3.1只能独立数据/统计branch，旧E0 DEV不得用于调参。选择模型后再做锁定DEV一次性泛化评价。所有真实负例保留，不为凑成功重复同seed。
**S2 Source DAgger**：T1先使用现有合格Oracle在新学生自己真实访问状态查询专家、收集valid H40、真实executed student prefix+专家必要接管；T2 Oracle未合格前只能做有界新source task-side recovery资格，不能造Dagger标签。比较相同数据/更新预算下的BC额外专家示范与Action-space DAgger；FlowDAgger要真实模型inverse↔forward roundtrip合格才允许独立低优先级研究。原S1拒绝学生的动作绝不进入真物理；专家真值不输入学生。
**S3 Vanda+fullbody**：现役Vanda Stage0/Stage1/编译成功后才用独立新版本做MuJoCo/ROS/EGL/冻结WBC/夹爪和双摄真实时钟资格。与CoreLab以事先固定reference比较。TF-R3旧native STOP保留，新目标pilot只允许具有明确定义task-side原因与再资格的不同版本，原WBC、腿policy、碰撞/限位与评分不许变。完全EEF10模型先source-only近场，BASE偏好标absent。新观测→H40→原SafetyGate→prefix8真实WBC/ACK→新观测才算fullbody工程闭环，真实物体任务得分另判。一个新native fault立即隔离对应版本；本E2累计第2个新target native fault停止所有新target物理。
**S4 Target data+adaptation**：每个已合格task只以真实WBC安全、物体完整成功、实测两路RGB/状态和实际提交参考/ACK为successful teacher bank；先2–5条证明来源，每task力争8 TRAIN+2 DEV独立成功并实际CPU loader；一个task先交不等另一task。只在真实target样本足额时进行source-only vs+5/+20 task demo的匹配适配和目标评价；缺样本不补零、不伪造，EEP10不训练不存在的BASE3，FB13只有真实非零BASE软偏好监督才独立研究。

### 科学独立分流（无需逐步再次问GO）
- Source T1成功，T2还不合格：继续T1的泛化/真DAgger和合格目标迁移；T2单独来源学习和oracle资格，不阻断T1。
- 源Oracle无法安全接管：结束该task DAgger，交明确失败原因；BC和Vanda已合格平台独立继续。
- Vanda无GPU/新MuJoCo数值资格：停止对Vanda全栈扩张，保留CoreLab目标权威，源/模型科学继续。不可用新的机器执行结果抹掉旧CoreLab native STOP。
- 目标Teacher无完整成功seed：目标数据/target微调保持NOT_RUN；源模型/target平台已有真实成果仍可交付。
- 正常S1/规划拒绝是有效结果，不是本Goal整体停止；**真实原生安全或状态完整性故障**必须在当前受影响物理/纠正分支隔离，不能通过更换host、模型、低层阈值继续试到成功。

## 只接受真实结束证据

不再只交CI数量/CPU_READY/loss曲线或首抓握画面。每阶段真实保存Model+data+processor stats source full SHA、E1父模型与新分支身份、TRAIN/DEV场景/语言/seed、学生/专家/actual动作溯源、原S1/native安全/物体原score、实际RGB/视频私有hash、独立reader结果、GPU/PBS/CPU/积分/存储/clock与所有attempt分母。GitHub只推脱敏小材料；raw、模型权重、私有WBC/vendor及凭据不入Git。每7天发布一次简短进度，越过原生安全、科学权限或budget立即停止相关分支并给出唯一阻断，不让“继续运行”成为无限授权。

交付到新的reports/PI0294/E2-expanded-science-<actual-run>/，最终FINAL_REPORT清楚区分source单项TASK成功、source-DAgger训练闭环、target工程闭环、target物体成功/teacher bank、少样本GR00T迁移、正式target部署（本次没有部署授权）。selected/deployment=NONE。

**执行顺序：先核对当前E1真实终态与Vanda实际任务，登记新E2权限和run；随后S1/S2与独立S3可在不抢资源时并行，S4按真实target数据到位条件启动。目标是完成科研证据链，不要求为了追成功而修改冻结控制器或抹掉失败。**
