# M_FREE V5 — 一次性独立安全/执行Profile准入决策书（待填写，未签发）

**状态：REQUEST_TEMPLATE_NOT_SIGNED；不授权任何新MuJoCo原生积分。**  
**目的：** 让独立责任人作出一个对已知风险与有界科学面板的明确决定，而不是对每个Codex工程补丁重复审批。必须由真实有权限者填写；执行Codex不得为自己签发或伪填通过。

## 1. 固定身份与执行合同

- 研究对象：Go2+ARX5 `M_FREE`，高层`SHIFT7_SINGLE`（不是旧FM+TIME HYBRID）
- 原证据：PR #398 `3497840724c6bfcf54ec4317a5ed605290310b14`（R1 180/180专家状态，R3连续实时失败，物理0）；PR #391原CN38/事故
- 待审核真实policy package manifest SHA：**[EMPTY — 从R1封存单模型读取，不能填旧HYBRID包]**
- 执行`source SHA`、完整source seal、site binding：**[EMPTY]**
- Scene MJB/layout（每family和seed）SHA：**[EMPTY]**
- 冻结WBC/MPC/腿/臂PD/solver/碰撞guard SHA、2ms配置：**[EMPTY]**
- 原完整EEF H40=40点×50ms、P10=10、clamp/反归一化/参考续接合同：**[EMPTY — 审阅确认]**
- 拟议时间profile（二选一）：`SIM_PAUSED_NATIVE` / `WALL_CLOCK_NATIVE`；若未选择，拒绝；前者不声称350ms墙钟通过
- STOP/取消/部分P10/上一**实际发出**参考/TTL使用哪个时钟的详细状态机及恢复边界：**[EMPTY]**

## 2. 无法忽略的旧风险（不可事后改成PASS）

- 三次旧`task_right_pad`／`task_table`非法接触（学生F1、专家F1、专家F2），原hard STOP后积分0；**不可从合法H40推导动态净空安全**。
- 最后连续告警前置：学生F1 **12ms**、专家F1 **20ms**、专家F2 **34ms**；专家F1首次316ms告警分段且非连续。原监测器误停 **1/38** 成功专家，原零误停筛选 **FAILED**。
- 保存receiver参考不等于实际controller-selected reference，该身份仍**UNKNOWN**。现有1204保存状态C++检查不证明实际native hook的时序/STOP最坏响应。
- STOP hook的装载点、控制器选择参考、时间戳/观测年龄、实测最坏响应、异常fail-closed、保护性误停成本、模拟暂停时STOP是否仍响应：**[逐项待审核，不可填“已通过”]**。

## 3. 只有三种有效准入决定

- **DENY_WITH_SPECIFIC_GAP**：明确至少一项事实缺失、预计风险、最短消除路径；**保持新增原生0**。
- **LIMITED_SAFETY_ONLY**：独立签发极窄的STOP/reference/非接触安全资格面板，事前绑定场景、代码、保护与最大原生请求；不得扩大到学生抓持、不得将结果混计到P1。
- **LIMITED_SCIENCE_READMISSION**：若并且仅若动态控制与STOP等真实安全证据、误停可接受性及profile都足以支撑风险决定，签发不可变源码/scene/WBC/profile下的**按阶段科学面板**：专家安全最多8、学生pilot最多8、DEV最多32、一次TEST最多40（条件性总≤88）。同一签发范围内不为每个episode重复人工review；阶段成功/失败阈值自动判断。

若`READMISSION`已有前签但任何绑定SHA/profile变更，签发范围外的变更必须补审；无需对未改变安全合同的E0/E1单纯工程修复重复审批。

## 4. 签发区（必须由独立责任方真实填写）

- reviewer/role/签名/时间：**[EMPTY]**
- 风险接受声明（含误停、碰撞、最坏STOP响应）：**[EMPTY]**
- 具体获准source/model/scene/WBC/guard hashes：**[EMPTY]**
- 获准profile与clock/TTL/STOP合同：**[EMPTY]**
- 签发面板上限/条件/STOP后隔离：**[EMPTY]**
- scope status：`NOT_SIGNED`。

没有真正签署且核对全部字段，唯一正确状态为`BLOCKED_SAFETY`，非物理工程工作可继续，但任何真实MuJoCo native/物理请求不得启动。
