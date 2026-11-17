# V5 一次性独立安全决定请求

**NOT_SIGNED；新增原生请求上限仍为0。执行Codex不是独立签发人。**

依据固定 [PR399@4ba0c36918](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4ba0c36918f570309558ba1c6283fb0b6d0438d8/docs/goals/2026-11-17/MFREE_SCIENCE_FIRST_READMISSION_DECISION_V5.md)。本页请求一次实质安全范围决定；不请求逐工程补丁或逐episode审批。

| 受审对象 | 实际身份与当前边界 |
|---|---|
| 高层 | `SHIFT7_SINGLE / M_FREE`；包 `ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb`；28 adapter＋统计＋tokenizer共30文件，当前逐字节SHA匹配 |
| 候选工程源码 | `980933e0896c247edeb7d9faf66c11fb987ce4b0`；235文件source seal `31ee0ca7c67611cc01e24de035daafbd68f73e07a506e79128938fcaffe98c93`；从PR398运行栈独立建立，非main |
| Profile | **SIM_PAUSED_NATIVE候选，未准入**；合同hash `5cc42afdaa10955ad7e437c3b06c66fd7b4cb1dfeb452a47632be091ad90c8b6`；当前只有非物理receiver替身，未安装native adapter或STOP hook |
| 时钟与参考 | query观测边界冻结native tick；墙钟STOP继续轮询；观测可用时刻独立保存，服务总等待上限10s含观测年龄；原native TTL 400ms、原50ms完整原子余量、原H40同后缀在第8原子前续期；不后移原TTL来源、不复活过期参考；不声称350ms实时资格 |
| WBC / scene | 当前回读原native、solver、控制器、腿策略与7配置SHA均与历史绑定一致；完整值见[G0绑定](G0_CURRENT_BINDINGS.json)。18个历史CN38 scene/MJB/layout/seed逐项回读；**它们是历史证据，尚无获准的新原生面板或site binding** |
| 接口证据 | Core复用F1/F2各一条已保存输入/ACTION，2/2原H40编译、20个mock原子；3点部分消费可追踪；恢复前STOP/时钟/hash/ACK重检。49项定向CPU回归。新增生成、训练、真实相机、原生请求、积分均0 |

**已知风险及最小事实缺口：** 三次right-pad/table碰撞保留；最后连续告警至接触为学生F1 **12ms**、专家F1 **20ms**、专家F2 **34ms**。F1专家首次316ms告警曾中断。原监测器对成功专家误拦 **1/38**，原零误拦筛选仍FAILED。1204保存状态C++检查不证明动态停止；receiver参考见证不证明MPC实际选中参考。**实际controller-selected reference仍UNKNOWN；拟议保护器的native装载点、检测→停止最坏延迟及暂停期间wall-STOP响应均未动态测量。** 现有Python替身STOP测试不能填补这些缺口。

**请独立责任人作出下列一种真实决定，并填写签发区：**

1. **DENY_WITH_SPECIFIC_GAP**：列出最小缺口、风险及消除路径，继续原生0。当前证据不能支持执行Codex自行判定全面接触准入。
2. **LIMITED_SAFETY_ONLY**：若需先测STOP/reference，只签事前固定的非接触资格面板。建议范围为已回读F1-T0、F2-T0各1个neutral-hold条件；不做接近/夹持/抬升，额外原生预算独立于88。**此建议尚非可执行许可**：责任人须先明确最终native adapter/保护器source与二进制SHA、scene/seed、site owner/lease、注入事件/响应上界、最多原生请求和tick、停止及隔离规则。未完整绑定时保持0；不能凭一句“允许测试”套用科学88。
3. **LIMITED_SCIENCE_READMISSION**：只有动态STOP、selected-reference、误停风险和profile证据足以接受时，签发绑定完整的有界科学面板：专家安全8→学生pilot8→DEV32→一次TEST40；每EP≤60s、2ms。任何硬STOP即隔离；pilot须各族≥1次完整安全移动任务，DEV≥24/32且各≥12/16，TEST≥32/40且各≥16/20。未开始与失败分别计数，不自动重跑。当前**没有**足够证据或签名将该分支设为GO。

**独立签发区：** reviewer/实际角色/签名/时间=`null`；决定=`NOT_SIGNED`；可接受误停成本与风险声明=`null`；批准source/model/scene/WBC/guard/site/profile及预算=`null`。最短下一步是由有权限的独立责任人明确拒绝或限定STOP/reference资格范围；无需重复批准无安全语义变更的工程修复。执行Codex不会伪填这些字段，也不会用CPU通过数替代签发。
