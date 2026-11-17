# V4 S0：三次碰桌的已存2ms证据 — BLOCKED_SAFETY_WITH_EVIDENCE

本轮仅从Core读取已封存的四个窗口与完整41任务汇总，并进行数值二次分析；没有重新积分／FK、装入hook、改变阈值、重评分成功任务或解除隔离。原38成功专家／41物理请求、三次非法接触与监测误拦1/38均保持。保存状态不恢复完整WBC/MPC历史，RP8仍失败。

| 原记录 | 首次告警→接触 | 最后连续告警段→接触 | 该600ms窗口最大右pad向下有限差分速度 |
| --- | ---: | ---: | ---: |
| 专家F2 `72H-F2-T4-A003` | 34ms | 34ms | 0.1914 m/s |
| 专家F1 `72H-F1-T3-A006` | 316ms | **20ms** | 0.2949 m/s |
| 学生F1 `00-F1-MFREE_C20_FINAL_FM1000-r201` | 12ms | 12ms | 0.3352 m/s |
| 成功专家误拦 `F2-T3-r2` | 无接触 | step19000–19024连续25样本 | 0.2754 m/s |

新定量发现：专家F1首先在19187–19193出现7个告警样本，随后消失，直到19335–19346再次告警。316ms是第一次历史告警的提前量，**不能当作连续危险信号持续到接触的余量**；最后告警段仅提前20ms。若有经过资格的锁存STOP，第一次告警理论上可触发停止，但当前hook未安装／最坏延迟未知，不能据此声称避免碰撞。最短12ms警告仍不足以替代停止端到端实测。

三次原contact pair都为task_right_pad/task_table。最终数据包的time_s比contact_time_s晚2ms，因此按STOP报告tick之差直接乘2ms会得到36／318／14ms；原34／316／12ms使用实际接触时间，**本轮保留原时间定义和结论**，未事后提高告警提前量。

表中的速度来自已存pad几何的2ms有限差分，是历史窗口最大值，**不是EEF命令速度、未来速度上界或原安全门违规的证明**。pad运动包含底座、姿态与臂跟踪的共同影响；不能直接归因于WBC某个参数。现有原记录显示TCP/BASE在末次query后均下降，但controller-selected reference仍UNKNOWN，receiver送达见证无法证明MPC实际选中了哪条轨迹。当前只支持“任务空间合法参考不足以证明动态接触安全”，不支持确定某一WBC或IK根因。

原S0完整汇总SHA `9ed2b01909cd15e582f6fdb81b8f6e76f0480fa56d9ca3d6ee9bacbd1174c347`、四窗口结果SHA `8f7b298cf8ce15a51e95eed61e198d3c17cb3bc92be9d6ef87cae9785b902f8d`。每事件实际MJB、journal、原结果及窗口SHA、告警段和几何量见[S0_ALARM_CONTINUITY.json](S0_ALARM_CONTINUITY.json)，计算脚本为`scripts/wm02/s0_saved_alarm_runs_v4.py`。当前读取到的F1/F2 expert/student共同隔离仍在；M_ASSIST继续隔离。

**独立READMISSION仍未签发。** 需独立责任者审阅controller实际reference、STOP hook顺序及最坏响应、scene/layout与冻结WBC/profile绑定、误停成本及有限资格面板。R1现已选择SHIFT7单模型，未来签发必须明确绑定其包和执行源码，不能沿用旧双adapter候选的未签署申请。当前新增原生请求／积分／抓持评价全0；P0专家8→学生pilot8→DEV32→TEST40均未启动。不得由本机分析或GPU实时资格自行放行。
