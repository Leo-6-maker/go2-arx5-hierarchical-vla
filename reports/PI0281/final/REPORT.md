# PI0281 终局：真实 baseline 已完成，WBC 资格未通过

授权：Issue #11 canonical 6040516315；release `360b412f55be8aec1568054d7d039ce7de2bc597`。原 PI0280 封存、旧微调继续暂停。selected/deployment 均为 NONE。

## 科学结果

| 固定面板 | 10000 | 10001 | 10002 | 10003 | persistent |
|---|---|---|---|---|---|
| 原 PI0280 direct EEF，复用 | FAIL | FAIL | FAIL | FAIL | 0/4 |
| 原始模型 joint 新观测闭环 | q6硬限拒绝，18.65s | 完整32s FAIL | 抓握/抬升，未放置释放 | 完整32s FAIL | 0/4 |
| FK(commanded q) EEF 实际回放 | 原角速度gate拒绝，6s | 完整32s FAIL | 抓握/抬升，未放置释放 | 完整32s FAIL | 0/4 |
| 既有专家 DROID/time-grid 桥接正控 | PASS | PASS | PASS | PASS | 4/4 |

全部质量结果按原物体谓词及28–30s的41采样点判定；提前终止不冒充完整32s。专家四条件皆41/41。真实 source Panda 是原始模型的 OOD 端口，不是 DROID 硬件成绩、Go2高层闭环或泛化资格。

同一10002条件，joint持续抓握28.95s、最大增高0.52243m；FK回放最大增高0.52289m。两者未形成 on-top，末窗未释放且不静止。其他模型条件无抓握。原 directEEF 的极短抓握和完整旧FAIL均保留。

10000 joint 在 dispatch 前拒绝 q6=3.758334875rad >3.7525rad。四 joint实际硬穿透均0。FK10001的实际穿透原值8.47131µrad，URDF复算8.47440µrad；安装IK目标另超joint4上限0.34846rad，不能把安装目标与实际状态混同。EEF路线原姿态gate与joint原始q硬门不同，不套WBC新门改写源质量。

## 保存数组及输入审计

完整复算240个A3及32个A2输出包，无新增 wrapper 查询。A3各条件H40 predictedEEF–FK(q)平均XYZ差40.66/70.92/28.68/57.42mm；SO3均差.308/.295/.112/.367rad。A2 H40均差97.01mm/.635rad。两输出独立预测，这不是自动的decode错误。

实际RGB历史、StateV1、TCP/FK、原统计和noise字节均审计。215个新joint查询全部复用对应旧实际noise；raw q→installed q误差0，gripper routing误差0。原始模型有效权重在加载和关闭时一致。未发现可证实的输入/解码/时钟缺陷，因此没有开放directEEF质量重跑。

## 冻结WBC：一个progress修复，姿态仍不合格

| 实际版本/固定条件 | 完整10s | 末2s位置RMS/max(mm) | SO3 RMS/max(rad) | 资格 |
|---|---|---|---|---|
| native hold | 是 | 9.063 / 11.838 | 1.105218 / 1.139202 | FAIL |
| native monotonic | 是 | 12.039 / 16.077 | .629326 / .663841 | FAIL |
| native return | 否，0.5s TRACKING_ABORT | 未观察末窗 | 未观察末窗 | FAIL |
| 真正消费repair hold | 是 | 9.063 / 11.838 | 1.105218 / 1.139202 | FAIL |
| 真正消费repair monotonic | 是 | 12.038 / 16.077 | .629325 / .663841 | FAIL |
| 真正消费repair return | 是 | 9.008 / 12.206 | .543153 / .575422 | FAIL |

所有WBC实际URDF硬穿透0，负soft margin仍保存。真正修复沿原16候选/.15m搜索窗/原SE3评分，取首个有序局部最小值；不是令s_progress=s_ref。原return在0.5s跳至.1m，s_ref=.001224m，timing=.098776m。真正修复完成10s，最终s_progress=.043294m、s_ref=.1m、timing=-.056706m；它阻止端点跳跃终止，但不能宣称完整空间progress已正确。

真实actor全history/preview/本体目标和sameTCP FK链一致（rot6D约1e-7、sameTCP角差约2e-6rad）。原姿态代理是无量纲sin(theta/2)，报告用真实SO3弧度。姿态门RMS≤.15/max≤.30仍失败，未授权或执行旋转偏置、gain调优、actor训练或第二修复。

前三个标为repair的原件实际未消费修复：实际WBC方法内重新导入原函数，先前接入写到未使用别名。原结果/代码/hash/消耗永久保留，外部deviation明确ENGINEERING_REPAIR_NOT_INSTALLED。仅修正同一规则的接入位置，使用原额度内3个同条件技术替换；每份真运行REPAIR_RUNTIME_BINDING显示bound=true、501次调用。无有效质量重试。

日志优化在native hold结束后用于未启动条件：相同payload/顺序，长期buffered handles，终局flush/fsync/linecount/hash回读。它不改变物理算法；原hold耗时920.42s，后续完整条件约32–36s，是IO等待减少，不能当作GPU利用率统计。

## 停止点和最小适配gap

修复三项没有全部合格，因此目标expert4和recorded-model4条件均为未执行依赖槽，quality=null；不是0/4模型FAIL。fresh targetVLA仍被0/8真实输入路线阻断，调用0。source闭环、FK回放和专家正控分开报告，均不冒充Go2高层闭环。

专家桥接4/4而模型路线0/4支持这四个开发OOD场景中的任务/观测分布gap；不证明容量不足或所需数据量。后继唯一草案：原始GR00T初始化、native flow-matching目标、先冻结VLM，对照有限的显式EEF/夹爪接口适配与标准projector+DiT。物体任务数据不按目标WBC成功过滤；不学习目标joint action。草案launch_allowed=false、NOT_AUTHORIZED，不自动训练。

## 原件、预算与交付

新增215policy/860heads、23physical starts（14source＋9WBC，其中3技术替换）、50301integrations、3559WBC controls、7100actor forwards、21449dispatches、7923RGB pairs。质量重试0；训练/梯度/optimizer/统计拟合/新teacher/新适配数据/目标VLA/真机/PBS/model下载均0。原PI0280的304/1216/7及历史账本不重置。

原technical replacement实际counter漏加、reservation已计3；ACCOUNTING_CORRECTION保存raw SHA/原值和owner有效3次计数，不篡改原账本，不重跑。源及WBC有限进程均已终局；quota UNKNOWN。资源细节见FINAL_RESULT及LOCAL_EVIDENCE_INVENTORY，GPU计时为完整进程占用上界，不是利用率。完整历史CPU总量未采集的部分标UNOBSERVED，不填估计。HF bulk延后，不阻塞科学终局；唯一原件仍在Corelab PI0281目录。

12条实际source视频及9条WBC原件/技术替换视频由保存RGB编码，无新render/physics。SOURCE_THREE_ROUTE_CONTACT_SHEET及WBC_CONTACT_SHEET已逐行查看first/middle/last：水印、成功/失败/未安装repair标注清晰；WBC相机仅原body/wrist视角，不伪造外部摄像成功录像。全部影片解码帧数/时钟/hash通过。

固定表：SOURCE_TASKS.csv、WBC_FIXTURES.csv。详证：FINAL_RESULT.json、GAP_MATRIX.json、COMPLETION_AUDIT.json、LOCAL_EVIDENCE_INVENTORY.json。运行身份和所有原始大数组/actor输入/contacts/hash保存在 `/home/liuyu-nfs/Projects/go2-arx5/PI0281`；不以Git代码或准备manifest替代已执行证据。

终局：COMPLETE_VALID_BASELINE_AND_WBC_QUALIFICATION_FAIL；STOP_FOR_MINIMUM_DECLARED_ADAPTATION_DECISION。selected/deployment=NONE。无自动后继、mail/monitor/广泛清理/merge。
