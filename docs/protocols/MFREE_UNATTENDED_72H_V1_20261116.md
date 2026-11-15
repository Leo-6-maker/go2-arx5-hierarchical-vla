# M_FREE 72小时无人值守采集—训练计划 V1

日期：2026-11-16（Asia/Singapore）  
状态：**SPECIFICATION_NOT_DEPLOYED；发布本文件不等于Corelab新作业已挂载。**  
配套：[机读规格](MFREE_UNATTENDED_72H_V1_20261116.json) · [一次性交接prompt](../handover/2026-11-16/START_MFREE_UNATTENDED_72H_V1.txt)

## 1. 用户目标和真实基线

用户明确要求：多环境长期采集与后训练挂在服务器后台，允许关闭自己的电脑；无需前台agent反复唤醒或监控。

现役#388核验HEAD为`efae461c548919f9e54956dde9dc31536adfcb3f`。00:54 SGT回执记录C20已5条合格/5条终态、2条在途；现有两族采集和18小时有限神经链已经服务器后台运行。此为旧快照，不是本次远程PID检查。[E1]

代码已有Journal/once、lease、独立世界和神经owner，可复用；但采集器仍硬编码每layout两成功、每族20次尝试，pipeline仍硬编码C20、1000更新、两族unit和ROS196，并在baseline8之后等待整批采集关闭。[E2][E3] **不能只把期限改成72小时就声称已支持C100与四环境。**

本会话能读写GitHub，未获得Corelab远程命令通道。以下是执行端一次性接手、实现最小扩展并实际挂载的合同；没有伪造systemd unit/PBS job ID或LIVE结果。用户许可不需要在每个新episode重复询问，但不包含修改WBC、原生安全、真机或RL。

## 2. 有限长期预算

从服务器实际注册UTC时间起最多72小时；不是对完成时间的承诺。站点allocation更短时取交集，不能延长他人allocation。到期停止新派发，预留原生清退与checkpoint时间。采集达目标可提前关闭，训练/评价在剩余预算内结算。

| 快照 | 合格完整TRAIN累计 | 每族×5个layout的每格成功数 | 累计采集尝试上限 | 模型训练 |
|---|---:|---:|---:|---|
| C20 | 20 | 2 | 40 | 复用现役FM1000，绝不重训同身份 |
| C60 | 60，含C20 | 6 | 120 | 同SLA_base独立FM1000/B32 |
| C100 | 100，含C60 | 10 | 200 | 同SLA_base独立FM1000/B32 |

总采集上限200次，**包含既有C20所有实际尝试、失败和部分请求**，不是新增200。新增目标最多80条，保留原先T0–T4、两族、±5mm几何与F2专家+8mm，不自动扩大场景、物体、摩擦或桌高。新reset/seed在运行前固定并与所有TRAIN诊断/DEV/TEST种子划分；不同seed不保证不同初态，报告状态hash、有效多样性和重复率。

每个cell最多20个预登记attempt ID。阶段边界分别最多4/12/20个attempt/cell；有资格数据不足则按实际N命名，不复制episode、不用别的简单cell补缺。若某阶段达到attempt cap仍缺合格样本，保留CN并关闭该自动扩展分支，不暗中增加新次数。

神经端最多一个并发训练或模型评价服务；三快照合计最多3,000次真实更新/96,000次B32优化访问，含已登记C20的1,000/32,000；此计划额外训练上限2,000/64,000。无自动第二loss臂、SF、RL、BASE head、JOIN10或长训20K。

资源上限：最多4个**总原生世界**（含既有采集、评价和新采集）；首轮维持已验证2世界。神经GPU累计占用最多36 GPU小时，单卡执行，渲染开销另计；72小时、用户/站点资源配额、数据增量512GiB任一先到即停止相关新工作。数字均是本轮设计上限，不是预计资源消耗。

## 3. 让笔记本关机不影响实验

计算、模型、venv、数据、spec、日志、进度数据库、ROS通信均须位于Corelab/已授权计算节点。不能依赖本地桌面进程、前台agent、SSH转发、反向隧道、本地代理/文件系统、临时交互token或远程桌面常驻。

优先复用已有受管systemd用户服务或站点PBS/Slurm。**后台shell、nohup、tmux、PPID=1或存在unit均不足以单独证明关闭最后一个登录会话仍存活。** systemd用户manager的存活还受linger及站点logind策略影响。[S1][S2]

执行端在Corelab读取并记录（以下是只读核对，不是已执行结果）：

```bash
loginctl show-user "$USER" -p Linger -p State
systemctl --user show "$REAL_UNIT" \
  -p Id -p ActiveState -p SubState -p MainPID -p ControlGroup \
  -p ExecStart -p StandardInput -p Restart -p RuntimeMaxUSec
```

`REAL_UNIT`取现场registry的实际supervisor，禁止抄写示例名。使用用户服务时必须确认Linger=yes及服务不绑定登录终端；若不具备，则使用站点已允许的batch/system级服务。不能未经许可改系统设置或凭短暂第二个SSH会话存活就宣布通过。

挂载后必须在前台终端退出后再独立读取PID/cgroup、lease、heartbeat和实际进度，验证有新事件产生；不得为测试断线杀死别人会话。最后一个用户会话的存活依据应来自linger或batch ownership，不能由“仍有另一个终端开着”冒充。

生成`DETACH_READY.private.json`，至少含：真实hostname、source/spec SHA、UTC及SGT时间、unit/job ID、MainPID或batch状态、最近heartbeat、last completed/next pending、资源锁、退出登录存活依据、确认无本机依赖、已关闭前台后的进度证据。

只有该回执成立才能回复用户“可以关机”。排队作业允许记为DETACHED_QUEUED，但不能记为已经训练。报告中的旧PID不代替新验证。没有执行权限时只交付配置/阻断，不声称挂载完成。

## 4. 后台结构：延用现有运行器，不依赖agent决策

服务器supervisor以固定状态机推进：

```text
核对旧owner与结果 → 两环境采集 → seal C20 → 现役FM1000与评价
                      ↓（相互独立，不热追加）
                   继续C60 → seal C60 → 新FM1000与评价
                      ↓
                   继续C100 → seal C100 → 新FM1000与评价 → 汇总退出
```

现役C20胶囊健康则保持原源码、manifest、18小时合同与身份；新长程supervisor先只读观察，接续未执行的扩展attempt，不抢旧锁、不重复已有C1/C4/C20。所有carryover用原完整结果及数据hash绑定。

旧baseline8的有效策略失败不阻塞新模型学习。不要让后继pipeline为了等待整批C100关闭才导出C20；每个snapshot一旦够额、完成原子封存及检查，即可独立进入训练队列。

如果现役链正常跑完，导入其终态作为C20节点DONE；若仅因原18小时期限关闭，则由新source/新lease冷接续未开始阶段，已执行的面板/梯度/物理记录不重复。UNKNOWN必须先核查真实进程和日志，不能凭超时盲重投。

主调度不需要LLM在线批准。碰到没有预先定义的修复或科学分叉，停止受影响分支并写BLOCKED，独立安全分支继续；不在后台自动改代码、改loss、更新依赖或挑checkpoint。

## 5. 多环境和资源公平

### 5.1 并行度

现役2世界计入全局上限，不再额外启动2个同任务。扩到4世界需：独立model/data/controller/history/ROS_DOMAIN_ID/IPC/输出；每世界单owner；共享attempt ledger原子领取；共用安全隔离状态；至少一个既有代表case证明新分配的世界语义一致，复用已通过的原生隔离证据，不重跑整套资格。

没有可用四路RAM/CPU容量就维持2路，不为了actor数耗尽主存。训练/评价排队时，在episode结束边界减少新采集派发，释放1–2个world及相应资源承诺；不杀健康episode，不能借用仍承诺即将启动的内存。

注册时定义神经任务资源优先级：快照封存后的模型/训练任务不得被扩采无限挤占。容量等待连续30分钟后，停止可选扩展world的新派发；仅在当前episode自然关闭后回收承诺。老C20运行器不支持撤销预约时，等其正常关闭，不伪造关闭状态。

### 5.2 预算与性能计量

动态按实际活跃/已承诺unit读取RAM、GPU、CPU、disk峰值，使用事前配置的余量；不只看GPU空闲显存。所有native世界与神经任务共同计算host承诺，锁定资源到真实子进程完成。未知memory读数或进程身份失败时，不派发新工作。

默认2路，只有4路的合格示范/墙钟小时实测增益>=15%、无新增隔离问题且为learner留足容量才保留4路；否则在安全边界回退。无8/16自动升级。每轮记录成功/全部attempt、覆盖cell、耗时、CPU/RAM/GPU峰值、写盘及数据大小，不只统计最快成功段。

模型/optimizer/梯度GPU驻留；专家采集不加载SLA；保存只读相机session、异步有界writer和已验证receiver identity缓存。输入一次必要H2D、小动作D2H；日志/检查点单列。不能将EGL读回称为零拷贝。

磁盘每次新派发前确保：足够该episode或checkpoint的保守峰值空间，且剩余>=max(50GiB,卷容量15%)。可根据已批准站点quota收紧，不自动删除旧权重、视频、失败数据或其他用户文件。通知上传失败不阻塞本地落盘。

## 6. 不可变数据、自动训练和评价

- C20/C60/C100是同10个TRAIN cell内的嵌套扩量，不是新任务或跨物体泛化。未监督/未消费mask、实时RGB、固定frame、实际0.5秒边界继续分开。
- manifest只列已关闭且校验通过的episode；tmp分片不进入训练，封存以atomic replace完成。snapshot记录完整episode列表、版本和数据SHA。
- 预先划分TRAIN/DEV，保留原DEV16不采回TRAIN；不按窗口随机划分，不用旧DEV结果选成功训练片段。
- 三个候选从同一SLA_base/公共初始adapter开始，各1000/B32、micro8×4、最高已资格精度；同已登记LoRA和FM32，dropout .25，50 warmup，5e-5余弦降1e-5，无EMA/额外loss。完整endpoint1000不可被DEV回挑的500替代。
- 每个snapshot只用自身TRAIN统计，随checkpoint绑定。旧PD4端点保持旧stats，不挂新stats冒称冻结基线。不同snapshot的结果是完整数据配方效果，不能说只隔离样本数量。
- checkpoint必须包括优化器、step、RNG、sampler、stats/dataset/source identity；每250更新及受支持的SIGTERM边界保存。只在完整恢复已有证明的前提下自动续训，且step累计不超1000。
- 训练前不把只检查71个C1 latent当成新统计完成；需冻结stats后的全数据FP32/codec检查。已有controller冻结资格不重复。
- 每个候选完成全态检查和新进程普通RGB恢复后，执行TRAIN8→DEV16。离线不是32/32也可进行受原门保护的评价；动作拒绝是策略失败，不是重新训练的自动信号。
- 旧baseline8只做一次。3候选×24+8=80个学生评价上限（包含现役原计划32），扩展净增48；全套200采集+80评价最多280个native科学请求，前置开发/异常物理成本也另计，不隐去。
- 不启动旧TEST或新增盲测；DEV重复使用只能视作开发学习曲线，不声称最终泛化。比较按family分层，安全、技术、策略失败与未启动分开。

安全隔离按原因传播：专家/场景/WBC共因安全问题隔离受影响family/几何版本，不能让另一worker绕过；仅某学生模型导致的碰撞按原MODEL_BY_FAMILY跨其面板传播，不假定所有新checkpoint已安全。M_ASSIST整模式隔离不自动解除。

## 7. 自动恢复：允许工程接续，禁止无限重试

| 事件 | 自动动作 |
|---|---|
| 任务失败/H40拒绝 | 结算该attempt，继续其他已登记条件，不重复该失败 |
| 真实接触/失稳/硬安全停止 | 立即停止本episode，隔离受影响范围，保留全部tick和STOP；不重投 |
| 短暂网络/通知失败 | 只重试上传或通知，幂等event ID，不重启物理/训练 |
| worker进程未知/lease过期 | 先确认PID、start time、cgroup和结果；状态不明即停派，不复制 |
| 已证技术中断且进程完全退出 | 最多1个登记技术后继；物理新请求另计，旧partial不删除；需相同科学参数 |
| 可恢复训练被正常抢占 | 最多2次已验证完整checkpoint续接，未提交梯度访问单列 |
| supervisor崩溃 | 最多2次重启，先reconcile；不让systemd重启整条非幂等旧pipeline |
| 磁盘/内存不足 | 暂停新派发；保留健康任务和数据；无自动删除/降阈值 |
| 新型语义/依赖/数值错误 | BLOCKED具体阶段，关闭受影响新派发，不自动pip upgrade或改科学代码 |

现有v1运行器见SUPERVISOR文件就拒绝重启，这是防重复的正确保护；不能删该文件让它“自动恢复”。新续接必须新lease/新spec、引用旧epoch，并将已确认DONE/UNSTARTED/UNKNOWN区别对待。

## 8. 最小实现差异，不能假装已经存在

1. `mfree_collection_v1.py`的2成功/20attempt硬编码改为新版本的可配置cell配额和全局原子attempt领取；并行worker共享隔离ledger。旧v1不改。
2. 新生产规格延伸既有scene catalog与seed bank，声明C20 carryover及nested snapshots；writer保存后atomic发布。
3. `mfree_pipeline_v1.py`后继只消费已seal snapshot，配置实际world/unit/ROS/GPU，移除固定两族unit依赖；C20健康旧pipeline不热改。共享learner owner去重每个dataset/config/seed run。
4. export/learner接受固定snapshot manifest，不强等全部采集结束；endpoint预算仍1000，恢复保留全态。
5. 复用Journal/registry/OwnerLock与站点job manager，增加reconcile与detach证据，**不另造一个需要聊天不断授权的调度框架**。

变更接缝最少测试：并发领取不重复；包含carryover的总attempt/预算；隔离跨worker；原子snapshot无tmp；owner未知不重投；模型资源等待后安全让位；checkpoint累计步数；退出登录后unit和进度真实存活。单元测试不替代该最后一项现场验收。

## 9. 一次性交接回执与用户可见汇报

执行端完成后台挂载后，一次性回复：
- 实际collector/snapshot/learner/evaluator/supervisor unit或job ID与状态；并行world数。
- 数据目标/累计attempt/当前成功/在途，及已存在C20的去重。
- 训练各run真实step/访问、排队或恢复点；禁止把登记1000步写成完成。
- UTC与SGT start/deadline、snapshot路径/hash、日志/控制路径和停止入口。
- detach存活依据和独立进度更新；无本地tunnel、无前台agent依赖。

STATUS以服务器原子JSON为准。每60秒轻量heartbeat，跨物理/训练/数据阶段才写摘要；GitHub只公开去敏摘要和hash，不上传私有模型/原始图像/路径/凭证。网络恢复后补发同event，不重复算实验。用户关闭电脑后，不需要手动点击下一阶段。

本计划部署前不能回复“新的72小时任务已启动”。本会话没有获得服务器shell，尚无新的detach回执，故新长程挂载状态仍为NOT_VERIFIED。

## 10. 固定资料

[E1] https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/efae461c548919f9e54956dde9dc31536adfcb3f/reports/WM02/mfree-c20-20261116/C4_PRODUCTION_ACTUAL_20261116.md
[E2] https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/efae461c548919f9e54956dde9dc31536adfcb3f/scripts/wm02/mfree_collection_v1.py
[E3] https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/efae461c548919f9e54956dde9dc31536adfcb3f/scripts/wm02/mfree_pipeline_v1.py
[S1] https://www.freedesktop.org/software/systemd/man/latest/loginctl.html
[S2] https://www.freedesktop.org/software/systemd/man/latest/systemd-run.html
