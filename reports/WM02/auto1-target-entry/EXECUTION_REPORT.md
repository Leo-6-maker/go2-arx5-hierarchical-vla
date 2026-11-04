# WM02 AUTO1：实际登记与完整本体初始化验收

**最新续轮：** 原WBC参考install/cancel/HOLD入口已通过，FIT单参数夹爪响应模型在已打开的8父释放诊断上将正regret由3/8降至0/8；累计目标12353tick，真实任务动作消费及目标WM仍未完成。见[续轮源码、命令、完整SHA和限制](G2_AND_OPENING_REPORT.md)。

2026-11-04，Asia/Singapore。接续未合并PR #355，统一采用PR #354固定提交 `930bb2ef0ff178771e48fb5f03ea3e37fa717582` 的AUTO1范围。main、旧capsule、旧S1/T2失败记录及CLOSED lease均未切换或重开。

**本轮G0实际登记成功；新候选完成全部六层冷启动／保持，并正常关闭。G2参考消费、目标WM和Actor仍未完成。** 这次新增7350个真实目标tick，其中成功候选7100、旧失败尝试250。对应284个合格初始化50ms区间和10个失败尝试区间；它们尚未作为有动作消费证明的目标WM训练样本。

## 实际身份与结果

登记源码 `3e434605201c8d9e47c274fe1a9e04b0650fcce6`，exit0；沿用原EXISTING_LOCAL_PC、4080-simulation、实际registry及Journal/onceIO。新campaign `wm02-auto1-20261104-01` 于 **19:54:21.739793 SGT** 激活，期限为实际登记后72小时。原onceIO的 `native_dispatch_admitted=false` 只描述该元数据操作不是跨站原生准入；不是伪造一个外部grant。新目标入口另行核验直接授权范围、实际profile/registry/lease、物理owner锁和冻结字节。活动campaign lease是有限范围元数据，**不是后台daemon**。

成功候选实际构建、运行源码 **`959337e68034fdb29f5cfa0819249cbd574e1e97`**。原始完整commit对象与GitHub snapshot的完整tree一一对应，见[源码映射](PUBLICATION_SOURCE_MAP.json)。本体二进制SHA256 **`e9aab70b6c1d2af76c6e178fed1fd8e86299dffb9624d15a85e58ba4d3f426ff`**；同episode场景MJB **`70663bf41a235f8f2baf179447720bd1ca845ce1abf847898ba884a46559fc55`**。outer、observer、trace、原leg policy、solver、WBC和配置的完整SHA见[G1实际回执](G1_ACTUAL.json)和[成本与冻结输入](COST_AND_INVARIANTS.json)。

| 独立阶段 | 原生tick | 仿真时间 | worker退出 | 三个子进程退出 | trace审计 |
|---|---:|---:|---:|---|---|
| cold-1 | 50 | 0.1s | 0 | 0/0/0 | PASS |
| cold-2 | 50 | 0.1s | 0 | 0/0/0 | PASS |
| hold-2s-1 | 1000 | 2s | 0 | 0/0/0 | PASS |
| hold-2s-2 | 1000 | 2s | 0 | 0/0/0 | PASS |
| hold-5s-1 | 2500 | 5s | 0 | 0/0/0 | PASS |
| hold-5s-2 | 2500 | 5s | 0 | 0/0/0 | PASS |

六次均精确暂停，无原生joint/contact/numerical安全触发，12个腿控制通道均有实际施加量；所有引用的INPUT/HISTORY及位置／速度发布消费版本审计通过，writer dropped=0。推理期间实际记录到PD重叠，最长推理28.221ms，运行期最大PD墙钟间隔18.820ms；这支持本次正常推理没有全程锁住PD，**不等于5ms周期无抖动，也不是trace扰动的因果消融**。历史首次腿部异常的唯一根因仍未证明。

本轮是 **arm HOLD、冻结腿策略活动、原MPC/WBC节点已构造**；尚未安装任务参考，不能将它改称WBC参考跟踪或目标操作闭环。每次先由原R1精确暂停，再正常停止自己拥有的进程，最终journal/trace计数无额外积分。真实安全STOP未触发，保持 **NOT_TESTED**；原延迟SIGINT的C ABI安全锁存夹具通过，两条outer各只调用假内核1次、产生1个journal包，不能代替真实目标安全停止资格。

## 实际失败及修复

所有原件和先前错误PASS状态均保留；机器回执为旧结果另列有效失败判定，不重写旧RESULT。

1. 父解释器缺ROS loader路径、私有xvfb路径未绑定、native预加载被传给显示脚本，三次均积分0。修复父exec环境和最终simulator注入边界。
2. 第一次50tick收尾缺writer summary；新增仅在精确暂停后封存trace的握手。原生控制回调不等待或写盘。并发夹具验证封存后被SIGTERM终止也完整保存1200/1200事件。
3. 显示进程组停止造成XIO；改为确认属于当前wrapper子树的精确simulator PID后停止，保留显示到子进程退出。gdb会建立独立进程组，旧诊断的匹配错误及受污染结果保留。
4. 零积分gdb抓到 `GetState → RobotControl` 的SIGSEGV；原PhysicsThread退出时释放mjData，PD仍可读取。新版本在PD／Policy／ROS读取者停止、物理线程join后才释放。
5. 后继仍出现FastDDS `DataReaderHistory` 空mutex的SIGSEGV。禁用额外ROS异步信号关闭后，控制循环均已正常停止，但全局 `/clock` 发布器与context的析构次序仍失败。最终使用相同SingleThreadedExecutor的显式生命周期，取消并join回调线程、释放ROS句柄和节点后才关闭context；保留原native SIGINT及原安全锁存。零积分待机关闭检查首次三进程均exit0，随后上述六层全部通过。
6. 构建worker曾误将原OwnerLock当作context manager；零编译、零物理失败，改用其原close接口后，以独立身份成功构建。三次新构建均14项编译／链接／loader／R1检查exit0。

还有一项需明确保留的工程遗漏：第一次启动OCS2时未隔离派生cache，重建了旧目录3个.so；27个生成C源文件字节不变。新旧库SHA完整保留，未删除或回写恢复。后继solver/controller各用独立cache副本。WBC／腿策略权重、控制数学、增益、solver、机械与安全阈值、物理dt均未修改。

## S域夹爪及G2的实际边界

源码 **`aa2a03fe1b632fc0803a65523811cc2e1d155f22`** 的只读审计exit0；原数据SHA256 **`c12293bc877758254a14d19ec112f147fe19c93ffacf4c8399ae8ebfcb42c808`**，prospective stream manifest **`d6593c1b37432164ca38fd88da1c60036b20b2b7126b7e9f767acce856c38c01`**。480个既有FIT/DEV窗口的H40开度映射误差0、消费行时钟误差0；未评分旧留出。用已执行命令从已知reset重建1472个前缀区间及640个已存分叉区间，内部finger target最大误差0。

这排除了这些记录的简单单位／消费行偏移；实际手指位置不能替代slew limiter内部目标。burn8且reset未知时保留区间，不用未来隐状态缩窄；这也不证明完整观测历史不可辨识。新增observer只重建参考记忆，**G1导出时小模型拟合尚未运行；随后已完成FIT单参数诊断，尚无新在线决策验证，见续篇**，不宣称修好了M4-B的物理预测。详见[独立审计](SOURCE_ACTUATOR_AUDIT.json)。M4-B保持冻结，原checkpoint **`9676312c5b810a757c465174ddc4302fc67b1f6896f8ba66dbed5242dee77f7b`**；M4-B本轮未重新推理或更新；原腿策略实际前向仍在trace中记录。

G2首次import exit1：缺少 `go2_x5_planning` 的进程路径。已定位原目标 `src/planning/go2_x5_planning`，绑定其12个原Python模块的SHA后，原TaskTrajectory、EETrajectoryManager、MpcObservation及receiver导入exit0；没有安装包、创建ROS节点或积分。**实际proposed→accepted→installed→consumed→achieved仍待实现及原生检查**，导入不算消费通过。下一工程工作是扩展现有薄adapter及顺序记录，再执行8项G2；目标Replay和WM入口不等待源域胜过规则、完整任务成功或任意中途恢复。

## 真实命令、检查与成本

实际通过原入口运行登记和有限worker，完整解析后的argv及逐次launch/terminal/lease回执留在私有运行目录。以下变量指向当时已登记的独立capsule与spec，没有未实现的训练命令：

```bash
~/Projects/GR00T-WBC/infrastructure/enter.sh 4080-simulation python -B "$CAMPAIGN/register.py"
# 六个worker各由systemd-run一次启动：CPUQuota=400%, AllowedCPUs=0-3,
# MemoryMax=8G, RuntimeMaxSec=600, Restart=no；LD_LIBRARY_PATH在exec前绑定。
~/Projects/GR00T-WBC/infrastructure/enter.sh 4080-simulation python -B "$RUN/worker.py"
# 其实际调用的已实现入口：
PYTHONPATH=src:. python -m scripts.wm02.commission_target \
  --spec "$RUN/EXECUTION_SPEC.json" --output "$RUN/native-01" --run-id native-01
PYTHONPATH=src:. "$PC_NATIVE_PYTHON" -m unittest discover -s tests/world_model -q
PYTHONPATH=src:. "$PC_NATIVE_PYTHON" -O -m unittest discover -s tests/world_model -q
```

成功候选六个worker均exit0；72项普通＋72项`-O`全部通过、无跳过。原ROS deprecated／线程优先级警告保留，不与MuJoCo原生warning计数混淆；未修改站点权限或他人任务。新GPU时间0、S采样0、WM更新0、Actor更新0、完整任务0。已测Core worker CPU **183.203790s**，PC只读夹爪审计 **1.662672s**；有记录的worker墙钟小计102.002958s。Core campaign存储约1.206GB，PC本轮命名产物约257.75MB；传输、测试与部分准备开销未完整逐项计量，无虚构账单或能耗。

导出时本campaign运行进程0、所有有限worker lease CLOSED；campaign的72h范围仍ACTIVE，无驻留调度器。新目标学习数据、WM、正式Actor及接触采集资格均未计完成。

下一条可立即执行的证据核验命令（不重跑任何已完成物理）：

```bash
PYTHONPATH=src:. python3 -m scripts.wm02.review_auto1_round \
  --report-dir reports/WM02/auto1-target-entry
```

下一科学阶段是G2原参考消费与8项诊断，随后G3顺序目标数据→G4目标200+1恢复→G5第二批100+1。缺失的G2采集CLI将先实现及测试；不将这些阶段描述成已执行。
