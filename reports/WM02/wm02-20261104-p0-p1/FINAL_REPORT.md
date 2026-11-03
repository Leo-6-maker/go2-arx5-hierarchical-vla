# WM02 第一轮实际执行回执

2026-11-04，Asia/Singapore。**P0 已得到实际目标身份与明确阻断；P1 已实现合同／薄adapter并完成 S 域原生检查，目标 P1 尚未完成。** 用户随后明确允许停止 GPU1 的727048；该进程已按原停止流程退出并独立核实 CLOSED。没有提交 WM02 长训练。

## 代码与隔离

交接 `ddc801f30e288365925c61c37f7ee445ddc29712`／PR344仍未合并。本轮从该SHA建立独立checkout和分支 `research/wm02-p0-p1-native-20261104`，没有假定main包含计划，没有切换活动source。

| 用途 | 完整源码SHA |
|---|---|
| 原生接触／恢复检查 | `27eed48ba5769c76b99feb80bd1bfdd026714e52` |
| 加速并行源域profile | `784f992216e0f6dadda6143ab6e913548a86c648` |
| 末次部分区间mask修正，CPU复测通过 | `a3886aa1d4714270841742671eb2f1a34c8dd165` |
| 明确复用的WM01源adapter版本 | `bf66a7c9bd59b2022113d6cd54e3d1c1a03ff2cd` |

仅复用已有source contract/native/storage/runtime/population seam，逐文件SHA见 [REUSED_SOURCE_ADAPTER.json](../../../experiments/wm02/REUSED_SOURCE_ADAPTER.json)。未继承WM01的模型、训练器、数据量或测试PASS作为WM02成绩。未复制第三方算法、OSC或私有WBC实现，未升级活动软件。

新增的实际入口为 `scripts.wm02.resolve`、`scripts.wm02.qualify`、`scripts.wm02.profile_source`。模型fit／Actor／reset-bank／distill仍未实现；没有提供空训练CLI。`plan.json`继续保留交接时设计快照，当前实现见 [RUNTIME.md](../../../experiments/wm02/RUNTIME.md)。

## P0：实际后端、冻结身份与第一阻断

现场 `4080-simulation --resolve-only` 返回原Python3.12环境；原目标登记／输入／startup配置指向 **MuJoCo＋ROS2／OCS2 Go2–ARX5链**，不是仓库的Isaac Gym构造器，也不是上游MuJoCo Warp＋OSC。原目标执行source为 `ee4f86d5fc9fd8a54008329394a0e0f1738526c6`。私有native副本不带可读Git HEAD，不能把目录中的`6dd13098`冒充完整Git SHA；以实际冻结文件SHA标识。

只读逐份核验9个原生核心成员和14个初始化成员，全部与原pin一致，包括：

- 冻结腿策略 `N1_s29_175632/policy.pt`：`b07c2d494a47cdae2a97fd9b2471d8f6f243049741e949414a5d099fa5148e3a`。
- WBC `task_floating.info`：`0b69bd58db364a01e87331b0e4e777a9684fda18a4f67fa60aee55174f72dcca`。
- 原OCS2 node：`fc985496715995778e1ba1580fddfd3f3922fd4b5c7e0954b66cef7b6ecd128f`。
- 原MPC node：`4a5f5aa6e0b9d4bde62f4d1da73135ef97e5268945a0cecf114d9190d3e04fd0`。

冻结参数／WBC／solver／原生安全均未修改。现场原目标lease为 **CLOSED**，包含 `native_safety_quarantine_preserved=true`；两个原目标unit均MainPID0。**首个确定阻断：S1/T2隔离没有当前解除与目标动态准入。** GPU1共享／停止PB并不替代目标动态许可。

后续工程缺口也明确保留：完整MPC solver/policy、WBC历史与receiver状态的可恢复导出器尚无；没有用qpos/qvel冒充完整目标恢复。已有full-body审计包含额外BASE位移判据，本轮站立局部任务的scorer/card绑定仍未完成，不擅自删判据或宣称目标评分已核准。目标8条件、1/4/16并行profile、抓持reset和奖励五类检查均未执行。

[脱敏RESOLVED_SPEC](RESOLVED_SPEC.redacted.json) 保存身份与缺口。原raw二进制／权重／资产／私有路径与SSH配置不进Git。

## P1：动作时间、记录与递推接口

56参数采用固定world-task增量、每步平移范数≤1cm、旋转范数≤0.05rad，按合法SO(3)组合；canonical四元数／native row-rot6d边界沿用原代码。H40的8个命令节点在t+0…t+0.35，尾32点保持末指令。现有MuJoCo wire converter在t+2.0追加末点，形成41点，不自行补一个measured anchor丢掉动作0。

实际冻结私有parser SHA `3fd59ca29b257a4fec2a66409d33099d0cc0befe4ef6a5039d80b44d03dadf90` 已仅在CPU执行：41个world采样全部核对，11个错误包被拒；原receiver保留action0、末点保持、不产生BASE通道。**这是实际parser代码上的合成合同检查，不是目标安装ACK／空间进度／物理资格。** 目标 `ACTION_TIME_MAP.native_consumption` 继续标为未知，不能把8个时间区间称为实际消费了前8行。

目标夹爪仍为q/.044机械fraction；保留V2原工作行程[.002/.044,.042/.044]，超界拒绝，无裁剪或隐藏缩放。源域夹爪合同单独保留。

实现了字段角色PREDICT／ANALYTIC_UPDATE／CONTEXT／RESTORE_ONLY、Actor白名单、8原子步历史移位和400ms决策节奏。下一步模型输入只能由预测量、解析更新和固定context产生，不能传入未来真实WBC缓存。GRU动力学尚未实现，不把接口当已训练模型。16原子步只在0和8调用Actor；部分块折扣使用`.997**实际步数`。零物理拒绝独立存事件，terminal后autoreset不能拼接，部分区间不作为完整50ms回归目标。

S域记录保留raw、完整H40条件、installed、逐2ms consumed、achieved、控制恢复引用、原始native state、实际tick和终止。未消费后缀不作物理标签；Teacher阶段只作分析元数据，不进Actor。

## S域真实接触与完整恢复结果

使用独立PC CPU环境：Python3.10.12、NumPy1.24.4、MuJoCo3.2.7；27份源资产原pin核验通过，原PI0294固定底座driver、Teacher、scorer与S1不变。两张已声明的WM01 TRAIN几何仅用于明确的S域接口诊断，不创建WM02正式TRAIN512／reset bank，不混入目标数据。

| 源诊断类别 | 尝试 | 实际结果 |
|---|---:|---|
| 保持 | 2 | 无硬安全；恢复一致 |
| 自由空间小轨迹 | 2 | 有界指令执行；恢复一致 |
| 抓持／小抬升 | 2 | native grasp持续true；物体高度约0.5230→0.5423m |
| 支撑后释放参考 | 2 | native grasp由true→false；物体约0.4602→0.45989m |

最后一类只执行了0.4秒释放参考，尚未覆盖完整撤离；保留此覆盖缺口，不能把条件名字当完成证据。没有任何新的完整32秒任务成功声明。

8状态各两条恢复分叉；完整native积分、IK积分、目标/cache、Teacher/history、RNG、forecast以及全部installed／consumed／achieved记录递归比较通过，最大误差0，接触／终止一致；原S1 live/shadow最大误差0，硬安全违规0。

接触检查产生 **672条50ms转移**：544条物理构造前缀＋128条恢复分叉；16800主积分＋16800 S1影子积分。CPU24.001200秒，墙钟24.459636秒。完整恢复的只读后审计另耗CPU1.045745秒、无新增积分。原件保持私有，完整原生状态没有复用缺字段的旧数据补零。

## 加速并行实测及容量负结果

新增CPU spawn进程并行，保持同一2ms物理／50ms原子步、原S1与接触，不改solver。每环境固定2秒、40条转移；吞吐包含进程启动、引擎构造和私有记录。

| 环境／进程数 | 实际转移 | 全周期墙钟秒 | 转移/秒 | 测量worker CPU秒 |
|---|---:|---:|---:|---:|
| 1／1 | 40 | 1.927838 | 20.7486 | 1.121498 |
| 4／4 | 160 | 3.335345 | 47.9711 | 8.060242 |
| 16 | 0 | 未运行 | 不适用 | 0 |

4进程为1进程吞吐的 **2.312倍**；这不是GPU WBC并行资格。实际CPU绑定4核，16环境预测RSS约5.24GiB，另保留1GiB余量时超过当时约5.47GiB可用RAM，因此未启动16环境；不是以降低安全／dt凑并行量。每worker实测最大RSS约336MiB。

本轮合计 **872条新S域原子转移、21800主积分＋21800影子积分**；13个独立源环境构造、0个完整32秒任务；模型／Actor更新0、模型加载0、WM02 GPU/PBS训练作业0。原生worker／接触／后审计已测CPU合计34.228685秒；父进程、测试、网络、准备开销未完整计时，不称全部总成本。目标转移／任务仍0。

## 用户指定停止727048及GPU1准备

用户明确允许停止该PID用于新WM。先核对uid、完整worker路径与proc_start346172649，已有完整1024／2048 checkpoint摘要；用原STOP sentinel＋**一次SIGTERM**触发原handler，没有SIGKILL、重启、删除或热改source。

独立读回：PID不存在、MainPID0、ExecMainStatus1、原RESULT写出`FAILED_BOOK_FINITE_OPTIMIZATION / TimeoutError`、lease由原owner关闭。外部原因明确为**用户主动停止的partial**，不是新的算法失败。实际停在2154更新／68928 TRAIN访问；最近完整checkpoint2048，之后106更新未形成完整checkpoint，不能称可恢复2154。1024／2048完成manifest SHA未变。原CPU86783.725865秒／墙钟38344.889151秒保留，不计作WM02成本。

GPU1独立读回free15899MiB；没有claim已取得长期GPU reservation，没有为未准入目标启动训练。GPU0他人工作、consumer08、WBC和其他活动capsule未改。详见 [PB_USER_STOP.json](PB_USER_STOP.json)。

## 实际命令、退出码与未完成项

下表命令在独立checkout／现有Journal中实际调用。`${WM02_*}`为脱敏别名；执行者私有`ACTUAL_COMMANDS.private.md`保存完整绝对argv，native wrapper与Journal还保存source绑定。不是待实现接口示例。

```bash
python -m scripts.wm02.resolve --binding "$WM02_PRIVATE/P0_BINDING.private.json" --output "$WM02_PRIVATE/RESOLVED_SPEC.json"
# exit0，输出TARGET_BLOCKED，不授予动态许可
python -m scripts.wm02.qualify --domain S --spec "$WM02_SOURCE/EXECUTION_SPEC.json" --output "$WM02_SOURCE/source8-01" --run-id source8-01 --profile local-PC-native --parser "$WM02_PRIVATE/ee_trajectory.private.py"
# exit0，8个源诊断
python -m scripts.wm02.qualify --domain T --spec "$WM02_PRIVATE/RESOLVED_SPEC.json" --output "$WM02_PRIVATE/target-blocked-02" --run-id target-blocked-02 --profile 4080-simulation
# exit78，无目标构造／积分
python -m scripts.wm02.profile_source --spec "$WM02_PARALLEL/EXECUTION_SPEC.json" --output "$WM02_PARALLEL/profile-01" --run-id profile-01 --profile local-PC-native
# exit0，实际1/4进程，16未运行
```

本轮重新执行20项world-model合同测试，普通和`python -O`均exit0；相关原SO3／SafetyGate／target bridge／wire identity接缝31通过、5跳过。私有parser在上文另作真实代码CPU验证；跳过项不冒称PASS。一个Torch输入梯度接缝检查得到24且冻结参数无梯度；不是模型学习结果。全仓库CI未跑。

保留的工程负结果：首次复用测试缺population模块（已按固定SHA补齐依赖）；Core没有rg（改用仅绑定源码树内查找）；首次P0的scorer字符串指向不存在模块，已改为候选审计函数并明确UNQUALIFIED。末次修正仅强化partial区间mask，CPU复测通过；没有热改已完成native capsule，也不将先前native结果改写为该新SHA执行。

下一条已经实现、无仿真副作用的可执行命令：

```bash
PYTHONPATH=src python -m scripts.wm02.qualify --help
```

继续目标P1之前，准确缺口是当前S1/T2独立安全解除／合法动态窗口，以及完整控制状态导出或经验证的相同前缀重建；GPU1空闲不能替代它们。P1奖励五类排序、目标接触／吞吐、P2数据／模型和P3以后均未完成，不能启动长训补盖这些缺口。
