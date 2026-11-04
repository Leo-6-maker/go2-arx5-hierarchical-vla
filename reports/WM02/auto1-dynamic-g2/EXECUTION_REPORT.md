# WM02 AUTO1：目标自由空间采集 → WM → 恢复 → 新批次续训；双环境试跑

2026-11-04，接续 PR #355 的固定公开提交 `08e0fd0ab64c42623842ccf3b1f87ff29ee9f538`，独立 checkout；未切换现役 source、合并 main 或修改旧 capsule。这里交付 **FREE_SPACE_TARGET_WM_ENTRY_VERIFIED**：真实目标数据与有限 WM 链路已通。接触入口仍未合格；未证明动作选择收益、独立父场景泛化或完整任务成功。

用户后续明确授权 GPU0/1 并行及更多环境。实际登记了同一 AUTO1 campaign 的两实例补充绑定，保留原绑定、原 owner 和原截止时间；一个父 owner 持有原 campaign/physical 锁，两个子进程继承已锁 FD。没有移除原锁，也没有新建队列服务。旧 S1/T2 lease 保持 CLOSED。

## 实际身份与数量

| 内容 | 实际结果 |
|---|---|
| 首批 A | 8/8 次原生通过，192 个 50ms 转移，8×600 任务 tick；初始化另计 |
| 首个目标 WM | 200 次主更新 + 1 次独立 199→200 恢复探针，全部误差 0 |
| 第二批 B | 首个 checkpoint 冻结后采集，4/4 通过，96 个转移；先评价旧模型，再续训 |
| 后继 WM | 100 次主更新 + 1 次独立 99→100 探针；模型、optimizer、RNG、cursor 恢复误差 0 |
| 双环境 C | 2/2 并行通过，48 个转移，独立 Replay；未用于以上训练 |
| 目标数据 | 336 个任务转移；已训练 A+B=288，C=48；每 episode 24 atoms/25 states |
| 父场景/DEV | 1 个原始物理父场景；全部 TRAIN，DEV=0。不同运行时初态不冒充独立父布局 |
| 本轮原生积分 | 80,609；前份报告 12,353；campaign 合计 92,962 |
| warmup 与诊断 | 本轮初始化/短冷启动 70,184 tick；其余 10,425 包含任务、诊断、失败及 TTL 边界。不能全部当学习标签；TTL 两次 partial 分别 22、3 tick，不能跨 episode 拼接 |
| WM / Actor / 完整任务 | 主更新 300，恢复探针 2（实际 optimizer 调用共 302）；Actor=0；完整任务成功=0 |
| 终态 | 导出时所有 40 个相关运行/slot 的 worker lease CLOSED，所有已登记 systemd worker MainPID=0 |

实际采集源码：A1–A2 `e93ddbbd667652121a801f4230b97f800f03c7c7`；A3–A4 `16124cddf67dd0bb779711b66d92b21d2fad2d36`；A5–A8 `49777cc1f24c044b5db87fd6885673ce265d9ff1`。B1–B2 `a852e37c998ac511b55f76cb963617856b4c4944`；B3 `e494a1bcc8532205998eec45fb2ba4c58c5a84c1`；B4 `c034892ff2da4437b40939485c62396eba4a863f`；并行 C `0199ea2a5afbf88f2d6d8e5daa064bda8731ec60`。

首训源码 `a852e37c998ac511b55f76cb963617856b4c4944`；续训源码 `05ec449b55054172f3b7e880dd0ec97b3a25df76`。后续并行摘要计数/同批失败关闭修复源码 `dde452045b53b12d093f597cb422b17c014989da` 未热补到旧运行；该修复只经组件测试。

| 数值数据 / 模型 | SHA256 |
|---|---|
| A Replay | `711768258c7de3bb131293da77d1c38ca3cea1a306ec7890de83b2fb256ce182` |
| A+B Replay（12 episodes） | `c29390ff49c66164f46d49b941e410fd276c95f4472635a256a140e65e2c515c` |
| C Replay | `a10d88a63d7e738a32f7130aa9ec8a097ea490a0d1e9c4ba4e3bf9bf9626ea77` |
| 首个 checkpoint-200.pt | `913d620b5ce33ce489126a670d7d2d119f4918b52215e8103f07496c2016a02c` |
| 后继 checkpoint-100.pt | `9a0ef30ba0e79412312c434de504f94b7476611ffd3b6d484b929926445dfd51` |

模型是新实现的 74,986 参数 history-conditioned GRU WM，实际 native schema 106 维：qpos34、qvel32、ctrl20、真实 TCP/object pose 及两夹爪位置。burn8/predict16，固定的预先声明目标及时间/缺失 mask；未输入未来 achieved state 或未来真实 renewal 源时间。缺少完整 MPC/WBC/ROS 隐状态，不称完整 Markov 状态或严格同状态分叉。normalizer 仅 fit 首批 TRAIN，续训固定；optimizer/RNG/cursor 连续承接，最终 cursor=2400。没有加载源域 M4 参数、GR00T 或 Actor。

实际目标 binary `e9aab70b6c1d2af76c6e178fed1fd8e86299dffb9624d15a85e58ba4d3f426ff`；构建源码 `959337e68034fdb29f5cfa0819249cbd574e1e97`。实际加载 controller `885042aee96eec2761e5cc9c1598194c94181bebe0de0548eb48b1110ddaa186`；solver `4a5f5aa6e0b9d4bde62f4d1da73135ef97e5268945a0cecf114d9190d3e04fd0`。场景 MJB `70663bf41a235f8f2baf179447720bd1ca845ce1abf847898ba884a46559fc55`；leg policy `b07c2d494a47cdae2a97fd9b2471d8f6f243049741e949414a5d099fa5148e3a`；outer R1 `246f0ab1a46956621eee6ac79b9b824b248d1dcd06dc07d8c026f72e002e3ae0`；observer `b0fd41b68702cfa889c28daa9be101ecc903200324255389572d377a062cf6b6`；trace `bd92a15c9ec4e435c3ee435878d94ed2e94075d38d6590d593c7c1201706eaea`；MuJoCo SO `848a26ad3daccac3a56d7269ac68167ed3497f048223f8720f4e56eb432eb6c6`。

## G2：真实负结果、修复和证据边界

1. 单纯 7D EEF 参考在 2,853 tick 触发原生 ILLEGAL_CONTACT：base_link unnamed geom12 对 task_table，距离 +0.0008377166584461393m、法向力 184.731765N。原生 STOP 后 **0** 额外积分，trace 丢包 0，子进程全部退出 0。保留该失败，没有放宽接触规则。
2. 原 native 14D 接口提供 EEF+BASE pose；7D 路径不启用已配置的 BASE pose tracking。新版本将实际初始 MPC BASE pose 作为 hold 参考，保留原权重、成本数值、控制律、solver、2ms dt 和安全阈值。此参考绑定变化会改变闭环行为，明确版本为 `MEASURED_EEF_AND_BASE_POSE_HOLD_V2`。不能据此断言已解释历史 T2 的全部腿部问题。
3. 修复源时钟浮点/ROS 整数纳秒比较、rolling trajectory 消费匹配、已删除 DDS/graphics SHM 误判、exclusive snapshot 重名及 renewal 重新锚定 tracking error 导致的 rate reject。renewal 保留预先声明的参考后缀；350ms 更新一次，原 source+400ms TTL 不变。
4. TTL 原生负例停在 step2700：source=5.0s、deadline=5.4s、真实 observation=5.4s，但模式仍 WBC。原因是 sim-clock timer 可能先于最终 observation 回调，暂停后不再触发。新增在真实 observation 更新后执行同一个原 expiry 判据；源码 `8b0f7feb63874f8adaac4a857cc07b933e84a8c9`。新运行 before2699=WBC，at2700=HOLD，after2701=HOLD，final2706=HOLD；无越过截止后仍处于 WBC 的推进。HOLD 与物理安全 STOP 不混称。
5. 已通过保持短/renew、+x/+y 1mm 自由空间条件。命令 1mm 不等于实际跟踪恰好 1mm。每个数据 atom 均有 exact installed target 与实际 solver policy payload 匹配及实际 arm request；**controller 最终选用的 policy ID 未导出，关联仍为 DERIVED**。不能提升为直接控制器选择证据。

受支撑抓持/释放撤离的四个条件尚未运行/合格；没有接触正例、站立操作任务成功或 contact event head。首个剩余目标操作缺口是这些条件的原生资格与实际接触数据。完整控制隐藏状态恢复仍未绑定；自由空间顺序采集不以它为前置条件。

工程失败也保留：TTL 首次私有 worker 误路由到 collection CLI，exit1/0积分；后继 WM 首次 baseline 子进程缺 PYTHONPATH，exit1/0更新，修复后新身份完成；双环境首次 env 重复 PYTHONPATH，exit1/0积分；第二次两个 cold child 均原生 PASS，但父进程使用 exclusive-create 更新 lease 而 exit1，修复并独立核实终态后结算，未重放原 Journal。其余逐次结果及完整源码 SHA 见 [NATIVE_RUNS.md](NATIVE_RUNS.md) 与 [NATIVE_AND_WORKER_ACTUALS.json](NATIVE_AND_WORKER_ACTUALS.json)。错误 overlay-base SHA 的一次本地准备命令在传输/物理前退出128，也不计原生失败或模型更新。

## 双环境：实际加速与限制

实际读取站点 profile 后在冷边界登记两实例：ROS220/221、CPU0–3/4–7，各自缓存、quota、trace、state、gripper、receiver、日志与 STOP。独立打开 canonical owner lock 会冲突；子进程验证继承 FD 的 inode、原锁仍被持有、实际父 PID/start、注册 spec/lease 与 slot pins。未更改站点 ENVIRONMENTS 或其他用户作业。

两个 50-tick cold child 通过；随后 2×2528-tick measured-base hold/reference 原生与父结算均 exit0；最后 2×3103-tick collection 均 exit0，24/24 solver-confirmed atoms，各子进程清洁退出。C Replay 数值构建验证 48 个 atoms、0 新物理积分。旧父摘要 `learning_atoms=0` 是硬编码遗漏；**实际量以两份原生 DYNAMIC_DIAGNOSTIC 和 Replay build 的 24+24=48 为准**，没有改写旧回执；后继源码已修正摘要。

并行采集总墙钟 15.9825s；近期相同 +x/−x 条件串行 B3+B4 合计 32.2315s，描述性比值 **2.02×**。没有随机交叉或严格匹配 benchmark，不当作普遍倍速保证。并发 hold 某次 inference 峰值 51.760755ms，采集两 slot 分别 19.168111/29.517260ms；观测到的、完全位于一次 inference 内的 PD 间隔峰值，hold≤7.722607ms、collection≤6.045006ms，原生 trace 无 mixed pos/vel、无 apply 缺配、无 STOP/丢包。该选定 PD 指标不代表全时间最大间隔，不能删除有意暂停时段来宣称新的实时性保证。

现场实测 20 逻辑核、约37.4GiB可用内存；GPU0/1 仍有其他用户任务，未抢占。此 native MuJoCo/ROS 执行链主要使用 CPU，CUDA 隐藏；WM 的 Tensor 显存峰值约68.6MiB。显存富余不能直接推出 16/32 个原生环境。当前实测资格仅 **2 环境**，未盲目扩到更多。

## 预测结果：有正结果，也有明确不足

B 是模型冻结后产生的新尝试，但与 A 同物理父场景，预登记为 TRAIN；以下是工程前瞻测试，不能称 held-out parent DEV。

| B 上 TCP position RMSE（mm） | H1 | H8 | H16 |
|---|---:|---:|---:|
| 冻结首个 WM，未看 B 更新 | 0.717 | 0.595 | 0.843 |
| Persistence | 1.217 | 2.989 | 3.326 |
| 局部数值 CV | 0.747 | 5.110 | 10.250 |
| 仅 A TRAIN 拟合的 ridge reference-residual | **0.254** | 0.730 | 1.125 |
| 续训 A+B 后，重新评价 B（训练集） | 0.353 | 0.287 | 0.296 |

冻结 WM 的 H8/H16 优于这些基线；H1 不如线性残差基线。物体在这些自由空间试验中静止，persistence 的 object position RMSE=0，而首个 WM 在 B 的 H16 为0.0412mm，明确更差。冻结 WM 的夹爪 H16 0.0575mm，也不如 persistence 0.0365mm。续训后 B 的降低是训练拟合，不是泛化证明。未评价动作排序、真实决策增量、contact events 或 Actor；不能据低 RMSE 宣称 WM 有控制价值。完整分字段 H1/H8/H16、基线身份与误差在 [TARGET_WM_METRICS.json](TARGET_WM_METRICS.json)。

## 实际入口、退出码和下一命令

实际启动沿用现有 `systemd-run --user`、有限端点、existing Journal/onceIO 和专属 worker；完整 argv/launch intent/进程命令与私有 spec 留在各自运行目录，未公开机器绑定资产。下面用已核验的私有绑定变量展开同一入口。关闭的旧 lease/输出不能原地重跑；新物理工作必须注册新身份。

```bash
# 原生 A/B worker 内调用；成功12/12为0，实际原生子进程命令另存 PROCESS_COMMANDS.private.json
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$WM02_LD_LIBRARY_PATH" \
PYTHONPATH="$WM02_SRC/src:$WM02_SRC:$WM02_NATIVE_PYTHONPATH" "$WM02_TARGET_PYTHON" -B \
  -m scripts.wm02.collect_target --spec "$WM02_RUN/EXECUTION_SPEC.json" \
  --output "$WM02_RUN/native-01" --run-id native-01

# G4 / G5 worker 内实际训练入口；最终各exit0，分别200+1与100+1
CUDA_VISIBLE_DEVICES="$WM02_GPU_UUID" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONPATH="$WM02_SRC/src:$WM02_SRC" "$WM02_CUDA_PYTHON" -B \
  -m scripts.wm02.train_target_wm --spec "$WM02_RUN/EXECUTION_SPEC.json" \
  --output "$WM02_RUN/fit-01" --run-id fit-01

# 两环境有限入口；parallel-hold-01、parallel-collect-01实际exit0
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$WM02_LD_LIBRARY_PATH" \
PYTHONPATH="$WM02_SRC/src:$WM02_SRC:$WM02_NATIVE_PYTHONPATH" "$WM02_TARGET_PYTHON" -B \
  -m scripts.wm02.parallel_target --spec "$WM02_BATCH/EXECUTION_SPEC.json" \
  --output "$WM02_BATCH/batch-01" --run-id batch-01
```

接下来的已实现、已准备好私有输入且尚未执行的命令，是冻结后继模型在独立保存的 C 新尝试上评价；无更新/物理。`WM02_REVIEW` 绑定本 campaign 的 `reviews/parallel-collect-01`，`WM02_SRC` 绑定已登记 `parallel-collect-01/source`，模型/数据 pin 已写入 `FROZEN_C_EVAL_SPEC.private.json`：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$WM02_SRC/src:$WM02_SRC" taskset -c 8-11 "$WM02_CUDA_PYTHON" -B \
  -m scripts.wm02.evaluate_target_wm \
  --spec "$WM02_REVIEW/FROZEN_C_EVAL_SPEC.private.json" \
  --output "$WM02_REVIEW/FROZEN_C_EVAL01.json"
```

后续实验优先独立父布局与受支撑接触入口；新 C 数据不会因数量增加就被改记为独立 DEV。源域 opening-response 的8新父前瞻试验尚未执行，不重复源域旧面板来代替目标进展。

## 测试与成本

最终普通和 `-O` 的完整 world_model suite 各 **97 项、0 skip、exit0**（实际 PC model-native Python；仅组件回归，原生资格以上述服务器实际记录为准）。最终源码后继批次摘要修复未作额外物理重跑。旧硬编码 `qualify --domain T` 的拒绝语义保留；测试输出中的旧权限拒绝不代表新 AUTO1 入口未执行。

相关 worker systemd 整个 cgroup CPU 累计 **778.532210 CPU秒**，包含失败与子进程；不含 staging、只读数据构建/评价和PC测试，故为总成本下界。200+100 主更新 CUDA stream elapsed 合计 **5188.679443ms**，不是 GPU 占用账单；两次 probe 单列。Torch allocator 峰值 **71,971,840 bytes**（驱动/context/其他用户显存不在内）。没有计费金额或能耗实测。完整成本边界见 [COST_AND_LIMITS.json](COST_AND_LIMITS.json)。

源码通过同一 PR 的正常追加提交发布，未 force push。GitHub API commit metadata 与执行时 local commit SHA 不同；每个完整 tree 逐一相等，执行过的原 commit 对象及公开 snapshot 对应见 [PUBLICATION_SOURCE_MAP.json](PUBLICATION_SOURCE_MAP.json)。原始私有数据、原生模型、WBC资产与 checkpoints 保留在原运行目录，不上传公开仓库。
