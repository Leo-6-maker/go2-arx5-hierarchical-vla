# WM02：恢复搬运动作效应，交付带时序记录的初始化候选包

从未合并 PR #350 的固定提交 `ed2755a10f4cb4de445ffadd872317ee27baab4f` 建立独立 checkout。**本轮没有新增物理采样；固定比较完成 M1/M2 各200主更新＋1续训探针。另保留一次工程失败前20主更新，本轮实际优化器调用共422次。Actor/Critic更新0，目标积分0，完整任务0。** 旧源码、旧capsule、WBC/腿策略、增益、solver、安全与S1/T2隔离均保留。

结果分开解释：**M2在已有FIT/诊断DEV恢复了H8搬运动作的正确方向与毫米级幅度；它尚未优于规则，时间响应也未校准。目标候选已实际编译、链接并验证动态库调用链；没有获得新的init-only动态范围，未进入原生初始化验收。**

可立即执行的只读交付检查：

```bash
python3 scripts/wm02/review_response_round.py \
  --report-dir reports/WM02/response-init-candidate
```

实际执行exit0，核验6个证据文件的SHA256；不构造原生执行器、GPU上下文或训练器。该命令不是下一次目标动态验收。后者的首个确定阻断仍是原owner独立有效的有限init-only范围，不能用本轮用户的GPU/工程授权替代。

## 执行源码与真实命令

| 运行 | 完整执行SHA | 退出码与实际效果 |
|---|---|---|
| 当前FIT B0/B1诊断 | `9215b93fec1eb6e04a714a40a53d57acc03e98c1` | 0；24 FIT父场景、120真实已存分叉；0积分/更新；OLS一次求解 |
| 首次M1尝试 | `efeed6247d13df0b920b1214fe7e7d53a7734a37` | 1；20主更新后进度文件重复创建失败；仅step1完整checkpoint，保留部分运行 |
| 修复后的固定200比较 | `486a237b23e8b97c157b152cd50d44fe9036e3b5` | 0；400主更新＋2续训探针 |
| 首次候选构建 | `55dfd41dc19f417f4957075abaf8045c76d3d7f8` | 1；主体编译/链接0，原进程ROS库路径未绑定，ldd检查拒绝 |
| 第二次候选构建 | `a0d9b960084178bd2ce7845faf07b412d2c23757` | 1；编译/链接/加载0，误用路径相等比较未版本化.so和实际SONAME |
| 最终候选构建 | `7ae02fd40bff0ca6e52255e4a23736f340f07bc5` | 0；主程序、SDK、两个可选外层、observer、trace全部实际构建；两条loader检查与原并发harness0；0目标构造/积分/腿策略查询 |
| 原批处理语义审计 | `44007235f77d3c9e18d0d51e8a01e2ab6e5fa33c` | 0；M0有1/96场景×horizon选择不一致；M1/M2一致，发现不当作通过 |
| 单候选路径独立重评分 | `7ae02fd40bff0ca6e52255e4a23736f340f07bc5` | 0；M0/M1/M2各96/96选择、全部输出严格一致；0积分/更新 |

私有绝对路径和argv留在实际各run的`LAUNCH_INTENT.json`、`EXECUTION_SPEC.json`、`COMMANDS.private.json`；不上传主机绑定信息或vendor源码/权重。以下以原绑定的路径变量表达实际调用。每次均是独立不可变源码快照、原registry和Journal，有限用户systemd worker，Restart=no。失败任务已由确切更新日志、FAILED.json、MainPID0/终态与CLOSED lease独立核清；原INTENT不盲重放。失败修复使用新身份，没有覆盖历史capsule。

```bash
# PC：实际B0/B1 worker，CUDA不可见。
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$R/source/src:$R/source" \
  "$WM02_CPU_PYTHON" "$R/worker.py"

# Core：实际神经worker运行环境，现有GPU0 UUID和owner锁。
CUDA_VISIBLE_DEVICES="$WM02_GPU0_UUID" OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 \
  CUDA_CACHE_PATH="$R/cache" TMPDIR="$R/tmp" \
  "$WM02_GPU_PYTHON" -B "$R/worker.py"

# Core：实际候选构建和CPU重评分worker，CUDA不可见。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR="$R/tmp" \
  "$WM02_BOUND_PYTHON" -B "$R/worker.py"
```

worker实际调用已实现入口的`main(argv)`，没有把文档接口当作现成命令：

```text
scripts.wm02.diagnose_action_response
  --spec "$R/EXECUTION_SPEC.json" --output "$R/collected-01" --run-id collected-01
scripts.wm02.train_response_models
  --spec "$R/EXECUTION_SPEC.json" --output "$R/matched-200-01" --run-id matched-200-01
scripts.wm02.build_init_candidate
  --spec "$R/EXECUTION_SPEC.json" --output "$R/candidate-01" --run-id candidate-01
scripts.wm02.audit_response_evaluation
  --spec "$R/EXECUTION_SPEC.json" --output "$R/audit-01" --run-id audit-01
```

神经worker为4 CPU/10GiB RAM/2400秒端点/4GiB CUDA allocator上限；候选为另一组4 CPU/8GiB RAM/1200秒，CPU重评分也在Core，AMD仅中继。没有另建调度或审批系统。完整run/spec SHA、各次退出码与成本见[执行身份](EXECUTION_IDENTITIES.json)。

## B0/B1：在哪一段失效

[当前FIT诊断](CURRENT_FIT_RESPONSE.json)使用本轮40布局V5数据、当前checkpoint，而非旧640窗口。所有FIT搬运锚点的五候选共用完全相同的起始state、8步history/reference/clock与hidden；保存每候选的TCP/物体H1/H8/H16实际与预测位置、分数、输入分布，以及真实状态上的单步误差诊断。完整预测数组仍在私有运行目录。

| 当前M0的FIT朝向−背离分量 | 实际TCP | 预测TCP | 实际物体 | 预测物体 |
|---|---:|---:|---:|---:|
| H1 | 0.294084mm | −0.083546μm | 0.210164mm | −0.262337μm |
| H8 | 7.936172mm | −2.742705μm | 7.942255mm | −1.511728μm |
| H16 | 7.899076mm | −3.384991μm | 7.905817mm | −3.050811μm |

**失效已出现在参考→TCP，并非只发生在TCP→物体耦合，也不是只有DEV泛化失败。** 精确FP32参数值临时提升FP64，仅作数值导数检查；原checkpoint不变。三个δ下有限差分与autograd一致，相对误差约1.4e−9～6.3e−9，但导数微弱且反向。计算图连通不能证明动作方向正确。

预先固定FIT父场景0–3×五动作20例，OLS直接拟合命令累计xyz/.004到真实ΔTCP/Δobject，不是新增WM训练。H8训练RMSE为45.63/49.12μm，余20 FIT为65.20/66.43μm，DEV8为50.71/54.31μm；H1/H8/H16全部32父场景方向正确。已存稳定抓持120分叉物体相对TCP位置漂移最大41.61μm、RMS31.27μm；刚性耦合近似只对这120例成立，不能推广到滑动、释放或建立接触。

[当前FIT损失与梯度](CURRENT_FIT_LOSS_GRADIENTS.json)重新用720窗口和当前条件/盲checkpoint分解：三阶段各240窗口、权重各1/3。条件模型连续项均值preclose .03985、transport .02973、release .02550；transport抓持1920/1920为正且无建立事件。4个预定梯度窗口/阶段只是探针，不是全FIT精确梯度。不能沿用旧640报告的阶段失衡解释当前模型，更不能据此宣称唯一根因。

## B2：固定200机制比较

原数据SHA256：`c12293bc877758254a14d19ec112f147fe19c93ffacf4c8399ae8ebfcb42c808`。原数据来源源码：`7e41cd69c3c215e91cb2f85688e5426d4a7f4d2c`。**没有新增布局、转移或目标数据，也未再次评价已见holdout。** 训练只用24 FIT父场景、720窗口；每臂51200基本序列曝光、相同seed、初始化、normalizer、batch256/micro32、AdamW、固定200端点。

M0保留当前200端点。M1仅改网络内部参考特征：当前TCP坐标下位置差/.01m、相对姿态、夹爪差/原物理范围；外部V2指令和原始数据不变。M2增加同parent/anchor/完整相同起始历史的H8 TCP＋物体效应差损失，尺度.004m、权重1；只用已抽取microbatch内第一块配对，不混不同父场景或第二块不同状态，共48074对。实现及曝光在训练前提交，见[冻结实验定义](../../../experiments/wm02/response_path_v1.json)。

| 模型 | checkpoint200 SHA256 | 主更新＋续训探针 | 参数量 | 重载1/200与199→200误差 |
|---|---|---:|---:|---:|
| M0 | `530a6e5b395e7eca76f12efcc6179b6af745b063faff4f6c0bdbc23b18b21f6f` | 本轮0 | 18,905,387 | 本轮不重训 |
| M1 | `404905951041604070b128c3a99d39f56eaefc54cf77c7523883d352820ba3e9` | 200＋1 | 18,905,387 | 0 |
| M2 | `1536f410178ce6e07bf0fa940e706e65c484a8acef5808c68f1313e21fedaa3f` | 200＋1 | 18,905,387 | 0 |

M0实际冻结推理副本SHA256：`4fdb580f9f85b2dbdcd6796ede621f6f11c76fca03ec42f1d867d4338692b69c`，父checkpoint身份保留。M1/M2初始参数SHA相同：`1f933cf8f0349eb198a9ed313adb462c7b0727c159025f65e2ac7bad8305be78`。

实际优化日志首10→末10更新均值：M1基本损失.48726→.02721；M2基本损失.46939→.02841，效应差损失.07117→.001066。M2改善效应差并未同时改善所有基本预测误差，不能仅按总损失挑选端点。

后续表格从独立CPU单候选路径重评分的[完整聚合](MODEL_RESPONSE_RESULTS.json)读取，未改模型/数据/事件/评分定义。数值以该JSON为准：

| DEV朝向−背离效应 | 真实TCP mm | 预测TCP mm | 真实物体 mm | 预测物体 mm |
|---|---:|---:|---:|---:|
| M0 H1 | 0.291499 | -0.000079 | 0.199939 | -0.000255 |
| M0 H8 | 7.944711 | -0.002650 | 7.952278 | -0.001304 |
| M0 H16 | 7.900684 | -0.003361 | 7.905551 | -0.003038 |
| M1 H1 | 0.291499 | -0.834852 | 0.199939 | 1.806727 |
| M1 H8 | 7.944711 | 0.350792 | 7.952278 | 3.377808 |
| M1 H16 | 7.900684 | 0.171356 | 7.905551 | 3.092689 |
| M2 H1 | 0.291499 | 1.990899 | 0.199939 | 3.615174 |
| M2 H8 | 7.944711 | 8.104549 | 7.952278 | 8.089836 |
| M2 H16 | 7.900684 | 8.350751 | 7.905551 | 6.504787 |

M1改善动作依赖，但TCP幅度仍明显低估，H1 TCP还反号。M2在H8恢复了合理幅度；H1明显过早、过大，H16物体幅度仍低估约18%。H8效应监督没有证明完整时间响应正确。FIT/DEV方向一致不能替代新任务、跨物体或目标域泛化，也不应直接启动Actor或默认延长到800/5000更新。

## B3：不把规则回退包装成WM收益

事件按名称绑定：left_object_contact、right_object_contact、native_grasp、illegal_contact；没有support预测头。身份一律使用原生float64 raw命令，1e−6内并列按原命令哈希。原批处理审计发现M0在FIT父场景5/H8的选项随batch1/5改变，最大Q数值差3.58e−7；M1/M2选择未变。旧结果和这项失败保留，未伪报通过。

修复为**每个候选始终使用相同CPU batch1算术路径**，事件、分数、tie tolerance和命令身份保持原定义。独立新身份重评分，三个模型各32父场景×3 horizon，批大小／置换下输出最大差0、选项96/96一致。这是工程重评分，不是新盲测试。[原失败及修复审计](EVALUATION_SEMANTICS.json)

DEV回退预先按其他7父场景的动作对Q差预测误差q90标定，不能使用当前父场景标签决定自己的阈值。H8/H16分别报告：M0纯WM平均Q regret约.0407/.0790，规则与hybrid为0，但回退8/8；M1纯WM、规则、hybrid均0，回退8/8；M2纯WM、规则、hybrid均0，回退0/8。**M2无需回退不等于优于规则**。全部是已有DEV的交叉拟合诊断，没有Actor更新或新在线兑现，也不是校准置信度的外部测试。

## A0：实际候选包与未完成的验收

[候选回执](INITIALIZATION_CANDIDATE.json)已绑定R1同步、observer停止锁存、两个可选外层、SDK版本记录及实际加载链。两外层各自组成一条链，不能同时叠加：

```text
task_actuation.so 或 task_actuation_r1.so
  → observer.so → 冻结 libmujoco.so.3.2.7
并预加载同一个 libwm02_init_trace.so
```

完整候选主程序SHA256：`7828f9f29e174ac7bf9e82edd897de577a2d365ab22e6dec438f0565b09ff488`。R1正常控制数学体保持不变；SDK只插入观测记录，移除记录后与原代码逐字节一致；未改两队列的消费行为。原SDK archive只有rl_sdk.cpp.o，替换的是新候选副本。MuJoCo实际SONAME字节与原链接pin相同，SHA256 `848a26ad3daccac3a56d7269ac68167ed3497f048223f8720f4e56eb432eb6c6`。20项冻结输入在构建后复核全部不变；目标旧lease仍CLOSED、SHA `db2c30f675acb9e8dc280e3fc69d277e1bf91db10cfc2476881eeac2895be407`。

新增记录覆盖：观测序号/单调与仿真时间、推理开始/完整输入/history/完成、发布的policy版本、PD实际消费的pos/vel版本、PD目标q/dq/kp/kd/tau、实际q/qdot/ctrl、模式/暂停标志、积分前后tick、STOP和后续被拒绝调用。65536条固定容量、32个完整payload内存槽（首4与滚动28）、每payload最多16384 float；由10ms后台线程写盘，emit路径没有新的锁/等待/同步写盘。已有原生控制/observer日志保留，不能声称整个旧程序已无同步日志。

实际源代码还显示pos/vel两条独立队列通过短路`&&`消费，存在消费版本分离风险。新增记录器的并发假队列harness准确记录到pos版本2/vel版本1，并证明推理等待期间PD事件仍能出现。**没有动态证据证明这就是首次腿部越界的原因，也没有改变队列语义来掩盖风险。** 停止后额外积分的原离线双包装器回归仍通过；记录器假事件读出STOP后0条STEP_AFTER，只是夹具，不是目标STOP验收。

实际loader doctor只解析dlsym，不构造mjModel、ROS控制节点或腿策略。其trace确为0事件，读取CLI退出0；因此本轮目标STOP未观测，STOP后真实积分数为NOT_MEASURED，而非“通过0步”。两次0.1秒冷启动、2/5秒保持、EEF/接触验收均未执行。任意中途MPC/WBC/ROS receiver完整恢复仍未合格，不能声称目标反事实分叉资格。

取得原owner有限init-only范围后，才可用新run身份绑定本候选执行两次冷启动，再按0.1→2→5秒保持递进。无Actor/GR00T、抓放或行走，沿用原生安全。**本次交接没有该许可，因此没有反复运行已知exit78的目标入口。**

## 测试、成本与终态

PC world_model：45项，44通过/1跳过；神经31项全部通过。均普通及`-O`模式通过。Core训练相关CPU测试两种模式通过；实际C++主体/SDK/四共享组件编译、链接、ldd、两loader链、原R1并发harness均0。冻结batch1审计在Core实际执行。完整仓库CI NOT_RUN；这些检查均不授予目标动态资格。

- B0/B1及当前loss：CPU84.895s、wall21.908s。
- 成功训练：worker wall122.537s；systemd unit CPU143.687s、wall126.106s。失败20更新尝试另CPU15.840s、wall12.642s，成本保留。
- M1截至主更新200的峰值CUDA allocated660,690,432B/reserved729,808,896B；M2为737,198,080B/918,552,576B。实际本进程NVML采样峰值1298MiB，4GiB allocator上限。训练所在GPU0实际总32760MiB；GPU1总16376MiB，本轮未使用GPU1。测得容量支持这套模型在16GB内的显存可行性，不将GPU0冒充16GB实测。
- 三次构建CPU25.294/26.220/26.702s，wall27.796/26.378/26.900s；两轮审计CPU77.039/171.170s，wall20.677/44.496s。各次精确值见身份JSON。
- 所有Core unit CPU合计485.952s，加PC诊断合计570.847s；不包含测试/暂存/传输开销，也没有可核验的货币账单。并行运行的wall不相加冒充总耗时。

最终7个Core worker全MainPID0、无重启、lease CLOSED；已知旧PID727048/3892803/1425203均不存在，没有我们遗留的GPU进程。其他账户GPU任务未停止。未重新安装环境、扩大数据/GPU、改WBC、排长训练或启动Actor。

下一科学问题是M2的早期时间响应与H16物体低估；先冻结有限的时间分辨诊断/损失比较再考虑训练扩展。目标线的下一实际步骤由有效init-only范围决定；当前可执行命令是前述只读交付核验，获准后的目标保持运行尚未声称完成。
