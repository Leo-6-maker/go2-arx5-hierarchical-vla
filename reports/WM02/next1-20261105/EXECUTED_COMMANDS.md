# NEXT1实际入口、参数和退出码

以下命令按公开展示需要把主机绑定路径替换为私有记录中同名变量；不是要求重新执行已完成实验。每次真实调用的完整argv、环境、spec和stdout/stderr保留在原`LAUNCH_INTENT.private.json`、`COMMANDS.private.json`和PC控制记录；run/spec/result SHA见相邻JSON。原生与模型worker均由原Journal.once执行其main入口，不重复派发未知状态。

| 实际阶段 | 调用 | 退出码与真实效果 |
|---|---|---|
| R0只读 | `PYTHONPATH=src python3 -B -m scripts.wm02.audit_stability_causality --replay "$C08_REPLAY" --references "$C08_REFERENCES" --replay-sha256 3493aa2ed00ceb2f9cd2c9f74a8c5a224f73be8da595be3d6abc4e74424a9a9a --output "$R0_OUTPUT"` | 0；新物理/更新0 |
| 原生单slot进程 | `systemd-run --user --unit="$UNIT" --property=Type=exec --property=RemainAfterExit=yes --property=Restart=no --property=RuntimeMaxSec=300 --property=TimeoutStopSec=60 --property=CPUQuota=400% --property=AllowedCPUs=0-3 --property=MemoryMax=8G --property=StandardOutput=append:"$RUN/logs/stdout.log" --property=StandardError=append:"$RUN/logs/stderr.log" --working-directory="$SOURCE" /usr/bin/env ... "$NATIVE_PYTHON" -B "$RUN/worker.py"` | 所有实际单slot退出0；CUDA隐藏，ROS220；环境完整值在原件 |
| 双slot进程 | 同上，CPUQuota=800%、AllowedCPUs=0-7、MemoryMax=16G，两个原隔离ROS域220/221 | 15个实际pair worker退出0；含双slot资格、10组TRAIN及4组DEV |
| 原生worker内入口 | `scripts.wm02.collect_target.main(['--spec', '$RUN/EXECUTION_SPEC.json', '--output', '$RUN/native-01', '--run-id', 'native-01'])` | 40个原生子执行均PASS；174920实际tick |
| pair worker内入口 | `scripts.wm02.parallel_target.main(['--spec', '$RUN/EXECUTION_SPEC.json', '--output', '$RUN/batch-01', '--run-id', 'batch-01'])` | 15/15 pair退出0；原件保留每slot实际记录 |
| 两次封存replay | `taskset -c 8-11 "$NATIVE_PYTHON" -B -m scripts.wm02.build_target_replay --manifest "$MANIFEST" --output "$NEW_REPLAY_DIR"` | 0；首次20 TRAIN/480atoms，后共享28/672；不把两份累加 |
| 真实动作差 | `taskset -c 8-11 "$NATIVE_PYTHON" -B -m scripts.wm02.audit_action_effect --replay "$S1_REPLAY" --output "$ACTION_GATE"` | 0；S1真实辨识PASS，0更新 |
| ridge/ARX | `taskset -c 8-11 "$TORCH_PYTHON" -B -m scripts.wm02.fit_target_baselines --replay "$MATCHED_REPLAY" --output "$RIDGE_DIR"`；`-m scripts.wm02.fit_target_arx --replay "$MATCHED_REPLAY" --output "$ARX_DIR"` | 各0；各1次TRAIN-only闭式拟合，神经optimizer更新0 |
| 三个模型进程 | `systemd-run --user --unit="$UNIT" --property=Type=exec --property=RemainAfterExit=yes --property=Restart=no --property=RuntimeMaxSec=600 --property=CPUQuota=400% --property=AllowedCPUs=8-11 --property=MemoryMax=10G ... /usr/bin/env CUDA_VISIBLE_DEVICES="$GPU1_UUID" ... "$TORCH_PYTHON" -B "$RUN/worker.py"` | RAW/SCALED/NO_ACTION各0；allocator上限4GiB；每次启动前真实查询显存 |
| 模型worker内入口 | `scripts.wm02.train_target_wm.main(['--spec', '$RUN/EXECUTION_SPEC.json', '--output', '$RUN/fit-01', '--run-id', 'fit-01'])` | 各200+1，全部严格restore0误差；不是科学Gate PASS |
| 冻结匹配评估 | `taskset -c 8-11 "$TORCH_PYTHON" -B -m scripts.wm02.evaluate_matched_target_models --spec "$FROZEN_SPEC" --output "$M0_EVALUATION"` | 0；M0科学Gate FAIL被正常输出，不以非零程序退出伪装学习崩溃 |
| PC冻结输入干预 | `PYTHONPATH=src "$PC_TORCH_PYTHON" -B -m scripts.wm02.audit_frozen_action_intervention --spec "$PC_FROZEN_SPEC" --output "$INTERVENTION_OUTPUT"` | 0；真实权重仅推理，新增优化/积分0 |
| C0静态复核 | `taskset -c 8-11 "$NATIVE_PYTHON" -B -m scripts.wm02.inspect_contact_approach_kinematics --config "$PINNED_CONFIG" --output "$STATIC_RESULT" --grasp-height-offset-m 0.02 --probe-gripper-geometry` | 0；当前step3703及新接触前缀step3503各绑定对应原件；无积分 |
| 分级参考静态证书 | `taskset -c 8-11 "$NATIVE_PYTHON" -B -m scripts.wm02.qualify_next1_contact_reference --config "$PINNED_CONFIG" --grade "$PREREGISTERED_GRADE" --output "$STATIC_CERTIFICATE"` | 原完整上方分块检查2，保留五次IK拒绝；连续克隆见证修复后新身份0；其余实际执行条件均先静态0 |
| 测试 | `PYTHONPATH=src "$PC_TEST_PYTHON" -B -m unittest discover -s tests/world_model` | 最终相关套件140 PASS，0；早期fixture/语法失败日志保留；不当作原生资格 |
| 终态只读 | `taskset -c 8-11 "$NATIVE_PYTHON" -B -m scripts.wm02.next1_status --binding "$NEXT1_BINDING" --output "$STATUS_OUTPUT" --gpu-query` | 0；40 native、600+3、28unit终态、95冻结文件一致 |
| 结算 | 私有一次性`close_next1_once.py`经原Journal.once、原owner与物理锁执行，仅更新原lease和canonical index | 0；NEXT1 CLOSED，原截止不变，旧NIGHT1未重开 |

所有工程失败原件、40个native RESULT、58个角色lease、28个unit退出、3套checkpoint、共享replay及模型完整私有评估都保留。新的只读输出须使用新文件名；上述采集/训练入口不能使用CLOSED窗口重跑。
