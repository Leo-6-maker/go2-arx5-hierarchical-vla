# PI0284 终局：执行完成，持续物体任务资格未通过

固定授权 fc9647b2183e5bb9d3943f4a57565ed9751e0739 / Issue #11 canonical 6053658526。
两臂各2048更新、四份完整1024/2048 Trainer、384离线行、32实际源域闭环及64 DROID查询全部完成，独立CPU复算通过。没有合格候选，CONFIRM未准入；source_candidate、selected、deployment均为NONE。

## 固定质量表

| 点 | TRAIN fit B0/B1（各24） | VAL fit B0/B1（各24） | TRAIN持续成功 | DEV持续成功 | 首次成功TRAIN/DEV |
|---|---|---|---|---|---|
| CTRL1024 | 0/24 · 0/24 | 0/24 · 0/24 | 0/4 | 未安排 | 0/4 · — |
| CTRL2048 | 10/24 · 10/24 | 6/24 · 6/24 | 0/4 | 0/8 | 2/4 · 3/8 |
| LR5X1024 | 1/24 · 2/24 | 0/24 · 1/24 | 0/4 | 未安排 | 0/4 · — |
| LR5X2048 | 7/24 · 12/24 | 8/24 · 7/24 | 0/4 | 0/8 | 0/4 · 2/8 |

fit为原prefix8 XYZ≤10mm、SO3≤0.10rad及夹爪MAE≤0.10的联合门。持续成功仍按完整32秒及28–30秒全部41个20Hz样本的原on_top/static/not_grasped门；首次成功不能替代持续成功。
四点的approach阶段均零fit。LR5X2048的TRAIN10001发生ANGULAR_SPEED_REJECT，原gate拒绝参考、未安装；保留为有效失败，没有重试。其余实际结果与每个失败分母见CLOSED_LOOP32.csv。

## 实际干预与完整性

固定EVENT1024全Trainer/Adam5120起点；两臂同16384 draws、9152 unique UID、相同实际输入/完整132维BF16噪声/time，microbatch4×accum2、两个实际loader workers。共同projector LR1e-4，仅DiT LR1e-5/5e-5不同。Native flow objective、统计、相机、decoder、S4、SafetyGate及Panda native PD保持原绑定。
每臂真实8 vs 4+fresh-restore+4 canary通过；两臂正式4096微批全量配对、四份14成员库存与冻结范围/Adam/S4审计通过。终点global Adam7168。Hopper 654134 Exit_status=0；Stageout_status=1不能改判为训练失败。
保存的详细梯度只有每臂48/4096微批（更新1–8及128倍数），其余UNOBSERVED；旧紧凑status中的INTERVAL16文字不正确。完整输入/噪声/time、loss及LR记录不受此影响。

## 生成与条件响应

K16/K4 TRAIN median J比1.12289，只有8/24改善，首次release夹爪回退0.030975>0.02；按预注册门冻结K4，没有K16物理panel。
自然观测只有1个合格不相交pair，少于6：所有弱响应判断INCONCLUSIVE。真实pixels/backbone features不同，噪声字节相同。parent/CTRL/LR waypoint7响应比例约1.656/2.156/1.958，不能将单pair写成因果视觉消融或足够的条件泛化证明；endpoint39、方向和残差见ENDPOINT_NATURAL_CPU原件。
DROID每臂32/32有限、原decoder精确复算通过；相对pristine的漂移只是描述性证据，没有完整多任务成功标准。

## 参考、安装、跟踪与任务

逐episode分开列出reference→installed、installed→同次控制后实际TCP、reference→实际TCP和物体原生成功；不使用两minimum的差值冒充tracking。物体lift高度只作描述，未补造历史漏存的lift资格门。
父模型首次丢失成功8.85s时仍on_top且not_grasped，但static=false（角速度0.725277>0.5rad/s）；同tick安装误差0.02069mm、实际跟踪7.116mm。8.95s曾恢复成功，随后再次失败。这不是单一接触/控制器/模型根因证据。
全部实际闭环为SOURCE_PANDA/native PD、非实时；不是师兄WBC、Go2迁移或真机。四个视频用同一预固定TRAIN10000、真实保存的BASE/WRIST和原物理标志，未新渲染或重跑。

## 预算与封存

正式4096 + canary32 = 4128 completed；新quality 3335/6400，heads 15644/102400；starts 32/64，controls 20904/45000，physics 104520/230000，RGB 20905/45000，dispatch 41808/90000。
双机科学GPU进程闭合区间并集 2.128120/12h；归档前新增实际占用 113.736462/120GiB。个人quota UNKNOWN；df不代表个人额度。CPU聚合core-hours未完整观测，不将PBS cput当双机总量或宣称CPU精确总量PASS。
四份Trainer均HF PUBLIC固定revision完整14成员readback；原RGB、唯一原始证据和本地恢复点仍受保护。compact evidence及真实视频独立封存；报告/归档问题不重训。

## 科学停止点

COMPLETE / VALID_QUALITY_FAIL；不补CONFIRM，不按中期点、噪声或episode追成功。原科学失败及前序预算保留。下一任务设计仅NOT_AUTHORIZED/launch_allowed=false；需要另行明确任务对照与预算，不自动扩量、恢复采集或WBC。

## 离线prefix误差（所有固定行，分组中位数）

| 点 | split/noise | XYZ mm | SO3 rad | gripper MAE | fit |
|---|---|---|---|---|---|
| CTRL1024 | TRAIN/B0 | 27.354 | 0.026252 | 0.011230 | 0/24 |
| CTRL1024 | TRAIN/B1 | 25.890 | 0.024673 | 0.010010 | 0/24 |
| CTRL1024 | VAL/B0 | 29.577 | 0.029195 | 0.023560 | 0/24 |
| CTRL1024 | VAL/B1 | 28.389 | 0.033294 | 0.015991 | 0/24 |
| CTRL2048 | TRAIN/B0 | 10.122 | 0.011831 | 0.003784 | 10/24 |
| CTRL2048 | TRAIN/B1 | 9.504 | 0.011150 | 0.005005 | 10/24 |
| CTRL2048 | VAL/B0 | 10.419 | 0.013150 | 0.008179 | 6/24 |
| CTRL2048 | VAL/B1 | 9.983 | 0.015781 | 0.007690 | 6/24 |
| LR5X1024 | TRAIN/B0 | 20.601 | 0.032507 | 0.007935 | 1/24 |
| LR5X1024 | TRAIN/B1 | 16.540 | 0.031588 | 0.005859 | 2/24 |
| LR5X1024 | VAL/B0 | 23.821 | 0.036573 | 0.014282 | 0/24 |
| LR5X1024 | VAL/B1 | 24.037 | 0.045937 | 0.021729 | 1/24 |
| LR5X2048 | TRAIN/B0 | 10.466 | 0.009311 | 0.005127 | 7/24 |
| LR5X2048 | TRAIN/B1 | 9.663 | 0.009457 | 0.009277 | 12/24 |
| LR5X2048 | VAL/B0 | 10.346 | 0.011588 | 0.004883 | 8/24 |
| LR5X2048 | VAL/B1 | 10.156 | 0.012407 | 0.007812 | 7/24 |

逐阶段/全H40/p95分解见OFFLINE_PHASE_NOISE_SUMMARY.csv。全部384行保留raw归一化OOB、原native inverse clipping后的物理误差及独立gate；原逆变换clip不等于生成有效。

## 四份完整Trainer固定HF版本

| 完整Trainer | PUBLIC固定revision | 完整树SHA256 |
|---|---|---|
| CTRL2048 @ 1024 | [efa8cca500bbb7b0f25a868d08654b495e16acbb](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/efa8cca500bbb7b0f25a868d08654b495e16acbb/arms/PI0284/CTRL2048/checkpoint-1024) | c2f1e8a2ceca060fc533a906982bcdbc0ca544f56dc63425e8dae6e5db46f752 |
| CTRL2048 @ 2048 | [fa84b4b7d054fae1426e117bc59405ea5e9e3e17](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/fa84b4b7d054fae1426e117bc59405ea5e9e3e17/arms/PI0284/CTRL2048/checkpoint-2048) | 6ab7615180306b744f3e266fec5dc96e1f810d178020e217bb9e746ae75d7108 |
| LR5X2048 @ 1024 | [038346f257479513cb00478466d5546717b966ae](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/038346f257479513cb00478466d5546717b966ae/arms/PI0284/LR5X2048/checkpoint-1024) | 2777645a2a67bf972085bbadd0b99f273970d03892c7c3d5838bbb87ccec5529 |
| LR5X2048 @ 2048 | [6677220016c4524eeda5a5bba5dbce801f9b9c67](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/6677220016c4524eeda5a5bba5dbce801f9b9c67/arms/PI0284/LR5X2048/checkpoint-2048) | 3ddf536579f80858793dbe970fe666c4c3122533942430e384a35c1a44e6ad6c |

## 完成封存与实际视频

- core: [固定证据树 3b4f8471931cc1696967125508e1a3a4d89b8150](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/3b4f8471931cc1696967125508e1a3a4d89b8150/evidence/PI0284/native-generation-conditioning-20261008/core)，完整逐文件回读PASS，10344原件。
- hop: [固定证据树 3ed802d1374507b849e312328ce59e394fbb5f67](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/3ed802d1374507b849e312328ce59e394fbb5f67/evidence/PI0284/native-generation-conditioning-20261008/hop)，完整逐文件回读PASS，270原件。

双端归档后实际新增占用 114.315495/120GiB；没有删除完整恢复点或原RGB。compact包内的TERMINAL_RECEIPT是封存时的科学终局，归档完成收据另列本报告/BUDGET/MODEL_LOAD_RECIPE。

- [CTRL1024：固定TRAIN10000真实双视图](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/3b4f8471931cc1696967125508e1a3a4d89b8150/evidence/PI0284/native-generation-conditioning-20261008/core/CTRL1024_TRAIN10000_ACTUAL_SOURCE.mp4)
- [CTRL2048：固定TRAIN10000真实双视图](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/3b4f8471931cc1696967125508e1a3a4d89b8150/evidence/PI0284/native-generation-conditioning-20261008/core/CTRL2048_TRAIN10000_ACTUAL_SOURCE.mp4)
- [LR5X1024：固定TRAIN10000真实双视图](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/3b4f8471931cc1696967125508e1a3a4d89b8150/evidence/PI0284/native-generation-conditioning-20261008/core/LR5X1024_TRAIN10000_ACTUAL_SOURCE.mp4)
- [LR5X2048：固定TRAIN10000真实双视图](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/3b4f8471931cc1696967125508e1a3a4d89b8150/evidence/PI0284/native-generation-conditioning-20261008/core/LR5X2048_TRAIN10000_ACTUAL_SOURCE.mp4)

保存的query latency中位数：父K4 0.13347s、K16 0.38925s；CTRL2048 S4 0.14503s、LR5X2048 S4 0.16642s。缓存/复用/加载上下文不同，不能直接当作实时闭环保证。旧48个K4 packet0保存的latency来自前序查询，未重跑；其Euler中间轨迹仍UNOBSERVED。

训练实际稳态4064微批：CTRL fwd/back中位0.406761s，loader等待0.031965s；峰值allocated 26203141120 bytes/reserved 31746686976 bytes。显存占用不是GPU利用率；本轮没有额外加速参数搜索。
