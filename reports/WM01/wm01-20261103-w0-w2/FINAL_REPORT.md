# WM01 实际阶段回执

2026-11-03，Asia/Singapore。W0 已完成实际源域采集和恢复验证。W1 首次 GPU 作业因设备 UUID 格式在模型构造前失败，已独立闭合；修复后唯一后继 `1515716.stdct-mgmt-02` 正在等待实际 A40 容量。W2 尚未运行，需要真实 5000-update checkpoint。本文不把排队、程序实现或测试通过当作模型训练结果。

## 身份与隔离

| 用途 | 完整源码 SHA |
|---|---|
| 用户固定交接，PR342 未合并 | `de64c252246e4746315f2a324535b5d48912783c` |
| 首次 native canary，保留失败 | `6e9c0d9758e248e59398edee76f359d4b5f1dd4d` |
| 通过恢复／S1 的 canary 和 TRAIN64 | `f208a11c6c20603d3e9f9a5ca67b2fac3313f20f` |
| CALIB16、reward V2、首 GPU 入口 | `953801963b926dad23d1c8d11a1574c46235a383` |
| 修复 GPU UUID 的后继 W1 | `1a7462086451af1c9416b63c504759ad2a92c13a` |
| 预备 W2 方向导数／分叉实现 | `bb4b53d8f55fe0c65f0b70193894789f931d02cd` |

独立 Git checkout／私有 source capsule、run ID、数据、配置和输出；原工作目录及其未提交内容未被切换、reset 或修改。Core 的 PB_CONTROL/M13 和 consumer08 保留。接管探针曾见 PB_CONTROL1504/4096；这是该次真实探针，不是此刻状态。旧 Vanda1514400 独立核为F/exit1，日志缺 native10 RESULT；没有重投该旧任务。

新物理运行在独立 PC CPU profile，Python3.10.12、NumPy1.24.4、MuJoCo3.2.7；27份源资产与 Core 实际绑定逐份 SHA 一致。它不是 Core GPU／源环境的部署资格。AMD 只承担原 SSH 管理路由。WBC、腿策略、gain、力矩、solver、目标隔离和原评分未改。没有加载 GR00T、训练 WBC、运行目标整机或真机。

## W0：原生链和真实数量

实际复用：

- engine：`scripts.pi0294.scene.make_engine`。
- driver：`scripts.pi0291_arx5_data.engine.Engine.step`，2ms、8个 actuator、固定底座。
- Teacher：`scripts.pi0294.teacher.make_teacher(T1)` → `scripts.pi0291_arx5_data.teacher.Teacher`。
- scorer：`scripts.pi0294.score.score(T1)` → `src/go2arx5_vla/target_sim/scorer.py::score_episode`。
- safety：原 `gate_action`、`scripts.pi0294.s1_preflight.predict/bind_install`、原逐 tick live/shadow1e-10检查。

T1 持续成功仍要求完整32秒、真实持续抓握／抬升、水平运输，以及28–30秒全部41个原生保持采样通过。没有使用 whole-body night_score。

| 人口 | 尝试 | 完整32秒／原评分成功 | S1拒绝 | 真实50ms区间 | 完整0.4s macro | 主轨迹积分 | S1影子积分 | CPU秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN canary | 4 | 4/4 | 0 | 2560 | 320 | 64000 | 70400 | 78.9944 |
| 固定 TRAIN | 64 | 53/64 | 11 | 34536 | 4317 | 863400 | 865163 | 1033.7679 |
| CALIB | 16 | 16/16 | 0 | 10240 | 1280 | 256000 | 256000 | 309.9865 |

Canary 另有6400个真实恢复分叉积分，已经包括在其额外恢复成本，不能从主轨迹计数推断遗漏。TRAIN 中32次原 Teacher、32次受约束命令扰动；11次拒绝均保留，没有补成功或把被拒命令当专家标签。零主轨迹硬安全违规。以上成功属于 Teacher 数据采集，不是 WM policy 成功率。

恢复保存234维 MuJoCo完整积分状态及独立IK state、ctrl/act、warmstart、目标、IK cache、RNG、原生 warning 和监测历史。在16个固定状态上恢复两次，后续状态／接触／终止一致；随后又把32条恢复分叉与先前**未中断原轨迹**逐值比较，包括完整积分、控制目标和接触，最大误差0。后者额外6400主积分＋6400影子积分、12.6109CPU秒。加入Torch的原生环境接缝又在一个抓持状态核2条分叉：额外400＋400积分，误差0，约2.47CPU秒。

首次失败 canary 保留：1119主积分＋400恢复积分＋1600影子积分；warmstart差3.94369e-9触发原1e-10门。修复去掉薄wrapper多余 snapshot/forward，原门没有放宽。此为无效工程尝试，不计成功canary分母。

实际字段／单位和原始SHA摘要见 [RESOLVED_SPEC.redacted.json](RESOLVED_SPEC.redacted.json)。首点按原合同绑定为t+0；每点消费25个原生tick；prefix8之后的H40尾部不作为已执行监督。

## 数据身份和负结果

旧73条闭合采集记录含真实观测／命令，但缺完整驱动和IK/RNG恢复记忆，完整WM转移复用0。没有把旧34563个VLA窗口装入WM replay。新TRAIN/CALIB/DEV/TEST数量64/16/20/50，先固定scene/parent再切窗；与审计人口无精确几何重复，15个近重复配对保留报告。TEST只做定义注册和去重，未构造原生环境、未进入replay或统计量、未评价。

新 TRAIN 视图：64次尝试，4317完整macro，4125个四步窗口，以及11个仅终止事件。安全预检拒绝没有未来状态／接触回归目标。统计量只来自TRAIN。

- 私有 HF 数据集：`leoyuer/pi0294-private-policy-delivery`。
- 固定 revision：`6ba73c0fcb957632bea7727780af35ddcdcbb51c`。
- 路径：`WM01/20261103/native-train64-v1/TRAIN_REPLAY_V1.npz`。
- SHA256：`1423ba5a6afd7a9e2c5fd05e9574162ce705ce8ce1032e2742c92b8ba60e73b2`；2,843,340bytes。
- CALIB 数值视图 SHA256：`ea6e9ffb60bd2fbe7eb99a6fe350929210d4877d615b675e2f7a37edf10d9060`，不参与拟合。

Reward V1 在真实TRAIN状态上失败：未释放势值0.5646858，高于完成释放0.5606848。优化前固定V2：对应势值0.3764572／0.7408916；原S1真实影子碰撞回报−0.9789224。解析／中心差分方向导数1.3894772／1.3894721。FP32差分舍入曾产生约0.00192误差，保留失败后用FP64数值检查，不改变生产FP32计算。未观察到本轮的非预期失抓案例；这不证明恢复能力或终止头已校准。

## W1：程序已实现，实际训练待后继分配

110维 privileged state、56维 task-space action；3×512残差SiLU ensemble，成员各675451参数，总2026353。使用合法SO(3)更新、几何loss、4步自回归损失、TRAIN归一化、独立bootstrap及优化器/RNG/replay恢复。CPU合成合同检查已验证非零动作梯度、WM参数冻结、无未来teacher forcing和恢复后的下一次更新逐值一致。

首作业1515661：PBS F/exit1，分配33秒、实际cput1秒、36CPU/240GB/1A40；WM模型构造0、更新0。实际Torch UUID缺`GPU-`前缀，NVML查询失败。旧身份CLOSED、结果和费用保留。修复后用同一科学配方／同一数据只提交一次1515716：200-update smoke后完整恢复到5000，额外4次ensemble恢复验证更新另计；walltime1小时。排队不计作已分配GPU或训练。

W1 完整模型、参数更新、峰值显存和原生 W2 成绩目前均不可声称完成。W3 actor 更新0。

## 真实命令、检查和下一动作

以下为实际Journal内部CLI，路径用已绑定run别名表达；完整绝对argv、进程、PBS提交和私有输出在执行者的私有 `ACTUAL_COMMANDS.private.md` 与各run manifest保存。

```bash
python -m scripts.wm01.collect --spec "$WM01_NATIVE/RESOLVED_SPEC.json" --run-id canary-01 --profile local-PC-native --output "$WM01_NATIVE/canary-01" --attempts 4 --restore-check
python -m scripts.wm01.collect --spec "$WM01_NATIVE/RESOLVED_SPEC.json" --run-id train64-01 --profile local-PC-native --output "$WM01_NATIVE/train64-01" --attempts 64
python -m scripts.wm01.collect --spec "$WM01_CALIB/RESOLVED_SPEC.json" --run-id calib16-01 --profile local-PC-native --output "$WM01_CALIB/calib16-01" --attempts 16 --split CALIB
```

实际检查：18项新合同/模型/诊断/UUID用例，其中新UUID接缝普通／-O各通过；其余16项普通和-O均通过。70项原trajectory/SO3/scorer接缝通过，原17项计划检查通过。不是全仓库CI、GPU smoke或模型决策资格。

下一条可直接执行的只读命令：

```bash
/opt/pbs/bin/qstat -fx 1515716.stdct-mgmt-02
```

实际5000端点经固定revision下载／SHA核验后，才执行预登记的64状态×5臂、每臂最多4macro的原生诊断。进入actor BPTT仍需改善比例≥60%、优于等范数随机、排序优于action-shuffle且零硬安全违规；不够就交付负结果。
