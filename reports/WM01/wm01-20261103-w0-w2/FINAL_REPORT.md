# WM01 实际阶段回执

2026-11-04 01:10，Asia/Singapore。W0 已完成。用户明确允许 GPU1 共享后，独立 Corelab 4080S 已完成200更新smoke及真实checkpoint恢复，峰值PyTorch分配135.76MiB。固定5000端点尚未确认完成，W2尚未运行。Vanda最后成功核验01:00:59仍Q，后续SSH查询超时，不能将旧排队快照当作当前状态。

## 身份与隔离

| 用途 | 完整源码 SHA |
|---|---|
| 用户固定交接，PR342 未合并 | `de64c252246e4746315f2a324535b5d48912783c` |
| 首次 native canary，保留失败 | `6e9c0d9758e248e59398edee76f359d4b5f1dd4d` |
| 通过恢复／S1 的 canary 和 TRAIN64 | `f208a11c6c20603d3e9f9a5ca67b2fac3313f20f` |
| CALIB16、reward V2、首 GPU 入口 | `953801963b926dad23d1c8d11a1574c46235a383` |
| 修复 GPU UUID 的后继 W1 | `1a7462086451af1c9416b63c504759ad2a92c13a` |
| 预备 W2 方向导数／分叉实现 | `bb4b53d8f55fe0c65f0b70193894789f931d02cd` |
| 离线留出误差／事件分析 | `84e6867fd7f1943aab6db72225b302cdbe6f68bc` |
| 实际 Core GPU1 共享smoke | `9018017d62753530c380fd81dc704413dd9ca8c3` |

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

Reward V1 在真实TRAIN状态上失败：未释放势值0.5646858，高于完成释放0.5606848。优化前固定V2：对应势值0.3764572／0.7408916；原S1真实影子碰撞回报−0.9789224。解析／中心差分方向导数1.3894772／1.3894721。FP32差分舍入曾产生约0.00192误差，保留失败后用FP64数值检查，不改变生产FP32计算。随后按原限速补了独立有界的抓握负例：0.4秒打开后仍抓持（目标slew尚未越过实际手指位置，保留结果）；1.2秒打开后真实抓持由true变false、物体高度0.52304→0.45989m，回报−0.19047，零硬安全违规。两次包含取得原TRAIN状态的完整重放，分别额外2200＋2200、2600＋2600主／影子积分，CPU3.117／3.469秒；不加入初始64或训练replay，不证明恢复能力或终止头已校准。

## W1：Core GPU1 smoke已完成，固定5000尚未完成

110维 privileged state、56维 task-space action；3×512残差SiLU ensemble，成员各675451参数，总2026353。使用合法SO(3)更新、几何loss、4步自回归损失、TRAIN归一化、独立bootstrap及优化器/RNG/replay恢复。CPU合成合同检查已验证非零动作梯度、WM参数冻结、无未来teacher forcing和恢复后的下一次更新逐值一致。

首作业1515661：PBS F/exit1，分配33秒、实际cput1秒、36CPU/240GB/1A40；WM模型构造0、更新0。实际Torch UUID缺`GPU-`前缀，NVML查询失败。旧身份CLOSED、结果和费用保留。修复后用同一科学配方／同一数据只提交一次1515716：200-update smoke后完整恢复到5000，额外4次ensemble恢复验证更新另计；walltime1小时。排队不计作已分配GPU或训练。

留出评价已实现持恒／固定动作打乱基线，H1/H4逐字段物理误差与component MAE、各Teacher阶段和接触分层、事件Brier／可靠性箱、ensemble成员两两分歧。CALIB只有成功轨迹，缺失败／恢复分层的真实样本；对应零分母明确输出，不能宣称终止头已校准。分析不更新归一化或模型。

Core共享smoke的真实模型与更新如下；固定5000模型和原生W2结果仍未交付。W3 actor更新0。

## 真实命令、检查和下一动作

以下为实际Journal内部CLI，路径用已绑定run别名表达；完整绝对argv、进程、PBS提交和私有输出在执行者的私有 `ACTUAL_COMMANDS.private.md` 与各run manifest保存。

```bash
python -m scripts.wm01.collect --spec "$WM01_NATIVE/RESOLVED_SPEC.json" --run-id canary-01 --profile local-PC-native --output "$WM01_NATIVE/canary-01" --attempts 4 --restore-check
python -m scripts.wm01.collect --spec "$WM01_NATIVE/RESOLVED_SPEC.json" --run-id train64-01 --profile local-PC-native --output "$WM01_NATIVE/train64-01" --attempts 64
python -m scripts.wm01.collect --spec "$WM01_CALIB/RESOLVED_SPEC.json" --run-id calib16-01 --profile local-PC-native --output "$WM01_CALIB/calib16-01" --attempts 16 --split CALIB
```

实际检查：21项新合同/模型/诊断/UUID/共享入口/留出评价用例，全部普通和-O均通过。70项原trajectory/SO3/scorer接缝通过，原17项计划检查通过。不是全仓库CI、GPU smoke或模型决策资格。

下一条可直接执行的只读命令：

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes vanda-simon /opt/pbs/bin/qstat -fx 1515716.stdct-mgmt-02
```

实际5000端点经固定revision下载／SHA核验后，才执行预登记的64状态×5臂、每臂最多4macro的原生诊断。进入actor BPTT仍需改善比例≥60%、优于等范数随机、排序优于action-shuffle且零硬安全违规；不够就交付负结果。

## 00:21容量快照与已启动的有限接续（历史观察）

最新实际观察 2026-11-04T00:21:39.158312+08:00：PBS `Q`，`Insufficient amount of resource: node_pool`，未给启动估计。有效站点项目与配额已核验；这次等待是容量问题。尚无实际 WM 模型、更新或W2结果，不能报告H1/H2通过或失败。

执行者已启动一次有限接续，读取同一1515716的真实结果；只在实际5000更新后，经私有HF固定revision与逐文件SHA核验，才调用已封存W2 source。接续不提交、取消或重投作业，任何实际失败保留后退出。PC原生接续截止 2026-11-04T05:22:07.131372+08:00，届时缺输入则W2记未运行；PBS自身另受1小时分配walltime约束。

W2固定人口：16场景×4状态；初态48有接触／16无接触，对齐、确认抓持、运输、释放各16。没有恢复阶段；全部320分叉与拒绝都保留分母。已备好冻结5000模型的CALIB全1232窗口误差分析与W2接触／阶段分层，但尚未执行。原活动capsule没有热修改。

## 01:10更新：Core共享GPU1实测，16GB容量足够当前WM

用户直接回答“允许GPU1共享”。新独立source/role/run/output仅执行模型smoke，无原生仿真、无跨站实时依赖。原PB owner及GPU锁保留，运行前后进程均存活；GPU0外部工作未触碰。使用现有Journal/onceIO和独立owner锁，没有改活动source，也没有复制HF或SSH凭据。

实际训练源 `9018017d62753530c380fd81dc704413dd9ca8c3`，Python3.12.3、Torch2.9.0+cu128、CUDA12.8。原同一FP32配方：3成员、共2026353参数、effective batch256、K4自回归。200次ensemble更新（600成员optimizer steps），另2次恢复检查更新（6成员steps）后恢复原200端点。真实下一次更新的参数最大差0、bootstrap样本索引相同。

- PyTorch峰值allocated **142350848bytes（135.76MiB）**；reserved **146800640bytes（140MiB）**。
- PyTorch allocator上限512MiB；此上限不包含CUDA context。进程结束前NVML测得**494MiB**，这是该时点读数，不是总显存峰值。
- 实际CPU29.252855秒、训练墙钟29.463394秒、整个worker31.366444秒。
- GPU1初始free2439MiB；smoke完成、进程退出后回到2439MiB。16GB总容量无需缩小当前batch或ensemble；本次只验证该WM配方。
- 新原生转移、任务、physics ticks、actor更新均0；使用既有TRAIN4317macro/4125四步窗口/11仅终止事件。

模型200端点：SHA256 `2c8da0afe073f22283624e144fb2c63869675418d07e6225d0eb529db0d2768d`，24374743bytes，已按SHA读回私有PC存储。没有发布smoke权重到Git或HF，也没有把200端点用于要求5000的W2。

保留的工程负结果：Core无HF token，直接读取私有repo返回401，GPU尚未启动；已有固定revision数值视图经缓存信任与SHA核验、沿既有认证项目SSH传入，不传原生资产或凭据。Vanda独立新读回请求超时，没有为这一步建立新的跨站运行依赖。实际fit有一次kernel-cache目录缺失警告，缓存被禁用，训练和恢复完成；未为该警告重跑有效smoke。终止头／CALIB决策价值仍未验证。

Core实际调用（全路径argv与Journal在私有命令记录）：

```bash
python -m scripts.wm01.fit --spec "$WM01_CORE/RESOLVED_SPEC.json" --data "$WM01_CORE/data/TRAIN_REPLAY_V1.npz" --run-id smoke-200 --output "$WM01_CORE/smoke-200" --profile core-wm01-shared-smoke --endpoint 200
```

机器回执：[CORE_GPU1_SMOKE.json](CORE_GPU1_SMOKE.json)。正式A40作业仍是独立1515716；没有取消、重投或用Core额外启动第二个5000端点。后续要切换正式站点须先独立处理该现有作业身份，不能在当前状态未知时双跑或热修改。
