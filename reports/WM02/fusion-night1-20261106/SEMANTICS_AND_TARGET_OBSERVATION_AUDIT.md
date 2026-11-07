# FUSION 动作拒绝定位与目标观察接口实测

现场只读快照：2026-11-06T13:11:49.764103+08:00。PR #366 未合并，主 goal ACTIVE。新实现源码 **`7c12085a6c858b914423f14b425539fa6e224f1e`**，包含结构化响应训练 CLI、真实目标观察绑定与保存状态采集 CLI；尚未部署或运行新 CPU worker。现役192评价面板仍是 **`0f235865a30f94cd877b7648598823d2039fa6ae`**，capsule/模型/评分/WBC未热改。协议见 [结构化响应与观察合同](../../../docs/protocols/WM02_FUSION_STRUCTURED_RESPONSE_AND_OBSERVATION_V1.md)。

## 首个确定阻断：原源域 SafetyGate 已拒绝真实轨迹

实际 FUSION 冻结 E1 查询仍只有1次，checkpoint manifest `383ad9b8fc3b81efffde132d4f06de92fb96fbdf9ec0b6c92c097e0f58eb2a8d`，动作数组 `0eb9040757d8fc4dc3922dc51ff2c839f043a69b754c159fa60301f2fe133a4e`。本次以原源域执行源码 `887e65629291f16f73863e2b111123007542d268` 的原 `gate_action` 只读检查完整H40，退出0，但决策为 **REJECT**：线速度最大4.9684424578m/s（阈值2）、角速度15.2314707762rad/s（阈值4）、平移路径5.6288115677m（阈值5）。阈值来源仍保留原 TEST_ONLY 等 provenance，没有放宽。没有源引擎构造、物理积分、轨迹修正、截后缀或新抽样。

另外只读复核了历史 E1 单次DRAW：其原记录已经是相同三类拒绝，未重新加载或查询模型。这条旧历史不计新增 FUSION 查询，也不能据此把原因唯一归于seed。当前事实是该冻结输出尚未具备源命令执行资格；不能绕过原Gate去测SOURCE_OUTCOME。

原 normalized 前10物理通道共有136越界分量，40/40点受影响；分通道为12/7/9/15/13/16/17/24/11/12。冻结源逆归一化实现自身会clip，这次只披露，不修改它。机械夹爪14/40点不在目标工作行程 `[.002/.044,.042/.044]`；没有套用Teacher逻辑开度变换。源命令issue首点是观测时刻，首个执行后果是50ms后；8点执行前缀只有400ms，32点后缀未执行。新decoder保留显式issue时钟，旧合同默认不变。

## 目标观察：输入已核实，真实图像采集尚未执行

T06原生资格记录的场景SHA `40408bbcbef2469396187c741750905a6ba9ed7161dc031d14bfab5fed62ddd2`、结果SHA `112c10f7d28ab346ff41245b9e3bb9ed5f7f5eb9dc464f042055d7d52d0f94f5`、原journal SHA `4d75842eb6059e76ae5567f813f7440671b32a45ac61caad9a898c4333211985`。采集入口的真实只读预检退出0，验证CLOSED/worker0/nativePASS、完整4103个2ms packet、源文件与加载执行器身份；不是重新积分或增加训练数据。

在3503/3703/4103三个原始packet上，两个相机均有任务cube的8个几何角点处于视锥内；此检查只做3次mj_forward，**未render，未验证遮挡或图像质量**。源body相机世界固定55°，目标body相机依附base_link、75°，不能仅改名为external_rgb就宣称视觉等价。

已从原源场景XML和decision3703实际BASE FK得到一次固定坐标关系，且核对完整base_link→x5_base_link→x5_link1静态安装链；map SHA `fc347a59fe6b82f758d198f2db93955ba2697a55e8ec24e490faeda53c2761b1`。这是坐标对齐，不基于TCP/object挑变换；不证明相机、任务或行为域一致。SOURCE_OUTCOME仍未解决。

现场依赖只读核验发现OSMesa缺失，EGL/GL可加载。于是新版本改用原已有CPU EGL Mesa软件路径并要求实际Mesa/llvmpipe身份，拒绝GPU设备选择；没有装包、改系统或启动renderer。新capture worker将仅输出三对实际保存状态图像及原packet/标定/绑定，明确标为replay；当前实际新图像、target查询及structured真实拟合均为0。

## 正在进行的原生对照与已有学习结果

当前面板已封存 **76次评价＋3次资格=79次原生PASS，324,137tick**，完成38个双slot pair，仍ACTIVE。不是192个成功，也未根据部分分数改模型。该活动panel的系统CPU快照约5807.006s、峰值内存3,948,982,272B，成本仍OPEN，待终态统一结算。

此前 FUSION 已封存64次nativePASS/262592tick；平衡数据SHA `e5139d792d8d2f28e49ea28d7f57fbabbd5e79378e5358b747b2896c851ed795`，48窗口/768真实后选择转移（32 TRAIN/16 DEV）。六模型实际12000主更新＋6恢复探针，另200次失败而无checkpoint的更新保留，总主调用12200。所有动作条件MLP均优于盲臂，但均不及强TRAIN静态响应；T06四臂0/4任务成功，不宣称WM独占收益。完整模型/checkpoint身份与成本沿用 [前一实际报告](BALANCED_RESPONSE_LEARNING_AND_NATIVE_REFIT.md)。

新结构模型只用TRAIN整父LOPO，在固定0.08124448952310876mm有意义阈值下选择ridge或退回静态；盲臂采用公平边际响应。若回退与既有静态动作相同，不为此重复一轮原生对照。现役面板结束后在原owner冷边界登记独立有限CPU拟合/图像worker，不复用CLOSED lease。

## 已执行命令、验证和成本

私有变量仍绑定原已验证route与交接目录，以下均为实际已实现且本次执行的入口：

```bash
python3 "$WM02_CORE_ROUTE" "$FUSION_PRIVATE/audit_original_source_action_gate.py" 45
python3 "$WM02_CORE_ROUTE" "$FUSION_PRIVATE/audit_archived_vla_semantics.py" 45
python3 "$WM02_CORE_ROUTE" "$FUSION_PRIVATE/preflight_target_capture_inputs.py" 45
python3 "$WM02_CORE_ROUTE" "$FUSION_PRIVATE/audit_rigid_base_frame_map02.py" 45
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src "$WM02_TEST_PYTHON" -B -m unittest discover -s tests/world_model -p 'test_*.py'
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src "$WM02_TEST_PYTHON" -B -m pytest -q tests/world_model/test_capture_fusion_observation.py tests/world_model/test_fusion_cpu_admission.py tests/world_model/test_fusion_observation.py scripts/pi0294/dec1/test_target_observation.py
```

上述命令退出均0；Gate的科学结果为REJECT，不能与进程退出混淆。完整world_model套件在功能提交066abdbbfd8af7907d1bc522fe6866226fba7618为352项/21.696s；随后实际依赖修复提交7c12085a的受影响回归54项/69子例、1.83s通过。此前63项FUSION测试在-O模式通过。旧fixture缺当前allocation.job_id已只补测试字段，生产准入未削弱；最初用unittest跑pytest函数收集0项，不计通过数，之后用pytest执行真实回归。CI未运行。

新增审计没有模型加载/查询/更新或积分。原Gate只读wall0.021594s、图像几何wall0.285678/CPU0.284155s、完整安装链坐标审计wall0.099968/CPU0.099731s、capture输入预检wall0.584719s、原数组审计wall0.005374/CPU0.004648s。其它小元数据检查保留私有回执，不将活动worker成本结为0；未申请A40/H200、分布式或GPU物理多环境。

下一条可立即执行命令仍为只读监测：

```bash
python3 "$WM02_CORE_ROUTE" "$FUSION_PRIVATE/read_response_learning.py" 45
```

原goal继续ACTIVE。等待完整panel封存时继续准备合法后继，保持唯一owner、原截止、2个CPU原生slot及原安全；GPU按用户最新授权共享，优先CORELAB4080S。
