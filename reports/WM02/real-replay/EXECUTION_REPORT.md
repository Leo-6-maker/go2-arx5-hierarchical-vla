# WM02 真实S Replay与GPU1首次预测实验 · 2026-11-04 SGT

**本轮已实际完成：原872条只读审计、可运行真实Replay Loader、物理单位V2编译器与冻结parser对照、GPU1真实S数据WM200主更新、完整重载与一次续训探针、独立场景留出评价。科学结论为 `INCONCLUSIVE_ACTION_CONDITION_VALUE_NOT_ESTABLISHED`。** 位置略优于持恒，但动作错配近似打平、姿态和抓持预测明显差于持恒；不扩大到5000，不做正式Actor更新。

| 身份 | 完整SHA |
|---|---|
| PR347分析基线 | `30b2db0be65dd481ea5988b743a465c52f71c0c2` |
| 实际GPU1主200更新源码 | `67721ff0bda5dee189bf4a73b6f8d43a454bd787` |
| 原端点元数据修复／恢复／评价源码 | `d5a87ae6eb7eb9fa48c0265944355357aeebfe1e` |

独立checkout／feature branch `research/wm02-real-replay-20261104`。没有切换活动source、修改main、重开旧队列或热补capsule。私有原数据、checkpoint、vendor/WBC资产与主机绝对路径均不进Git。

## P0：原件与真实Replay

[机器审计](REPLAY_AUDIT.json)：29个原项目流按完整SHA固定；仅允许既有数值NumPy构造器的受限legacy reader，不加载任意类；跨环境传递的是`allow_pickle=False`数值NPZ。每条复核原始split、50ms/25tick、reset/terminal、qpos/qvel/TCP/object连续性、原始物理动作、完整H40、安装行和25个实际消费命令／时钟。原件逐份复核未改。

- 原始872条均为有效完整50ms记录；去重为496个不同的积分pre/post＋命令＋时钟观测对。**496也不是独立episode数量。** 保留13个原episode ID、6个原parent ID、16个恢复分叉文件；通过相同物理模型／场景或parent连接归并后只有2个物理场景组。
- 8 burn-in＋8预测得到420个不同连续窗口，FIT210／诊断留出210；8＋16得到388窗，FIT194／留出194。恢复分叉各沿自己的前缀连接，重复窗口去重，不能串接两条分叉。
- 分区在训练前固定：按物理组字典序最后一组留出，所有同parent、场景和重复布局只能在同一区。**原TRAIN身份不改；这是TRAIN来源的场景留出，不是正式DEV或TEST。** 每区进入窗口的不同转移ID240，交集0；归一化只用FIT的240个不同FP32状态，不读留出统计。
- 原始接触／grasp／释放onset各4，去重后各2；持续grasp转移294；失败与超时0。仅有两个场景，不能据重叠窗口计算独立episode置信区间。释放只属于旧有限诊断，完整32秒成功0。
- 原872条`reward_train`均为None。真实成功、空手搬运、早放、碰撞、抓住不放五类奖励资格尚不齐；本轮不伪造这五类真实记录，不把以前评分器的CPU反例当奖励原生验证。
- 360条实际Teacher H40的后32行不是末点持恒。保留真实完整H40作为动作条件，**不从前8个raw标签重建或覆盖它们**，也不将未执行尾部当物理监督。未来Actor持恒尾部的分布覆盖仍需新数据；S驱动只消费前8行，不能据此声称T域OCS2完整preview效应已验证。

S独立schema是50维：6 arm q＋6 qd、TCP位置／旋转／速度、夹爪fraction／speed、物体位置／旋转／速度；没有BASE、腿或补零目标标签。context仅实际goal_xyz、table_top和cube_edge，不编造mass/friction。四事件为实际左右接触、native_grasp和illegal_contact；Teacher phase、控制缓存与未来状态不进入模型。绝对时钟和当前reference elapsed解析提供。未来实际已发指令轨迹是离线预测的外生条件，不是在线Actor想象，未使用未来WBC缓存。

8＋8数值数据SHA256：`2e41ac69be155c69377516fba7427e392eede60dc03e26d0f9b644640dc40714`。

8＋16评价数据SHA256：`48dba6cb0c2be055f82a4e082bae2a6ca1f2221f00430e92045aa3ad1eaaec17`。

## P1：编译合同修复

新`WM02_PHYSICAL_WORLD_DELTA_H40_V2`接收米／弧度物理量，世界平移、左乘旋转、原生row rot6d、t+i×50ms及完整40行；latent→物理只在独立Actor变换做一次。目标夹爪直接输出机械fraction的工作区间[.002/.044,.042/.044]，原生编译器仍拒绝越界、不裁剪／重标标签。非SE3 anchor也拒绝。

[实际CPU对照](COMPILER_V2_ACTUAL.json)：非单位姿态、非零43.15s锚点、完整H40、尾部及41点原wire；40行与原生编译器max差1.11e-16、绝对时间0、冻结parser姿态max差2.34e-15。边界、坐标、旋转和动作梯度测试通过。另实测发现FP32饱和latent的位移／转角范数及夹爪上界会因舍入超出原生double边界；V2 latent构造在允许范围内预留8个dtype机器epsilon，physical校验按double做拒绝判定。正负80饱和、机械工作端点和舍入越界反例均通过；这是动作构造定义，不裁剪已发raw／监督标签。

原冻结parser SHA `3fd59ca29b257a4fec2a66409d33099d0cc0befe4ef6a5039d80b44d03dadf90`，没有ROS节点或积分；独立GRTGRIP1执行未启动。

旧`neural.py`的synthetic compiler把dimensionless数值再次按.01/.05缩放，并非原生raw ABI；旧Actor的[0,1]夹爪及零tracking residual仍只属于原V1合成实验，不用于真实Actor。V2另有测量／预测TCP对已知active reference的可微残差函数，明确使用sin-angle代理；Replay保留实际command相对TCP的数值残差。没有宣称旧Actor观察、奖励或想象链已取得原生资格。保留旧200回执及其输入身份。

## P2：真实S模型与完整恢复

复用5×2048 SiLU／LayerNorm＋GRU256结构，只将输入／输出绑定到S schema，实际 **18,905,387参数**；不是将93维Go2模型中的BASE置零。FP32、AdamW lr2e-4／wd1e-5、clip1、有效sequence batch256、micro32累积8次。FIT只有210个重叠窗；51,200次访问不等于51,200独立物理样本。

首次GPU1 unit实际完成200主更新并写出checkpoint1/199/200后，严格`weights_only=True`重载因provenance的`numpy.str_`场景名失败，OS exit1／MainPID0／lease CLOSED。**这是真实的checkpoint元数据工程失败；没有重新运行主200。** 其首步恢复验证实际晚于主训练才被尝试，未达成原定先验证1步的顺序，准确保留该缺陷。

新源码将group转成builtin str，今后fresh run在更新1后立即验证再继续。另建有限恢复身份：先核原checkpoint SHA，在`weights_only=True`下仅放行已知NumPy scalar／dtype／StrDType构造器，把2个字符串元数据转成builtin；model/optimizer张量误差0，原文件保留。派生checkpoint可由没有额外allowlist的严格加载器读入。恢复unit主更新0，从199额外执行1个相同第200步作探针，参数和optimizer max误差0；1和200端点均完整恢复。总真实GPU WM optimizer.step为**200＋1**，Actor/Critic均0。

原checkpoint200与派生checkpoint200完整SHA／字节见[RESULTS.json](RESULTS.json)；派生身份：`38ecc892222c90513b3f91daa4c0ec0afd24a583af62ffd3a2a1216241d5ee89`。保存并恢复model、optimizer、TRAIN normalizer、Python/NumPy/Torch/CUDA RNG、cursor51200和来源。旧进程的初始同32窗loss未持久化，因此不补造其值；原trace首批更新前loss0.44269、第200批更新前0.01144，批次轮转，不能称同一切片比较。恢复后的FIT32 loss为0.005338。

固定200端点后才评价场景留出，不按留出结果选择checkpoint。下表是**8步burn-in后各预测步的物体位置RMSE**，单位mm；H16留出194窗，其余210窗。窗口重叠，不作独立episode统计。

| 预测原子步 | 正确动作WM | 错配动作WM | 状态持恒 |
|---|---:|---:|---:|
| 1 | 2.129 | 2.125 | 2.317 |
| 8 | 9.709 | 9.837 | 10.337 |
| 16 | 18.614 | 18.935 | 19.651 |

动作错配为同一留出场景内，按固定window hash顺序将完整reference轨迹循环错位一行（含burn-in），保留当前状态／context／clock；这是固定动作条件消融，不是获准物理反事实，也不是算法臂或新任务成功。

负结果很明确：H1错配甚至略优；H8／H16正确动作仅略优于错配。物体姿态RMSE在H8为0.01525rad、持恒0.00556；H16为0.02539、持恒0.00688。H8 grasp Brier为0.20359、持恒0.03810；native_grasp阳性1184次**窗口访问**中WM漏检654次，持恒漏检36次。这些访问包含重复物理事件，不能当1184次独立抓持。失败／超时类别没有正例，不能评价其预测能力。[完整1/8/16步误差与接触计数](WM_REAL_S_ACTUAL.json)。

目前模型可拟合训练窗，但没有建立可靠的动作条件、姿态或接触预测。应先补独立场景／接触动作多样性及奖励五类原生记录，再讨论新版本训练；本轮不自动扩到5000或拿合成reward更新Actor。

## GPU1与实际成本

[GPU身份与用量](GPU1_ACTUAL.json)：GPU1 UUID `GPU-845348f1-1511-dc19-0877-cc97658e3edd`，PCI `00000000:83:00.0`，NVML16376MiB；实际CUDA属性16709255168字节，运行只暴露该UUID、allocator上限4GiB。真实S200主训练进程NVML采样峰值916MiB，恢复／探针／评价进程1132MiB。恢复阶段Torch峰值allocated794,644,992／reserved815,792,128字节；原训练阶段Torch peak在元数据失败前未持久化，不能拿恢复值冒充它。

GPU0是另一个UUID／PCI02:00.0，NVML仍报告32760MiB且VBIOS不同；保留观测，未核实容量差异的物理原因，不改驱动、固件或设备配置。GPU1 model-only真实训练／完整重载已实际完成，不等于sim＋ROS2／OCS2／WBC／渲染共驻资格。其他用户工作未动。

两unit分别exit1／0，CPU56.314020／17.787469秒，墙钟56.679412／12.362262秒；相加为本轮两有限unit **74.101489 CPU秒／69.041674墙钟秒**。包含该unit的CPU回归与错误收尾，不包含完整预审、PC测试、artifact staging和发布，不能称全任务总成本。恢复程序自身CPU10.641697／墙钟5.372959秒，已包含在第二unit，不能重复相加。两unit MainPID0、lease CLOSED，无新PBS和现役GPU训练。

## P3及剩余阻断

[目标API与替代路线](TARGET_RECOVERY_PATH.md)基于本轮再次核对的5份实际源文件SHA，保留原MuJoCo/WBC/OCS2。原native lease SHA仍为`db2c30f675acb9e8dc280e3fc69d277e1bf91db10cfc2476881eeac2895be407`，CLOSED且隔离保留。**首个T域阻断仍是S1/T2未解除与缺少当前动态准入。** 完整控制恢复、真实消费／ACK及站立T1全身传感仍需绑定；没有反复运行exit78、写permission=true、启动目标节点或重编译WBC。

本轮新真实S/T采集0、目标物理0、完整任务成功0、Actor/Critic更新0、PBS0；旧872条不是新采集。真实数据WM训练是新增效果，但还没有real collect→WM→Actor→collect闭环。实时源采集器、正式任务reward资格、原生Actor观察／想象绑定仍未实现或验收，不提供虚假collect CLI。

## 实际命令、测试与下一条命令

已实现入口：`scripts.wm02.replay`、`scripts.wm02.check_compiler`、`scripts.wm02.train_real`。GPU有限worker经原Journal调用`train_real.main(args)`；不是又运行了一份独立shell训练。实际argv／绝对路径、helper SHA、原生unit退出和中间失败留在执行者私有`ACTUAL_COMMANDS.private.md`。关闭的stage／Journal不重放。

```bash
PYTHONPATH=src "$WM02_CPU_PYTHON" -m scripts.wm02.replay --manifest "$WM02_PRIVATE/OLD_S_MANIFEST.private.json" --output "$WM02_PRIVATE/audit-08-v1"
# exit0，真实872条输入，420个8+8窗
PYTHONPATH=src "$WM02_CPU_PYTHON" -m scripts.wm02.replay --manifest "$WM02_PRIVATE/OLD_S_MANIFEST.private.json" --output "$WM02_PRIVATE/audit-16-v1" --prediction-atoms 16
# exit0，388个8+16窗
PYTHONPATH=src "$WM02_TORCH_CPU_PYTHON" -m scripts.wm02.check_compiler --parser "$WM02_PRIVATE_PARSER"
# exit0，实际冻结41点parser；零积分
# train_real.main(['--spec', SPEC, '--output', OUTPUT, '--run-id', RUN_ID])
# 原200 worker exit1；新身份finish_existing worker exit0，主更新0＋恢复探针1
```

PC神经12项普通／-O通过，含真实schema metadata严格重载、去重／场景隔离、raw/consumed/reset拒绝、post-terminal边界与future-state不泄漏；原world_model30项通过。Core两身份各运行原神经6项普通／-O、V2数值对照，均exit0；完整legacy审计在原CPU环境做，不声称Core跑了依赖SciPy的新12项。全仓库CI NOT_RUN。先前CPU测试发现alive mask原位修改会破坏BPTT，改为新tensor后通过，发生在首GPU启动前。元数据失败发生在主200后，旧错误保留。

下一条可以在本机执行的新入口是对真实原件的**16步预测窗口审计**（新输出目录，不触发模型或物理）：

```bash
PYTHONPATH=src "$WM02_CPU_PYTHON" -m scripts.wm02.replay --manifest "$WM02_PRIVATE/OLD_S_MANIFEST.private.json" \
--output "$WM02_PRIVATE/audit-next-16" --prediction-atoms 16
```

其中`WM02_PRIVATE`在执行者私有命令记录中已经绑定；公有报告不披露其绝对路径。该命令是已实现的真实数据审计，不是下一阶段科学闭环。下一项科学增量应为新的独立S域接触动作／奖励记录，需新有限采集身份；不能重开旧source lease或把尚未实现的采集接口当命令。
