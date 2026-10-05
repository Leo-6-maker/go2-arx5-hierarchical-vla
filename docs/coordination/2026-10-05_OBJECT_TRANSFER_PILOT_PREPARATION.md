# 单物体搬运 pilot：实现准备与后续训练加速

状态：PREPARATION_ONLY；物理/教师/渲染/模型/训练 launch_allowed=false。源路线 bc7e59a5716152c6500724e92b45dd1d7b83ad82，Issue11协调5994002129；本文件不是新科学GO，不修改PI0274。
用户另已直接允许后续提交采用多worker或其他训练加速。该许可适用于保持所批准科学条件的工程优化；不改变现役PI0274的封存入口，也不自动授权新的物体任务训练。

本准备稿已对齐随后发布的GROOT_OBJECT_MANIPULATION_MAINLINE_CHARTER_V1（章程固定提交14ca272185df8973a5c314f44f08f4a738cc43c7；README入口ed398fe）。object_task为真实物体搬运；high_level_contract为RGB/实际EEF夹爪/语言→EEF与夹爪；frozen_wbc_identity为NO_WBC_RUN、新版本待绑定；task_data_provenance为独立物理工具教师；single_intervention为首次引入物体交互任务、不同时搜索学习配方；task_evidence_and_controls为教师/模型工具域、正确/模型参考WBC域分别对照；budget_and_stop为第5节的待授权pilot、失败不补样；claim_boundary为准备已完成、物理学习和迁移尚未证明。章程不增加本稿执行预算。

## 1. 职责和已发现的接口细节

GR00T选择语言指定的物体与容器，持续产生EEF参考和夹爪动作；WBC跟踪任务空间参考并协调身体；独立夹爪控制器执行开合。轨迹是接口，成功须以实际物体完成抓取、搬运、释放和撤离判定。
末端跟踪/底座平衡属于WBC；高层持续给出错误参考、松手时机错误、选错目标不由WBC修正。旧虚拟模型漂移也不能归因于WBC，因为该执行器没有调用WBC。

已读源：src/go2arx5_vla/adapters/groot_to_wbc.py、trajectory/preview.py、contracts/trajectory.py、data/readiness.py、docs/adr/ADR-003-parallel-gripper-channel.md、scripts/pi0267_virtual_renderer.py、scripts/pi0273_execution/quality_virtual.py、reports/PI0258/CONTROL_A0.md。

三个必须处理的具体点：

1. StateV1与当前物理标签使用两行row-rot6d；TrajectoryChunkV1.from_rot6d及adapter.adapt_xyz_rot6d使用canonical列约定。新入口必须先用原rot6d_rows_to_matrix转SO3/quaternion，再构造canonical chunk，不直接把row值传给该便利入口。非单位旋转CPU检查覆盖此边界，不改旧adapter。
2. adapter.adapt在当前controller_timestamp选择一个夹爪值；一次安装返回值不足以执行整个开合序列。新runner须保留当前已接受chunk的并行夹爪序列，在每个控制tick用同一时钟消费；夹爪绝不追加到WBC18D或9点preview。原生空间进度、chunk替换、history和夹爪消费时钟必须用实际sink收据绑定，不能用时间推进冒充空间推进。
3. Corelab真实ROT统计文件SHA256 0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9：state/action.gripper_position均min=max=q01=q99=0.5，std=0。随后用原合格codec和实际StateActionProcessor做独立CPU探针：0/.25/.5/.75/1分别编码为-1000001/-500001/-1/499999/999999；解码误差均0、有效gripper mask为1。旧无clamp仿射在1e-6分母下可逆，但幅度极大。新TRAIN-only夹爪尺度须经真实数据processor roundtrip后冻结，不能静默沿用或裁剪。旧统计不修改；该探针不是完整VL/实际物体数据准入、梯度或训练结果。

## 2. 首任务规格草案

指令四组合：Move the {red|blue} object into the {left|right} bin, release it, and withdraw. 场景均有红蓝两个可抓刚体、左右两个宽松容器。颜色与位置交换，两个对象/容器都存在，避免固定位置代替语言选择。暂不加门、书本插入或多物体整理。

下列数值是待独立pilot冻结的提案，未实测：物体40×30×30mm、质量0.05kg；两指夹爪最大内宽80mm；容器内口160×160mm；桌高0.65m。任务坐标world+z向上，工具TCP为新工具卡明确定义的掌面中心；不把新工具卡误标旧实体x5 TCP。Go2迁移时另绑定实际TCP变换。

最多30s任务窗；physics dt=1/120s，20Hz双RGB/state记录，H40/dt=.05s，完整episode结束前另保存至少1.95s实际终末记录。位姿驱动速度上限0.10m/s、角速度0.5rad/s，工具6DOF用有限力/力矩驱动，初始提案20N/2Nm；夹爪每指力上限10N、宽度变化上限0.04m/s。有限力质量/接触/控制参数在首次物理调用前冻结；不为成功事后调参。物体摩擦0.8、恢复系数0.0为首版提案，非真实物体标定。

物理实现复用Corelab现有IsaacGym/PhysX栈，但新建隔离动态工具环境：6个受控根关节+2个手指关节、重力与真实接触对象；不得调用旧place()设置物体或工具的运行位姿。仅初始化可设初值，tick阶段用关节target/受限驱动推进simulate/fetch_results。禁止焊接、parent、吸附、根据成功真值写物体pose。原零物理renderer不改、不作为动态成功证据。

允许：手指与目标接触、目标与桌/目标容器接触。失败分别记错物体抓取/搬运、目标掉落、危险非目标碰撞、超界、穿透>2mm、未抓起、未释放、未撤离、超时、技术错误。允许接触类别与原SafetyGate分开；任何原gate REJECT不安装，不能绕gate取得任务PASS。涉及新接触的准入规则须在未来pilot中单独冻结。

完整成功草案：选中物体曾离开源支撑面至少80mm，且该抬起过程中两指目标接触连续至少.3s；搬运到指令容器；完整物体OBB在容器内；实际手指开度至少70mm且无工具-物体接触；TCP离物体至少100mm；固定任务终末28—30s全部tick物体留在容器、速度<.02m/s；非目标物体位移<10mm且未被抓起；无危险接触/穿透。末尾额外2s用于完整H40标签记录，不挑最好的2s成功窗口；全部保存段的安全事件保留。上述评分只在评价端读真值，不作为policy阶段推进或隐藏控制输入。这些尚未运行的任务规则须随未来pilot冻结，不改变旧几何判据。

## 3. 算法教师草案：状态触发，不按帧强推

| 阶段 | 命令 | 实际转移证据 | 失败 |
|---|---|---|---|
| PREGRASP | 张开，移动到目标上方80mm | TCP距离≤5mm、角误差≤.05rad持续.2s | 不到位/错误目标 |
| APPROACH | 张开，对齐抓取位置 | 同阈值且目标位于两指之间 | 穿透/超界 |
| CLOSE | 保持位姿，受限闭合 | 两指目标接触持续.2s，实际内宽合理 | 空闭/非目标接触 |
| LIFT | 闭合，抬高100mm | 实际物体离桌≥80mm、相对工具漂移≤5mm持续.3s | 滑落/夹空 |
| TRANSFER | 闭合，有限速度至容器上方 | 实际TCP到达且物体仍随抓握接触携带 | 掉落/危险碰撞 |
| LOWER | 闭合，下降至放置位置 | 物体实际支撑接触且竖直速度低 | 未支撑/挤压 |
| RELEASE | 保持位置，张开 | 实际开度≥70mm且工具目标接触解除 | 黏连/未松手 |
| RETREAT | 张开，撤离 | TCP-物体≥100mm，之后观察2s物体保持 | 带走/掉出容器 |

教师可以离线读取真实目标/接触，只产生标签及诊断；学习策略运行不调用教师或phase。阶段超时固定为4/4/2/4/6/4/2/4s，整体30s优先；首pilot不重抓或重复reset追成功。无效proposal/episode保留分母，技术恢复沿原状态记账，不能伪装新样本。

## 4. Episode / H40 / processor 契约

先按scene family+对象实例+布局分DEV/TRAIN/VAL，再采集episode，再切H40；同族全部command在同split。完整未来40条command不足的窗口标无效，不能拼其他episode或补零hold。

| 数据域 | 内容 | actor可读 |
|---|---|---|
| inputs | 同一实际tick的body_rgb/wrist_rgb，实际TCP row6d＋测量夹爪开度StateV1，语言 | 是，仅四键 |
| commanded_labels | 未来原教师命令XYZ/row6d/夹爪宽度、独立命令时间、安装状态 | 训练标签 |
| achieved | 实际TCP/DOF/根/夹爪/物体状态及源时间 | 实际TCP/夹爪仅经inputs白名单 |
| evidence | phase、目标ID/pose、接触力、容器占用、mask/depth、相机矩阵、成功、来源hash | 否 |

保存传感时间/clock_id；CPU草案要求sim lockstep图像和State同tick，H40时间t+.05k。原始宽度按米保存；新工具提案规范u=width/.08，0闭1开，不等于硬件现有极性已验证。measured_u进入state，commanded_u进入action，不得互换。新stats仅由TRAIN拟合/冻结，开放/闭合/携带/释放全部非恒定且有效mask前10维，其余122维结构padding。实际132D processor encode/decode、dtype、mask、所有阶段、跨进程恢复身份仍为未来数据准入必做，当前CPU例子不能代替它。

相机两路保持原输入尺寸；body rig可独立移动并保存world外参，wrist随工具；不能从TCP猜base相机或叠字补观测。第一版相机卡从已供标定起步，缺失参数列UNKNOWN；不在后续成绩上调rig。移动工具域与Go2 base移动分别报告。

## 5. 独立预算草案：先验证任务物理，不启动大训练

建议16预列scene families：4DEV/8TRAIN/4VAL，各四command，64教师episode计划；DEV先16、TRAIN32/VAL16。不同分区无对象实例/布局共享，冻结列表后不补容易样本。最多30s/episode=230400 physics ticks、38400双RGB记录；包括额外终末1.95s的总保存预算按32s/episode=245760 ticks/40960记录。先不训练、不推理、不WBC。所有新预算需独立pilot授权；本准备包实际新增这些调用均0。

教师和数据合格后，后续学习协议再冻结唯一初始化、新夹爪processor、更新数、精度/loss和评价预算；不根据PI0274质量自动选最好的模型。一个有界任务拟合检查应同时报告pose与gripper，不隐去抓空/错目标。再做教师vs学习策略同工具评价，之后正确参考vs学习参考同冻结WBC评价。各层不互相冒充，不按WBC成功筛上层数据。

## 6. 已获允许的后续训练加速实施顺序

先用现有程序真实阶段耗时/GPU占用定位，不新增模型查询测吞吐，不更改当前健康任务。保持批准的样本、有效batch、noise、精度、loss、step和scheduler。

1. 固定对照JSONL在启动时验证hash并加载一次，当前追加JSONL用完整行增量游标；保留审计内容与失败条件，禁止为速度关闭审计。当前逐微批read_lines完整文件属重复解析。
2. 数据准入时冻结并缓存CPU处理结果，key含原样本/语言/processor/stats版本；不缓存跨步随机噪声，不读取VAL拟合缓存统计。
3. 后续新运行采用固定访问索引的map-style数据集或明确分片iterator；worker从2开始，必要时4，in_order=True/不shuffle，prefetch_factor=2，pin_memory=True，CPU数据worker不接触CUDA。worker不得写消费账本；main只在实际消费时记账。数据loader使用单独固定generator，不能消耗模型RNG；保留实际optimizer-boundary cursor和checkpoint RNG。
4. 在原已批准canary预算内做连续与save/resume比较，逐样本/标签/epsilon/tau/S4/梯度/mode/optimizer一致；未通过退回单worker并修工程，不消耗新的科学更新尝试。比较允许的数值容差必须沿科学协议，不能自定放宽。
5. 保存计算重叠可用在线独立HF归档进程，checkpoint已完整且不再写时读取；只有fixed-revision full-file readback后可退役，唯一resume源不得提前删除。更大batch、改变梯度累计、TF32/精度切换、compile/fused kernel属于可能改变数值的单独验证项，不能仅因显存空闲直接开。

加速是否生效以相同有效更新的walltime和CPU/GPU等待数据报告；目前未实施多worker，未宣称吞吐增益。

## 7. 准备交付与未完成项

scripts/object_transfer_preparation.py复用现有StateV1、canonical chunk、SO3与adapter，提供CPU格式/row旋转/时钟/夹爪调度/泄漏负例。它不导入Torch/IsaacGym、不调用教师、不连接控制器；合成fixture明确SYNTHETIC_CPU_FIXTURE，不是RGB渲染或抓取证据。
scripts/object_transfer_gripper_codec_audit.py在独立进程中复用真实StateActionProcessor/原codec，CPU隐藏CUDA、4线程/16GiB且拒绝神经Module及Optimizer调用。收据reports/object_transfer/preparation/GRIPPER_CODEC_AUDIT.json保留源hash、前10有效mask、结构padding和0..1五点实际编码/解码。没有新任务statistics拟合或live normalizer改动。
scripts/object_transfer_task_score.py实现独立物体结果评分草案，使用120Hz实际状态、完整OBB、真实接触/夹爪状态及原gate结果。CPU合成轨迹覆盖14种有效失败及非有限证据，保留错误物体/未抓起/未松手/未撤离/掉出/危险接触等原因。它不执行教师或physics，逻辑PASS不证明真实抓取可行；还须由未来环境完整性证明对象没有被手工搬动/吸附。
剩余未知：新WBC固定版本、真实夹爪极性/宽度/消费周期、实际TCP、相机标定、接触/负载能力。按旧manifest查实际已有材料，仅向师兄索取差异，不要求真机示范才能准备。新环境、教师和完整VL/真实物体episode processor准入尚未运行。
selected/deployment=NONE；PI0274继续原预算，不新monitor/mail/merge。
