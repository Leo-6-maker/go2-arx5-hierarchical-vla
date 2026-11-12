# SLA＋冻结 WBC：Corelab 并行数据生产与 H200 后训练推进计划 V3

日期：2026-11-13（Asia/Singapore）
状态：**PLAN_PUBLISHED / IMPLEMENTATION_AND_NATIVE_RESULTS_PENDING**
继承：[V2 计划，固定 b079368][v2]；执行基线：[PR #380，固定 aaf80b4][pr380]。
配套：[机读矩阵](SLA_CORELAB_PARALLEL_DATA_H200_V3_20261113.json) · [执行交接](../handover/2026-11-13/SLA_CORELAB_PARALLEL_V3_CODEX_START.md)。

## 0. 目标、用户授权与推进原则

用户本轮明确要求：“给出继续推进的详细计划，允许corelab上并行多环境采集数据，注意采集效率，不要CPUGPU来回搬运，推送到github”。

本计划把这项授权落实为三条可同时推进的工作线：

1. **Corelab 原生多环境专家采集**：每个世界保持冻结 MuJoCo/WBC 控制合同，生产完整、可学习的抓持／放置及就绪新技能示范。
2. **常驻 GPU learner／teacher／推理服务**：训练与视觉推理的大张量留在 GPU；只在明确输入、动作输出、持久化边界传输。
3. **策略结果驱动的数据迭代**：D100→D300→条件性 D1000，接学生偏离状态的专家纠正；同时检查数据规模与优化预算，交付 checkpoint 和完整闭环结果。

**Corelab 多环境采集已经获得用户许可。** 接手执行端可在本项目实际可用资源内实现、验证和运行独立 actor，不再逐个询问 2／4／8 环境是否获准。旧“整个 Core 只有一个物理 owner”的全局串行解释，对这项新工作改为“每个独立世界有一个控制 owner”。共享 checkout 仍由一个写入者管理，每世界的 WBC／solver／时钟／安全保持独立。

许可来自上述用户消息；文档发布不是作业已启动或性能已达标的证据。H200 按既有有效授权和实际站点 allocation 推进，不把 Corelab 许可伪造成其他集群已分配资源；现场配置检查不是新增逐阶段人工审批。已获授权的工程修复、数据分波和真实验证应连续推进。

保持 WBC 权重、增益、cost、solver、腿策略、动力学、2 ms 物理时钟与原安全条件。保留健康现役作业和历史结果，不热改运行 capsule；不改他人任务、系统服务、共享存储基础设施或真实机器人控制。AMD 登录／共享存储端不承担采集、训练或压缩。

## 1. 固定事实与第一项现场动作

本轮读取到的最新实际报告为 **2026-11-13 00:02:56 SGT**；不是接手时实时状态。[阶段回执][stage]

| 项目 | 已有事实 | 不能推断 |
|---|---|---|
| 原 S1 | 500 更新、16,000 样本访问；独立 RGB 重建通过 | 已有泛化抓取成功 |
| 原 S1 TRAIN4 | 2/4 终态；策略拒绝和初始化失败保留 | 其余条件已结束 |
| S1 DEV16／S2／G DEV16 | 快照中均未运行 | 现场现在仍为 0 |
| V2 logit 数据 | 1,450 窗口、58,000 点 CPU 派生检查 | 新表示已通过原生夹持 |
| V2 batch | CPU micro1／8／16 同 fixed32 随机 bank；35 项软件检查 | 真实 SLA CUDA 梯度或吞吐已通过 |
| V2 模型更新、RGB 生成、物理请求 | 全部 0 | H200 训练已经开始 |
| EEF_STREAM_V2 | 尚未实现 | 中间路径已能完整消费 |
| 大卡与 site entry | 未完成；未验证新 SLA allocation | 集群没有可申请资源 |

先读既有 unit、注册组、checkpoint、终态和当前资源。原 W0 队列曾处于 WAIT，且有有限期限；其首次准入／总期限在旧 PR 中分别登记为 11/13 09:39／18:39 SGT。[现役原队列][pr378] 若已结束或超期，保留实际终态；只接续未执行的工作，未知状态不重复投递。后继采集可以在独立世界和资源范围推进，不以原 S1→S2→G 全部完成为前置，但原比较的人口、顺序、身份和负结果不改。

公开入口 main 固定为 6843163；V2 计划尚未合并，实际执行栈也未整体进入 main。后续实现从核实后的执行 SHA 建独立 worktree，通过正常 PR 集成，不把本文件当成已有 CLI。

## 2. 已确认的效率问题与改动位置

以下是固定代码审计，不是 profile 测得的占比。实现先复用已有接缝，再测真正的端到端耗时。

| 实际接缝 | 目前行为 | 首项修复 |
|---|---|---|
| [fusion_paused_observation][capture]＋[BodyBaseCameraCapture][body] | 持久 camera 分支只匹配 SourceCameraCapture，body recipe 每次建 recorder、save、close | 新 body session 接入真正的持久 model／renderer 路径 |
| [FixedBodyRenderer][renderer]／[FrameRecorder][worker] | 每 capture 读模型、创建上下文，生成原 body、wrist、新 body；旧 body 主要服务审计 | 保留所需 body/wrist；参考像素审计移到资格／抽样路径 |
| [capture][capture] | 强制 Mesa llvmpipe CPU 渲染；PNG／state 同步写入、fsync，再读回绑定和重复 hash | 不可变内存 snapshot 直接进入策略／writer；原文件 reader 留作兼容 |
| [settle_atom][perf]＋[consumer][consumer] | 未接入事件模式时每 50 ms atom 固定 spin(0.15) | 接通已存在的 ACK、step、solver、gripper 就绪判定 |
| [student batch runner][student] | 条件串行，反复启动 native／GPU pair | CPU actor pool＋一个常驻 GPU 策略服务 |
| [batch_v2][batch]＋[runtime_v2][runtime] | 每 update 建线程池；逐 microbatch 等梯度并取 float 指标 | 持久 loader、device 累积和有限标量同步 |
| [FeatureBank][features]＋[spatial runtime][spatial] | 每样本读／hash npz，CPU pad 512→768 token 后上传；S2 仍有逐样本路径 | shard 验证缓存、有效 token 批量传输、GPU padding／同卡教师互操作 |

固定 150 ms 等待若每个 atom 都触发，30 秒／600 atom 单此项就累加 **90 秒墙钟**。这是代码路径的算术，不是新实测速度。不得直接删除等待而丢失消费确认。已有事件 opt-in 的角色限制需要在新版本中正确接通，不能假定加环境变量就已生效。

renderer 持久化必须保留已有 reset_render_scene 经验：MuJoCo scene camera 历史可能影响同状态图像。复用 context 不代表所有 scene 临时状态都可复用。

## 3. 数据驻留合同：必要边界只传一次

### 3.1 原生控制、图像与模型各留在适合的位置

| 数据／计算 | 默认驻留与传输 | 生产路径要求 |
|---|---|---|
| 原生 MuJoCo／OCS2／WBC／现有 CPU 腿策略 | 各 actor 的 CPU 进程；现场先确认实际 build | 不为“全 GPU”逐 2 ms 上传再下载控制状态 |
| 专家采集 | CPU 任务专家＋原生控制，CPU RGB snapshot→writer | 不加载 SLA；没有模型推理就不制造 RGB H2D |
| 学生 CPU 图像输入 | 共享内存 uint8→固定 pinned staging→一次 GPU 上传 | GPU 内完成已资格的预处理、模型与解码，不用 PNG 文件作在线输入 |
| GPU 模型／梯度／优化器／flow 中间量 | GPU 常驻 | 热路径不 offload 到 CPU；checkpoint 单独计量 |
| 动作返回 | GPU→CPU 一份紧凑 H40／必要审计字段 | 全 flow／逐层结果不回 CPU；CPU 控制器仍执行原安全检查 |
| 冻结空间教师 | 与 learner 同卡时在 GPU 内交换 target；或离线 cache 分块上传 | 不走 GPU→NumPy→GPU；不 detach 学生可训练分支 |
| 持久化 | immutable snapshot→异步 writer→本地 shard→授权持久存储 | 与控制解耦，有界缓冲和背压，完整提交后才供训练 |

核心验收是 **不必要的 CPU↔GPU 往返为 0**，不是伪称原生 CPU 控制链完全没有设备边界。CPU 内 staging copy、GPU 内 layout copy、输入 H2D、动作 D2H 和归档读回分别统计。

两张 RGB224 uint8 的像素量为 301,056 B；H40×7 FP32 为 1,120 B，EEF10 为 1,600 B。完整 raw32 的 H40 也只有 5,120 B。保留原要求的 raw／decoded 审计，不为省少量动作字节删证据；可一起打包为每 query 约 ≤16 KiB 的动作与审计 payload。该大小是工程目标，实际以字段清单为准。上传字节按真实渲染分辨率计算，301,056 B 仅在已经是 RGB224 时成立。

### 3.2 相机后端选择

先使用已经验证的**持久 CPU renderer**建立生产线；这使专家采集的像素保持 host resident，学生只需一次 H2D。renderer、model、只读资产不逐帧重建。

EGL 加速单独测。经典 MuJoCo Renderer 的 NumPy 输出通过 mjr_readPixels 读回，设置 EGL 不能证明 GPU tensor 零拷贝。[MuJoCo API][mjr] 因此：

- expert-only 模式若 GPU 渲染更快，可一次读回供持久化，且不再把这份图像上传给无关模型；
- 学生模式优先选择 CPU 图像一次上传，或已验证的设备图像直接进入 GPU policy；
- EGL→host→GPU 的大图回环不作为消除往返的合格终态。仅保留为有计数的参考／迁移诊断，若需采用必须明确其真实成本和未达项；
- GPU 原生图像路线需要实际 CUDA/OpenGL interop 或 device renderer，实现前不得宣称已存在；
- GPU writer 所需读回是独立归档分支，不应回流成为 policy 输入；
- 后端或 resize 实现改变像素时单列 camera/preprocess 版本，不用新域结果覆盖旧实验。

不为追求渲染后端先迁移整个物理／WBC；参考 CPU 路线可持续产出。

### 3.3 进程、stream 与 buffer 生命周期

CPU actor 发送共享内存 slot 与 actor/episode/query/native_step/version。本 actor pool 在每张实际使用的 GPU 上由一个常驻服务管理 CUDA 资源与已加载策略；不为每 actor 复制 SLA，不要求整张设备只有一个 CUDA context，也不改变其他已授权进程。JAX／Torch／渲染互操作按实际 context、device 和 stream 验证。CUDA 初始化后不 fork；需要 CUDA 子进程时采用现场支持的 spawn／forkserver。[PyTorch multiprocessing][torch-mp]

起步为每 actor 4 个 host observation slot、GPU 输入当前批＋下一批、每 actor 最多 1 个未完成 policy query。共享内存未必已 pinned，允许一次 CPU staging copy；在初始化或 loader 中建立固定 pinned pool，不在每帧主线程临时 pin。copy／compute 使用明确 event 依赖。每slot有generation ID与消费者引用，只有policy输入传输和writer读取都完成才可复用；writer取得独立不可变副本时，先将副本计入字节限额再释放原slot。不得仅凭policy回复释放writer仍引用的帧。non_blocking 本身不证明传输与计算重叠。[Pinned memory][pinned]

writer 队列按**字节和帧数**双限额，site 配置必须给出非零上限；起步 1 GiB 总 pending snapshot，依实际可用 RAM 下调。发布下一snapshot前先取得buffer／writer容量，不能先无限分配再等待高水位处理。高水位停止派发新 episode，必要时在现有安全暂停边界背压；不在控制积分中等待网络。超出帧时钟／完整性要求就把 episode 标为不完整并保留原因，不静默丢帧后继续计合格。

## 4. Corelab 采集实现：先让单环境快，再并行

### P0：绑定现场与最小 profile

交付一个简明现场表：实际执行 SHA、旧任务终态、CPU 核／内存、项目可用 GPU、模型／数据版本、输出所在真实文件系统。只检查相关路径，不能把共享挂载的 symlink 当成本地盘。没有本地可写存储时如实记录，有限 staging 方案单独测，不能假装写入不在关键路径。

先在一个既有可完成 TRAIN 条件记录初始化、reset、积分、solver、相机、ACK、写盘、模型时间；覆盖 native 子进程 CPU，不能只计 Python 调用线程。profile 不依赖 H200。暖启动与冷启动分开；有重叠的 spans 不相加冒充总墙钟。

### P1：持久相机、内存传递、异步 writer、事件就绪

按以下顺序修改现有模块：

1. 把 BodyBaseCameraCapture 接入持久 session；episode 内保留 MjModel、渲染资源和固定输出 buffer，按不可变 snapshot 重建正确 scene 状态。
2. 直接输出 body/wrist 内存图像、state 和 native step；同步不经过 PNG→读回。原 hash／calibration 身份在加载和版本边界核验，逐帧 hash 在 writer 中生成并绑定同一 immutable snapshot。
3. 为新采集角色接通既有 settle_atom 事件条件，保留 pause ACK、精确 step、WBC 模式、solver 和 gripper consumption。timeout 保留有限终态。
4. writer 从 snapshot 异步生成 lossless shard、metadata 和 manifest；压缩、fsync、传远端均移出控制调用。发布前完整 flush，缺写入的 episode 不进入训练。

最小验证：32 个已保存状态的新旧像素／标定／step 对照（不新增物理）；随后 4 个完整 TRAIN episode 验证图像、命令消费和任务语义。纯资源复用应保持原图像；若存在渲染差异，保留差异记录并另版本验证，不悄悄放宽。

建议默认存储为无损 uint8 数组分块＋数值命令／状态表，保持当前 reader 的可导出兼容；压缩放 writer。每文件目标 256 MiB、上限 1 GiB，按 frame chunk 与 episode manifest 组织，长 episode 可引用多个 chunk。一次 hash 写入、一次转移后核验；完整 immutable shard 在可信缓存中不逐样本重 hash。新方案无需搭建新的数据湖或调度平台。

### P2：多环境隔离与生命周期

每 actor 独立拥有：

- MuJoCo world、solver/controller/leg policy 历史和 reference；
- ROS 域／节点／端口、native 时钟、query 编号、随机数；
- renderer context、可变 codegen/cache、tmp、输出与日志；
- 所需 CPU affinity 和线程预算。

只读资产可以共享，控制状态不可共享。先核查当前 ROS 实际版本和隔离机制，不能照抄不适用的 domain 参数。CPU 预算覆盖 solver、BLAS/OpenMP、软件渲染和 writer；防止 N 个 actor 各自开满所有核心。初始线程预算由实际可用核数计算并留出服务／writer 核，不硬编码机器容量。

第一版做到**episode 内持久＋独立 episode 并行**。跨 episode 的 persistent actor reset 是后续优化：必须重置 qpos/qvel 之外的 MPC warm start、WBC/腿历史、消息队列、reference、时钟、RNG 与相机状态。未证明完整 reset 时保持每 episode 原生进程重启；这不阻止 episode 内相机优化与多世界并行。

单个 actor 故障只关闭其世界并保留终态，不重启健康 pool。global/shared checkout 的写入 owner 与每 world 的控制 owner 分开；不删除原安全锁来获得并行。

### P3：持久策略服务与专家纠正

专家采集无需 SLA。学生／纠正 rollout 使用常驻服务，先支持 batch1／2／4，静态 shape bucket 避免每个实际 batch size 反复 JIT。padding 有明确 mask，不能改变归约、RNG 或动作。

按 actor 独立保存采样噪声；动态合批顺序不能改变同一 observation 的随机流。请求绑定 policy hash，回复核对 episode/query/native_step；超时回复不得喂给下一 episode。最多等待 5 ms 合批作为起步配置，按 measured p95 与任务节拍调整，不以无限等待凑 batch。

在 16 组固定 TRAIN 输入上检查常驻服务与原 provider 的数值／动作合法性，再做 1／2 actor 各 2 个完整闭环。控制使用本机或实际合适的近端推理资源，不能默认每 0.5 秒跨站点向 H200 同步 RPC。只有远程 p95、带宽与断线行为实测满足合同才采用；否则 H200 负责训练，发布 checkpoint 到 Corelab 常驻推理服务。

## 5. H200 learner／teacher：一次上传、整段 GPU 执行

### P4：完成真实 batch 资格与驻留优化

基于现有 V2 修复而非重建 trainer：

1. 把只支持旧 500 steps／16,000 visits 的 LR、visit map 和数据规模硬编码改为显式 run config；旧配方保留原 reader／日程。不能只改 max_steps 就宣称 3k／20k 已正确支持。
2. 保留同一 fixed32 RNG bank 的输入生产边界和 state-token dropout 语义；比较 microbatch1／8／16、有效 batch32，缺显存时只测实际能运行的两项。
3. 持久 loader＋有界当前/下一批预取；解码／tokenization／无损统计可以缓存，GPU 上传后不为每 microbatch重新拿回 host。有效 batch staging 能放下时一次上传后在 device 切片；否则按不重叠 microbatch 各上传一次。“一次”按sampling visit及实际缓存驻留期计量；缓存未命中允许必要H2D，同一window的合法重复访问保留各自visit、noise、time、dropout和权重，不为省传输而去重训练人口。
4. 模型、梯度、Adam 和训练激活留 GPU；gradient accumulation、finite flag、active7/padding25 指标在 device 归约。取消逐 microbatch 的 float(loss) 和全局阻塞，保留每 update 一次小终态确认；非有限时不更新权重或提交 data cursor。常规指标每 50 update 返回，完整快照另计。
5. 保留真实 loss／gradient 对照、连续4与独立2＋2恢复、冻结参数 fingerprint、新进程普通 RGB 重建；CPU 两参数测试不能替代模型验证。
6. 预热／编译与稳定更新分别计时，在计时边界等待设备实际完成。不能用异步 dispatch 返回时间当吞吐。[JAX 计时][jax-async]

真实 paired qualification 沿用 loss rtol=1e-3/atol=1e-5、gradient cosine≥0.999、relative L2≤0.02。每实际采用设备起步最多40资格／计时更新、1,280 visits、32次RGB生成，属于测量预算而非反复人工审批；改变科学配方后另命名实验，不能靠扩大容差混过去。只测试会实际使用的设备，不为占满机器扫所有卡。

梯度一次诊断导出可以存在，归入 qualification 成本；热路径梯度／参数 D2H 为0。模型可训练层的 VLM embedding 不可离线冻结缓存；冻结视觉模块的缓存也必须绑定真实输入和版本。

### P5：空间教师与缓存

先完成与 FM 路线匹配的 batched alignment；alpha=0 应保持公共参数梯度与 FM 一致。projector-only 更新不能冒充学生吸收空间知识。

两种实现按 profile 选一种主生产路径：

- **同 GPU 在线教师**：PyTorch OmniVGGT 与 JAX learner 共驻；GPU 图像与 target 通过现场支持的 DLPack 交换。明确相同 device/layout，能用 copy=False 时让隐含复制显式暴露；保留 producer 至 consumer 完成，不原地改写共享 buffer。空间教师冻结，学生分支仍可反传。DLPack 不是跨进程队列通信协议。[JAX DLPack][dlpack] [DLPack 规范][dlpack-spec]
- **离线冻结特征**：只对实际监督 anchor 计算，按 shard 缓存有效 token、mask、teacher/preprocess/image 身份；批量预取到 GPU，再做 padding。旧 FeatureBank 的逐样本 npz/hash 路径不直接扩大到 D1000。

现有两相机有效 target 为 512×2048 BF16，即2 MiB/anchor；约60,000锚点仅有效特征约126 GB，尚未算模型／激活。**不默认全部驻一张 H200，也不为10 Hz所有原始帧无条件计算教师。** 采用可度量的 cache 上限、命中率和按需计算。缓存不能与改变 crop／颜色／相机的增强错配；本轮基础矩阵继续无额外图像增强。

JAX与Torch共享显存时，显式规划模型、teacher、激活、预取和cache预算，依据现场版本配置allocator并测峰值。避免两框架抢占后退回逐步CPU offload。若同卡不划算，可先离线生成有效target，再进行FM＋alignment；不要把教师准备当无空间监督首策略的阻断。[JAX 显存][jax-memory]

### 5.1 如何证明没有隐藏往返

有限 profile 时对 JAX 启用 log_explicit；正常热路径禁隐式传输，明确标注输入／动作／日志／checkpoint边界。disallow仍允许显式device_put/get，不能单凭guard说传输为0；各loader线程也要正确配置。[Transfer guard][jax-transfer]

结合GPU profiler与代码计数，分别输出 H2D/D2H/D2D bytes/calls、同步等待及用途。图像、teacher feature、动作、指标和持久化分开；渲染API和Torch不由JAX guard全部覆盖。

验收目标：

- 模型／optimizer／完整梯度热路径 D2H=0；
- 每个visit的未命中输入只做一次必要上传，驻留期间不绕回host；保留重复抽样语义，无逐flow-step／逐layer／逐physics-step大张量host bridge；
- CPU来源学生每query一份观察上行、一份动作／必要审计下行；
- teacher跨框架交换无NumPy中转；
- PNG写盘后再读取不在控制关键路径；
- 无界prefetch／writer backlog=0；必要CPU/GPU边界如实计量。

## 6. 动作、时钟和数据语义必须同时成立

继续采用V2动作合同，新旧版本单独报告：

- GRIP_LOGIT_V2：原nominal标签保留，epsilon=1e-4，inverse latent normalization→stable sigmoid→原physical映射；实际夹爪state不改。偏差先通过专家原生夹持验证，再运行R1_500。
- EEF_STREAM_V2：H40、50 ms命令与2 ms物理保留；起步执行10点／0.5s再观测，保留中间SE(3)路径和夹爪事件。现有2s baseline仍可运行；新技能需要曲线时不得继续把路径拉直。
- reference TTL仍为0.4s，原≤0.35s续期独立于policy查询；0.5s重规划不能覆盖或延长TTL。续期只沿既有有效预测后缀／规定终端参考，不新增一次隐含策略查询。
- 不重复截取2s平滑曲线的起始慢速段；连接处约束速度／加速度、reference安装ACK和原安全。
- 每个future knot必须能定位到真实消费的专家命令，先还原issuing query坐标，再统一到当前训练anchor。未来实际TCP、错误学生命令、未消费suffix不自动成为专家标签。
- nominal窗口有任一点来源不明／未消费／被拒绝则整窗排除；纠正片段按独立专家来源规则纳入。
- 任务结束后真实执行并记录至少2s安全保持供终末窗口，不合成未来padding；尾段失稳不计合格完整示范。
- 原始RGB/state建议10Hz；训练anchor按已选0.5s节拍2Hz；命令20Hz。相机记录必须绑定native step，不能重复最后一帧冒充10Hz。各频率均是待实现合同，不是已有数据。

先复用V2的4布局×2重启×2节拍=16个专家条件；只有接触反馈仍不足才追加0.2s的8条件。这个变更验证一次后给所有数据actor复用，不为每个actor再跑一遍全部资格。

## 7. 数据生产：有配额、覆盖与学生反馈的连续流水线

### P6：可学习的抓持／放置数据

默认PICK_HOLD与PICK_PLACE：10个可达位置/姿态层×5个物体×2任务=100层。D100_new／D300_new／D1000_new每层分别1／3／10条，保持同一支持域。PLACE专家未ready时明确使用单任务版本，不把后来加入任务解释成纯数据量收益。

| 版本 | 合格完整episode目标 | 本波新增成功 | 本波尝试上限 |
|---|---:|---:|---:|
| D100_new | 100 | 100 | 200 |
| D300_new | 300 | 200 | 400 |
| D1000_new | 1,000 | 700 | 1,400 |

每波在成功配额或尝试上限达到时提交真实版本。优先调度欠覆盖strata，不用容易样本补满总数。记录attempted／technical-complete／task-success／usable-with-H40／完整覆盖数，不能把100个episode自动称100条无接管成功示范。

数据内容：同步body/wrist RGB与时间戳；实际EEF／grip；指令；专家nominal与消费区间；实际整机反馈；task/scene/object/seed／相机／frame/TCP／单位／专家与控制器版本；接管与阶段；完整outcome与首次失败。仿真真值只进入专家和审计，学生保持可部署输入。

每episode上限默认60s任务＋最多2s记录尾段；固定nominal、相机与WBC合同。采集worker有有限campaign、walltime和writer资源上限，分波结算后在已有授权内继续下一有依据波，不需逐波人工GO。不得无限重复同一故障。达到数据量也要检查分层覆盖与有效窗口，而非凑文件数。

### 7.1 writer、版本与Corelab→H200交付

数据按episode提交manifest、按chunk形成不可变shard。writer状态为writing→committed或incomplete，训练仅读取committed。episode的committed manifest只在全部引用chunk/shard已封存、flush且checksum／索引可验证后原子发布；正在追加的shard不进入训练可见版本。可提前封存较小chunk，避免等大文件填满。转移走已有授权路径，源／目标按shard校验一次；保存transfer manifest，失败从未完成shard继续，不逐batch跨站点读PNG/npz。

H200先把当前版本stage到其实际训练节点可用存储；共享FS无法避免时，明示并测cache-miss I/O，不能谎称local NVMe。Corelab继续生产下一版，learner本run固定数据snapshot／stats／sampler，禁止一边append一边改变训练人口。

不随意删除原RGB、失败episode或旧权重。空间不足时停止新wave并报告所需容量，使用用户已有清理安排。

### 7.2 采集、训练、纠正的衔接

D100_new完成即启动首个短学生诊断，Corelab并行补D300。新技能10–30完整种子就训练首版，不等待D1000或SLA胜过GR00T。

D300四臂＋同配方D100完成后，若DEV48增加≥4个成功或发现明确可采集状态缺口，才扩D1000；4/48是工程投入标准，不是显著性结论。若TRAIN仍欠拟合，先看优化预算曲线，不能把短训练结果当数据无用。

纠正采集从固定候选在TRAIN-only场景偏离的状态开始，第一波最多400学生rollout、目标至多200个成功专家纠正片段。接管前错误动作不作BC；没有正确纠正的失败只用于分析。正式DEV/TEST不回灌为训练轨迹。

沿用匹配续训：同父候选分别100%nominal和80%nominal＋20%correction，各1,000×B32；相同优化器初始化／统计／LR／评价。只有纠正臂的额外收益才能归因数据类别。

## 8. 训练设计：先筛配方，再验证训练充分性

### P7：保留V2核心比较

D300四臂分别3,000×B32=96,000访问：双LoRA／完整action expert两种范围，分别有无空间loss。各从同SLA_base开始，RGB224、同state dropout、相同数据和动作合同；参数白名单与LR沿用V2。D100和条件性D1000各补一个相同recipe的3k端点，L300复用。

**此固定计算预算曲线不声称各数据量都已充分收敛。** 不在选择recipe的同时解冻完整视觉骨干、改batch64或加入随机增强。有效batch改变另立配置，先把单卡真实batch32跑好。

### P8：选中recipe的可选优化预算曲线

在标签／接口正确且TRAIN仍可学习时，允许一个明确后继，检查总步数 **3k→6k→12k→20k**。只给已选recipe、已冻结数据版本做这项研究，不把所有矩阵候选各续20k。

执行方法：

1. 从选中3k完整checkpoint恢复，保留Adam、RNG、sampler位置与固定normalization。
2. 运行前固定追加17k的尾段LR：从该参数组第3000步已用LR连续出发，按cosine到第20000步降至该起始LR的20%，不再warmup；所有中间端点都使用这条同一尾段。各参数组保持比例。
3. 固定6k／12k／20k检查点及各段有限walltime，保存中间状态但不在DEV挑未登记步数。
4. 每检查点固定TRAIN8＋DEV48；相对较早候选收益≤2/48视为接近，优先较低成本、原安全与分任务无退化者。连续两个新检查点无DEV改善且TRAIN也不再改善，可结束该分支。
5. 根据DEV选出的U（3k／6k／12k／20k）用于后续两臂纠正和两个确认seed；不能让seed0训练20k、另外两个只训3k，再报告“三seed同方法确认”。
6. 两个确认seed各从同基础权重重建3k前缀＋固定20k尾段到所选U的前缀，不能按U重新缩短cosine日程；若采用纠正，再各执行同一1k纠正配方。独立TEST只对最终固定端点打开。

这会引入DEV模型选择，最终TEST保持独立。原V2 3k核心成绩原样报告；长训练单列，不追溯更改已登记比较。

### 8.1 预算与计量

| 工作 | 最大新增更新（完整分支） | B32样本访问 |
|---|---:|---:|
| 继承V2主矩阵（S2、R1、6核心端点、2纠正、2确认seed） | 29,000 | 928,000 |
| 选中seed0从3k延长到20k | 17,000 | 544,000 |
| 另外两个确认seed匹配相同延长 | 34,000 | 1,088,000 |
| **含长训练的主线最大值** | **80,000** | **2,560,000** |

80k不是自动排满的任务列表。记录seed0探索到的最高步数U_probe与最终选择的U_selected，完整分支预算为29,000＋(U_probe−3,000)＋2×(U_selected−3,000)。只有两者相等时，3k／6k／12k／20k分别对应29k／38k／56k／80k；例如探索到20k后选6k应计52k，选3k也已花46k。未选中的探索成本不因选择撤销。记录actual_completed_updates，已完成原S1_500不重做、不重复计入；已执行的后继从实际剩余量扣减。预算是研究工作量，不是新的人为权限门槛。

原V2主学生评价最大934条件保留，seed0长训练额外最多3×(TRAIN8＋DEV48)=168条件，合计最大1,102；性能／采集／教师节拍／新技能另计。两个确认seed的中间checkpoint仅保存／恢复，不另外逐点打开TRAIN8＋DEV48；若另评则单独计入。新数据短诊断为独立pilot，不混入矩阵分母或重复充作正式DEV。D100首个500×B32 pilot另计16,000访问及10个TRAIN诊断条件。

最终继承三seed各TEST80：每任务24个常规、16个挑战条件；工程目标为每seed每任务常规≥20/24、挑战≥10/16，完整终态全部报告，阈值不是已取得成绩。策略引发的硬安全停止计策略失败并遵守原停止行为；技术故障单列，不能删除失败后只报成功子集。

另外：每采用设备资格最多40更新／1,280访问；每ready新技能10–30种子首版500×B32；共享四技能起步5,000×B64沿用V2作为独立后继。无需把这些追加成所有资源同时启动。

## 9. 性能实验与明确验收

### P9：从性能测量选并行度

固定8个TRAIN性能条件×2独立重启=16个episode/配置。相同任务、录制频率、动作／相机版本和完整成功定义：

| 配置 | 条件数 | 回答的问题 |
|---|---:|---|
| 原串行参考 | 16 | 当前端到端基线；已有完全兼容profile可复用 |
| 优化后1 actor | 16 | 持久化／事件／writer单独收益 |
| 优化后2 actors | 16 | 第一档并行与隔离 |
| 优化后4 actors | 16 | 主要生产配置候选 |
| 优化后8 actors | 16 | 条件性扩展，4有收益且容量允许才跑 |
| 优化后16 actors | 16 | 可选，8仍有收益才跑 |

最大96个性能episode；8／16不是硬件容量承诺。旧参考与优化版若科学接口改变，分开做“旧合同性能基线”和“新合同容量扫描”，不能把降低频率、换短任务或换成功定义当优化收益。

并行工程目标相对于**优化后1 actor**：2 actor≥1.5×、4 actor≥2.5×有效完整示范/小时。相邻倍增收益<15%即采用更优实测档位并定位瓶颈；不为达到actor数字强行扩并发。小样本profile报告波动，不宣称统计显著。

每档提交：

- attempted、technical-complete、完整成功、usable H40和分层覆盖；
- 有效完整示范/墙钟小时，冷启动／暖运行分别报告；
- 初始化、reset、solver/physics、renderer、ACK、writer、model服务耗时；
- 相机到命令及queue p50/p95、native step与图像一致性；
- H2D/D2H/D2D bytes/calls与同步等待；
- 全native进程CPU、峰值RSS/VRAM、writer backlog与I/O；
- 重复样本／丢帧／跨actor污染／完整reset情况。

PerformanceTrace既有分位数是有限样本，报告注明采样范围；并行墙钟以整个campaign起止计算，不把每actor速度相加冒充端到端。

### 9.1 与H200训练并行时的联合压力检查

使用获选actor数，运行至多32个新TRAIN专家episode，并同时进行已有授权learner run或其有限profile段。复用正式训练更新，不额外暗加步数。检查共享存储、转移和writer背压：若有效数据产量或训练steps/s相对各自独立基线下降>15%，错峰传shard／限writer并发／节点local缓存，而不是丢帧或停安全确认。

资源不同时可用时标为待测；Corelab生产和H200离线训练各自继续，不互等heartbeat。成本分别报CPU-hours、各GPU-hours、跨站点传输量与wall hours；不把GPU利用率高当任务能力。

## 10. 复杂技能与GPU物理的推进条件

柜门、推物、书架专家可与抓放主线并行，复用既有资产和原评分。每个就绪技能先10–30条完整种子、最多60次尝试，再500步首策略；未就绪的专家只阻断该任务。柜门从可达无锁门开始，书架区分已持书插槽与桌面取书全链，部分成功单独报告。

全链专家与学生已有真实成功后，按覆盖扩到100–300，再决定1,000–3,000；5,000条共享四技能数据是后继规划，不是本轮立即采满命令。共享策略使用同权重、同可部署输入，通过指令选择任务；接触恢复和语言选择要有实际数据。

MJX-Warp已有多世界GPU RGB/depth渲染，但原生CPU WBC不会自动迁移。[MJX文档][mjx] 只有profile表明physics／渲染是主要瓶颈、预期节省覆盖迁移成本时才开展。若GPU physics每2ms向CPU传整状态求WBC再传回控制，仍需计时验证，不能称GPU全链。

GPU后继先验证控制、接触、摩擦、reset和相机时间绑定，再比较32／64等批量；保持原生MuJoCo作为参考评价。渲染context的nworld、生命周期和BVH／render token按实际版本处理。在线RL另立研究问题，本计划先交付可工作的模仿学习与纠正循环。

## 11. 实施顺序、可review的交付与时间估计

| 工作包 | 可并行关系 | 必须交付 |
|---|---|---|
| P0现场／profile | 先做，随后各站独立 | 最新现役终态、真正资源与耗时分解 |
| P1相机／内存／事件／writer | 与P4 learner并行 | 优化单actor真实完整episode，控制路径无PNG回读 |
| P2 actor pool | 在P1新合同成立后；不同任务专家可并行 | 2／4 actor实际隔离、完整结果与吞吐 |
| P4/P5 GPU learner/teacher | 不等D1000；teacher准备不阻断FM | 真CUDA梯度／恢复、驻留trace、RGB可部署包 |
| 动作consumer＋P6数据 | 与相机工程共用边界，防重复修改 | 可消费轨迹、D100/D300合格版本 |
| P3学生服务 | 与专家生产并行开发 | 固定policy常驻、跨actor结果不串线 |
| P7/P8/P9 | 数据一版完成立即训练／评价 | 核心对照、按需优化曲线、并行产量 |
| 纠正／三seed／TEST | 固定方法后 | 真实checkpoint、逐条件成绩、成本与失败分类 |

第一轮工作目标是**两条完整新episode＋实际GPU训练更新／或明确站点阻断＋一次可部署RGB生成**，而不是增加一批文档和CPU检查。单站不可用不阻断另一站的已授权工作。

原生优化建议拆成三个实现PR：①body session／memory observation／async writer；②事件wait／actor隔离与pool；③batch驻留／teacher互操作与常驻推理。数据reader／consumer可以随必要依赖提交，避免修改同一文件的多个并发写入者。每PR只测试改变的接缝和一个完整原生流程，共享关键代码变化才扩大回归。保留历史reader与reference，避免新建重复调度器、审批器和大日志系统。

时间由实测计算：

- 采集小时＝目标合格episode／实测合格episode每小时；每个wave重估剩余量。
- GPU小时＝GPU数量×(加载编译＋更新数×实测step秒＋推理秒)/3600。
- run walltime／RPC timeout／writer上限在现场绑定为有限值；初次测量使用实际调度窗口，不能无限sleep。
- 保留20–30%加载／写盘波动余量作为排程估计，不冒充额外数据或效果。

## 12. 每次交付的简洁回报

只突出能解释当前科学进展的内容：

1. 实际source／dataset／policy／consumer版本与本次新变化；
2. 有效完整示范数／全部尝试、覆盖、失败阶段；
3. actor数和有效示范/小时，最大开销及CPU/GPU传输统计；
4. 实际训练更新／样本访问、参数范围、steps/s与显存；
5. 真正被WBC消费的候选轨迹、完整TRAIN/DEV/TEST成绩；
6. 已固定checkpoint与普通RGB推理包，下一项有依据工作。

引用既有manifest和有限终态即可。公开提交代码、配置和非敏感摘要；原始RGB、权重、vendor/WBC资产及主机私密路径继续保留在授权存储。每次端点保留失败，不通过重命名版本、抽样成功或删除难场景美化结果。

## 13. 固定证据与一手技术来源

[v2]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/b07936899b10f196653f617f95c5cd8b124e5e99/docs/protocols/SLA_GO2_ARX5_SCALEUP_V2_20261112.md
[pr378]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/378
[pr380]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/380
[stage]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/reports/WM02/sla-scaleup-v2-20261112/CURRENT_STAGE_ACTUAL.json
[capture]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/src/go2arx5_vla/world_model/fusion_paused_observation.py
[body]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/src/go2arx5_vla/world_model/native_body_camera_capture_v1.py
[renderer]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/src/go2arx5_vla/world_model/native_camera_transfer_v1.py
[worker]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/scripts/pi0294/task_native_worker.py
[perf]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/src/go2arx5_vla/world_model/native_collection_performance_v1.py
[consumer]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/src/go2arx5_vla/world_model/stable_task_prefix_v1.py
[student]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/src/go2arx5_vla/world_model/task_learning_student_batch_v1.py
[batch]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/src/go2arx5_vla/sla_adapter/batch_v2.py
[runtime]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/src/go2arx5_vla/sla_adapter/runtime_v2.py
[features]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/src/go2arx5_vla/sla_adapter/spatial_assets_v1.py
[spatial]: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/aaf80b4eda7ecb2816c0c251d3535a4e54ea62c9/src/go2arx5_vla/sla_adapter/spatial_runtime_v1.py
[mjr]: https://mujoco.readthedocs.io/en/latest/APIreference/APIfunctions.html#mjr-readpixels
[mjx]: https://mujoco.readthedocs.io/en/latest/mjx.html
[dlpack]: https://docs.jax.dev/en/latest/_autosummary/jax.dlpack.from_dlpack.html
[dlpack-spec]: https://dmlc.github.io/dlpack/latest/python_spec.html
[torch-mp]: https://docs.pytorch.org/docs/stable/multiprocessing.html
[pinned]: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
[jax-async]: https://docs.jax.dev/en/latest/async_dispatch.html
[jax-transfer]: https://docs.jax.dev/en/latest/transfer_guard.html
[jax-memory]: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
