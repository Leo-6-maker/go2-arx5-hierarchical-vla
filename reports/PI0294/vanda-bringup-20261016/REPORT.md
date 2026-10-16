# Vanda 独立开发平台：实际准备与排队收据

2026-10-16，Asia/Singapore。原生状态快照：19:44:56；Hopper 原 owner 收据刷新至 19:45:03。

**已完成资源核验、固定推理资产和 CPU processor 准备；A40 实测尚在排队。完整 GR00T＋冻结 WBC 闭环尚未建成或验收。**

本轮直接依据用户在当前 local-PC 对话授权使用 Vanda 项目 GPU 额度，实施有限的平台工程检查。沿用现有唯一 local-PC owner，未派生其他账号/agent。此授权与现役 E0、已关闭 Teacher/WBC scope 分开登记；不借用旧预算或重置其时钟。

## 实际完成

| 项目 | 原生证据与结果 |
|---|---|
| 连接与账户 | `vanda-simon` 密钥登录通过；当前 amgr 项目查询有效，无需重复登录或保存密码 |
| 项目资源 | `CFP04-CF-007` 当前有效、账户为成员且余额查询成功；项目 scratch 可写并已实际容纳固定模型 |
| 调度约束 | 付费 `batch_gpu` 单次上限 168h、最多 2 GPU；当前用户该队列并发上限 4 作业。`interactive_gpu` 当前配置上限 12h，不能沿用旧说明文件的 2h |
| 实际分配形状 | 本轮每次申请 1 GPU/4 CPU/32GB；站点将其规范为 1 GPU/36 CPU/240GB。检查进程另限制最多 4 个活跃 CPU |
| A40 登记 | PBS 节点快照登记 98 个 A40 物理 host、196 个带 GPU 的 vnode；其中付费 `gpu_node` 池 80 GPU。登记总数不是本账户当前可用/独占数量，也不是 GPU 实测 |
| 存储 | 个人 Home/scratch 接近配额上限；新增模型/运行产物写项目 scratch，未清理或移动历史资产 |
| 固定官方源码 | 从 Hopper 原固定源只读导出独立 Git 对象和原提交，Vanda `HEAD=51d4c89f72fda44cbf77285c6a8114b52676b8a1`、tracked diff 干净；不依赖旧 parent 对象库。LFS 保留源指针，未下载无关 demo payload |
| 固定 M2048 | 9 个推理必需文件，12,576,324,408 bytes，逐文件 SHA 全部匹配。未下载 optimizer/scheduler/RNG 训练恢复成员，未加载模型 |
| 固定 Cosmos | 复用 Vanda 已有 `9ce19a195e423419c349abfc86fd07178b230561` snapshot；10 个必需文件逐一匹配 Hopper 固定副本 SHA |
| CPU processor | 固定官方 processor＋原 statistics＋原 TRAIN 第 0 窗口真实已存观测实际处理通过；model loads=0、physics=0。诊断进程 CPU 18.8451s |
| 新平台入口 | Vanda 新建标准 `Projects/GR00T-WBC` 导航、私有 registry、run manifest、output binding 和独立 lease。大型运行根在项目存储。使用明确的 Vanda bootstrap adapter；不声称尚不存在的通用 workspace CLI 已部署 |

固定 M2048 来源：`leoyuer/pi0294-public-artifacts`，revision `4c3097652bb81e43ea40a2b0b423086ec6ea7091`；prefix 和预期 SHA 直接复用原 R3 公共收据。原 v3.0 statistics、EEF10 和 processor 字节不重拟合。

固定 VLA 数值模块来自不可变 `ffcb6887fb22d5e6a32a38463536c7e36992986e` capsule：438 成员、3,523,614 bytes，独立 source seal。保存的 TRAIN 观测包 95,439 bytes，仅作接口/延迟检查，不是 live closed-loop、新示范或成功数据。

初次 CPU processor 检查因冷启动导入在 85s 墙钟上限处中断；另一次带有限截止和堆栈记录的诊断实际通过。堆栈显示耗时在共享文件系统上的 Torch/Dynamo 模块读取。前一次中断保留，不把它改成首次即 PASS，也不据此更改模型或数值算法。

## 已实际提交的两个独立作业

| Run | Job ID | 快照状态 | 最长资源 | 内容 |
|---|---|---|---|---|
| `20261016-vanda-stage0-01` | `1462438.stdct-mgmt-02` | Q，`node_pool` 资源不足 | 1 A40 × 15min | GPU/驱动/CUDA、软件导入与容器能力；无策略加载或机器人运行 |
| `20261016-vanda-stage1-01` | `1462451.stdct-mgmt-02` | H，等待 Stage0 afterok | 1 A40 × 30min | 一次 M2048 加载、1 次预热＋16 次保存观测查询；H40 finite/维度、解码、显存和延迟 |

两项顺序执行，单卡申请合计最多 45min。每个作业单次提交、禁止 scheduler 自动重跑；独立 manifest/lease 在各自登记后 24h 失效。失效后即使获分配也拒绝模型工作；该检查不代表 PBS 会自动撤销仍排队的作业。

Stage1 还会在加载前核对自有 RUNNING PBS job、项目/单卡/时长、Stage0 的真实 GPU/CUDA 结果、独立源码 seal、固定官方源、模型/processor/Cosmos 字节及 TRAIN 观测身份。沿用原 `DETERMINISTIC_CUDA_MATH_SDP_V1`、bf16 autocast、原四步 sampler 和原 `ResidentModel.generate`；原 Hopper/E0 admission 本身没有被放宽。

16 次重复保存输入得到的 p95 仅是小样本接口延迟读数，不是在线生产尾延迟、任务能力、任务成功或实时 WBC 闭环资格。新 adapter 不安装或执行任何预测动作。

## WBC 兼容性与尚未完成项

Vanda 宿主是 RHEL 9.4，未见宿主 ROS 安装。现有旧 WBC 工程/项目容器为 RoboDuet/IsaacGym/IsaacLab 路线，不能认作当前 CoreLab 冻结 MuJoCo＋ROS 的合格副本。

现有 Ubuntu 24.04 容器可以运行；实际检查没有 ROS、cmake 或 colcon，只有编译器。root-mapped user namespace＋Apptainer fakeroot 的 `id` 检查成功，说明有可尝试的容器构建路线，**不等于已完成 ROS 安装、WBC 编译、EGL 渲染或物理资格**。

已只读获取 CoreLab 现有 registry 的冻结 WBC/资产角色索引，保留当前参考身份。后续需补齐匹配的 Ubuntu/ROS 运行容器，绑定 CoreLab 当前实际源、机器人 XML、策略/依赖 manifest，再做保存状态/提交参考的等价性检查。不得拿本机旧最小交付包或旧 Vanda RoboDuet 冒充当前冻结执行环境。

下列目前仍 NOT_RUN：实际 A40 模型加载/H40/显存/延迟；GPU EGL 渲染；新平台 WBC 构建/运行/等价性；VLA＋WBC live episode；Recovery Teacher；DAgger/FlowDAgger；任何优化器更新。旧 WBC/Teacher 安全 FAIL 保留，不自动重开旧物理任务。

## 验证与现役任务

两个新 adapter 语法检查通过；本地临时 fixture 共 13 项定向检查通过，覆盖正常准备、过期、权限缺失、额外 GPU/query、源码改变、登录节点拒绝，以及 admission 前没有导入 Torch。它们是工程 guard 检查，不是 GPU/物理 PASS。

部署与实际运行的 SHA 分开：

- Stage0 entry：`ffcb6887fb22d5e6a32a38463536c7e36992986e`。
- Stage1 entry：`76aa85e3a23f596989e6cb56a470bdb2acf1cab1`。
- Stage0 manifest SHA256：`11bcd9f92e726850061ff8d150f0c3358a01e8992c1f05e8521f71d1a12dc126`。
- Stage1 manifest SHA256：`a1f9758d17abda80a96219d256f0328f48e399425c2d130dafe20fea5f962a7f`。

Hopper 原 owner 的 19:45:03 收据仍为 `670049.hopper-m-02` Q，runtime `9a699d28c6a585df1abb5aa906e41e8d8c025bdc`，model loads/query=0。本轮未取消、复制、迁移或热更新它；未改 CoreLab 源/参数/统计/安全或旧终局。Vanda 另一项原有排队作业保持原样。

原始项目 roster/余额/调度输出、主机私有路径、观测 RGB、模型和完整日志仅保存在私有运行根；本报告是脱敏小收据。GitHub 发布与排队本身不等于真实 GPU 或完整闭环已经通过。
