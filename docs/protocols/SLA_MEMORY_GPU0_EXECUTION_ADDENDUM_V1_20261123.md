# SLA_Memory 冻结迁移：GPU0 执行推进补充协议 V1.1

> 2026-11-23 · PLAN_ONLY，尚未由此文件启动推理、仿真或训练。基线：PR #428 的 V1 协议；本补充覆盖其中与 2026-11-23 P0 静态核查相矛盾的三处技术解释。实验目录由现场报告为 /home/liuyu/ISSAC-GR00T/work/sla-memory-frozen-transfer-v1，本文未直接读取该主机的本地文件。**Gate 0 仍是 NOT_RUN**。允许目标：主机 GPU0 的独立冷启动与上游 O0 推理；这不等于授权抢占其他运行作业或自动放开 P3 原生环境。

## A. 新的不可逆事实与更正

- SLA_Memory ARX X5 真实模型每次 **12 张图像** = 三路当前（cam_high, cam_left_wrist, cam_right_wrist）+ 主相机 9 张历史（episode 起始帧、过去 8 至 1 秒）。当前三路 mask 全为 true；不存在原生的“缺一只手腕相机不使用”输入配置。任何黑图、复制图、虚拟臂都要标明 OOD。
- 官方 RobodojoReal transform 的输出经过逆变换后是 **每只手各自的 ARX X5 arm-base frame 下绝对 EEF**，来自 RobodojoOutputs.target_to_local_arm_pose；模型内部采用双臂中点，不能对已经还原的 16D 输出再加一遍 midpoint offset。单臂到 Go2 world/TCP 必须用真实的 Go2 arm-base pose、固定工具标定、姿态约定。校验 arm side。
- 官方 source fps=30；RobodojoMemoryBuffer 用**更新次数**而非观测时戳回看 30、60、…、240 帧。若向其直接每秒写20帧，则8秒历史变成12秒。O0 保持源数据30Hz；P1/O2 需另立“真实时戳对齐”观测桥，预先声明实现和消融，不能默默给高层伪造新视频。
- 若真实模型 action shape=(50,16) 且 t0=0，源最后一个点 t=49/30=1.633333s，H40最后点39×0.05=1.95s，所以 index33..39 共 **7/40=17.5%** 不在原生预测覆盖范围。真实首点时刻、形状必须在 O0 读取后才能确认。严禁把 source 50 点直接当 H40 或静默延长到 1.95s。
- 模型清单按静态核查：排除 train_state 后19个文件、12,441,551,063字节；这是**未下载的发布清单**，不是本地校验完成。上游 XPolicyLab source=thriving191228/XPolicyLab@c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454；权重 HF revision 7f6880569c3be777291e646fad1b4951245e8553；需同时记录 tokenizer/assets/norm stats，不能仅仅核对 params/。
- 官方 README：attention_impl=xla 支持通常的 CUDA GPU；xla_cudnn 需要 Hopper，**GPU0 若是 4080S，不改成 xla_cudnn**；extra_frames_first=false、train_config_name=train_robodojo_real、action_type=ee、env_cfg_type=arx_x5。部署文件默认 SIM/true，必须以隔离的 runtime copy 覆盖，记录真实模型装载回执。

## B. GPU0 资格与 P0-0 现场环境审计

“4080S 32GB GPU0”需实测：标准 RTX 4080 SUPER 通常是16GB VRAM；32GB 可能是主机 RAM 或其他实际 GPU。以 driver 读取的型号、UUID、total VRAM 为准。不得据此预先宣称该卡能装下 12.44GB 权重和12图激活。

在指定主机由拥有执行权限的 owner 进入独立目录，先保存现场快照（以下命令供现场执行，**未在本次会话执行**）：

~~~bash
set -euo pipefail
WORK=/home/liuyu/ISSAC-GR00T/work/sla-memory-frozen-transfer-v1
mkdir -p "$WORK/logs" "$WORK/evidence" "$WORK/vendor"
{
  date -Is
  hostname -f
  nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv || true
  free -h
  df -h "$WORK"
} | tee "$WORK/evidence/P0_GPU0_HOST_PREFLIGHT.txt"
~~~

过门：GPU0 的真实 UUID、显存、在用进程和 owner 使用范围记录；没有干扰其他进程、没有未知显存占用；模型文件、HF cache、源码、局部源数据与日志的空间预算有余量；如有现役进程使用 GPU0，停止，不 kill/reset/抢显存，另行协调。对 16GB GPU 的 OOM 应写 RESOURCE_BLOCKED，不自动抢占 H200或改权重量化。应记录 driver、CUDA/JAX版本、系统 RAM、实际 GPU 电源及显存峰值，而不只记录 nvidia-smi 单一瞬时截图。

## C. P0-1：只部署 pinned source 与原模型，先不开始模型 forward

以下在独立 work/vendor 下，不能写现役 repo checkout 或已存在的训练 capsule：

~~~bash
WORK=/home/liuyu/ISSAC-GR00T/work/sla-memory-frozen-transfer-v1
XPL="$WORK/vendor/XPolicyLab"
UPSTREAM_SHA=c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454
if [ ! -e "$XPL/.git" ]; then
  git clone https://github.com/thriving191228/XPolicyLab "$XPL"
fi
git -C "$XPL" fetch origin "$UPSTREAM_SHA"
git -C "$XPL" checkout --detach "$UPSTREAM_SHA"
test "$(git -C "$XPL" rev-parse HEAD)" = "$UPSTREAM_SHA"
cd "$XPL/policy/SLA_Memory"
bash install.sh
bash download_checkpoints.sh robodojo_real
~~~

下载脚本是官方明确的真实 checkpoint 分支，不是 training（SLA_base）分支；已经 pin 模型 revision 7f6880…。上游下载会写其隔离仓库下 checkpoints/SLA_Memory_robodojo_real_ckpt。若已有独立只读缓存，优先审计并复用，不重复下载。不要重算官方 norm_stats 或拿旧 GR00T stats 替代。先完成完整哈希（包括 params、assets/norm stats、tokenizer），比对 HF resolved revision 和预期文件结构；保存 sha256 清单和总文件字节数，排除 HF 下载辅助 .cache 文件；**本地校验前不可声称通过**。

源域观测数据另 pin **RoboDojo-Benchmark/Robodojo dataset 的实际 commit SHA**，目录预计为 data/RoboDojo_real/pack_and_pour_fruit/arx_x5/data；先列出服务端该前缀实际文件、大小、许可和下载总量，选定 episode 0/1 并记录来源，不要运行 download_robodojo_data.sh real 下载整个 Real 数据集。若单文件包含多 episode，仅下载必要源文件。找不到真正的 ARX X5 三相机、双臂原生源样本，则 Gate 0 无法通过；debug 合成输入只证明 wiring 而非 O0 source-positive。

## D. P0-2：GPU0 O0 冷启动与诊断调用

固定 runtime copy，不能更改权重、训练数据或动作语义：

~~~yaml
policy_name: SLA_Memory
train_config_name: train_robodojo_real
extra_frames_first: false
attention_impl: xla
env_cfg_type: arx_x5
action_type: ee
ckpt_name: checkpoints/SLA_Memory_robodojo_real_ckpt
gpu_id: 0
eval_batch: false
~~~

上面的字段为“执行时必须达到的有效配置”，其 checkpoint resolver 路径须依 XPolicyLab model.py/deploy 流程验证；不能把它误当已运行命令。JAX 只使用设备0，使用 isolation 的 HF_HOME/JAX 编译缓存；起步为 batch=1、单独一个服务、只允许真实 O0 的 12 图和 16D 双臂状态，不通过降低输入图数解决 OOM。cold compile 与 steady inference 分别计时。监控 nvidia-smi 峰值，不通过改变图片分辨率/减少 history/量化权重伪装相同模型结果。

可用以下环境保护（由现场根据已使用服务配置调节）：

~~~bash
WORK=/home/liuyu/ISSAC-GR00T/work/sla-memory-frozen-transfer-v1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$WORK/hf-cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu --format=csv -lms 500 \
  > "$WORK/logs/P0_GPU0_VRAM_MONITOR.csv" &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT
# 在此运行已提交并审计的 O0 source-forward 入口；严禁直接用旧 sim 配置。
~~~

控制预算：最多先选择2个完整、独立、真实 ARX X5 的 pack_and_pour_fruit 源 episode；每条在预注册的早期接近、中期接触/抓持、后期放置各取1个查询点，至多6个主要真实前向＋最多2个重复性/reset诊断调用。每条 episode 从首帧逐帧更新原30Hz history，再仅在选定时刻请求 chunk。不同 episode 前 reset，并验证第一帧不会泄漏前一条 episode 的历史。源 HDF5 是示范动作，不是策略闭环执行；与 expert 比较只用于诊断。

必须逐次留存：真实输入 camera source、shape/rgb顺序、尺寸/图片哈希、12张图索引及目标秒差、三路mask、prompt、双臂 state、source action index/epoch、模型输出实际 shape/时间索引与两臂位姿/夹爪；JAX 模型与 norm stats load 信息、每次 forward 真实 GPU UUID、cold/steady latency、主存峰值、VRAM峰值、运行报错、源码/数据/权重 SHA、实际 optimizer updates=0、MuJoCo ticks=0。

建议 Gate0 判定拆分：PASS_LOAD(完整加载)、PASS_FORWARD(真实12图与16D输入产出有限动作)、PASS_SOURCE_CONTRACT(原始双臂/时戳/历史/逆转换合法且事件有意义) 三个字段。三个均通过后才允许 P1；不能以单次虚拟 prompt 返回有限数组等同于“复现官方任务成功率”。如果 O0 原生输出无意义，先诊断 source 准备与模型路径，不允许宣布 Go2 迁移失败。OOM/数据缺失分别报 RESOURCE_BLOCKED/DATA_BLOCKED，仍保持 Gate0 NOT_PASS。

## E. P1：先建立时间保真消融，后建立少一侧相机与手臂消融

实验 profile 和统计口径先封存：
- O0 原生：30Hz 三当前 RGB +9历史 +双臂，冻结 checkpoint；
- O1a 源域同一窗口在原30Hz下，去掉未选腕/手臂的可辨信息，使用**预先固定**的占位/复制 profile（原模型三路mask仍真，结果是 OOD，不能称为原生mask支持），左/右臂选择与指令条件全部记录；
- O1b 同一源域的20Hz观测，以真实时戳重采样生成 t-8…t-1 历史，和**未修正的每30帧索引方案**做成对对照，诊断12秒偏差对输出的影响。不得在 O0 的30Hz源输入上改缓冲算法后仍叫 O0。
- O2 Go2真实 Body/Wrist，相机标定、原生20Hz时戳、实际单臂状态；head/body 与手腕的几何差异、第三相机替代及虚拟臂 profile 明确标识。目标 observation 只能来自可部署传感器；仿真真值仅用于评估。

主分析：O0/O1成对动作差、selected arm EEF空间偏差、gripper事件时间差、相机敏感性与模型是否产生可解释目标行为。只要 O1/O2输入不可自洽，或者同一指令中物体与腕角色不对应，就记 OBS_CONTRACT_BLOCKED，不允许继续做目标WBC，也不凭一两个“动作看着正常”的样本过门。O1/O2不构成官方原生真机成功率复现。

## F. P2：两套坐标、一套安全门、两种时间网格（零 MuJoCo）

- 官方 real 模型 inverse 后输出各自 arm base 绝对16D，不加双臂中点；用同一时刻真实 Go2 的 world←arm_base 及固定 EEF-tool/TCP 标定生成 world←TCP；姿态使用 wxyz vs xyzw 与矩阵/rot6d约定，先 round-trip 再生成未来参考。
- 分离“确定性物理单位转换”和“会影响任务行为的 reference compiler”。不能代改目标、夹爪接触事件或补充 Agent 规划。保存 source16、arm-local16、world SE(3)、EEF10、H40、SafetyGate结果和真正安装结果的哈希。
- 源30Hz默认50点，target H40@50ms；如 O0 确认为 t0=0 且50点，则 H40 index0..32在源覆盖内、index33..39（7点）不在，需选择**一个事先注册的带标签 suffix 策略**（如最后合法参考显式terminal hold），单列17.5%的尾部与可能影响WBC的轨迹预览；不能改成50@20Hz或默默拉伸时间。若原执行协议不允许编译后缀，判 ACTION_TIME_ABI_BLOCKED，另立新协议，而非直接绕过。
- 首点 achieved TCP、上一实际 P10 消费指令、下一请求的速度/加速度、40mm/s夹爪接缝、空间弧长预览与STOP语义全部以冻结实际代码验证；与 expert reference 使用同一个 Gate。不能修改阈值、不加平滑、拒绝保留证据。
- 预注册16个 TRAIN/offline 状态窗口，按接近/合爪/提起/搬运/松爪/终止分层；输出每种拒绝的 observed、threshold、首个 failure stage、发生频数与相对 raw 程度。若16个都无法合法准入，则不进入物理。

## G. P3：条件性的冻结 WBC MuJoCo Pilot（本次不自动启动）

P0-P2 过门后，重新核对真实执行代码/权重/原 SafetyGate 与 world owner，以新run_id隔离。最多2次接口技术卡＋4条相同场景下 Teacher→WBC＋4条 Frozen SLA→WBC，共10个原生请求；原 PR #428 预算暂估367,030 ticks，现场以真正冻结任务卡重算。严格保留 2ms physics、原 H40/P10/TTL/40mm/s门和接触STOP；第一次只做稳定站立，不能自动扩展移动底座、加 Agent、训练适配器或访问 DEV/TEST。

必须确认 F1/F2 的任务、物体、语言与官方 pack_and_pour_fruit 能否构成源技能匹配；若无，单列 OOD task，不能拿目标StackCube 0%解释为官方已经学到的抓放技能根本失效。全部请求保留 init seed/模型请求数/物理ticks/STOP receipts/任务阶段成功/原参考→安装→achieved EEF误差和物体最终结果。

## H. 核心停止条件与失败分类

- GPU0未知PID占用、额外GPU被选中、无足够磁盘/主存：RESOURCE_BLOCKED（不 kill 他人进程）。
- 模型下载不完整、SHA或norm/tokenizer不一致：MODEL_ID_BLOCKED；拒绝继续。
- 真实源域样本缺失或只有合成debug：SOURCE_O0_DATA_BLOCKED；Gate0 NOT_PASS。
- 12图原生真实 forward OOM/NaN/config错误：FORWARD_BLOCKED；记录显存峰值，不能改模型后把原实验算 PASS。
- 缺视角/单臂 OOD失配：OBS_CONTRACT_BLOCKED；转入另立 embodiment adapter。
- arm/world/TCP/time/H40/gripper不合法：ACTION_ABI_BLOCKED；离线结束，不进行 WBC。
- 任何安全 STOP、非法碰撞、STOP后积分、原阈值超限：原停止，保存拒绝，无隐式 retry；正式资格样本以预先注册的独立 run_id 计数。
- 任何阶段无证据保持 NOT_RUN/BLOCKED；pilot成功只支持某一组 source-matched技能在声明接口下的迁移，不支持零样本通用性、连续实时硬件或多任务泛化。

## I. 当次报告与下一项决策

必须交付并一一对应: P0_GPU0_HOST_PREFLIGHT, SOURCE_AND_WEIGHT_MANIFEST.actual, P0_REAL_GPU0_O0_REPORT, P1_OBSERVATION_ABLATIONS, P2_EEF_H40_SAFETYGATE, P3_NATIVE_PAIRED_PILOT, FINAL_TRANSFER_DECISION；每份同时MD/JSON或结构化账本，含路径/sha/设备uuid/代码版本/真实运行次数/失效类别/实际消耗。已经完成的 P0_STATIC_PREFLIGHT 与 manifest.preflight 只作为**前置事实记录**，不可改名为 Gate0 PASS。

**当前下一步唯一能推进判断的实绩：**先在现场确认 GPU0 设备/占用及 owner → 下载指定 revision 的19文件并回算SHA → 获取 pin 住的两条原始 ARX X5 三相机+双臂示范 → 12图 O0 cold forward → Gate0 三分项报告。GPU0资源可用仅表示选择这块卡；不得因可用GPU就跳过对照或直接 P3。
