# PI0292 — ARX5 数据补齐、原生 GR00T 基线与受控闭环执行计划 V1

状态：USER_REQUESTED_BOUNDED_SCIENCE_GO_FOR_EXISTING_LOCAL_PC_OWNER。发布不等于已远程启动。
日期：2026-10-13，Asia/Singapore。现役分支：`pi0267/executor-decoupled-recovery`。
审阅 HEAD：`e3cdb22a1802fcae1d97e670487d5bc60b5e5c0b`。
科学记录：Issue #11；已放行的 Corelab 整理清理独立记录在 #47。

## 0. PI 决策与本轮科学问题

接受 PI0291 为 **ARX5_DATA_PARTIAL_NATIVE_CPU_AND_HF_VERIFIED**：场景、56 条实际 TRAIN、原生 CPU 数据处理和不可变私有交付成立；原 64/16 队列未完成，Trainer 未资格。不要重新采这 56 条，不重跑原 8 条 pilot，不把 PARTIAL 改成历史成功。

本轮要回答：在同一固定底座 ARX5 场景、统一 EEF10 接口下，直接从 NVIDIA 官方预训练初始化、采用原生 flow-matching 训练，能否学会教师抓放，并在未参与训练的新初态上闭环完成任务？

执行链为：**稳定工作路径和队列 → 补齐未尝试数据 → 原生 Trainer 适配 → 真实模型工程 smoke → 单一 B0 配方训练 → 离线归因 → 有限在线闭环对照 → 固定版本交付**。各门通过即继续，不每阶段重新请求 GO。工程失败可以在下文预算内修复；科学质量不通过不能换 seed、换 checkpoint、加 loss 或自动加步数冲分。

这是新 PI0292 的条件式有界授权。旧 PI0291 的禁止新增 physics/model/PBS 不再限制本协议明确允许的新任务，但旧时钟、账本、原件、FAIL 和准入结论均不改写。**不新增任何文件删除授权，不运行/修改师兄 WBC，不做移动 Go2、真机、DAgger、额外任务、视觉骨干解冻或超参数搜索。**

## 1. 已有证据及不可变依赖

- PI0291 科学原始源：`57a02b4923e5001df5fd4b3b5958c372b9cff541`，121 个 sealed source hashes；原 run `20261013T023717-localpc-arx5`。
- 固定底座真实六轴 ARX5、有限力矩、双指接触、MuJoCo 3.2.7、Mesa CPU 渲染；教师 pilot 8/8、新方向资格 4/4。
- 原正式队列 56 TRAIN 首次尝试全部物理成功/审计/准入；`T291056..T291063` 与 `D291000..D291015` 未尝试。原 92 行账本含 4 commissioning、8 pilot 和 80 个正式计划条目。
- 33,656 TRAIN 完整 H40 窗口；全窗口数值 codec、真实跨 episode/阶段 RGB batch、0/2-worker 对照已验收。不要声称所有窗口都完整运行过视觉网络：原轮模型执行为零。
- 原 TRAIN-only statistics SHA256：`43469f3818119c075774d91c5b08bbc2a46a0323a572a8e58614daa48155fd2d`。原生产者 native manifest SHA256：`1d2dd42c681197493570128aa09f60c864e3cc1be80f9a3fa1d6af3b175189d7`。
- 私有 HF dataset：`leoyuer/pi0291-arx5-fixed-base`；完整 revision `c25eeadf9be0aa7d46bb6f7b65e260cbf3a305ea`；prefix `PI0291/20261013T023717-localpc-arx5/v1-57a02b49`。81 文件、3,589,423,752 B，全集合/hash 核验、Corelab 完整样本/native 回读、Hopper 小 READY 已通过。
- 原生 NVIDIA GR00T 源码固定为 `51d4c89f72fda44cbf77285c6a8114b52676b8a1`。目标基模为 `nvidia/GR00T-N1.7-3B`，**模型权重 revision 仍需现场核验并封存，不把名字当完整身份**。
- 原队列 child exit120，恢复后确认已退出，但根因未证明。Python 文档说明解释器退出清理/标准流 flush 错误可能返回120；这仅是排查线索，不是本次 SSH 中断因果已经确认。

local-pc 的 `/home/liuyu/Documents/.../PI0291_DATA_CLOSEOUT_REPORT.md` 不在本 ChatGPT 环境；本计划依据 GitHub 对应 FINAL_REPORT 和实际收据，不声称读取了那份本地文件。

## 2. 单一 owner、清理协调和新运行目录

唯一既有 local-pc Codex 负责调度、seal、准入、Git 提交及累计资源；允许子任务并行写 adapter/离线测试/只读审计，但同一时刻只有一个物理执行 owner。不得通过 issue mention 再开云端 owner，不重启已终止的 PI0291 dispatcher。

用户已另行放行 Corelab 整理清理。本轮不重复审批或替它扩大删除范围。S0 与维护执行者确认：
1. 读取实际最新目录迁移/删除清单及当前进程；保护 PI0291 原数据、两份机器人模型依赖、固定官方 processor/cache、科学代码、Python 环境、HF 凭据和本轮新目录。对这些路径建立具体的依赖保留清单和活动 lease。
2. 新的人类入口可为 `~/Projects/GR00T-WBC/`，但必须由现场路径表确认；共享 NFS、4080S 节点本地 SSD、Hopper scratch 分别定位。不要把 AMD 的 SSD 断链当丢数据，也不要为目录统一搬动所有 raw 数据。
3. 如维护正在改变某个依赖，只暂停该依赖的读写；在确认稳定的私有运行快照上继续独立工作，不以“所有历史垃圾尚未删除”为科学任务永久阻断。
4. 切换目录在作业边界完成；记录迁移前后实际解释器、Git/worktree、editable install、asset resolution、stats/hash。不要直接移动 Home 中的 venv 后假定可用；原路径仍被进程使用则由维护保留。
5. 科学执行路径稳定后不再移动它们。资产/源码可复制到只读 capsule 并逐文件验证；凭据只引用现有机制、不进 Git/HF。

输出 `WORKSPACE_HANDSHAKE.json`、`ACTIVE_DEPENDENCIES.json`、`HOST_PATH_BINDINGS.private.json`（后者仅私有）、`ADOPTION.json`。全部新原始结果进 `<verified-science-root>/PI0292/<run_uuid>/`，GitHub 仅代码和脱敏小收据。

## 3. 新预算和连续执行约束

T0 是 owner 首次采用本固定 release 的真实时间。持久化 run UUID/T0/截止；全轮 deadline=T0+24小时，含排队和网络等待。最后60分钟只收尾，不能因重连/新子进程/重发任务重置预算。未在窗口内获得计算分配就交 BLOCKED/PARTIAL，不借别人的作业执行。

| 项目 | 累计上限 |
|---|---:|
| 环境连续性教师 canary | 2 次 starts |
| 原未尝试数据的新授权首次执行 | 24 次：16 DEV 先，8 TRAIN 后 |
| 真实 relay 工程物理资格 | 2 次 starts |
| 微调模型 TRAIN 布局闭环 canary | 2 次 starts |
| 新确认集：教师 / B0 / B1 | 每臂8次，共24次 starts |
| 全轮新物理 | **54 starts / 864,000 integrations / 34,614 对物理观测** |
| 单物理 episode | 最多32 sim秒、dt=0.002秒、16,000积分、641对观测 |
| 零积分相机/回放诊断 | 最多64对，单独计数 |
| 新 TRAIN 主训练 | **1个配方，最多4096 optimizer steps，32样本/step** |
| 隔离模型工程 smoke | 最多16个实际 optimizer updates；不成为主训练母本 |
| 离线 action-generation | ≤1152 次（B0、S2048、S4096，各≤192窗口×2噪声） |
| 在线神经 action-generation | ≤1440 次（18条模型episode×80查询） |
| 其他模型 action-generation smoke | ≤32 次；总生成查询≤2624 |
| Hopper GPU | 1张 H200；实际累计 allocated GPU-time≤8 GPU-hours，含空闲等待 |
| PBS submissions | ≤3，最多1个自有活动神经作业；提交不是开始/完成证据 |
| Corelab | CPU软件渲染；本轮GPU=0；科学CPU≤96 core-hours，同时≤8核 |
| Hopper CPU / local-PC CPU | 各≤64 / ≤8 core-hours；与维护旧账分开 |
| 新增峰值空间 | Corelab≤40 GiB、Hopper≤180 GiB、local-PC≤8 GiB |

所有失败、早停、工程物理尝试也消耗所属配额，不相互挪用。不额外分配“免费物理预检积分”。新数据量包括复制/下载/压缩暂存的并存峰值；共享 inode 不重复算占用，但不当独立备份。每阶段容量预检：预测剩余峰值+至少10 GiB（Corelab）/20 GiB（Hopper）可用余量，且满足可核实个人 quota。quota 无法读取则记录 UNKNOWN、做有限实际写入检验，禁止声称 df 就是个人额度。

GPU优先预算：工程smoke≤0.5h、主训练≤4.5h、离线与闭环≤3h，合计8h。按实际吞吐预检；预算不足时不压缩样本曝光或偷偷更换候选定义。记录 CPU measured 与 conservative upper，无法量测不能写0。

## 4. S0 — 修复任务可持续执行能力，而非猜测网络故障

先核对 PI0291 已终止与全部原件。新 runner 的 stdout/stderr 直接写远端持久日志文件；采集不依赖 SSH 观察连接的 stdout 管道。使用既有许可的用户级会话/任务机制，不安装 daemon、不提权、不改 VPN/防火墙/隧道。

先用无模型、无物理的假任务完成最多两次工程验证：断开观察连接后继续记账；重新连接只读原 session/lease/state；重复启动被锁拒绝；过期响应、重复 ACK、预算触顶和磁盘写入错误能留有明确失败记录。执行 owner 退出/日志 flush 报错和物理任务成功必须分开记。

工程预算守卫与科学源解耦：旧 `resource_guard.py` 写死 PI0291 截止与 ID，不能把新动作灌入旧账本。建立新 PI0292 dispatcher/guard 和 source capsule，允许改 authority/输出路径/计量/日志/外层 adapter；engine/scene/teacher/score/相机/动作语义的数值保持原冻结版本。记录完整 diff 和实际新运行 SHA，不谎称全部121文件仍逐字相同。

旧 guard 每2秒递归枚举整个run的逻辑占用，可能随文件增多拖慢；允许在新守卫使用保守预留、逐产物增量计量与阶段完整核对代替高频全树扫描，必须包含未知写入上界并证实不会漏预算。此优化不改变任何 physics tick、teacher reference 或数据内容。

输出 `RUNNER_RELIABILITY_TEST.json`、`SCIENCE_SOURCE_EQUIVALENCE.json`、新账本和准确 resume/status 命令。只读问题未定因可以继续，但耐断连资格不通过不得启动长队列。

## 5. S1 — 补24条未尝试数据，保留旧实验事实

### 5.1 身份与顺序

使用新 attempt UID，例如 `PI0292/<run_uuid>/SUP-D291000`；记录 `origin_plan=PI0291`、原 logical ID、完整原 card/seed/group/hash 与本次授权。仅补 `D291000..D291015` 和 `T291056..T291063`。原账本对应条目仍是原窗口内 NOT_ATTEMPTED，新 successor manifest 显式关联两轮，不能回填覆盖旧数据。

原 seed/card 从私有 sealed 原件或固定HF原件读取，以Python精确uint64比较；公开 seed 必须十进制字符串，不从可能被JavaScript舍入的显示值重建。若现场发现名单内某ID已被另一任务启动，则不重复执行，停止该ID并报告身份冲突。

先运行两条新身份的环境连续性 canary：复用原 `C29103` 与 `P29104` 布局，只验证迁移/runner以后真实动力学和相机没有回退，均排除TRAIN/DEV。必须2/2完整物理成功、原审计通过，且同初态独立FK/相机/力矩约束成立。失败不另补seed或调teacher；冻结源的环境问题可做零积分定位，未恢复则停止扩采。

之后顺序固定为 **DEV16→TRAIN8**。这是预先声明的采集调度，不是根据DEV成绩挑训练数据；不许用DEV修改教师、几何、控制器、训练配方。相比原来的先TRAIN后DEV，本次优先确保验证集不会被时间耗尽挤掉。

### 5.2 数据准入

复用原完整32秒T1物理谓词、0.3秒双侧抓握、2cm抬升、10cm搬运、8cm撤离、28–30秒41点保持、全程安全与原图/时钟/指令审计。实际桌高0.44m、原TCP与夹爪极性不变。源frame/统计误差门不因失败放宽。

24条全部首次执行并封存结果后生成 **新ARX5 bank v2**：原56条成功TRAIN原封不动引用+新增准入TRAIN+新增准入DEV。TRAIN64计划中准入≥56、DEV16中准入≥14且全部计划已明确终态，才允许进入主训练；所有失败/拒绝/未尝试另列，不能靠删分母达到门槛。

原 PI0291 永远保留 PARTIAL。本轮可给 `ARX5_BANK_V2_QUALIFIED`，并注明跨两个授权窗口和版本来源。无有效DEV时可完成adapter CPU工程，但不启动本轮主训练/模型确认。

### 5.3 统计量与交付

先完成准入，再从新bank所有已准入TRAIN按原规则拟合**一次新版本**统计：每个合法state anchor一次，每个完整H40窗口/point等权。原56条统计/hash不能被覆盖；新旧统计差异和对应TRAIN ID完整保存。DEV/8个确认布局/commissioning不参与拟合。

641 observations / 640 committed commands / 601 complete H40，offset0..1.95s，prefix8，绝对EEF10落盘、官方只做一次相对转换。保留原语言全文、原像素，不用短task label、视频重编码或padding补命令。所有TRAIN完整窗口再次通过原官方数值误差门（XYZ最大分量误差≤1e-5m、SO3≤1e-4rad、gripper≤1e-5、独立SE3相对矩阵差≤1e-8）；DEV实际窗口独立报告preclip、inverse clip与误差，不能重拟合消除DEV越界，也不能把0窗写PASS。

私有HF以新prefix追加：保存原56依赖的固定revision引用、新24原件/失败、合成manifest、统计和native导出；不覆盖旧prefix。新机器消费时读取全部实际需要的原RGB与numeric源，不能只下载小NPZ或READY就说数据已在Hopper。全集合/hash验证，完整真实样本回读，READY最后写。

## 6. S2 — 正式训练 adapter；不重做已合格采集器

优先为现有 NPZ+PNG 数据实现一个薄的**训练专用**Dataset/Trainer适配层，复用固定官方 processor、collator、模型和原生loss。LeRobot标准导出不是本轮必做前置；禁止同时更换数据格式、图像压缩、统计方法和训练目标。

现有 `native_dataset.build_processor()` 会调用 CPU guard 并替换 `torch.nn.Module.__call__`。**训练进程不得导入/调用这条guard路径，也不能删除旧guard。** 在新模块构造参数等价的官方processor，CPU审计用独立子进程。旧native loader仍是数值和像素对照参考。

须实际实现/验证：
- `get_dataset_statistics()` 返回新bank冻结TRAIN统计，不触发自动重拟合；启动前后统计SHA一致。官方factory可能调用stats生成，采用显式adapter注入/子类入口，不暗改原upstream源码/全局monkeypatch绕过。
- 只有 `processed` 进入官方collator；episode/split/seed/teacher/stage等identity留在审计侧。保持原语言、相机次序、absolute→relative一次、10有效维/132padding。
- 确认传入模型的是官方collator实际产生的schema；原实现没有state_mask就不伪造“官方已有”。动作mask只在10有效维为1。
- TRAIN样本为全体合法H40窗口的固定seed循环随机排列，无额外事件重采样；每周期覆盖全部，不使用旧factory默认0.1采样率悄悄降数据。worker任务分片、sampler cursor、seed/reset_seed、断点RNG恢复可复现。
- 固定world_size=1；测试workers=0/2样本与张量等价。未测试多GPU就不称DDP已资格。通过参数、数据地址与split的负例防止DEV进优化器。
- 官方源码factory目前断言 `eval_strategy=='no'`，与教程部分文字存在差异；按已核对源码保留训练eval_strategy=no，独立离线DEV评估，不简单打开一个不兼容的flag。
- 新adapter与原CPU参考对同一批TRAIN/DEV真实raw输出逐字段/数值对照，验证像素、state、标签、normalization、inverse decode一致。默认保持已验收的无图像随机增强、state_dropout=0、min/max和clip设置；这些是本基线明确保留的项目条件，不称完全官方默认。

输出 `TRAINER_ADAPTER_CPU_QUALIFICATION.json`、精确源码/解释器/实际可执行命令、配置展开值和测试结果。仅代码存在、pytest fixture PASS或CPU dummy张量不等于模型训练资格。

## 7. S3 — 基模身份与Hopper真实smoke

仅Hopper合法新PBS allocation内加载/执行神经模型。沿用已安装环境、既有认证和作业模板，严格使用allocation分配的GPU；不在login节点、Corelab或他人allocation运行，不因queue慢偷换场所。数据和adapter先合格再申请长GPU时段。

官方基模：`nvidia/GR00T-N1.7-3B`。现场读取现有缓存的repo/full commit、完整shard index、两份safetensors及config/processor/tokenizer provenance，逐文件SHA与固定官方来源对应。候选曾声明6,910,361,856权重bytes及index SHA `407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59`，这些是核对线索，不替代当前完整验证。若本地候选不一致，可以在本轮空间/流量预算内从确认的同一官方基模固定revision取得缺失文件；不能替成DROID/LIBERO或历史AGG/ACTION模型。

固定 `skip_weight_loading=False`，记录missing/unexpected keys与各模块来源。NEW_EMBODIMENT=10的初始行同样记录哈希/初始化来源；不擅自换成别的已训练本体tag。不声称VLM冻结就证明预训练行为全保留。

隔离smoke使用新bank TRAIN，不读DEV优化。最多16个总实际optimizer updates，计划第一轮为两条匹配路径：连续4步，与2步保存→重建进程→恢复再2步（共8个实际updates）。主训练启动前重新加载 pristine 官方模型，不继承smoke权重。

smoke必须证明：官方native loss为finite；有效action mask参与归约；trainable action/projector梯度和至少一组权重确实变化；冻结VLM权重未变；真实有效batch和optimizer/scheduler/RNG/sampler状态保存/恢复；同顺序样本及噪声下恢复与连续路径在预先冻结的数值容差内一致。样本顺序、噪声、RNG、global step、scheduler和sampler cursor必须严格一致；数值固定使用权重/optimizer数组allclose(atol=1e-7, rtol=1e-3)、loss allclose(atol=1e-5, rtol=5e-3)，非有限值必拒绝。记录最大差异与未通过字段；不根据观测反向放宽。工程测试不拿loss下降或任务成功做准入。

未通过时在剩余smoke预算内最多一次同类工程修复；保留原失败。不重装整套CUDA/依赖、不动旧CPU guard。输出 `PRETRAINED_IDENTITY.json`、`MODEL_SMOKE.json`、`RESUME_EQUIVALENCE.json`；之后才给 `TRAINER_MODEL_INTERFACE_QUALIFIED`。

## 8. S4 — 一条原生 B0 配方，停止历史小任务补丁循环

命名：**B0**=官方pristine权重+本轮相同已冻结processor/统计/route；**B1**=下列单一配方完成4096步的结果。B0并非旧AGG4096或baseline1000；它不受训练的零样本成功保证。

| 配方项 | 本轮固定值 |
|---|---|
| 初始化 | S3验证的官方N1.7-3B同一完整revision |
| loss | 官方masked flow-matching；不加XYZ/SO3/速度/夹爪辅助loss |
| tune_llm / tune_visual | false / false |
| tune_projector / tune_diffusion_model | true / true，按官方实现；记录实际trainable参数集合 |
| 数据 | 新bank已准入TRAIN，单任务单来源；DEV不优化 |
| state/action | 10有效维、132padding、row-rot6d relative EEF＋absolute gripper |
| normalization / augmentation | S1新TRAIN统计；维持CPU已验收min/max、无随机图像增强、state_dropout=0 |
| precision / optimizer | bf16；AdamW，lr=1e-4，betas=(0.9,0.999)，eps=1e-8，wd=1e-5，grad norm clip=1 |
| schedule | cosine，warmup ratio=0.05，主训练总长4096 |
| 实际每optimizer step样本 | 32 |
| 优先microbatch / accumulation / GPU数 | 2 / 16 / 1 |
| workers / train seed | 2 / 29200 |
| 快照 | 完整S2048、S4096；权重、optimizer、scheduler、RNG、processor、stats、sampler、配置与来源齐全 |
| 正式训练次数 | 1；没有依据DEV自动续训或改配方 |

**批量语义必须单独验证。** 固定官方 `FinetuneConfig.global_batch_size` 是梯度累计前的跨GPU batch，不是累计后batch。因此在它的入口上2×16×1=32，不得把global_batch_size写32再accumulation16而实际变512。OOM只能在主训练第一步前从2×16改为1×32并保留有效32，记录选择；主训练中不按结果改批量/学习率。

4096×32=131,072次样本曝光；如果64TRAIN全准入则有38,464窗口，约3.41个窗口池等量曝光。这不是3.41个独立任务，也不是保证收敛。日志单列独立episodes、anchors、阶段/闭爪/开爪事件曝光、原生loss、梯度、实际step及速度。事件元数据只用于账本，不能混入模型。

不改原生loss、不追加history/目标坐标/teacher stage、不自定后处理平滑。保存所有已产生的S2048/S4096，不设置会自动删除唯一完整checkpoint的轮转。必要中断时保存新本轮完整恢复点；同一训练链可在剩余PBS/GPU/24h预算内恢复，不重置已用steps。到截止未到4096，B1=NOT_AVAILABLE；不得临时把表现最好的中间checkpoint改叫最终候选。

## 9. S5 — 离线评价：先看是否学会，再决定能否接近闭环

S4前封存probe清单和脚本。TRAIN从准入ID按数字排序均匀确定16个episode，DEV使用全部准入（最多16）。每条取6个阶段锚点：0、首次close、lift、carry、release、hold对应的合法首anchor；事件只决定评价采样，不成为网络输入。去重规则与0..600范围封存；实际不足则如实少于192，不补一个“更好”窗口。固定两个噪声bank29201/29202，并按episode/anchor派生噪声，三个profile使用相同噪声哈希。

B0、S2048、S4096每个profile最多192×2=384次原生生成，共≤1152。使用固定官方4步采样、native decode和本轮同一stats，保存解码前后轨迹。S2048是诊断快照，不参与选优；不因DEV曲线选择它覆盖预注册S4096。

评价分别报告：H40与真正执行prefix8的XYZ误差（逐点欧氏，median/p95）、SO(3)弧度、gripper MAE与闭/开事件时序、首点跳变、相邻速度/角速度、SafetyGate各原因计数、TRAIN/DEV分层和每episode宏平均。全REJECT行仍保留误差，不能只统计幸存轨迹。官方clip、模型出界与数据归一化误差分开报告。

先将相同专家committed H40经相同decode/时间绑定/SafetyGate做阳性对照。若专家参考都不满足本次安装契约，先定位端到端契约，不让模型训练替错误背锅。不可把SafetyGate是否接近目标混成同一个量。

本轮不把某个跨任务固定MSE当“已经会抓取”的证明。TRAIN/DEV均差、TRAIN好DEV差、位姿准但gripper差，分别写明确归因。极差结果同样保留为有效负结论，禁止自动进入调loss循环。

## 10. S6 — 在线闭环是新反馈查询，不是播放预生成动作

### 10.1 Relay工程资格

Corelab负责MuJoCo；Hopper在有效PBS内驻留模型；local-pc通过两条既有SSH连接中继。采用显式 **NON_REALTIME_PAUSE_ON_OBSERVATION**：物理暂停等待推理，收到合格prefix8后推进0.4sim秒，再采新真实obs重新查询。网络等待不算仿真时间，但算GPU分配/墙钟预算。

每个packet包含run/episode/query/ref UUID、physics tick、snapshot/hash、实际TCP anchor、model/config/processor/stats身份及ACK。传输仅两路原RGB、原语言、actual EEF/gripper及合法时钟；物体真值、goal坐标、teacher stage不得出现在学生请求。结果绑定该次观察anchor只解码一次；只执行前8点，其余32点不保留为已提交动作。不能使用旧预测回放冒充新feedback。

先做CPU假端点协议测试（无需模型/physics）：stale/duplicate/reorder/CRC损坏/超时均拒绝且不推进physics。再跑两条已知TRAIN布局上的真实物理teacher-reference relay资格，教师由Corelab现有反馈算法产生，经相同消息/安装接口转送，**明确不是GR00T成功**；2/2原评分通过才允许模型闭环。任何控制命令/ACK重复不重复推进。

资格前从实际队列模板确认 compute节点可经既有合法路由访问，不开外部端口、server tunnel或修改集群网络。每query墙钟超时60s，每episode墙钟上限15min；超时即暂停/封存该attempt并终止，不用教师接管后仍计模型成功。无连接则交OFFLINE_ONLY，不把模型搬到Corelab。

### 10.2 模型执行和冻结约束

模型输出按原SafetyGate契约验证有限数值、SO3、frame/TCP/time、速度和首点连续性；同时使用原engine关节/力矩/穿透/速度安全约束。封存具体Gate配置与代码；不降阈值、不投影轨迹、不平滑/重定时、不让教师代选目标。原生inverse clipping是已声明处理，不用额外clip掩盖非法输出。Gate拒绝就在本attempt记REJECT，不能继续采噪声直到接受。

合法EEF参考交同一个独立ARX5 IK/有限关节driver；它不是原WBC。不能更改物体qpos、weld、隐藏外力，不以teacher phase控制模型抓放，不在首成功后停止32秒保持检查。

B1先在两条已进入TRAIN的布局（T291000/001）上跑新attempt canary，记录为开发条件，不叫未见泛化。两条都无真实安全违规、且至少1/2 persistent success，才进入新确认集；否则封存离线与canary归因，不增训练。B0差不否决工程资格，也不要求靠teacher帮助B0达标。任一模型出现真实安全违规立即停止该模型臂剩余尝试；Gate在执行前正确拒绝属于受保护的模型失败，不等于发生物理安全违规。所有未尝试case仍保留在计划分母。

### 10.3 八布局三臂确认

`CONFIRM_LAYOUTS.json` 随本协议发布，8个新布局，依旧窄分布forward T1；均不加入TRAIN/DEV或统计。先完成独立静态几何/初态合法性检查；与已有完整card比对不存在精确初态重复。若有冲突或不可达，停下报告，不换seed生成更简单布局。

三臂：同一反馈教师、pristine B0、固定S4096的B1。每臂每布局只1次，初始场景、实际相机、driver、评分、32秒保持要求一致。教师8条先运行；若少于7条通过或出现任何真实安全违规，停止模型确认并报告环境/teacher范围问题，不筛掉失败case。模型case顺序固定，奇数case先B0、偶数case先B1，不按输出决定顺序。全部8仍在每臂分母，安全拒绝/运行失败分别记录；不可用Gate早停的视频伪装完整运行。

每臂输出8条逐case证据与总体成功数，teacher/B0/B1配对差异、Gate拒绝、物理违规、首失败阶段、generated→installed→achieved误差、完整原视频。n=8仅是探索性系统验证，不足以证明广泛泛化或统计显著。

最高探索性正状态：teacher=8/8、B1≥6/8 persistent success、B1无真实安全违规，且B1成功数高于B0。它只表示本轮窄分布、固定底座的有希望的适配结果；不是部署/跨本体/多任务资格。所有其他完整结果仍可返回 `B0_COMPARISON_EVALUATED`，明确负结果而不伪造成功。

## 11. 交付与状态机

阶段顺序：S0 →（S1与S2 CPU可并行）→ S3 → S4 → S5 → S6。维护工作不在科学预算内另开实验；资源/依赖不冲突时可并行，但迁移/删除与本轮持有的活动依赖串行。每个阶段先写机器可读gate，再让owner校验输入身份与剩余额度进入下一阶段。

必须有：
- `ADOPTION`、`WORKSPACE_HANDSHAKE`、`DEPENDENCY_LEASE`、`RUNNER_RELIABILITY_TEST`、`SCIENCE_SOURCE_EQUIVALENCE`。
- `SUPPLEMENT_ATTEMPT_LEDGER`、`BANK_V2_ADMISSION`、`BANK_V2_MANIFEST`、`STATISTICS_PROVENANCE`、`DATASET_CARD`、两台机器实际数据回读。
- `TRAINER_ADAPTER_CPU_QUALIFICATION`、`PRETRAINED_IDENTITY`、`MODEL_SMOKE`、`RESUME_EQUIVALENCE`、`TRAINING_RECIPE_SEAL`。
- `TRAINING_RECEIPT`、两个完整checkpoint身份/归档和`EXPOSURE_LEDGER`；所有未运行/缺失状态明确。
- `OPEN_LOOP_PROBE_SEAL`、三profile真实records/summary、`RELAY_QUALIFICATION`、canary和三臂确认逐case原视频/physics/request/ACK证据。
- `RESOURCE_LEDGER`、`FINAL_REPORT.md`、`RESUME.md`、`ARTIFACT_INDEX.json`；固定代码、权重、processor、stats、dataset revision、准确已执行命令和下一条可执行命令。

状态分开：WORKSPACE / SUPPLEMENT / DATASET / ADAPTER_CPU / MODEL_SMOKE / TRAIN / OPEN_LOOP / RELAY / CLOSED_LOOP / HF。
`PASS`必须指向实际收据，`BLOCKED`必须给具体缺项，`NOT_RUN`不能写成失败或通过。S5负结果、S6未过门不推翻S1的数据资产成果。

所有新大数据/完整checkpoint仅到已验证用户私有HF repo/prefix；新建必要private repo允许，公开发布不允许。保留完整resume状态、metadata和失败证据。量测backups与空间后再上传，READY-last，固定revision全集合/hash与实际样本回读。GitHub只小文件/代码，不输出凭据/私有路径/mesh。

最终#11发实际报告、report/run/source SHA和各阶段状态，普通fast-forward推送后逐文件回读。#47仅发依赖/迁移协调结果，不把维护回收字节写成科学成果。`selected=NONE; deployment=NONE`；内部B1仅是预注册评价候选。

## 12. 局部阻断、停止与下一轮边界

- 清理未完成：在已稳定且得到保留的依赖上继续；不能强删旧目录来满足新计划。
- 无DEV/数据门不通过：保留新旧全部原件，完成可做的CPU adapter，不主训练。
- Adapter/smoke失败：先修确证工程错误，不换模型/语义；预算耗尽交BLOCKED。
- 主训练不足4096：保留最后完整恢复点，完成已有离线诊断；B1确认不自动拿中间快照替代。
- 离线好闭环差：分别定位网络/安装/跟踪/任务转换，不自动加loss、DAgger、history或WBC补偿。
- Deadline/资源上限：停止新增任务，关闭本轮自己的模型会话/物理owner并核验实际终态；无权限或状态不明则记录UNKNOWN，不杀其他用户。

本轮结束后的候选研究方向是共享多指令/多任务、恢复覆盖、冻结WBC组合以及同高层权重跨执行器；**本协议不授权它们**。只有当前原生基线的实际归因结果，才能决定下一轮该改数据、条件化还是执行适配。原WBC的独立泛化证据是另一条主线，不由本轮固定底座成绩替代。

## 13. 固定来源

- [PI0291 canonical终局](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6136292691)
- [PI0291 FINAL_REPORT](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/e3cdb22a1802fcae1d97e670487d5bc60b5e5c0b/reports/PI0291/20261013T023717-localpc-arx5/FINAL_REPORT.md)
- [Trainer缺项](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/e3cdb22a1802fcae1d97e670487d5bc60b5e5c0b/reports/PI0291/20261013T023717-localpc-arx5/TRAINER_INTERFACE_READONLY_REVIEW.json)
- [原生数据源码](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/e3cdb22a1802fcae1d97e670487d5bc60b5e5c0b/scripts/pi0291_arx5_data)
- [官方固定FinetuneConfig](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/configs/finetune_config.py)
- [官方fixed factory](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/data/dataset/factory.py)
- [官方fixed微调指南](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/getting_started/finetune_new_embodiment.md)
- [Python sys.exit说明](https://docs.python.org/3/library/sys.html#sys.exit)
- [既有维护实际结果](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47#issuecomment-6130933143)

以上是制定依据；阈值、预算、训练配方和8布局三臂对照是本轮预注册设计，不是已取得的实验结果。
