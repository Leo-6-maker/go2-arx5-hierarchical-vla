# SLA ARX5-rich V6 · Codex执行手册

阅读 [计划](SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6_PLAN_20261126.md) 与 [机器登记](SLA_HOPPER_ARX5_SOURCE_EXPANSION_V6_REGISTRY_20261126.json)。本文件给出具体执行顺序、接口和交接，不是已经运行的脚本。复用现役JAX/SLA engine、数据reader、Orbax、Journal及PBS合法提交路径，普通工程修复自主进行。不要另造调度平台，不用整repo下载验证，不等待所有数据源才开工。

## 0. Hopper实际对账与接力（同时准备源数据）

1. 读最新#520增量、PBS实际job/终态、owner、资源授权；历史17:09回读564/1000不是实时。健康735204继续，已有2K/R0不重复。按现场可用的原授权命令只读核对qstat历史/现役、完整STEP1000/访问账本、存储quota、项目额度与截止。确认唯一owner。
2. 原1K完整→Core原评价；R0已登记从STEP0250的27工程步仅执行一次；原2K条件满足则恢复同一训练状态和bank剩余1000。无明确原V4续训证据不自动绕门；未运行时CPU准备与Core继续。
3. 复用既有有界接力服务/脚本，在合法站点部署一次性阶段状态：VERIFY_COMPLETE→HANDOFF_EVAL→ELIGIBLE_NEXT→SUBMIT_ONCE→RECONCILE_JOB_ID→ACTIVE→SEALED。每次提交锁定阶段、父完整SHA、dataset/bank/recipe、目标步数、实际owner及预算。UNKNOWN提交先按intent/已知scheduler历史独立查，不重复提交。回报实际jobID和独立读回；用户关闭客户端不应中断已获授权的服务器owner。若站点不允许常驻CPU服务，则用已有许可的作业收尾提交或有限后继，不申请空闲GPU伪装守候。旧Vanda暂停。每capsule≤6h、保存余量≥15min、每100更新/30min先到封存及端点，原安全文件空间、期限、额度保留。

## 1. 实际获取数据（不在AMD做重计算）

在原授权环境只读列出存储/网络、CPU/内存/实际GPU；优先复用现有下载、元数据、局部镜像、原文件hash。示意Python API（运行前查现场已安装版本，不在本次PR执行）：

```python
from huggingface_hub import HfApi, hf_hub_download, get_hf_file_metadata, hf_hub_url
api = HfApi()
info = api.dataset_info(repo_id, revision=chosen_ref)
rev = info.sha
# 根据实际目录meta、任务和episode/file映射产生有限 allowlist
# 先读取文件真实size与许可，再由已授权执行器下载单文件
meta = get_hf_file_metadata(hf_hub_url(repo_id, path, repo_type="dataset", revision=rev))
# reserve_download(meta.size, scratch, final, checkpoint_reserve)
path_on_disk = hf_hub_download(repo_id, filename=path, repo_type="dataset",
                               revision=rev, local_dir=authorized_staging)
# hash raw bytes, bind source revision + file path + episode identity
```

路径、token、repo revisions、函数reserve_download仅示意，按实际合法账号/已安装接口实现，不打印token、不自动接受OpenNeoData条款、不绕访问限制。优先5个源族各最多8条实际pilot，合计40；许可/语义/下载失败保留记录，改走其他READY源。先核精确文件size，pilot总在途暂存≤10GiB、Mix500新增初始≤40GiB（且服从更严实际配额）；下载/解码峰值、快照、两套完整状态和原存储门先预留。不要下载Table30全库、OpenNeoData整库、Ville的1TB+汇总或默认ALOHA RoboTwin。对已有Raw/HF缓存引用按hash去重，不自动删除任何旧文件。若实际预算不足，以更小许可小包先训练。

| 来源 | 首次具体读法 |
| --- | --- |
| RoboDojo | 先读现役200条实际manifest/source revision、任务和train/dev/test；列官方real/arx_x5实际目录与数据格式，仅增未见的真实训练episode。复用现役reader和已核实RGB顺序，检查新revision变动，源动作语义不升级Go2 COMMAND。 |
| Table30 | 复用#375的元数据/FK/PTS原件；先读取task_info.json/episode_meta/states.jsonl、真实wrist/global视频；旧两个可核任务drawer/coaster优先。只robot_id实证ARX5；v1与v2分别schema，不能把30任务全当ARX。许可须确认。按PLAN决定严格源或SOURCE_RECORDED_REFERENCE_WEAK_PROVENANCE（不是旧门PASS，最多100条首批）。 |
| OpenNeoData | 首先权限和条款；只读arx5_single/meta/info.json,tasks.parquet,episodes索引、stats；选真实episode后获取对应data parquet及third/wrist MP4所需完整文件；核episode偏移/时间，不取tactile、不取其它embodiment。先区分observation.eef_pose与action.eef_pose。 |
| RoboTwin | 官方默认脚本是ALOHA，不能当ARX；按Traly卡实际ARX-X5子目录、原始任务/demo_clean manifest、meta/info/episodes/task映射获取少量parquet/MP4，确认原许可证、episode来源。action_ee下一帧参考，不当实控命令。原/镜像去重。 |
| Ville | 优先原作者真实小库，读LeRobot2.1 meta及视频；咖啡prop、footbag、rice-cooker子技能，先确认TRAIN成功；eval/fail/dAgger_test剔除，明确纠正段另池。若只有关节数据用经验证FK/TCP转EEF，真实20Hz等保留，不拿预计算DINO替代RGB。 |

将每个source的许可、commit、原始字段语义及实际文件身份写成manifest。未确认许可不得准入；gated账号未知直接标BLOCKED，其他就绪继续。新split按episode/session/scene先绑定，TRAIN-only不能冒充独立泛化。禁止从Go2 DEV/TEST回流数据。

## 2. Reader与训练ABI

建议沿现有训练入口增加独立source adapters/registry（名字由实际代码树决定，不假装已存在）；复用codec_v1、现有scale20k域训练、真实mask、已有stats、数据worker及engine。不要把源数据改写为Go2的H40或删掉双臂监督。定义可检查的统一样本对象：

```text
dataset_id, revision, source_episode_id, session, split, task_id, query_index
observation_clock, RGB keys + original frame/PTS/offset, image mask
measured_or_documented_state, frame/TCP calibration, actual previous command if available
native_time_grid, source_action_semantics, valid_time_dim_mask, episode boundary
domain codec, normalization hash, label provenance tier, success/correction identity
```

每域按实际state和action定义做SO(3)、FK、TCP、夹爪、原时间与相机检查；OpenNeo rot6d按列、Table30 v1 RPY/v2 xyzw、RoboTwin xyzw；实际频率确认后候选近30Hz H50、20Hz H40，旧域固定不变。pad32只为shape，缺臂/BASE/尾端全部正确mask，不将未知夹爪当0，不复制最后一帧或末点；mask影响noisy target和loss。单源stats TRAIN-only；新IO从兼容父叶或确定性初始化，旧域叶保持；每个实际domain固定8window CPU/Processor探针，报告源语义是否可信。许可、单位或时间不可辨则局部BLOCKED，不阻塞其他源。

在H200合法冷边界加载完整父、新domain真实梯度与全配方6步（2warm+4timed）、每新域至少2实际覆盖，最多24工程更新，完整保存冷恢复、固定推理/下一步。确认有限值、有效mask、旧冻结叶、global clip/Adam、inactive private leaves不变、实际访问与GPU/CPU峰值；旧1e-4/1e-3门只记录，不为普通浮点差耗时反复试。实际逻辑不正确则修。工程父独立，正式从封存科学父初始化；4worker/2预取、整域microbatch、持久JIT按实测，显存不足回退小MB不改变128 effective batch。

## 3. 数据版本与训练接续

先争取Mix500：旧200 RoboDojo＋新100 RoboDojo＋100 RoboTwin＋100单臂（合格Neo/Table/Ville任意组合）。已有200+新增≥100且≥3任务组可先开V6真实课程；若只有双臂新增则明示dual-only。没有新增READY不启动伪扩充。逐步扩展400/500/500/400/200＝2000 ARX5 TRAIN，留出另计；每次扩源封存独立version并记录来源/标签/实际统计与采样成员，不热改1K/2K。每个文件收方真实读取、完整manifest hash和immutable READY，原批准通道；禁止将私有数据公开、Vanda暂停。Core V5继续采集与评价，不能因源域数据解码挤占物理必要资源。

V6父：原V4 M2000如确实完整且回归合格，否则M1000如合格，否则已验证MAIN3.8K。新tree新optimizer，不拼旧Adam。SOURCE primary 16 Go2/64 single/32 dual/16 LIBERO，fallback 16/0/96/16；ADAPT primary64/32/16/16，fallback64/0/48/16。每组固定native domain轮转、分层采样、不把不同H和统计硬stack；group权重B/128，源子域按实际B比；aux4个真实完整支持Go2 H40槽、10步全反传；SOURCE λ按旧目标/aux相对比换算，若旧Go2 0.5归一化则×0.25，ADAPT恢复。新峰值先锁为V4峰值0.5×，warm100、5000总日程余弦；1000后有证据续到累计5000；若数据新版本在边界改变，记录新课程/父及实际累计预算，不冒称固定续训。10K须条件证据。每次实际端点同步Core评估；新SOURCE不等待Adapt200齐备。

Adapt100可用即从选定完整SOURCE端点开始目标联合适配；不足100先如实命名；ADAPT1K→条件3K→条件累计5K，默认peak 0.5×SOURCE声明峰值、warm100、3K日程；后续5K另锁LR。保留源回放，Recovery按真实支持与阶段。旧52反复训练不作为数据扩充。原V4/V5评价人口、任务安全不变；V6最多5端点×6原生卡，条件按计划；未见Drawer/Insert/组合复用V5固定评估，不先采对应目标训练。实际源DEV指标和原生成功分开。

## 4. 最小交付和启动指令

首个提交回执必须包括真实Hopper状态与2K接力是否已部署、可用源/实际revision/许可、真实pilot读取和排除原因、首个新增READY manifest、首个实际新源访问和训练job。以后每阶段报告真实消费、端点、恢复、资源与模型闭环；不以计划存在、数据下载或loss下降替代。健康任务继续，失败留证、普通修复自主，真实授权/资源限制先行。

```text
Codex：执行V6 ARX5-rich源域扩充。先对账现役Hopper 1K/条件2K/R0，保留固定课程与唯一提交owner，落实合法自动接力并回报真实jobID。同时在允许的CPU/存储上并行小样本准入RoboDojo、Table30/v2、OpenNeoData arx5_single、RoboTwin ARX-X5和Ville；先核实际许可、revision、相机/动作/clock与mask，不等待gated源，不整库下载。先旧200+新增≥100（优先Mix500），逐步Mix2K，单臂缺失明示dual-only。保持各源独立codec/stats，Go2真实消费COMMAND不变。H200新SOURCE 16/64/32/16+4aux，1K→累计5K条件接力，完整端点与Go2回归；Core V5继续采集，Adapt100/200就绪后新ADAPT 64/32/16/16+4aux，1K→3K/条件5K。旧严格数值等价只记录；真实逻辑、有限梯度、恢复、安全与闭环不可省。保持Vanda暂停、不删除历史资产、不重复健康/未知job。各阶段实际授权/存储/截止满足才派发，持续发布真实读取、访问、作业与科学结果。
```
