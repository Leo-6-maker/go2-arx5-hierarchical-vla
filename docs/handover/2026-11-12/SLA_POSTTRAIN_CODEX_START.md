# Codex启动与实现交接：SLA本体后训练V1

这是用户要求执行的新研究任务。主协议为 `docs/protocols/SLA_GO2_ARX5_POSTTRAIN_V1_20261112.md`，声明性规格为 `experiments/sla_go2_posttrain/v1_design.json`。JSON中的null和RUNTIME_NOT_ADMITTED表示现场尚待绑定，不可原样交给旧GR00T launcher。

## 可直接交给Codex的启动指令

接续 `Leo-6-maker/go2-arx5-hierarchical-vla`。完整读取本分支根入口 `SLA_POSTTRAIN_START.md`、上面主协议和JSON。最新用户要求是直接使用 `haofuly/SLA_base` 做Go2＋ARX5本体后训练。不要再写一份替代计划后停止；在已有合法执行环境内，按依赖实施下面的真实数据转换、训练与闭环评价。若没有远端权限，如实交付实现和具体阻断，不冒充已启动。

### A. 先核现场与资产，保留已有成果

1. 读取最新#376及合法后继、实际ACTUAL、当前owner/PID/Journal/队列，检查GR00T A500是否正在运行。旧正文可能过时：`893db1f...`的T0已证明新相机SFT2000四条件4/4成功，T1官方GR00T10次非候选训练资格已完成。不要重跑这两项，也不要打断健康作业。
2. 文档分支基于main，main不包含完整现役执行栈。为SLA创建独立实现worktree/feature branch，以实际已审阅#376执行版本为基础；不force、不盲合历史、不热改capsule。
3. 在用户空间独立获取上游 `OpenHelix-Team/Spatial-Forcing@405ca8879c032a05f0a778adc7d86f4e221c5eba` 的 `SLA/`。固定HF `haofuly/SLA_base`的真实revision、逐文件SHA、Orbax参数树、tokenizer与许可；不能用benchmark权重冒充base，也不能假定公开即无限再分发。教师资产仅为S2获取，不阻塞S1。
4. 默认Core4080S单卡，重GPU任务串行。先核真实空闲/容量；无新大卡作业授权则不要自动qsub。CPU工作不得堵住GPU科学推进，也不要用AMD登录/存储机做重计算。

### B. 实现最小的数据与模型适配

5. 复用sealed新body/wrist TRAIN100，保留100episode、1450专家窗口、1500原capture、50干预非BC和原Teacher名义H40。原图/状态/标签不改，建立独立SLA视图与manifest。不得静默删除缺文件或把future-achieved改为专家命令。
6. 图像映射body→base_0_rgb、wrist→left_wrist_0_rgb；right_wrist为空且False mask。实际224resize/pad后按阶段查可见性；原GR00T cache不复用。首轮无随机几何或颜色增强。
7. 实现唯一 `codec_v1`：现役frame/TCP解码→episode post-hold固定ARX安装任务帧→绝对7维state与chunk-relative7维动作→32维容器。Δp=p_k−p_0；ΔR=R_k R_0^T；inverse R_k=Exp(Δr_k)R_0。夹爪action绝对nominal，state仍actual。不要重加145mm TCP，不跟随物体建立坐标，也不要执行中移动每个knot的anchor。
8. target统计仅用TRAIN100、S1/S2共享；H40标签时刻0…1950ms，完整消费2s；SLA内部num_steps=10，不能偷偷用GR00T四步或截默认50点。上游原32维FM保留，记录active7/padding25分解。
9. 全量codec往返、专家正对照、相机时间/旋转/夹爪测试；若有意有损转换必须报告。仍使用原basis、消费者和原生安全，不用新的controller。

### C. 真实训练资格与S1

10. 使用PaliGemma rank16/alpha16和action expert rank32/alpha32 LoRA，以及action_in_proj/out_proj、time_mlp_in/out；SigLIP和全部非LoRA主干冻结。显式正向参数白名单并核nonzero梯度、frozen指纹。新增LoRA初始函数应与原SLA一致，不能悄悄随机化预训练投影层。
11. 完成真实加载、固定8窗口两组噪声自由生成、完整有效batch32更新及连续4对独立2＋2恢复（8个非候选更新）；若使用SigLIP缓存再做完整RGB/cache两路线各1更新对照。只缓存真正冻结的视觉塔输出，不缓存有LoRA的VLM；保持所有必要梯度。ordinary RGB导出不依赖训练cache或teacher。
12. 首先实测16GB峰值和稳态耗时；允许数学不变的remat/生命周期修复，不允许CPU训练激活offload、暗改horizon/batch/分辨率/白名单、无限原样OOM重试。容量未通过就明确阻断，并继续有独立价值的CPU/已合格推理。
13. 从干净SLA父及同初始LoRA正式启动S1：500更新，micro1×accum32，seed20261112，lr峰5e-5、25步warmup后cosine至1e-5，AdamW wd1e-5、clip1.0、EMA关闭、state token dropout0.25。dropout不删除实际动作anchor。沿用3200exposure列表×5轮共16000访问。
14. 保存0/250/500，500是固定主端点；250用于恢复，不根据闭环挑最好checkpoint。评估RNG独立。完整保存optimizer/scheduler/RNG/cursor/统计/tokenizer/codec/相机/权重身份，fresh process普通RGB重建。
15. S1完成立刻在新身份执行TRAIN01/07/13/25各一次、最多15查询。raw与basis轨迹均保留，原生拒绝/失败/成功全部统计。不能因Table30或S2教师尚未就绪停在checkpoint，不使用Teacher fallback或best-of-N。

### D. 空间对齐对照与开发评价

16. S2不从S1继续，而从相同干净SLA父、共同初始LoRA开始；除空间loss外，全部公共配方、访问与随机性与S1一致。
17. 固定OmniVGGT版本，用相同body/wrist RGB、无额外相机参数/深度/物体真值生成教师特征；可离线分批缓存后卸载。第12层固定对齐、末层教师、alpha0.5、bilinear、VLM norm、padding mask，非自适应门控。先证实alpha0公共梯度/更新与S1一致，alpha>0梯度到部署保留的VLM LoRA而不是只有projector。
18. S2同500/16000后恢复与ordinary RGB导出，推理移除teacher/align_projector但保留实际LoRA。教师不可用不影响S1交付。
19. 独立SLA_DEV16：8个未训练且不重叠旧TEST的布局×2重启/方法，先登记输入身份/seed/顺序再看任何候选输出，完成固定人口。G用已完成官方GR00T A500时加入工程比较；尚未完成不阻塞SLA。不同模型不宣称seed相同就相同噪声。
20. 原任务评分：真实抓持、抬升≥20mm、持稳≥1s、原V3站立及硬安全；旧2mm/0.5°诊断另列。报告首失败、raw/basis/关节/接触拒绝、实际消费、物体/夹爪及延迟。硬安全按原规则停受影响分支；技术问题与策略失败分开；不事后挑人口、换seed或放宽阈值。

### E. 收尾与后继

固定每一臂500结果，不自动长训。若SLA普通后训练有效而S2无增益，保留简单方案；若全部失败，先依据codec/拟合/执行/接触的具体证据定位，不盲加数据。后继pi0.5匹配初始化、25/50/100数据效率、多seed独立TEST、PLACE/长程技能按主协议的新版本推进。

保持WBC/腿策略/增益/solver/动力学/2ms、原采集与安全冻结；不引入WM/RL、主动行走、多GPU或实机。沿既有合法执行owner自治修复项目代码，不增加逐函数人审或重复调度层。权限、配额、唯一writer和每次有限walltime仍须真实满足。

每个冷端点在实现PR发布 `QUESTION/CHANGED/ACTUAL/NEGATIVE/DECISION/NEXT`、UTC/SGT、真实加载/更新/访问/物理分母及成本。模型、原始私人图像、凭据和host-bound元数据不公开。GitHub提交不算远端执行。

**现在先读回最新现场；并行开始SLA资产固定和TRAIN100 codec实现，完成首个有资格的真实模型更新，然后推进S1_500与TRAIN4。**

## 实现提交建议（有限工作包）

| 提交 | 主要文件 | 验收 |
|---|---|---|
| I1 | sla_adapter/codec_v1、dataset_v1及测试 | 1450往返、相机mask、时钟、任务frame/TCP及夹爪 |
| I2 | config/checkpoint、上游锁定环境、scripts/sla/qualify_v1 | 实际参数与梯度、容量、4对2＋2、ordinary RGB |
| I3 | train wrapper、S1采样/日程、报告 | 500真实更新和完整可加载端点 |
| I4 | provider与原consumer薄接线 | TRAIN4完整原生行为，无新控制器 |
| I5 | 教师cache、alpha0/alpha>0校验、S2 | 受控空间对齐及部署移除教师 |
| I6 | 新DEV16注册和完整评分 | 独立人口及逐方法结果，不修改旧A/B面板 |

这些接口尚未在本计划分支实现。不要尝试运行虚构CLI；先实现薄封装并在实际资源环境验证。复用原 `task_learning_endpoint_basis_v3.py`、native pair、采集、reference安装与评分；只替换神经provider和数据/动作转换。JAX/OpenPI与PyTorch/GR00T使用独立环境和私有模型包。
