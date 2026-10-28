# GR00T＋冻结WBC：数据与训练主线重新收敛

日期：2026-10-16，Asia/Singapore。
类型：**RESEARCH_ROADMAP_PROPOSAL / REVIEW_ONLY / NO_NEW_EXECUTION_AUTHORITY**。
审核仓库基线：`6af24da77cc11951366acd07c75939c9648a7fdb`。

本文件响应用户关于原始解耦设想、zero-shot不足、目标本体仿真以外的数据来源及下一步训练的提问。它不取消/热改现役TF-R3，不启动新owner、GPU/PBS、Teacher或物理，不恢复旧时钟/配额，也不覆盖后来已明确采用的资源/工程规则。使用现有唯一owner、统一入口、不可变运行源码；新实验按实际权限合并为一次可执行安排，不能把本文的研究建议冒充已采用任务卡。

## 1. 结论：保持原架构，拆开三个研究问题

保留 `GR00T -> task-space EEF/gripper reference (+ optional learned BASE intent) -> frozen WBC -> actual feedback`。WBC、腿策略、控制算法/增益/约束不由VLA团队优化；独立保持RMS不再作为任务数据的全局门。真实安全、动作/观测身份、时间和物体成功仍检查。

应分别检验：
1. **控制解耦**：VLA不监督腿/臂关节或力矩，不为每个任务重训WBC。使用目标本体仿真成功示范不破坏这项主张。
2. **数据迁移/少样本适配**：源环境示范学到的EEF行为，能否在冻结目标执行器上迁移，少量目标数据的收益有多大？这需要source-only和source+target对照。
3. **严格本体无关/零目标示范**：更强的主张，不能从前两项推导。源数据已有固定底座ARX5，不可称为完全未见过ARX5；固定根与自由根/不同执行器也不是两种机器人本体。目标训练、Teacher调试、接口标定、模型选择和评价接触必须分别披露。

近期主线应为：**已有成功源数据固定版本 + 先评价M2048 + 源域任务学习 -> 目标整机小样本适配 -> 同一WBC的独立任务评价**。人类视频为后续明确对照，不是现在必须重建的world-action预训练项目。Teacher是动作监督的一种生产工具，不是研究必须从零解决的通用自动专家课题。

## 2. 当前已有资产，不再写成“从未固定数据/从未微调”

以下均依据上述固定ref：

|资产|已核验事实|不能扩大为|
|---|---|---|
|源v3.0|158准入episode，126 TRAIN/32 DEV；75726/19232 H40窗口；各自真实RGB、实测EEF/夹爪、提交的EEF10；已交付|完整整机示范|
|源v3.1|295准入episode，248 TRAIN/47 DEV；149048/28247窗口；包含v3.0；独立统计与CPU资格|295条额外数据，或已在M2048使用|
|M2048|由M512完整状态增加1536次更新到全局2048，累计消费65536个TRAIN样本；VLM冻结，537个tensor变化|已学会双任务|
|原模型评价|最终报告中固定16个条件均NOT_RUN/UNOBSERVED，实际0/0 observed|实际模型0/16失败|
|目标整机Teacher|TF-R2 S7真实接触后安全FAIL；TF-R3已进入反应式Teacher及参考时序修订；最新进展仍无完整成功seed|仍只停在早期6条件等待，或已具备新整机bank|
|FB13|在已读取证据中未有合格整机训练bank/正式模型成果|M2048已经有BASE三维能力|

证据：
- `reports/PI0294/20261013T142457Z-localpc-pi0294/DATASET_CARD.md`
- `reports/PI0294/20261014T121640Z-incoming-pi0294-r3/M2048_TRAINING_RECEIPT.json`
- 同目录 `FINAL_MODEL_DEVELOPMENT_REPORT.md`
- `reports/PI0294/tf-r3-midpoint-20261016/REACTIVE_PROGRESS.md`（报告时有在途任务；本次未SSH实查）
- #11 comment6207669304记录后来TF-R2采用及资源规则更新，不能自动恢复更早的累计配额门。

数据卡中的历史HF交付阻断只代表当时状态，后续M2048独立归档已另有收据。数据存储当前能否读取仍由owner按已注册私有manifest核对，不因为旧路径名称相同就宣称相同版本。

## 3. Zero-shot不足：先区分接口有效性与学习不足

官方GR00T文档把预训练本体上的base-model推理与NEW_EMBODIMENT数据适配区分。EEF维度相同不表示预训练动作语义相同。

只做一次紧凑的兼容性审计，复用已通过测试而非重写基建：
- 精确model/processor/embodiment-tag及新本体投影参数的初始化来源；是否误把未训练映射层的输出叫任务零样本能力。
- RGB视角、通道、resize、标定与状态key；输入必须是实际状态。
- 位置/旋转表达、工具TCP、四元数顺序、单位、夹爪方向与范围、normalizer；relative转换恰好一次。
- H40时间网格、真实执行prefix8、ACK、过期与恢复；当前源码已有起点采样与终点采样不同版本，不能混用而仍说20Hz一致。
- 把一段已知成功的高层命令经同一数据处理及执行接口往返检查。该测试只检验接口，不证明未见场景策略成功。

项目中对使用训练所得stats或接口标定的base-model评价，应标明 `no-task-finetuning with calibrated interface`。不要用未知/随机新head去做误导性的fair zero-shot对照，也不要为了得到合法输出绕过安全。

## 4. 数据来源：任务知识不必全部由目标WBC生产

|来源|用于学习什么|对目标整机是否必须先重执行|
|---|---|---|
|已有v3.0/v3.1固定底座ARX5|直接学习T1/T2任务空间动作，作为source baseline|源域训练不需要；宣称目标成功需要目标评价|
|其他仿真/机器人成功示范|额外操作技能、视角和物体变化；有明确EE/command语义时可独立source训练|不需要每条先经Go2过滤；若命名为目标域重定向成功数据则必须|
|少量人辅助仿真示范|在目标接口给出成功参考，缩短全自动专家开发|必须是真实仿真执行和录制，不能拖动实际qpos/物体造结果|
|MimicGen/SkillGen式扩增|物体相对片段复用、目标环境连接|源示范可来自别的机器人；新生成的目标成功结果须重新执行|
|UMI/带测量工具的人类示范|可测的EEF类轨迹、夹爪/视觉|无需自有目标机器人遥操作，但工具/相机/时间适配不可省|
|普通人类视频|交互语义、物体关系、阶段/latent priors|可独立表征学习；不能直接补出米制EEF13真值|

**关键纠偏：若每条source示范都必须先在目标WBC执行成功才能用于任何VLA训练，数据路线就人为重新绑定到了目标执行器。** 源示范按源域真实任务/标签质量准入；目标域数据另建bank，不混淆来源。源域会有视觉/控制差距，应通过source-only与few-target实验测量，而不是声称差距不存在。

其他机器人joint action转换EEF时，必须区分实际FK轨迹和所发目标轨迹。未知控制语义不能仅用FK就造“实际提交的EEF命令”；适配失败的维度可用于表征研究或明确mask，但不伪装成完整BC标签。不能把未执行源预览全部当作目标实际执行过的动作。

ManiSkill目前更适合复用规划与交互采集原语，而非整体迁移仿真引擎。MimicGen不要求源必须是我们自采目标本体，但需要目标环境接口与实际执行。人工辅助只作为有人的独立数据来源，不暗塞入无人值守Goal。

## 5. 立即固定的数据决策

### Source Baseline V1：直接引用已有v3.0，不重采、不重拟合

为当前M2048评估固定：原TRAIN/DEV名单、原数据成员/revision、原stats、原processor、原checkpoint与执行器版本。第一项交付是能实际读取的绑定清单，不是创建又一份大数据副本。先不混入v3.1，因为新统计和样本分布会破坏M2048归因。

从原TRAIN中预定T1/T2各少量episode作学习诊断；完整source DEV继续保留。固定数据后的学习改进才单独命名模型分支。v3.1仅在基线行为明确后成为独立数据实验；不把它的stats热塞M2048。已有codec clipping报告按原件披露，不因本计划把它改为绝对零误差。

### Target Adapt V1：单独采集与封存，未有样本不伪造

每项技能首批2–5条成功用于证明来源成立，5–10条独立成功可做少样本诊断；20–50条/任务作为第一轮目标域适配探索。后续100–200条/任务仅在学习曲线支持时扩采。这些是建议实验档位，不是新硬门、成功承诺或本次执行额度。

目标域保留自由根Go2/真实足地/夹爪/物体/机身和腕部RGB。多次复制同一reset不叫独立多样性。整机T2先成功可以先开展T2适配；不能以T1未好为由阻断已有合法数据。任务失败bank不作为成功BC标签，可用于带真实结果的辅助预测或以后纠正学习。

每个版本锁定：来源/工具/本体/执行器、scene-group split、camera及time、观测state与action语义、成功/安全、mask、数据成员、TRAIN-only统计、训练和评价用途。开发/恢复/外观兄弟样本不跨split。固定源模板复用于TRAIN/DEV时需披露，不能声称未见技能模板泛化。

## 6. 实验顺序：现在最应执行的是模型行为评价，不是再存一份checkpoint

### E0：现有M2048的来源域评价

在相同原源环境、相同D0/S1或已明确版本化的固定保护层下，评价M2048的TRAIN与DEV，按需复用M512对照。原16面板尚未运行的事实保留；新run不覆盖旧NOT_RUN。保存原模型raw动作、安装动作、实际状态与物体结果，S1拒绝计任务失败且报告特权仿真保护身份。

先看少量TRAIN行为是否能走到接触、抓取/推动和释放，再扩原DEV。没有新的目标整机成功数据也可进行这一研究，但必须有独立有效GPU/源仿真执行权限，不能在CPU准备中偷偷加载神经模型。

判读：
- 源TRAIN行为成立、源DEV较弱：优先数据覆盖/视觉与语言条件；不简单判定需要全量解冻。
- 源TRAIN也不成立：使用固定5–10条小集做过拟合诊断，核对codec、mask、语言与关键事件，再决定继续原v3.0训练。
- 只有源loss下降、没有物体任务行为：不盲目扩大到v3.1、RL或video-world-model。
- 源Teacher成功而同模型执行失败：区分模型生成、安装、跟踪与实际接触，不把低层故障统称模型没学会。

### E1：源训练策略到完整整机，零新增整机任务梯度

同一来源域合格策略，固定权重接入完整Go2＋ARX5。兼容性/实际任务安全前提保留，不要求额外hold RMS。先近场完整物体任务，不把复杂移动与新视觉模块同时引入。

该结果称 `source-trained, zero additional fullbody task demonstrations`，不是完全没见过ARX5。物理调试暴露与任务选择继续披露；最终确认集留封，不按确认结果修Teacher。来源和目标环境不需要逐条一一复刻源初态，但任务定义和对照要预定。

### E2：少量目标示范适配，与source-only直接比较

以同一source模型出发，固定相同接口/保护层/评价池/更新预算，比较0/5/20条目标成功示范每任务的可行档位（实际样本不足就缺省，不假造）。target-only可作为有预算时对照。为了不把样本量和计算量混在一起，各档位做等更新预算并报告样本重复次数；独立种子能支持时做至少多个training seeds，不以单次成绩宣称统计定论。

数据/统计改变时单列branch；不得说加载旧weights并重置Adam就是完整精确resume。重复采样小数据的过拟合要通过固定DEV监测。选择一次模型后才打开最终确认；同一冻结WBC不按条件调参。

### E3：BASE意图是独立能力实验，不取消既定EEF13方向

保留EEF13长期接口。先用原EEF10模型完成源域/近场比较，同时给BASE标 `preference absent`；它不同于对BASE发送零速度软保持。旧模型不能被解释为已学会3维BASE。

待有非零BASE真实标签、匹配观测及同一固定ABI，训练独立13D分支，并在明确需要平移/转向的任务中比较同一模型BASE偏好开/关（省略与零分开）。源EEF10缺BASE标签时用验证过的逐维mask和明确缺失state处理，不能零填作为正确专家。可先保留两个独立schema，不强迫整个实验立刻换state19/EEF13。

若目标任务只在近场零BASE下成立，如实报告；不以自由根漂移当作模型移动规划贡献。BASE观测和动作扩展属于上层能力，不能把这三维直接发到腿cmd_vel绕过既定WBC接口。

## 7. 微调配方：先最小可靠基线，再单项改进

现有官方flow-matching动作头、H40、原20Hz和prefix8作为基线，复用当前已资格数值/加载/恢复设置。先以冻结VLM、训练projector/动作DiT为实际历史baseline，而非宣称所有本体的普遍最优方案。

新目标图像若出现明确视觉域差且数据足够，可单列视觉adapter/LoRA或少量vision层微调，与原baseline做匹配对照；不同时改loss、全量解冻、数据、action ABI与执行器。全参数微调/RL/WING整栈预训练不是当前首选。

正常源v3.0精确续训保持其原Adam/scheduler/RNG/sampler；新schema/统计/数据干预显式暖启动身份。动作监督是记录的实际提交命令，观测是实际状态；若额外做可行性正则/轨迹连续性约束，作为单独算法因素比较，不静默投影模型动作后把改进全归给模型。

训练中启用固定的分阶段验证，不只看总loss：位置/旋转误差、gripper开闭时机、语言反事实、阶段覆盖、动作块边界、预执行拒绝及任务闭环。不能把不同合理动作路径的逐点MSE当作唯一正确性。

## 8. VLA侧的可验证创新，而不是重新发明Teacher

近期优先检验：**源任务空间策略在冻结全身执行器上需要多少目标示范？** 高低层解耦和人类视频学习均已有先例，接口换名不是算法创新。

方法候选只挑一个：在原GR00T上增加由RGB/语言/可测历史预测的交互进度/事件辅助表征，使接触、推动、释放不被漫长approach窗口淹没。阶段真值可用于训练监督，不在推理给oracle。方法必须超过相同数据/算力的普通BC，并报告语言配对、未见布局和样本效率；不是仅做一个encoder示意图。

人类视频的合适后续：保持机器人动作监督预算固定，比较无视频、普通视频特征、交互对齐视频特征。目标是减少目标动作数据需求而非宣称普通视频直接给出EEF13。WING当前核实到官方摘要，本文不重复此前尚未在本轮全文复核的GPU数、DCT维度或表格细节。LAPA官方明确latent动作最终仍需映射到真实动作。

若要写更强跨本体claim，需要不同真实机器人/执行器域、固定共享策略及严格目标数据记账；只切两个场景或固定/浮动底座不够。失败事实不应重写，但任何单次实现失败也不等于否定整个架构。

## 9. 两端真正并行与避免新前置

|时段|CoreLab|Hopper|
|---|---|---|
|现在|现役TF-R3按已有采用继续；source环境评价服务与整机物理分配由单owner隔离调度|准备/执行已有M2048 source模型评价，须自己的有效GPU权限|
|source学习|封存既有数据和少量针对性source扩增；整机任务种子开发独立|固定source版本上的学习诊断/条件式续训|
|目标种子成立|小批独立target示范；合格后1->2->4worker逐级隔离|读取已经冻结的target小包做适配，不等下一批|
|模型比较|评价优先，采集在episode边界降并发|同一DEV的source-only/few-target模型比较|

旧固定臂4worker的2.86x仅为历史实测，不代表整机MPC环境加速。DataLoader worker数与环境worker数分开。源与目标运行共享资源而非两个无协调owner。

不再要求全自动Teacher、多任务全量、公开归档、整机全部场景、低层hold RMS都先成立才能有任何VLA工作。真正安全/身份/数据错误仍停止相关执行，不因为急着推进而忽略。具备相同目标的重复阴性检测应收敛为一份原因和下一决策，不重复三轮只复述blocked。

现行新增TF-R3记录中有后来用户采用/资源更新；本研究不重新加已撤销小门、不移走在途任务，也不发无限预算。该路线若采纳，由owner把源模型评价和目标Teacher两项调度合并成一份当前可执行任务安排。

## 10. 下一轮应交付的最小清单

1. SOURCE_BASELINE_BINDING：引用v3.0原名单、stats、processor、M2048和执行器版本；不复制raw。
2. MODEL_BEHAVIOR_MATRIX：source TRAIN/DEV的真实模型结果，NOT_RUN与失败分开。
3. TARGET_DATA_CARD：当前实际成功数、来源、实际相机/动作、失败库和哪些子任务仍缺数据。
4. COMMON_ACTION_CONTRACT：EEF10核心及独立BASE扩展、时间/normalizer/合法输入和可观测反馈。
5. SOURCE_TO_TARGET_EVAL_PLAN：固定source-only和few-target比较、oracle/工程暴露披露与最终确认。

交付到GitHub的仅为自研代码、脱敏小报告和来源引用；私有raw/视频/WBC/模型不公开。不用本研究文档修改历史FAIL、active代码或selected/deployment。无实际结果就保持NONE。

## 11. 原始方法来源与工具读取边界

- NVIDIA官方NEW_EMBODIMENT教程：https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md （本次读取文档blob707e893c3dad98f58541f402d253c07806d3bfff；文档不是当前项目runtime升级指令）
- NVIDIA官方README：https://github.com/NVIDIA/Isaac-GR00T （新本体适配、预训练本体推理；不擅自切latest）
- UMI on Legs：https://umi-on-legs.github.io/ （人类工具示范/任务空间策略与WBC分离，明确单向EEF接口不保证可达）
- MimicGen自定义环境：https://mimicgen.github.io/docs/tutorials/datagen_custom.html
- ManiSkill交互采集：https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/teleoperation.html
- LAPA：https://github.com/LatentActionPretraining/LAPA
- WING官方摘要：https://arxiv.org/abs/2610.03607 （本次全文/项目页获取失败，仅使用已检索的官方摘要，不复用未经核对的详细数值）

本次Chrome读取尝试返回缺少originating browser context，未读取用户标签页；Zotero插件已读其技能说明，但当前会话未连接用户本机Desktop API，未读取/导出/修改其文献库。配套BibTeX来自公开原始来源，供本机Codex/Zotero后续使用，不声称是用户文献库导出。

**最终建议：不要推倒解耦架构，也不要把人类视频、Teacher、BASE扩展和新模型一起上。先以已经训练过的source模型取得可解释行为结果，再测量目标域适配需要多少数据，同时保留完整Go2＋ARX5作为最终评价域。**
