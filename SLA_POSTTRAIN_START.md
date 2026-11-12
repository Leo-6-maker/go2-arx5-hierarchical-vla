# SLA 本体后训练 · Codex唯一入口

用户请求（2026-11-12）：利用SLA预训练模型，对完整Go2＋ARX5＋冻结WBC进行目标本体后训练；制定详细计划并推送GitHub供Codex执行。

## 顺序读取

1. [详细科学与执行协议](docs/protocols/SLA_GO2_ARX5_POSTTRAIN_V1_20261112.md)
2. [机器可读设计规格](experiments/sla_go2_posttrain/v1_design.json)
3. [准备状态与实现细节](docs/reviews/2026-11-12_SLA_PLAN_READINESS.md)：明确LoRA零初始化的合法零梯度、500项日程及已做/未做验证，作为主协议相应术语的精确解释。
4. [完整Codex启动prompt与实现工作包](docs/handover/2026-11-12/SLA_POSTTRAIN_CODEX_START.md)
5. `AGENTS.md`的操作安全与仓库工作流、现役[#376](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/376)最新ACTUAL和站点owner。

## 当前已经知道什么

审阅执行HEAD：`893db1f49945dda8b318e846a58fd19ed7543940`。T0新body/wrist的SFT2000＋endpoint basis四TRAIN条件已4/4真实成功；T1官方GR00T实际缓存/训练/恢复/普通RGB资格已通过。不要继续引用过时PR正文中的1/4未读回或0资格摘要，也不要重复已完成资格。

这两项不是SLA结果。**SLA权重尚未实际加载、SLA优化更新0、SLA物理评价0。** Hub发布入口已核验，实际权重revision、SHA和使用条款待Codex固定。不得把本次文档发布计作远端作业开始。

## 本轮决定

- 使用 `haofuly/SLA_base`，上游代码固定 `405ca8879c032a05f0a778adc7d86f4e221c5eba` 的 `SLA/`。
- 复用新机身＋腕部 TRAIN100；首先不新增示范、不等Table30，不做多本体持续预训练和WM。
- S1：原FM＋SLA目标PEFT，500更新／16000访问，然后立即完整TRAIN4与新开发评价。
- S2：同一父和公共LoRA初值、同数据/预算；额外空间对齐（冻结OmniVGGT，alpha0.5），独立500更新。教师不进入部署，S2不阻塞S1。
- LoRA只改变新SLA的低秩分支；密集VLM、视觉塔、原GR00T与WBC均不额外解冻。新方案与原GR00T A/B实验身份分开，原A500可作为工程基线；健康运行保留。
- 模型内部采用SLA原相对EEF/SO(3)表示和10步采样；经过可审计转换返回原物理EEF10/H40。时间、nominal夹爪、basis与评分保持明确。
- 默认Core4080S，真实容量实测；不以发布计划为由提交新的大卡、HF付费云或分布式作业。

## 执行与文档分支不同

本分支从main `6843163d...`创建，只发布计划/规格/启动prompt。main尚未包含完整#376执行栈。Codex应从最新审阅的执行版本建立独立实现worktree与PR，不盲合旧提交，不force、不热改在途capsule。

`v1_design.json`是声明性设计，`RUNTIME_NOT_ADMITTED`及null字段尚待现场解析，不能直接当成可运行配置。新CLI路径是待实现接口，不是已经存在的命令。

本次最新任务在**新SLA分支**范围内增加了上述明确的LoRA和空间监督研究；旧GR00T实验的相机、loss、白名单和评价不被修改。操作权限、原生安全、2ms、冻结WBC、唯一物理owner、旧DEV/TEST及私人资产边界始终保留。

## 首个应交付的结果

固定SLA资产并完成codec → 真实单卡更新、保存恢复与普通RGB资格 → S1_500 → TRAIN4。每个端点给出真实更新、模型调用、物理请求分母、失败原因和下一动作。不要在checkpoint已经完成后仍只停留在软件检查或等待Table30。
