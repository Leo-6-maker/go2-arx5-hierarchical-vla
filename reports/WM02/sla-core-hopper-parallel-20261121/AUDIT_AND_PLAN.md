# Core/Hopper 并行计划：证据审计与设计决定

日期：2026-11-21，Asia/Singapore。此文为**源码与协议审核＋新计划设计**，不是新服务器实绩。

## 1. 信息可信范围

| 信息 | 本轮依据 | 结论范围 |
|---|---|---|
| 11月21日10:55现场进度 | 用户提供的最新完整快照 | 作为最新计划基线；本轮未SSH/PBS刷新 |
| 当前main与旧入口 | 回读`6843163d641eb24a29c3b1557f868765346ba66c` | main仍是较早文档，不能当最新训练实现 |
| SCALE执行／报告 | PR418固定`c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6` | 可引用历史报告与共享实现，不假装它包含用户后续所有现场回执 |
| ADAPT实际worker | `1432c8dc88c57adf39e8fa363f5438c5e63aa8fd` | 现有训练语义、合同与数据银行 |
| ADAPT传输／site | `124af5eca57d0713714c2c74ed7f7456ec0b10d6` | 推理29数组与单GPU站点校验，不是完整20K恢复工具 |
| 性能工程 | PR419 `2caa24b2447e41f6a64323fadc1a428c72d583db` | 可复用预取与同步改造，旧A40结果不等于H200已测 |
| 既有expert producer | `55030f539c448728133d8ce7cf8dcc5afa4be725` | 复用来源，仍需新scope与受影响接口验证 |
| Hopper直连VANDA | 用户明确说明 | 计划采用直传；现场核实现有获准账号和队列，不修改网络配置 |

本次发布新增服务器作业、训练更新、离线生成、原生请求、tick均 **0**。计划中的数字均为新预算、目标或引用的历史结果。

## 2. 关键源码和报告

- [现有ADAPT合同](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1432c8dc88c57adf39e8fa363f5438c5e63aa8fd/src/go2arx5_vla/sla_adapter/adapt5k_contract.py)、[训练循环](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1432c8dc88c57adf39e8fa363f5438c5e63aa8fd/src/go2arx5_vla/sla_adapter/adapt5k_training.py)、[数据实现](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1432c8dc88c57adf39e8fa363f5438c5e63aa8fd/src/go2arx5_vla/sla_adapter/adapt5k_data.py)：旧5K和银行／阶段约束需显式新版本，不能仅修改max_steps。
- [Go2推理传输](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/124af5eca57d0713714c2c74ed7f7456ec0b10d6/scripts/sla/adapt5k_endpoint_transport.py)：明确检查29个package参数数组、旧Goal和既有Eval，不能恢复140数组训练状态。
- [单GPU站点验证](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/124af5eca57d0713714c2c74ed7f7456ec0b10d6/src/go2arx5_vla/sla_adapter/site_v2.py)：按实际GPU UUID、PBS owner／project、内存、node-local cache和期限校验；不是只允许A40。
- [原六任务评分](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6/src/go2arx5_vla/world_model/multitask_score_v6.py)：2ms完整状态、HOLD60／PLACE90原时限、20mm抬升、1秒、80mm底座、完整放置支撑释放撤离。
- [Hopper历史正常退出](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6843163d641eb24a29c3b1557f868765346ba66c/reports/PI0284/execution/HOPPER_JOB_EXIT0.txt)：证明历史H200 PBS执行；不可当作当前grant。
- [Hopper旧接续说明](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/6843163d641eb24a29c3b1557f868765346ba66c/reports/PI0276/execution/HF_DATA_HOPPER_CONTINUATION_ISSUE11.md)：已有nohup随SSH退出而终止的经验，所以采用真实调度器持久任务。
- [TGT2K／5K配对](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6/reports/WM02/sla-multisource-goal-v2-20261120/TGT2K_5K_PAIRED_AND_MIX_PUBLIC_WATCH_ACTUAL.md)：更多更新并不保证接缝双门单调改善，不能以训练步数替代闭环评价。
- [数据与吞吐对照](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/c8642aa602d3e34dd4dcda5a55ff6ae7c7c53ae6/reports/WM02/sla-multisource-goal-v2-20261120/EMBODIMENT_NEXT_DATA_AND_EVAL_DECISION.md)：mb4和两卡工程结果有边界，不能直接宣称高吞吐后端已资格。
- [ADAPT2K公开固定身份](https://huggingface.co/datasets/leoyuer/sla-goal-v2-checkpoints/tree/5c460f195f6509f2f4da55dea32c356b45e00d37/ADAPT5K/step-02000-2bff547b6f2c)：计划固定父端点。完整文件内容及冻结底座仍需现场交接验证；不能从目录短前缀推导完整hash。

## 3. 主要困难与本计划的回答

### 训练慢：先消除串行计算，再讨论更多卡

已有循环global batch128但microbatch1，逐个处理和回读；辅助还有四个完整十步反向。H200大显存允许测试更大的FM和aux批，但没有实际H200基准前不能承诺倍率。固定global batch、域均值、四辅助和精度，先在一张H200比较六个有限候选，失败可回落mb1，避免又把主线锁在工程完美门后面。

### 数据少：按独立轨迹和任务覆盖补齐

D2的2,031窗来自26条轨迹，不是2,031次独立成功。大量公开源数据能支撑通用视觉动作表示，不能自动提供新本体腕端碰撞、携物净空、左右放置的完整监督。新TRAIN240以六任务/F1F2/场景变体为单元；单条合格就可入库，不等所有格都通过。120条是采集目标，历史低产率下可能达不到，必须保留真实分母。

### 科学验证断层：离线闭合与完整任务分开

ADAPT2K及时24/24、联合9/24尚不能证明接触安全或抓持。固定新DEV、最终CONF和第一失败机制继续保留，22mm宽度只诊断时序。新增VAL按scene family与TRAIN隔离，避免拿TRAIN闭合窗当留出泛化；旧TEST不参与模型选择。

### 等待链过长：删除不必要依赖，保留真实必要条件

新H200只等自身资源、完整父状态、现有D2和正确worker；采集只等自身scope/owner/合法人口。新数据晚到只影响下一课程边界，部署失败只影响该模型学生运行。旧推理导出器的FORMAL424要求在新包schema中拆开：独立冷态REF16和Core原严格RGB16与三域Eval并行，源域缺包不拖住学生分支。未完成424/48明确标待评价，不伪造旧回执；共同完整性故障仍冻结受影响下游。

### 后台易断：持久调度器与完整恢复状态

旧推理包不可继续训练；旧合同不可冒用Hopper身份。新PBS接续读取完整seal与真实phase，独立记录主臂、工程、控制、提交重放及未提交尝试。旧A40后继继续其原Goal；不把UNKNOWN作业当未启动再双投。

## 4. 新设计的因果解释边界

主臂新20K同时改变硬件后端、训练长度、LR阶段和后续数据课程，是新的谱系，不能当作原ADAPT5K的无改动复制。H200工程在相同父状态和相同输入/RNG下比较，保证配方含义与数值误差边界，正常归约差异不等于科学配方变更。

第一次新数据接入保留共同父状态，主/控各向前2K，只比较新数据及预登记采样配额课程整体作用。后端、LR、源输入、更新数与辅助定义匹配；不能声称隔离了“纯数据量”效应。主臂不等控制，不为增加统计显著性临时扩样或选最优端点。

新VAL离线评价以专家成功示范为条件，须报告24条原始专家尝试与实际可用子集，不能声称覆盖全部失败场景。新DEV/CONF是有限本体仿真验证，不是全部长程语言规划能力评测，不代表真实机器人可靠性或连续实时资格。数据/吞吐/闭环三类结论必须分开报告。

## 5. 尚待现场绑定，不能预填通过

- 真实Hopper／Core／VANDA账号、project、queue、资源allocation与GPU UUID。
- 父端点完整140数组清单、完整checkpoint hash、冻结底座和数据content hashes。
- 新H200执行树SHA、环境与真实恢复输出。
- 新人口数值几何及与历史数据的身份交叉审核。
- 实际采集成功率、准入数、H200吞吐、GPU显存峰值、闭环成功。
- 实际T0、deadline、后台job/service身份、PBS依赖及断开交互后存活。

这些是现场执行产物。任何缺项应在真实终态回执中明确记录，不通过写一个READY文本、变更旧goal哈希或把计划数字写成实测来消除。

## 6. 交付与静态核验

[完整协议](../../../docs/protocols/SLA_CORE_HOPPER_PARALLEL_V1_20261121.md)、[机器计划](../../../docs/protocols/SLA_CORE_HOPPER_PARALLEL_V1_20261121.json)、[逻辑人口](../../../docs/protocols/SLA_CORE_HOPPER_PARALLEL_V1_POPULATIONS_20261121.json)、[启动prompt](../../../docs/handover/2026-11-21/START_SLA_CORE_HOPPER_PARALLEL_V1.md)。

发布前仅做计划一致性检查：JSON可解析、预算逐项加总、人口唯一与划分、硬依赖无环且没有旧A40/Core训练完成依赖、相对链接、根入口历史保留。结果见同目录`PLAN_VALIDATION.json`。这不计GPU/原生测试通过，也不替代现场资格。
