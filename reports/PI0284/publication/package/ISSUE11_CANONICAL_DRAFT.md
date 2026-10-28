# PI0284 proposed canonical — 原生生成质量与任务条件化对照（发布前模板）

**This is a prepared draft, not an existing Issue #11 record.**
Proposed authority: `PI0284_NATIVE_GENERATION_CONDITIONING_GO_V1`
Fixed release: `{{RELEASE_SHA_AFTER_SUCCESSFUL_PUSH}}`

Protocol: `docs/protocols/PI0284_NATIVE_GENERATION_AND_CONDITIONING_V1.md`
Limits: `reports/PI0284/design/REVIEW_AND_LIMITS.json`

审核起点e193847f3feeb2a4bf1e5028f0fd3204f65eb0a0；已刷新终局e873939fa6a78698211b18a8995cfce3f664f831 / #6053296607（2026-10-08 13:49:46 SGT）。MORE和EVENT均已完整结束，持续0/4。EVENT TRAIN拟合6/24全部来自晚期保持，首次释放仍全部失败；10003有44个连续20Hz成功采样点，之后失去成功，最终物体静止释放但不在目标上。不得继续沿用104步旧快照或重启已完成EVENT；所有旧预算、失败、恢复点保留。旧09:00限制已据用户指令移除；新旧有限预算不混用。

## Gate与新权限
1. 采用PI0283实际终局并核验EVENT1024完整恢复，parent master ac8fad1a39aad656fac3f0bdafaa856b984d67cd55a0df111103ae79a80f8e2a。它已明确未合格，不重跑EVENT或旧失败面板。只有在协议已提交、canonical和文件回读成功，且前序科学执行已终结后，才启用下面的新科学预算。
2. 固定权重和真实输入，DIAG48四份固定完整噪声，比较原生Euler K4/K16；不改变H40、20Hz、前8点/.4s、decoder、精度、安全和评分，不挑噪声或平均动作。按预注册TRAIN指标锁定K，改K是公开的推理策略变化，不是已验证的修复。若零更新变体开发合格，先确认，不再训练。
3. 原TRAIN实际初态中最多8对近似同TCP但不同任务目标参考的自然观测对，测动作是否响应布局变化。没有伪造state或拼图；不足则INCONCLUSIVE。弱响应只是局部证据，不等于视觉或语言全被忽略。
4. 专家状态仍拟合不足时，只选一个配对比较：弱响应成立则CTRL(p=0) vs STATE20(p=.2)；否则CTRL(DiT1e-5) vs LR5X(DiT5e-5)。两个独立分支同一真实parent/Adam、同16384个样本和实际flow噪声，各2048步；不得同时加dropout和升LR。原93集、原生flow-matching、row10+DiT范围、冻结VLM/其他row、stats、image与EVENT采样家族保持一致。
5. 若专家拟合已合格但闭环差，本goal止于归因与恢复方案，不盲目继续拟合，不自动采teacher。
6. 固定TRAIN/DEV，只有同一endpoint≥3/4和≥6/8，才能冻结一候选执行新的预注册30100..30115 CONFIRM16；至少12/16原persistent成功才能叫source qualified。原CONFIRM不动，新ID必须先验无曝光，碰撞不换seed。最多一个候选确认，失败不回训、不测第二候选。

## 有限预算与资源
正式更新≤4096（两臂各2048），全已完成≤4208；模型请求≤6400、实际head≤102400（K16不能仍记成S4）；物理最多64 starts，质量重试0。12 GPU-hour/96 CPU-core-hour/120GiB是上限，不是用时或成功承诺。两worker缓存、现有合法H200+Corelab、本地完整恢复和精确直传；HF不入关键路径。一个owner与最多三个互斥分工worker。无新数据/teacher/多任务/WBC/target/真机。

## 研究解释
原生训练速度场loss低不代表四步自由生成可靠；K16更准也不保证真实任务成功；STATE20更好也不证明shortcut是唯一根因；LR比较结果不是通用训练步数定律。源场景PASS仍不是多任务或Go2资格。保留原始DROID32观测保留探针（两endpoint共64调用）和冻结参数证据。selected/deployment=NONE。

执行前须回读两份文件与canonical并登记实际release，不得沿用本模板占位值。
