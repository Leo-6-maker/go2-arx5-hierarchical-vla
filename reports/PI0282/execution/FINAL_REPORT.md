# PI0282 R2 终局：原生 ACTION 已完成，任务质量仍不合格

**结论：VALID_QUALITY_FAIL_NOT_USABLE。** 最佳开发点 ACTION2048 已实际训练、加载并完成固定评价。持续抓取有明显进展，持久 StackCube 成功为 **0/4**，未达到 TRAIN ≥3/4，因此 **DEV／CONFIRM 未准入、未查询**。`source_candidate=NONE`，`selected/deployment=NONE`。这是实际源 Panda 的 EEF＋夹爪闭环，原 native PD、原 SafetyGate；不是 WBC、目标机器人迁移或实时部署。

Authority：Issue #11 **6044490956**；release **fdcde0e29f7eb689d0eea0f4dd20586309e77326**。原历史失败、全部工程日志和原始数组保留。主 owner 独立收口，无辅助签字依赖。

## 固定质量结果

每点 DIAG48＝24 TRAIN＋24 VAL；每点 TRAIN4 固定条件10000..10003。512的四个安全拒绝均保留为有效失败；1024／2048均完整32任务秒。没有质量重试或更换种子。

| 点 | 持续抓取 | 描述性2cm抬升 | on-top | 首次原生成功 | 持久成功 | TRAIN拟合门 | TRAIN XYZ中位／p95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACTION512 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/24 | 56.599 / 105.427 |
| ACTION1024 | 1/4 | 1/4 | 0/4 | 0/4 | 0/4 | 0/24 | 41.426 / 71.878 |
| ACTION2048 | 4/4 | 2/4 | 2/4 | 2/4 | 0/4 | 1/24 | 17.335 / 46.363 |

2cm抬升列是从实际数组复算的描述性指标（高于初始高度2cm、持续30个100Hz样本），**历史原始 lift predicate 未记录**；它不代替原成功条件。持久成功必须完整32秒，并在任务28..30秒全部41个控制采样同时满足 on-top、static、not-grasped。CPU独立复算每个原生谓词、实际控制安装、每次当前 RGB／StateV1 输入及全132维 noise 字节身份，三个面板均通过正确性检查；正确性 PASS 不是任务 PASS。

## ACTION2048：成功事件极短，验收窗均已失去 on-top

| 条件 | 首次成功 s | 首次成功丢失 s | 最大实际抬升 mm | 验收窗on-top | IK hold计数 | reference→installed RMS mm | installed→同次控制后actual RMS mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 26.75 | 26.8 | 91.41 | 0/41 | 171 | 9.256 | 10.073 |
| 10001 | 8.3 | 8.35 | 90.62 | 0/41 | 0 | 0.068 | 15.941 |
| 10002 | None | None | 13.08 | 0/41 | 0 | 0.071 | 17.274 |
| 10003 | None | None | 2.81 | 0/41 | 246 | 12.151 | 12.104 |

10000仅在26.75s的一个控制采样出现原生 success，26.80s失去；10001在8.30s出现、8.35s失去。它们不是持续放置成功。10001没有 IK hold，也未维持成功，故不能把全部失败归因于 IK。reference、安装目标与同次控制后的实际 TCP 误差分别报告，没有用两个 minimum 的差当 tracking。

2048 TRAIN前缀 SO3中位0.01943rad，夹爪 MAE中位0.00452，但夹爪p95仍0.70449，合取拟合门仅1/24；各阶段也未过半。归一化输出越界（原 inverse 裁剪前）TRAIN DIAG中位20.5%，实际闭环四条件均值约10.56%..15.57%。这是保存输出的诊断，不证明统计量错误，不授权改 normalizer、精度或 clipping。训练损失下降与真实抓取改进同时存在，仍不足以产生稳定任务完成；没有证据支持将失败概括为“完全没有学习”或“唯一控制根因”。

## 训练、恢复与预训练范围

- 正式2048更新／16384 chunks；原始累计2080＝正式2048＋已通过canary16＋工程丢弃16。两微批×4，effective batch8；无额外子阶段更新。
- 连续8与4＋保存＋新进程恢复＋4真实 canary 已通过；四次技术恢复全部记账。两次早期初始化失败在首次 native forward 前结束；原数值失败权重退役前保留了32组超容差原张量对及完整小证据。
- **完整1024与2048 Trainer**各14文件逐文件 size／SHA 审计，含模型、Adam moments、scheduler、RNG、cursor、processor、statistics及mapping；保存位置在模型加载说明中。
- 两点实际 CPU 检查：560冻结参数张量等于原始值；14个projector仅row10改变，其他31行精确不变且 moments 为0；456共享DiT张量改变。470个optimizer参数实际step与各checkpoint一致，projector decay0。
- 4096实际正式微批顺序、16384样本与PLAN逐项一致，四阶段各4096。完整 noise/time hash 存在；详细梯度证据272微批，其余使用有限全局grad检查及interval16。**实际逐步mode日志 UNOBSERVED**，不补造。DROID32保留性模型查询未执行，不能声称共享预训练行为全部保留。

## 为什么没有 C／F／R 后续训练

实测512物理面板完成后，按协议保守评价预留（1.5×实测p95×28个所需条件＋export/load＋15分钟收口）冻结约12210.63秒的预留；由此双端新optimizer截止提前到 **05:30 SGT**。不得因候选失败而把这个已冻结截止向后改。

- **C**：1024→2048物理漏斗确实改善，科学触发成立；结果闭合时已过新训练截止，`NOT_LAUNCHED_TIME_RESERVE`。
- **F**：拟合不足触发成立；CPU计划与carry-state草案准备好，实际更新0，`CPU_PREPARED_NOT_LAUNCHED_TIME_RESERVE`。
- **R**：拟合不充分，且on-top／first success只有2/4，不满足两条准入规则，`NOT_ADMITTED`。
- INTERFACE未启动。没有把未执行的分支写成质量FAIL或已穷尽假设。

实际512物理反馈到达时，正式2048已经完成。工程环境恢复、不可变模型输送及串行评价使它没有形成训练期反馈；这是本次执行的时序局限。HF完整归档是可延期项，本次却为跨端模型交付使用了HF路径并产生等待，不能宣称为最短路径。已有结果不因这项记录缺口重训。

## 交付与累计预算

最佳开发／最新完整恢复：**ACTION2048**（不是合格source_candidate）。公共HF repo `leoyuer/pi0246-gate-pass-campaign`：

| 点 | fixed revision | 交付类型 |
|---|---|---|
| ACTION512 | ff03f42e5178a367cb99fac054ec498412f1fc09 | 原始base＋完整changed tensors；HF model-only，本地已退役，无完整Trainer |
| ACTION1024 | 60027054d3f62e5ae0b2b3e70be54e76fe9a2980 | HF model-only完整回读；本地完整Trainer＋Core推理视图 |
| ACTION2048 | 4f32c9fd680a1db69ef1fdc0f1411e16a801ccdc | HF model-only完整回读；本地完整Trainer＋Core推理视图 |

每个prefix `evidence/PI0282/R2-delivery/POINT`。完整Trainer的HF上传延期，本地唯一完整恢复源受到保护；没有把model-only出口称为exact-resume。2048 mixed master identity：`cd845d21b9a953a02d74dc49c1f15b5d5136cfce2e915c5f8539ae7eb6b8dc77`。模型、processor、mapping、decoder和可运行原命令见 [加载说明](MODEL_AND_RUNTIME_LOAD_RECIPE.md)。

实际累计：**874模型请求／3492 S4 heads**；**4160 native训练forward attempts**，无loss失败更新；**16物理starts**（4专家codec回放＋12模型），8648控制、43240 integrations、8652 RGB pairs；新教师调用0，WBC／目标机器人／硬件／TEST／DEV-F均0。所有五个PBS job已F；Core有限评价wrapper返回0且PID消失，当前goal科学子进程均无。其他项目GPU进程未操作。

保留新输出合计 **73.845 GiB**，低于120GiB cap；个人quota仍UNKNOWN。GPU start→obit／Core首个专家wrapper→最后模型wrapper区间的保守合计4.657 GPU小时，非利用率测量。已记录PBS cput3282秒、源panel子进程CPU2618.50秒；全体短进程CPU未完整采集，声明线程资源的保守调度上界104.40 core小时，不能当实际CPU利用率。

## 真实录像与停止位置

- [2048真实双相机：首次成功后持久失败](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/52ea271e3120499e8e844ae7328d7d3c91ca402e/evidence/PI0282/R2-delivery/videos/ACTION2048_TRAIN10001_FIRST_SUCCESS_THEN_FAIL.mp4)：660原始帧，10fps输出330帧，真实状态标注；无新render／physics／model。fixed-revision全文件回读SHA `e06ae7228581c0577db77792f8b528aaa719c0e91006cb859bfeba0849fe0527`。
- [512实际安全拒绝录像](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/d258d5dae4335088125025d3c08d3c0683ef1943/evidence/PI0282/R2-delivery/videos/ACTION512_TRAIN10001_ACTUAL_FAIL.mp4)。

基线及已执行固定评价完成，科学质量FAIL；限定续训分支未执行，不声称完整因果对照。DEV、CONFIRM、target transfer均未准入；不自动追加新训练，不等待09:00制造更多失败。留存当前完整恢复、原数组、负结果、精度／输入／noise／scope证据，交下一次独立科学决策。邮件、monitor和merge保持关闭。
