# PI0286 终局科学报告

**原两臂均完成；没有达到原资格门的模型。** 原裁决函数返回 `NO_DEVELOPMENT_QUALIFIED_CANDIDATE`。新 CONFIRM_C 未启动，selected/deployment 均为 NONE。

原 release：`eae6dbc1f9281b204a4e28e5f7c100d705837318`，Issue #11 canonical `6072813602`。用户“完成两臂，允许越过14:00”的接续授权已在 canary 与正式训练之间的停止边界绑定；只延长执行时间，未改变配方、门槛或累计科学预算。早先 DEADLINE_PARTIAL、工程错误、旧模型质量 FAIL 和所有原件仍保留。

## 1. 实际完成与质量裁决

| 原固定终点 | 正式更新 | 真实恢复 canary / 训练 CPU | TRAIN | DEV | DIAG96 原 gate | 新确认 |
|---|---:|---|---:|---|---|---|
| C0_REPLAY1024 | 1024 | 均 PASS | 2/8 | 未运行：TRAIN 未达 7/8 | 95 ACCEPT、1 WARN、0 REJECT | 未运行 |
| C1_CORRECT1024 | 1024 | 均 PASS | 8/8 | 5/16，未达 13/16 | 96 ACCEPT、0 WARN、0 REJECT | 未运行 |

两臂所有已运行条件均保留 first-valid 结果、原分母和独立 CPU 复算。C0 的 DEV 不是 0/16。安全 gate 接受输出，不代表物体任务成功。未选择候选，也未使用新确认集来诊断或训练。

父模型原 CONFIRM_A 为20/32，未达26/32；12个失败中六个获取/搬运历史不完整、六个在后段放置/保持失败。其32条件和有效 FAIL 永久保留，已曝光确认集未用作本轮监督或新盲确认。本次接续不改变历史裁决。

实际闭合和原裁决：[owner 收据](OWNER_ACTUAL_COMPARISON_CLOSE_AND_NO_CANDIDATE.json)。科学终止收据 SHA256：`ae3e4d81072a1dc664b21cb908972e8185fb0fa8668e1ddb55c1f8e550e6bb1e`。

## 2. 学习对照实际检验了什么

两臂分别从同一完整 DOSE6144 / Adam6144 父模型恢复，保留 Adam，使用原 fresh cosine1024/warmup52、LR 5e-5/5e-6、micro4×accum2、native masked flow loss、EEF10/H40/K4、原 processor/statistics 和冻结范围。没有顺序 C0→C1 适应，没有新 loss 或精度试验。

C0 实际消费 8192 个专家 chunk；C1 消费 4096 专家＋4096 真实纠正 suffix。32 个固定纠正请求全部忠实恢复，28 个物理成功、27 个 codec 准入；四个物理失败和一个真实 SO3 codec 失败未进入正监督。历史零 preclip 附加门错误及 15/32 旧收据保留，V2/V3 的纠正和采样解释单独封存。

两臂五类行为曝光相同，为 `[2048,2048,2048,1638,410]`。2048 个对应训练微批的实际 noise/time 哈希全部相同。TRAIN8 的初始场景、真实双 RGB、state、语言，以及 640 个实际 noise/time/K4 包逐项一致；后续观测允许随政策改变。

配对 TRAIN8 中，两臂共同成功 2 条，C0 失败而 C1 成功 6 条，反方向 0 条。这支持本次固定 TRAIN 条件上的改善；它不是独立泛化或部署证据。旧父模型 TRAIN7/8、DEV13/16 使用不同噪声命名空间，不能将其与本轮分数差直接解释为严格配对的退化或收益。

C1 所有 2048 微批都混合两种数据来源。已保存 native loss 的 first/last16-update 均值约 0.132817/0.026623；没有单独保存的 per-source/per-example loss 无法从混合标量还原。32 个已保存梯度样本及实际 LR 已分析，没有新增梯度查询。

实际输入与随机配对：[训练消费](TWO_ARM_CLOSED_EXPOSURE_NOISE_IDENTITY.json)、[TRAIN8 输入/噪声](TWO_ARM_TRAIN8_INITIAL_NOISE_PAIRING.json)、[C1 曝光与 loss](C1_CLOSED_SOURCE_PHASE_LOSS.json)。

## 3. 真实物体任务失败

C0 的六个失败：两条缺少持续抓取/抬升历史，三条未进入目标区，一条到达目标区后仍抓持物体，未完成放置、释放和撤离保持。

C1 DEV 的原历史谓词漏斗为：16 请求→14 持续抓取→12 抓取且抬升→11 获取/搬运历史齐全→5 完整稳定放置、释放、撤离及任务成功。11 个失败中，2 条缺持续抓取、2 条缺持续抬升、1 条缺搬运历史、6 条未完成后段放置/保持。后段多个失败谓词会重叠，不能相加为独立原因。

保留关键反例：285200 最终 hold41/41，但缺抬升历史，仍 FAIL；285209 到目标区后又离开；285210 直到 29.84s 才进入 hold，仅通过 4/41，仍 FAIL。没有删除晚期失败或改任务标准。

区分生成参考、实际安装、同一已发命令后的 achieved TCP 和完整任务。例 285215 的参考→安装 XYZ RMS 为 0.015448m，安装→同命令 achieved RMS 为 0.006640m；这些量不能互相替代，也没有证明唯一神经或控制根因。师兄 WBC 未在本轮运行或修改。

完整逐条件及原谓词：[C0 失败](C0_ACTUAL_FAILURE_AND_DIAG_SUMMARY.json)、[C1 DEV 失败](C1_ACTUAL_DEV_FAILURE_FUNNEL.json)、[独立 CPU 原件](../evaluation/ACTUAL_CORE_SMALL_RECEIPTS_READBACK.json)。

## 4. 离线指标与闭环结果分开

| 固定 DIAG96 保存指标 | C0 first8 | C1 first8 | C0 H40 | C1 H40 |
|---|---:|---:|---:|---:|
| pooled XYZ RMS，m | 0.014747 | 0.017892 | 0.027544 | 0.030207 |
| pooled SO3 RMS，rad | 0.087603 | 0.095711 | 0.128302 | 0.144208 |
| mean gripper MAE | 0.047532 | 0.056448 | 0.017121 | 0.026856 |

C1 的固定 TRAIN 闭环成功改善，以上教师标签误差未降低；两个结果都保留，不能用一个替代另一个。C0 唯一 WARN 是角速度 2.1214rad/s 超过原 WARN2.0 门，非 REJECT。两臂 decode error 均为 0。没有修改统计量、decoder、gate 或历史 raw-target clipping 案例。

## 5. 完整模型、归档与实际视频

| 模型 | HF PUBLIC 固定 revision | Hopper 本地分类 |
|---|---|---|
| C0 | `b0b8c65de8d5bbdf0946ccdda9ecb23348cf16c9` | HF-only 完整 Trainer；本地仅 metadata，不能 resume |
| C1 | `d35dcda18def3d0691b55957ce461fe6b8449437` | 完整 Trainer＋固定 HF，继续保留本地源 |

PUBLIC repo：`leoyuer/pi0246-gate-pass-campaign`；各 prefix 为 `arms/PI0286/<arm>/checkpoint-1024`。每臂完整 14 文件约 20.47GB，含模型、Adam、scheduler、RNG、processor/statistics 和 Trainer 元数据；native CPU、上传 manifest、固定版本全字节读回逐项一致。Core 仅有 BF16 model-only 评价视图，不能充当 Trainer resume。[owner 独立库存核验](FINAL_MODEL_INVENTORY.json)。

C0 在完整读回后仅退役三个本地大文件，释放 allocated20,463,648,768B；小证据保留。一次本机归档收据跨臂目录复制错误已留 deviation、恢复正确原件并重新核验；远端原件未变，没有重训、重上传或重复模型读回。[修复原件](../archival/receipt_copy_deviation_20261009_1428/REPAIR_RESULT.json)。

新视频完整覆盖 C0 TRAIN8、C1 TRAIN8、C1 DEV16，共32条件，含全部失败。只编码已保存 BASE＋WRIST RGB，完整解码和8页目视核验；没有重新渲染或仿真。固定 HF 版本、成员哈希、观看地址见 [最终交付清单](FINAL_DELIVERY_MANIFEST.json)。早先48个 learner/teacher 视频保留其当时标签，不伪造后续训练状态。

## 6. 累计预算与工程处理

正式2048、canary32、技术完成更新0，合计2080；4092实际 query、16368 heads；81 charged source starts、53268 control、53267 RGB、266335 physics、106536 dispatch、15680 teacher。81 starts 包含原 learner285109 工程不完整及同条件替换，不是质量重试。预算直接来自闭合 ledger，不按 episode 数推算。

Hopper/Core 同设备 reservation/service 区间 union 分别为 5083.990s / 3147.902s；不重复相加嵌套 child 或 PBS walltime，也不当作 GPU busy 利用率。正式 job 均 F/Exit0。planner 调用、完整 CPU core-hours 和剩余 CPU额度未在原 ledger 完整记录，明确 UNOBSERVED，不冒称全部资源计量 PASS。[闭合预算](PI0286_CLOSED_FINAL_BUDGET.json)。

科学闭合后的 owned-storage 快照为70.73GiB，低于100GiB输出/cache cap；它不是个人磁盘 quota 或共享盘 free。新视频/小归档产生少量后续容量，另有带时间的实际收据。未大清理或触及其他项目。C1 同逻辑路径 SSD transfer/materialization176.786s，C0 NFS702.947s；工程提速不改变模型和评价契约。

## 7. 下一研究判断与停止边界

本次有限对照得到真实负资格结果，同时发现 TRAIN 改善未转化为原 DEV 门通过。下一个尚需单独授权的问题应明确纠正覆盖、获取历史和后段放置/保持的迁移能力，并保留等预算对照及真正未曝光确认。现有结果没有证明应改 WBC/gains、换 normalizer 或直接再加步数。

后续 T1/T2 共享方案必须单独冻结物体任务数据、EEF/夹爪监督、对照、累计预算和两个任务各自的完整闭环成功标准；不能以混合平均分掩盖 T1 失败。当前不启动共享训练或新增 T2 模型查询。

本轮不再启动训练、教师、模型/physics 查询、其他候选确认或 T2/shared 优化。源 Panda T1 PLACE_V2、非实时；不是 Go2/WBC、本体迁移或真机成功。正式 selected/deployment=NONE。详细技术表和全部负结果见 [完整比较](SCIENTIFIC_COMPARISON_DRAFT.md)。

归档后补记：55 文件、43,196,338B 的 endpoint 包在固定 PUBLIC revision `8927fcf7c6bbdae5b8f85531162298c51fd9edba` 匿名全文件字节回读 PASS。Owner 再核验了14个本机交付成员、45份原 source/CPU 小收据及两臂各14个 Trainer 文件身份。新视频/归档后跨主机 owned-storage 为70.8043GiB，个人 quota 仍 UNKNOWN。[固定版本证据和视频](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/8927fcf7c6bbdae5b8f85531162298c51fd9edba/evidence/PI0286/recovery-vs-replay-20261009/endpoint-closed)。
