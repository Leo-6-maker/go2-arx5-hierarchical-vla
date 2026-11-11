# 原生相机／公开 ARX5 首轮：实现与执行交接

## 任务身份

协议：`NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1`。先读 [完整协议](../../protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md)、[JSON规格](../../../experiments/native_arx5_transfer/native_camera_public_arx5_transfer_v1.json)、[条件顺序](../../../experiments/native_arx5_transfer/dev_panel_v1.json)。

用户已要求详细方案和GitHub新分支；本次发布只完成仓库整理和协议，不是现场运行回执。后继执行遵守用户已有授权、当前GPU／物理owner与活动作业状态。不要请求重复的日常工程GO，也不要把本handoff当成虚构的现场授权或已经通过的资格。

## 1. 从哪个代码版本开始

main的文档整理merge为 `6843163d641eb24a29c3b1557f868765346ba66c`。参考执行源码：

`f0e19c8812896f98053e74b165d6c52e85327b69`

该执行快照相对整理前main含730个后继commit、1,522个blob变化，来自#371及其祖先分支；本次没有把整栈合并main。

实现负责人应在独立checkout/worktree建立本轮实现分支，以固定执行快照复用功能，并带入本协议文件／commit，记录实际基线及最小diff。文档分支继续作为协议来源。之后若集成main，使用单独正常PR明确执行依赖，不force、不重写历史、不在活动胶囊内pull/checkout/hotpatch。不要直接运行旧recipe期待新parent/source自动适用。

## 2. 最少必要的实现工作包

| 工作包 | 复用位置（均在固定执行SHA） | 本轮需要新增／适配 | 完成证据 |
|---|---|---|---|
| W1 body观测版本 | fusion_source_camera_capture / fusion_paused_observation | 固定BASE安装、真实state渲染、副本隔离、body映射external_rgb | 覆盖报告＋100/1450不变量 |
| W2 source数据 | 原native_dataset的Processor合同／SO(3)工具 | Table30 metadata reader、真实时钟、RPY/TCP／反馈语义、source width(m)、episode split／H40 | source admission与完整分母 |
| W3 官方父权重profile | raw_parent_delivery / minimal_model / minimal_adapter | 新官方父身份，保留目标category10、七接口＋DiT末两层，移除旧SFT父身份的隐式依赖而不改历史profile | 参数／frozen指纹＋实际loader回执 |
| W4 cache与训练 | task_learning_features / core_local / gpu_rows | 新相机cache、source可变N、500/25新日程、三阶段完整保存／恢复、target配对trace | 匹配更新、GPU数据与普通推理 |
| W5 原生评价 | endpoint_basis / endpoint_policy / matched_native | 新DEV注册与80条件槽、8→40门、全终态和阶段分表 | 原评分／消费trace与报告 |

这些是实现任务清单，不是已经存在的新CLI。新launcher与实际命令行由实现PR登记并现场验证。保留旧模块的历史身份，不通过放松旧校验让新实验冒充旧C500或CORR配置。

## 3. 第一批只读事实核对

集中核对一次相关版本，不扫描整台主机或全历史：

- 当前Core所有权、用户自己的训练／物理job是否已经终态、GPU与节点本地存储可用性；
- 参考执行源码与实际checkout差异、已存在的官方七文件可信audit；
- TRAIN100实际manifest与全部capture／监督窗口、原wrist、状态／时钟、Teacher消费证明；
- 已预览相机的fovy口径与安装版本，现有SFT2000端点和其Processor；
- 已在允许存储中的Table30同pin文件及有限获取成本；
- 旧人口可隔离的输入身份manifest，不读取TEST结果；
- 用户Teacher24最终19/24的完整终态回执是否已另存；找到则登记，不重跑补成更好结果。

只缺某个来源回执时先补对应事实，不反复重做与缺口无关的训练或全历史资格。

## 4. 具体推进顺序

### P1：相机与目标数据

固定body=[0.25,-0.24,0.45]m，看向BASE=[0.38,0,0.34]m，BASE+Z，verticalfovy75°，原腕部配置。对1450监督窗口与所有capture做离线覆盖；原图和Processor后都逐阶段检查。目标geometric mask只作审计。

通过后生成新body图、复用原wrist；100episode、1450window、状态／action／mask／stats不变。缺图修复或明确停在数据准备，不删除。重渲染不增加原物理轨迹的样本数。

### P2：公开数据，可与P1并行CPU准备

先有限的任务／archive成员清单，再split，再窗口。公开README没有充分解决EEF是否actual feedback、RPY/TCP和时钟；这些必须由具体episode／采集端证据补齐。若只有setpoint且无可信实际joint/FK反馈，不能标成ACHIEVED。

源gripper保留m；source Processor自己统计，不转换为target nominal。t0…1950ms；视频PTS或可信frame-to-state映射；无跨episode/外推/末帧填充。源统计与采样按协议固定。

### P3：早期反馈与新接口资格

相机通过后做SFT2000的TRAIN01/07/13/25各一次整机诊断；先确认原25布局manifest的ID映射。保持其原Processor，不拿4项当A/B。

完成官方起点白名单、原路径与cache匹配、source真实短批、GPU显存／吞吐、fresh-process恢复、普通推理重建。资格更新独立，不加入候选。

### P4：有限训练

默认A_TARGET500→B_SOURCE500→B_TARGET500；目标阶段seed20261111、同16,000访问／噪声trace、重置optimizer/scheduler/RNG/cursor。B只承接source白名单权重。三阶段共1500optimizer calls／48,000访问；真实次数另报。保存250/500；仅500主评价。

新scheduler明确500/25、cosine到0、首个使用LR=4e-6，不沿旧helper的8000/200或0.1floor。执行前生成LR数组hash核对。

### P5/P6：新DEV

绝对scene原点、物理身份／旧人口重叠、原timeout/query上限／评分和新endpoint资格未封存前不开新DEV。按JSON固定顺序先16请求，不能依据A结果调整B。

B至少多2/8且无新增硬安全／技术问题、无未解决技术问题才激活其余64请求。全部结果保留；首8计入40不重跑；Stage1、Stage2、合并分别报告。原拒绝算策略失败，技术与未启动分别计数。

## 5. 不能作为通过的证据

- 一个相机内存预览，不能替代全TRAIN覆盖。
- hash字段填写完，不能替代实际文件身份。
- 原CORR/SFT旧资格，不能替代新官方起点／Processor边界检查。
- prefix冻结，不代表可缓存DiT前30层输出或用no_grad阻断接口梯度。
- 末点basis平滑，不代表原始H40已执行或一定可行。
- source loss下降，不能替代target整机成绩。
- 新DEV名字不重复，不代表其物理场景独立。
- 计划合并main，不表示模型已训练或运行checkout已更新。

## 6. 最终交付

按 [报告模板](../../templates/NATIVE_ARX5_TRANSFER_REPORT_TEMPLATE.md) 发布QUESTION/CHANGED/ACTUAL/NEGATIVE/DECISION/NEXT，附相机、target/source manifest、参数/cache资格、A/B端点引用、全部原生请求与资源成本。私有payload留在项目允许存储，只向公开GitHub推去敏摘要、代码和可公开manifest。

未通过扩样门时封存首轮，给出失败谱系与下一项有依据的独立实验。不要自动追加同分布更新、public replay、WM、视觉解冻或大卡作业。
