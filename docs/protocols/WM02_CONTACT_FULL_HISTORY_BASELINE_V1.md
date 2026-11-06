# WM02：完整历史接触事件基线 V1

2026-11-06。独立、有限CPU开发诊断；实现已完成，本文不表示实际拟合或派发已经发生。原 [90项FIR协议](WM02_CONTACT_TIMING_BASELINE_V1.md)、模型与结果保持不变。

新变体保留原神经replay的九行×103项，即原顺序的 **927项实际观察历史**，包括原生事件和夹爪内部状态；不把部分状态称作完整仿真器或纯视觉输入。每个H1/4/8/16/32/64头只追加本时域内的绝对opening程序前缀，输入维度为927+h。AC读取预声明程序；NO_FUTURE_PROGRAM重复anchor实际 `executed_reference_EEF10[-1]`。replay须将该值从原始anchor行逐行绑定到 `blind_plan`，模型拒绝非恒定或与该anchor值不符的盲计划。候选未来程序不得改变盲臂的特征或基底。

旧FIR90只用历史子集和命令增量，其盲臂重复内部 `memory.opening`；这与原神经replay的当前执行参考不是同一来源，不能默认相等。新变体使用原神经历史和盲计划语义，但线性架构、事件目标、概率链接及每头有效样本归一化仍不同于神经模型。因此不将比较解释为仅架构改变，也不重写旧“同信息”实验的数值。历史仍为原样观察，不额外添加phase、episode绝对时钟、parent、goal或close-start；未来观测与超出本头时域的程序后缀不能进入输入。

信息合同保持 `PREDECLARED_PROGRAM_CONDITIONAL`：静态程序在anchor8已知；已选择Policy程序须有实际决策hash且只在anchor111后可知。未来程序尚未安装/消费，`INSTALLED_REFERENCE_ONLY=NOT_RUN`。概率线性输出明确裁至[0,1]并统计原始越界数；不裁动作、参考或观察状态。

拟合数据固定为原C1实际TRAIN manifest，SHA256 **`4409578eb684119319ecea6ace03d20c70373859fac76866e27ef0c9b2b31dcd`**：28个episode、8个物理父；仅原TRAIN close56/64，close72不拟合。按原descriptors重建纯TRAIN，保持所有原角色。每臂、每头只用自己有效TRAIN标签行拟合均值/尺度、事件均值基底和固定 **ridge=1**；不搜索ridge，不读DEV进行选择。两臂六头共12次闭式求解、2个最终模型、0次优化器更新。小样本dual与实际数据primal为同一个ridge目标，实际求解次数单独记录，不称神经训练。

两臂必须全部拟合、保存并精确重载所有数组及TRAIN预测，封存新 `TRAIN_FROZEN` 后，才能读取完整matched包、评价bundle、旧神经或FIR权重。新NPZ、freeze和ACTUAL使用独立FULL927 schema，不能与旧FIR互载。旧FIR比较须核原freeze的TRAIN manifest、重建replay、身份列表、recipe、ridge及两臂模型hash与新拟合输入一致；只加载，不重新拟合旧模型。

全部评价为已曝光的回顾开发诊断：C09_COMMON36、C10_TIMING12、原TRAIN72 holdout与DEV close72，不是独立TEST；两个cohort重叠两条episode，不合计为48条独立episode。保留全部12个固定神经比较器：PREFIX36三seed×两臂的较少训练暴露须标记，MIXED46三seed×两臂与28条TRAIN匹配；另保留原两臂FIR90。不得挑seed或只报告赢的时域。报告四事件、各时域/父/时序层的Brier、误报/漏检、概率越界及原六时域网格转变时间；网格时间不是原生first-passage，重叠anchor不是独立试验，漏检不记作零MAE。

实际入口已实现：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$WM02_NATIVE_PYTHON" -B -m scripts.wm02.pilot1_neural_worker --spec "$FULL927_SPEC"
# 包装器内部调用以下入口；该入口仍检查真实admission：
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$WM02_NATIVE_PYTHON" -B -m scripts.wm02.fit_contact_timing_full_history --spec "$FULL927_SPEC" --output "$FULL927_ROOT/fit-01"
```

除原owner/registry/lease、有限deadline及原C1全部输入pins外，新spec必须固定以下字段：

| 字段 | 值 |
|---|---|
| execution_site | `CORE_FUSION_CONTACT_FULL_HISTORY_CPU` |
| profile | `wm02-fusion-contact-full-history-offline-cpu` |
| kind | `FUSION_CONTACT_TIMING_FULL_HISTORY_BASELINE` |
| recipe | `WM02_CONTACT_TIMING_FULL927_MATCHED_TRAIN_RIDGE_V1` |
| feature_recipe | `FULL_927_HISTORY_WITH_WITHIN_H_PROGRAM_PREFIX_V1` |
| blind_source | `ANCHOR_ACTUAL_EXECUTED_REFERENCE_EEF10_LAST_GRIPPER_COMPONENT` |
| observed_history_features / ridge | `927` / `1.0` |
| numeric_mode / device / physics | `NUMPY_FLOAT64` / `cpu` / `false` |
| arms | `[ACTION_CONDITIONED, NO_FUTURE_PROGRAM]` |
| information | `PREDECLARED_PROGRAM_CONDITIONAL` |
| GR00T_updates / WBC_changes | `0` / `false` |

`frozen_fir`须包含 `freeze:{path,sha256}` 与恰好两项 `models:[{arm,path,sha256},…]`；path绑定实际封存C1产物，不能以当前新输出冒充旧模型。原freeze SHA为 `345dddd31e2cd6e8cbf200b0dda011d072a1e30c9dbd45d75ecb645e4db27e5b`；AC模型SHA为 `4938b946ce70e0f2373958d05817b582a2bf7acfa6605d15a4043d48ad928416`，盲臂为 `e19ca6281591aaed9af84e21c160ab84838ef8fed50e493384c30cf4f4975323`。其余必需输入为TRAIN manifest/path/hash、`expected_TRAIN_episodes=28`、`expected_TRAIN_parents=8`、matched_training_package、两个evaluation_bundles和原12项frozen_models。

实际运行须登记fresh role/root和新的完整source delta，保留原FUSION owner、Journal/once、registry CAS和父deadline reserve；原192面板、两个7c CPU bridge、C1和成功assets02须先核MainPID0/CLOSED及各自封存记录。C1/assets02还须原worker退出0及结果身份精确匹配。旧lease/root/capsule均不复用或热改，未知/已有身份先reconcile，不自动重试。拟定CPU12–15、BLAS四线程受四CPU约束、1200s工作端点加60s宽限、6GiB上限，CUDA为空；条件不满足时无派发效果。

新增物理积分、真实转移、VLA查询、GR00T/Actor/WBC更新均为0；不评价自由滚动、原生控制收益或任务迁移。所有结果和失败均封存，后续实际报告另行记录真实成本、闭式求解数及精确恢复误差。
