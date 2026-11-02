# 四任务raw共享基线：数据冻结与采样实现

依据[PR317第5节](../protocols/CONTROLLED_FINETUNING_AND_POLICY_DELIVERY_20261102.md)。首版使用原raw GR00T、M32、单帧、原masked flow matching和连续夹爪；不等待所有单书机制支线或BOOK达到80%。当前实现属于CPU数据/配方准备，**不是已运行共享训练或新任务结果**。

## 数据与身份

实际source07共同数据为46条完整专家episode＋13条专家后缀／34563窗。旧共同manifest仍是官方处理合格的候选，`training_admitted=false`。`shared_task_data.freeze`在现有owner/Journal和容量核实后，核对该manifest、原官方全部窗口/真实RGB资格、原TRAIN统计及既有DEV定义摘要，将相同数值字节复制到一个新版本，保留原raw RGB引用。新版本只开放TRAIN reader，不放行模型/GPU；原候选、D0/D1/D2和统计不修改，不重新拟合统计。

新版本提供`TRAIN_POPULATION.json`、`TRAIN_DEV_SEAL.json`和完整`TASK_DEV_POPULATION.json`：原T1/T2各24及DOOR10定义与另外封存的BOOK10合并绑定。原文件分别保存在`SOURCE_TASK_DEV_REFERENCE.json`及`BOOK_ORIGINAL_TRAIN_DEV_SEAL.json`，不改条件或噪声。完整轨迹和专家后缀分别计数，不能当59个独立完整场景。每个新数值文件和统计/DEV副本均核原SHA，全部完成后才发布新manifest；中断残留不被当成有效数据集。

## 实际曝光与恢复

`shared_task_sampling.TaskBalancedSampler`与原NativeDataset的(cursor,index)及`make_sampler`接口一致。每批32固定T1/T2/BOOK/DOOR各8，任务内以episode循环排列均衡，episode内以窗口无放回循环覆盖；不将BOOK阶段强套给推物或柜门，也不向模型提供阶段/接触/Teacher元数据。采样使用独立PCG64流，不消耗模型全局NumPy随机状态。

实际59条目CPU枚举262144次访问，每任务65536次，全部34563窗至少被访问。10步资格与完整日程前320次相同；游标1、319、32768、65537、131072、196608、262144恢复检查通过。顺序绑定manifest、原数值SHA、row索引及全序列摘要；调整元数据顺序会改变恢复身份，不能沿用旧游标假称相同输入。

## 配方与尚未准入的执行

`recipe`固定raw父revision/7文件身份、目标category10及末两层DiT(30/31，77847684参数)、有效batch32(micro1×accum32)、8192日程/warmup256、LR1e-4、WD1e-5，保存2048/4096/6144/8192。资格10步沿用同8192学习率和采样前缀，正式实验fresh从raw开始，不接续资格权重。16个固定TRAIN诊断输入应四任务各4，在零更新及四个保存点共80次生成；这仍不是闭环成绩。

该配方不同于4096步book日程，8192中的4096点不是原4096终点的等价比较。旧E1/E2、b248及原DEC1不改。新recipe未接入现役book启动器，不会因数据reader开放自动启动训练。仍待完成Core原raw交付/绑定、零更新模型结果、实际训练器接线、原生前后向/完整状态恢复与设备容量、四任务评价入口。固定模型和条件后再进入实际共享学习与逐任务评价。

[真实source59完整CPU曝光](../../reports/PI0294/design/controlled-finetuning-20261102/SHARED4_SOURCE59_TASK8_CPU_EXPOSURE_ACTUAL_20261102.json)

## Core实际采用与DEV引用修正（22:40 SGT）

原raw七文件已在Core节点本地完成全流校验和独立stat身份复核。新共享59数据reader以原NativeDataset的for_optimization=True实际构造；6377253数值字节逐份SHA一致，统计/RGB原引用保持，原官方资格按相同字节复用，不声称重新运行Processor。Core NumPy1.26.4无Torch/CUDA完成全部采样，与PC序列一致；子进程OS0且回收，零模型/更新/物理。

首次冻结发现原共同DEV文件只有T1/T2各24与DOOR10，BOOK10存在另一份原封存TRAIN_DEV文件。共享模型尚未启动；保留初版、费用和结果，在新独立V2中补齐BOOK原条件/噪声，完整绑定68个既有DEV定义。新数据schema为V2；初版不作为共享模型输入绑定。DEV定义完整不等于各条件原生可执行或模型成功，未合格候选和旧STOP仍保留。

当前D2继续392/4096、12544访问。raw零步模型、训练器/完整checkpoint接线、原生10步和GPU容量、四任务政策评价入口仍待完成；数据采用不会自动启动共享训练。

[Core实际交付、V2采用、修正成本与未完成项](../../reports/PI0294/design/controlled-finetuning-20261102/SHARED4_CORE_RAW_INPUTS_AND_FROZEN59_ACTUAL_20261102.json)
