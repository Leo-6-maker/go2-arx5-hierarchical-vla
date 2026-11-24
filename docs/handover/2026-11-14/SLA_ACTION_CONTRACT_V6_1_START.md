# V6.1 执行启动 prompt：动作语义、双 mask、原生回放与小窗口拟合

日期：2026-11-14（Asia/Singapore）。本文件是执行交接，不是已启动作业。

先读 [详细计划](../../protocols/SLA_ACTION_CONTRACT_REPLAY_V6_1_20261114.md) 与 [机读矩阵](../../protocols/SLA_ACTION_CONTRACT_REPLAY_V6_1_20261114.json)，再读 V5 长程计划、仓库工作流与现场最新状态。

## 可以直接交给执行端

继续 `Leo-6-maker/go2-arx5-hierarchical-vla`，本轮先交付标签经学生部署链的原生回放，以及覆盖阶段和重规划接缝的有限拟合。不要再建立一条替代在途 V6 的训练主线，不要只扩写计划。

1. 核对 PR #384 及后继的真实训练/原生/队列状态。计划固定阅读 SHA `4273dcdf143a0f9bac28afc5f831eb597a240412`，V6快照为2026-11-14 12:25:57 SGT：92更新/2944访问，不是现在的实时状态。V6原3K/96K、数据/统计/输入/consumer/评测保持不变。旧N20的0/8和所有负结果保留。未知H200作业不盲重投，健康Core V6/SF/BOOK不抢占、不热改。

2. 将 `eef_stream_v2.expert_program`（EEF_STREAM_V2）的重构明确归于专家规划。捕获函数返回的 `program.decoded_H40` 作为该query最终计划及其版本/hash；不要把重构前proposal作为计划监督，也不要让学生raw入口调用专家重构。下游复用同一 `FiftyMsProgram`、certifier、参考安装、TTL与P10消费，不复制一个近似adapter。

3. 新数据sidecar分别保存 `supervision_mask[40]` 与 `consumed_mask[40]`。前者控制新计划监督，后者只按原query/command/tick的实际完整消费确认填写。计划40点/实际P10时是40个监督true、10个消费true；后缀不能作为实际执行、成功或动力学转移证据。TTL续期不增加消费点，后来query相同数值不回填旧query。旧V6 schema、标签、统计不原地添加语义。

4. 完整新验收顺序固定为：统一语义并划分episode → 拟合/冻结TRAIN统计 → 编码/归一化/pad32/FP32序列化 → 部署反归一化/夹爪解码/frame转换 → 完整原native检查 → 原生回放 → 新训练。V6已有1220/1220及FP32检查直接复用；不在小窗口重新拟合normalizer。只验证归一化前数组不能算通过。

5. 从V6 N20 TRAIN固定S16：五阶段各2窗口＋6个不同重规划/事件接缝窗口，尽量跨4 episode/4布局，模型输出前封存。保持全N20 TRAIN统计。做RP8：五阶段各1＋三接缝条件。使用专家原前缀/完整控制状态恢复到合法anchor，经过学生文件/codec/raw compiler注入FP32往返标签，完整H40检查、实际P10。qpos/qvel单独恢复不代表WBC状态恢复；不匹配就明确失败，不偷偷重锚或筛选重试。oracle结果不计学生能力。

6. RP8支持执行合同后，以独立 `MICROFIT_S16_REFERENCE500` 从SLA_base训练500步/B32=16000访问，沿现有LoRA/500步/25warmup/highest/micro8×4配方，FM-only，保持25% token dropout且实际anchor不变。每batch16窗口各2独立noise。固定0/100/500，终点500，不在DEV挑checkpoint。16窗口×2noise检查32个raw H40；报告XYZ/SO3/grip与速度/加速度/准入，目标32/32合法。然后固定4个TRAIN全任务诊断，不用Teacher替代模型，不把拟合当泛化。

7. 在途V6继续原独立TRAIN8/DEV48。需要后继SF时复用身份完全相同的V6 FM端点，只增加同数据/初始公共权重/访问/统计/consumer的空间臂。旧2EP SF不当作20EP匹配对照。参考误差输入、历史、EMA、平滑损失等都不是本轮在途run的改动。

8. 计划蒸馏PD1是条件性独立研究。最终计划与mask记录准备好后先4条、再20条完整新episode（8布局配额3/3/3/3/2/2/2/2，最多40尝试）。同episode导出真实消费参考与最终计划两个视图，各自冻结正确TRAIN统计，全链验收后各500步FM对照；不要用PD新观察直接对比旧N20便归因监督表示。无完整历史final plan就新采，不用当前planner补写旧计划。部分mask需真正进入新loss，并验证无效标签不会通过flow输入污染有效梯度；旧loss不热改。

9. LeRobot导出后置，不阻塞回放/拟合/在途V6。匹配SLA锁定依赖，以同一S16在两条reader上输出相同RGB/mask/state/tokens/actions/dual-mask为准。多频率通过真实timestamp映射，不把2Hz锚点表当20Hz action行，不重复帧伪造观测。保留高效分片生产路径。

10. Corelab沿已有独立多世界授权与当前4-world配置；RP8先1个状态诊断世界，确认独立控制历史后再并行。GPU只用真实许可allocation，参数/Adam/梯度常驻，有界缓存和异步writer。元数据identity不导入SciPy/JAX/Torch/Flax。只做变化接缝测试，不重跑相机/GR00T/WBC旧资格，不新增重复调度器或审批系统。

本轮新增必需预算：500候选更新/16000访问、RP8＋学生TRAIN4共12原生诊断。可选同数据SF与PD两臂全跑时，候选总4500/144000，mask资格最多8/256另计；并非自动排满。已运行V6及其评测单独保留，不重复累计。

下一次回报必须有：V6真实状态、RP8逐条件结果或精确首失败、S16实际更新和32条预测合法性、原生学生结果、双mask/完整episode/尝试分母、数据/统计/模型/adapter身份、实际CPU/GPU成本。不能用文件数或排队代替实验结果。

## 静态检查命令（不是训练CLI）

```bash
python -m json.tool docs/protocols/SLA_ACTION_CONTRACT_REPLAY_V6_1_20261114.json >/dev/null
git diff --check
```

具体运行命令从现役代码/已有profile及新增实现读取，不能把计划中的拟议脚本名当作已经存在的launcher。遵循现有所有权和普通PR工作流，不force、不自动merge、不更改真机/WBC安全范围。
