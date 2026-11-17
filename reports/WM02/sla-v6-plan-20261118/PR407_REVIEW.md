# PR407审核：已完成的证据、仍未识别的因果与下一研究边界

日期：2026-11-18（SGT）。固定审阅PR407 `1d78f2b758fb9b0dc5bbc05ecfc5a0aa3b1e63d3`；V5.2终态 `14886acdf739484cac914578e72736e63cedef62`。本次是GitHub报告/源码审阅及算术核对，不是SSH现场核查、原始HDF5独立重读或模型复跑。未修改既有报告、执行代码或安全规则。

## 1. 审核结论

V5.2终态可信地支持：当前体系能够重复完成移动抓持，但B0/B1/B2均未满足既定晋级要求。RealMix V1停止正确；其结论应严格限定为“尚不能进入当前command＋SHIFT7混合协议”，不是“公开真实数据无训练价值”。

| 模型 | F1 | F2 | 综合 | H40拒绝 | 接触STOP | 技术失败 |
|---|---|---|---|---|---|---|
| B0 |2/4|3/4|5/8|2|1|0|
| B1 |3/4|3/4|6/8|2|0|0|
| B2 |2/4|4/4|6/8|1|1|0|

B2对B1：一个条件胜（F2/T3）、一个条件负（F1/T2），总成功差0；F1/T3两者均失败，但B1为拒绝、B2为碰撞。B1对B0有两胜一负，总差+1。n=8、一次训练seed、post-init全控制状态不完全一致，不能宣称显著性或普遍优越。B2在本轮F2较好，但不能把旧F2 0/4变为新4/4的差值全部归因于训练，因为冻结B0新面板已3/4。

计量复核：`24*5703 + 13773*25 + 36 = 481233`；旧2接口11906 tick未重复计入。1379/1384=99.6387%的query通过完整H40，但综合任务只有17/24；该query率是混合模型、不同停止长度的汇总，不能当独立风险概率推算长任务可靠性。STOP后0积分是本轮仿真trace事实，不是最坏墙钟停机或真实硬件认证。

## 2. 目前最有信息量的执行缺口

5/7失败在安装之前被完整H40拒绝，4个首端、1个内部。开发父P=B1的两条拒绝分别为F1/T3首端0.279878684、F2/T3首端0.253554552m/s²；原门0.25。B2/F1/T2的内部k34为0.261575275，首尾合法。它表明“clean proxy上的端点辅助项”不等于“完整生成轨迹的全段约束”。

B1没有观测到碰撞，不等于证明它没有接触覆盖缺口：两条拒绝提前终止，后续行为属于右删失/未观测。原E3未触发是当前协议的合理决定，不应被写成永远不需要纠正数据。也不能拿历史专家近邻充当这些学生状态的精确反事实。

因此，只保留一组有明确终点的B1全轨迹约束对照；它不是整个项目的永久主线。源域学习与多任务准备不以它全部成功为前提。

## 3. 源数据审计：三个结论必须分开

首12条、12585帧、六类字段逐值满足action[t]=state[t+1]，末action复制末state。字段行数`(12585-12)*6=75438`一致。该观测强烈支持“移位记录状态标签”的解释，但没有采集器证据不能断言state一定是已实现物理反馈、action一定从未发出，亦不能假定硬件同步已验证。

这足以拒绝把它当作“已有dispatch/ACK的目标命令”。但对于新的source-only预测问题，可以如实定义监督为FUTURE_LOGGED_STATE_REFERENCE：从当前记录RGB/state预测未来记录的pose/signal，不伪造命令履历。未知绝对TCP、未知Go2标定限制物理迁移与单位主张，不必永久阻止在一致源坐标中进行有边界的离线学习。

未来标签与未来输入不是一回事。未来记录可以是监督，输入仍只能取anchor及更早信息。raw action已比同索引state超前一帧；必须保留这一offset，禁止再shift一次。末尾复制不是新观测：构造未来窗口时排除越界复制标签。

## 4. 表示审计中此前摘要遗漏的关键分母

条件为记录夹爪标量确可解释为归一化开度，并按0.002+0.040*g映射目标等效宽度；它不是已测源毫米。

| 条件性数值诊断 | ZOH | 线性敏感性 |
|---|---:|---:|
| 全部双臂重叠窗口 |1568|1568|
| 动态窗口 |661|663|
| 动态窗口必要下界失败 |521（78.82%）|506（76.32%）|
| 单调性下界失败 |134|139|
| 速度下界失败 |512|492|
| 合并下界失败 |522|506|
| EEF内部加速度违规 |1076|1074|

ZOH的134+512−124重叠=522。源EEF任一违规1078中1076包含内部违规；不能把来源冲突主要解释成首尾补零问题。线性诊断未消除现象，但单位/时钟/TCP仍未认证，所以不能宣称测得真实硬件违反某速度限制。

`necessary_width_error_bounds`对自由输出对使用半差下界、固定context使用全差下界，符合最大范数必要条件的逻辑。通过该下界并不证明属于SHIFT7的更小函数族。当前结论不是泛化的“标签坏了”，而是“在明确条件下，大量关键动态行为不能被当前单事件/限速族忠实表达”。

因此不继续下载48条来碰运气通过旧表示，不只保留静止窗口，不调宽0.2mm容限，不把新decoder伪装成旧配比变量。

## 5. 对上一计划的修正

PR406的0/10/25%比较保留为BLOCKED，不自动续接。下一个V6把两类任务分开：

1. B1上的一次全H40约束实验，检验已观察到的执行缺口。
2. 真实数据在其原生双臂/原始频率下的未来记录参考学习，不要求它先具备Go2命令资格；之后用共享LoRA参数初始化迁移，而不是直接拼接标签。
3. 在新的、逐时刻夹爪表示中学习完整PICK_HOLD/PICK_PLACE和不同对象/目的地。这个科学变更必须另有版本、共同新统计和匹配对照，不复用B2为新方法的“等算法Real0”。

本体无关的强主张另行验证：共享高层权重冻结、换执行器或本体；V6的目标域后训练不冒称零目标数据迁移。WBC独立参考跟踪与更换执行器的研究不得永远等待单任务刷满，但本次不更改WBC或重训低层。

## 6. 固定来源

- E2终态：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/14886acdf739484cac914578e72736e63cedef62/reports/WM02/sla-mobile-f2-v52-20261117/E2_FINAL_ACTUAL.md
- E2拒绝数值：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/14886acdf739484cac914578e72736e63cedef62/reports/WM02/sla-mobile-f2-v52-20261117/E2_FINAL_REJECTIONS.json
- 源语义：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1d78f2b758fb9b0dc5bbc05ecfc5a0aa3b1e63d3/reports/WM02/sla-robodojo-real-sim-v1-20261118/SOURCE_SEMANTICS.md
- 表示机读结果：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1d78f2b758fb9b0dc5bbc05ecfc5a0aa3b1e63d3/reports/WM02/sla-robodojo-real-sim-v1-20261118/CODEC_COMPATIBILITY.json
- 审计实现：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1d78f2b758fb9b0dc5bbc05ecfc5a0aa3b1e63d3/scripts/sla/robodojo_semantics_probe_v1.py
- 已有通用逐时刻夹爪codec：https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/1d78f2b758fb9b0dc5bbc05ecfc5a0aa3b1e63d3/src/go2arx5_vla/sla_adapter/gripper_v2.py
- 上游loader仅加载原字段，不证明生产者：https://github.com/XPolicyLab/XPolicyLab/blob/main/utils/data_loader.py （此次返回blob 1a794dc09cc365b6cf4c8a4c7a8806a988befaec）
- 官方SLA模型学习未来动作序列的FM实现：https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/models/pi0.py
- UMI的手持示范/相对轨迹接口提供“没有机器人dispatch也可学习轨迹”的方法先例，不证明本RoboDojo数据已标定：https://umi-gripper.github.io/
