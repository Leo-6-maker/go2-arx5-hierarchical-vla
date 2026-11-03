# WM02 来源审计与代码适配图

日期2026-11-04。以下区分本次读取的事实、设计选择与待现场确认事项；不是模型或原生运行验证。

## 固定来源

- 本项目main：`0a27753bbda8ed75ab6b74aeb64755e0427362c7`，tree `90baeb7eb2e46c5227993769905211d1d2896100`。
- WM01：`de64c252246e4746315f2a324535b5d48912783c`，PR342在本次读取时open/unmerged。WM02从main独立新增文件，不要求先合并旧计划。
- 上游：`emoPointer/supermimic_manipulation@eda7da480c6d92a68cc41c9628b7efdae08cae23`；`docs/supermimic.md` blob `8797e8ef6c433e3a1cd72bc6e1c195dcd9683583`。
- 本项目 `AGENTS.md` blob `cb7d85b78e3025255a283b76ffbfb5e65a77e3a7`；目标环境实现blob `cb125db60f3b26956ffb85065d7370a5b4802954`。

来源链接使用上述不可变提交；若执行时上游变化，不能静默替换。先评估diff并另固定版本。

## 已核实事实与迁移含义

上游说明其算法来自第三方 `supermimic_vibed`，不是SuperMimic原论文官方操作复现。其Actor/Critic、WM、reset/reward适配分别存在；WM预测约化物理状态而不是生成视频。BoxInBin配置说明MLP为5×2048，约17.2M；其Actor读取关节与物体真值。上游还使用任务特定的想象奖励代理，因此不能照搬后声称与PPO同奖励。[算法说明](https://github.com/emoPointer/supermimic_manipulation/blob/eda7da480c6d92a68cc41c9628b7efdae08cae23/docs/supermimic.md)。

本项目 `target_sim/environment.py::build_native` 导入Isaac Gym、私有WBCEnv与HistoryWrapper；说明目标后端必须现场确认，不应一概称MuJoCo。当前公开源码不是私有控制器内部实现的完整审计。[源码](../../src/go2arx5_vla/target_sim/environment.py)。

本项目 `AGENTS`/V7允许同范围工程自主修复并保留目标S1/T2和实机限制；已有报告状态不是实时服务器观测。WM02不借用旧运行额度/时钟或推断当前GPU空闲。[规则](../../AGENTS.md)。

## 复用与新增

| 已有位置 | WM02用途 | 禁止替代/尚需工作 |
|---|---|---|
| 上游 `src/clab/algorithms/supermimic/networks.py` | 增量MLP、归一化、Actor/Critic基础 | 再接目标schema/history；不能直接用OSC权重 |
| 上游 `agent.py` | WM多步loss、冻结参数保留输入梯度、想象更新 | 改为400ms Actor/50ms WM；加入真实critic校准；明确方法变体 |
| 上游 `buffer.py`、`runner.py` | 序列replay与采样/学习分离参考 | 不复制整个调度器；恢复需包含replay与当前项目控制状态 |
| 上游 `tasks/supermimic.py`、`tasks/*/supermimic.py` | task adapter与代理奖励的边界参考 | 改成WBC-native，不引入其OSC硬假设和不同安全阈值 |
| 上游 reset 实现 | 物理有效中途状态理念 | 目标reset必须在冻结WBC下重建验证；不直接加载原bank |
| 本项目 `target_sim/environment.py`、`bridge.py` | 已有目标后端与外部参考安装 | 绑定真实部署版本；测试preview/history，不改控制律 |
| 本项目 `target_mujoco/` | 仅在实际运行确实使用该后端时复用 | 不与Isaac路径混搭或宣称等价 |
| 本项目 `contracts/trajectory.py`、`trajectory/` | SE(3)、时间、rot6d边界 | 固定canonical列编码/native行编码；使用已有正确版本 |
| 本项目 `scripts/pi0291_arx5_data/engine.py`、`pi0294/multitask/` | S域受控诊断/既有任务几何 | 不改名为目标冻结WBC |
| 现有entry/profile/owner/Journal/HF通路 | 作业绑定、持久化、批次交付 | 不新增scheduler、daemon或跨站实时依赖 |

拟新增库 `src/go2arx5_vla/world_model/`：native_adapter/state_schema/action_compiler/replay/recurrent_dynamics/task_reward/imagination/actor_critic/train_loop/native_eval/visual_student。CLI仅薄包装 `scripts/wm02/`，测试放 `tests/world_model/`。这些路径目前是实现任务，不声称存在。

## 六个必须先解决的接口问题

1. 实际WBC是哪一版、在哪个后端、有没有合法目标动态资格？
2. H40首点绑定与8个控制区间怎样对应？空间进度/预览实际看了哪些点？
3. 完整物理/控制状态如何恢复？模型哪些字段预测、解析递推、仅保存？
4. 未来WBC history不能读取真实答案时，recurrent WM能否保持多步一致？
5. 操作奖励代理何时与原native contact/success不一致？是否会奖励空手/不释放？
6. 目标每秒真实转移是多少，完整成本由仿真、WBC、WM还是数据交付主导？

## 许可、软件环境与保留

本次没有复制任何上游实现。开始vendor/adapt代码前检查上游根许可、`REFERENCE_LICENSE`和相关资产/权重许可证，保留版权与来源；不能把局部MIT说明推断为全仓库资产可任意再分发。私有WBC/机器人mesh不进入公开Git。

上游mjlab/MuJoCo Warp环境与当前冻结WBC环境可能冲突。先复用数学/算法模块，不在活动环境全量 `uv sync` 或升级torch/MuJoCo/CUDA。需要独立依赖时建立项目既有机制允许的隔离环境，并明确环境身份，不能改全站软件。

## 发布验证

本次成功读取GitHub来源并人工核对设计及算术关系；本会话的container/Python执行调用不可用，因此没有程序化JSON检查、链接检查、CPU单测、CI或运行性能数据。文件写入后通过固定commit与Git blob身份回读；这只验证发布，不验证训练程序。Codex首项检查应解析plan.json、检查文档/配置一致，再实现实际合同测试。不得引用WM01旧测试数量作为本轮PASS。
