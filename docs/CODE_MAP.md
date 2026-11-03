# 代码地图 · 原生执行复用，WM研究隔离

读取基线：`0a27753bbda8ed75ab6b74aeb64755e0427362c7`。**本次不搬迁任何历史Python模块，不改import，不复制物理驱动。**

## 已有、优先复用

| 位置 | 用途 | WM01边界 |
|---|---|---|
| `src/go2arx5_vla/contracts/`、`trajectory/`、`adapters/`、`safety/` | frame/TCP/time、row-rot6d、轨迹安装与安全 | 复用数值语义；新增policy参数化不回写旧ABI |
| `scripts/pi0291_arx5_data/engine.py` | 有限力矩源ARX5驱动、`step`、`snapshot` | 固定底座，不是Go2 WBC；读取但不改gain/IK/力矩规则 |
| `scripts/pi0294/multitask/source_engine.py` | `make_engine`、BOOK/DOOR接触场景扩展 | 用现成资产与源任务；不移植一个新模拟器 |
| `scripts/pi0294/scene.py`、`scripts/pi0294/pv1/` | T1/T2源场景、执行/评价链 | W0绑定实际使用的Teacher/score与full SHA；不能选名字相似的历史版本 |
| `scripts/pi0294/multitask/` | 现役四任务、恢复/采样、共享训练相关实现 | WM独立数据与输出，不改正在运行的PB/M32/shared59 |
| `scripts/pi0292/train_adapter.py`、`scripts/pi0294/generalization/` | 官方GR00T数据/Processor/训练与恢复 | 只在W5复用；不在W0加载3B模型 |
| `src/go2arx5_vla/target_mujoco/`、`wbc/` | 目标本体冻结执行 | 本轮不改、不运行隔离目标 |
| `src/go2arx5_vla/workspace/`、`scripts/workspace/`、`infrastructure/` | 现有entry/profile/registry/owner | 不新增调度器、双端实时控制或预算账本 |
| `scripts/pi0294/replay_viewer/` | 已记录实际轨迹与图像查看 | 诊断，不改评分或物理状态 |

`Engine.snapshot()`已经提供qpos/qvel、EEF/物体位姿和速度、夹爪、接触及安全量；但完整恢复还需驱动器目标、IK缓存、RNG、MuJoCo积分状态和任务监测记忆。WM适配器补记录与恢复测试，不给旧驱动器加补偿。

## 本次已提供

| 路径 | 状态 |
|---|---|
| `experiments/wm01/plan.json` | 机器可读设计，不是可直接训练的配置 |
| `experiments/wm01/check_plan.py` | 纯标准库静态一致性检查，无GPU/仿真/网络副作用 |
| `experiments/wm01/tests/test_plan.py` | 检查器单测 |
| `docs/protocols/WM01_TASKSPACE_WORLD_MODEL_POLICY_V1.md` | 科学与数据合同 |
| `docs/handover/2026-11-03/WM01_CODEX_EXECUTE.md` | Codex实现顺序、验收、接续 |

## 拟新增，由Codex实现后才能称可运行

```text
src/go2arx5_vla/world_model/       # 尚不存在，不是本次已实现代码
  state.py                       # WM/actor/critic不同字段白名单
  transition.py                  # 原生区间、终止、split与读取
  action.py                      # 8×7参数→原EEF10/H40；消费prefix8
  dynamics.py                    # 小残差ensemble与事件头
  reward.py                      # 相同训练reward；原生score仍独立
  policy.py                      # 小actor/critic
  imagination.py                 # 参数冻结但保留输入梯度的短BPTT
  train.py                       # 共用PPO循环，WM臂增量想象更新
scripts/wm01/                    # 拟新增，薄CLI，不是另一控制平台
  collect.py / fit.py / diagnose.py / train.py / evaluate.py
  native_adapter.py              # 包装原生Engine/完整恢复，不fork驱动器
  export_teacher.py              # 后继W5，不导出想象RGB
tests/world_model/              # 拟新增的合同与梯度测试
```

先核查已有实现再增加；同一职责只保留一个新实现。对旧模块必须改的工程缺陷用独立patch与受影响回归，不借机改WBC、score或老面板。

## 文档与历史

`START_HERE`短入口 → `CURRENT_SCIENCE_STATUS`事实摘要 → `docs/protocols/`科学定义 → `docs/handover/`执行交接 → `reports/<experiment>/`真实结果。长日志不再逐条复制到根文件。

原地图完整保留：[整理前版本](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/docs/CODE_MAP.md)。历史stageNN、PI模块、数据引用和source capsule继续按原路径工作；不以“旧目录多”为由删除科学证据。
