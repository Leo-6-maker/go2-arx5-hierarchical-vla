# SLA 移动底座后训练：V5.2 接手入口

2026-11-17（SGT）。本分支交付实验计划，**未启动训练或仿真**。

## 已有结果与下一目标

[PR #403](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/403) 的固定审阅 HEAD 为 dbf222b41cb666bff4dd5412b864d023aec69399。冻结 SLA SHIFT7 已有3次真实移动抓持：学生3成功/8尝试，4科学失败、1IPC技术中断；F1有效3/3、F2 0/4，pilot未过。当前是已知布局的 SIM_PAUSED_NATIVE 能力，尚未取得盲测或350ms连续实时资格。

下一步先分析已存319动作的首尾加速度与接触几何，再从同一SHIFT7做普通FM500/边界loss500匹配对照，完成B0/B1/B2各8次真实闭环。证据支持时才采12次纠正示范、做两臂各500数据对照。冻结单一候选后确认pilot，再按原DEV32/TEST40目标推进。

## 顺序读取

1. [V5.2详细协议](docs/protocols/SLA_MOBILE_F2_POSTTRAIN_V5_2_20261117.md)。
2. [机读矩阵、预算与晋级条件](docs/protocols/SLA_MOBILE_F2_POSTTRAIN_V5_2_20261117.json)。声明性规格，不是已可运行配置。
3. [Codex启动prompt](docs/handover/2026-11-17/SLA_MOBILE_F2_POSTTRAIN_V52_START.md)。
4. [PR403最终实测](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/dbf222b41cb666bff4dd5412b864d023aec69399/reports/WM02/sla-sim-only-v51-20261117/FINAL_RESULT.md)与执行分支最新合法后继。
5. 执行版本的 AGENTS.md、V5.1纯仿真规则及现场真实owner/容量/许可。

main仍是文档/集成入口，完整执行栈在#403及合法后继。不要直接在main假装运行现成V5.2命令，不盲合长分支、不热改胶囊。旧根入口下面的历史GR00T/相机计划不替代本次用户指定的SLA移动学习主线。

完整H40/P10/TTL/2ms、WBC/腿/MPC/solver/臂PD、碰撞/STOP与原评分保持。新训练器须真正加载SHIFT7，不伪造原parent1000合同；新纠正数据须真实标mask/时钟并保持C20统计来源。每个科学失败保留，技术修复只续接未启动条件。

本次计划最大新训练2,000更新/64,000访问，含所有条件分支最多142原生请求；这是保守上界。首交付为D0、两个500端点和24次配对闭环。用户交给Codex执行后，在实际授权环境内自主完成有界工程与实验，无需逐补丁重复GO；本文件发布本身不代表作业已启动。
