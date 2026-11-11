# 实验路线图 · 原生相机与公开 ARX5 迁移

更新：2026-11-11。具体数值、准入与停止规则以 [首轮 V1 协议](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/research/native-camera-arx5-transfer-v1-20261111/docs/protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md) 为准。

| 阶段 | 工作 | 产物／继续条件 |
|---|---|---|
| P0 | 固定官方起点、执行代码来源与新观测／标签合同 | 文件身份、依赖清单、未决事实清单 |
| P1 | TRAIN100 全阶段相机覆盖；实际状态重渲染 | 覆盖报告；100 回合／1,450 监督窗口及非视觉数据不变 |
| P2（可并行） | Table30 ARX5 元数据／归档预算、episode split、时间与坐标转换 | 源 manifest、≤3,200 TRAIN／≤400 保留窗口；真实数量与全部排除原因 |
| P3 | SFT2000 四 TRAIN 布局诊断；新起点与缓存训练资格 | 原生诊断、参数白名单、GPU 显存／吞吐、完整恢复与普通推理验证 |
| P4 | A：官方→目标500；B：官方→源500→目标500 | 三个有限训练阶段；目标配对随机输入；250／500 完整端点 |
| P5 | 新 DEV 四布局×两重启／方法 | 16 请求完整分母；同一 endpoint-basis，原始 H40 指标另列 |
| P6（有条件） | B 比 A 多≥2/8 成功且无新增硬安全／技术问题，接续其余布局 | 新增64请求；最终40／方法；首筛8直接计入；分阶段报告 |
| P7 | 失败谱系、成本与决策 | QUESTION / CHANGED / ACTUAL / NEGATIVE / DECISION / NEXT |

目标阶段保持纯目标数据；本轮不加 WM、视觉解冻、公开回放、更多同分布更新或新大卡。后继实验必须由失败机制与本轮证据决定。

## 历史路线

- [旧 Goal V7](protocols/GR00T_WBC_SCIENCE_GOAL_V7_20261031.md)
- [整理前路线图](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/docs/EXPERIMENT_ROADMAP.md)
- [V2 PR #372](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/372)
- [V3/V3.1 PR #373](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/373)

历史协议及已运行实验的原始定义保持有效；旧路线不自动成为本轮追加作业。
