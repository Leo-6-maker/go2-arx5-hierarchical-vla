# Go2 + ARX5 + GR00T + frozen WBC

研究目标：让视觉语言策略通过可审计的 EEF 轨迹接口控制完整 Go2＋ARX5，在冻结 WBC、腿策略和原生安全条件下完成操作任务。

## 当前工作：原生相机与公开 ARX5 迁移

2026-11-11 的首轮问题是：**固定目标 TRAIN100 和目标训练预算，先用公开 ARX5 数据适配，能否改善目标整机的闭环抓持、抬升与保持？**

- 共同起点：官方 GR00T-N1.7-3B 固定 revision；目标输入为随 BASE 运动的机身 RGB＋腕部 RGB＋原 EEF 状态＋语言。
- A：官方起点→目标 500 更新。B：同一起点→公开源 500 更新→同一目标 500 更新。
- Core 4080S；七类状态／动作接口的目标类别切片及 DiT 末两层；冻结视觉／语言骨干。
- 本轮限于站立整机仿真；冻结 WBC、腿策略、动力学、2 ms 时钟及安全。WM、视觉骨干解冻及新大卡作业不属于本轮。
- [详细实验协议（独立研究分支）](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/research/native-camera-arx5-transfer-v1-20261111/docs/protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md)；[当前入口](START_HERE.md)；[事实与证据状态](CURRENT_SCIENCE_STATUS.md)。

## 仓库与运行版本

`main` 是整合与发布入口，更新文档不表示已部署、已训练或已通过整机资格。当前复用的训练／评价实现位于 [固定执行快照 `f0e19c8812896f98053e74b165d6c52e85327b69`](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/f0e19c8812896f98053e74b165d6c52e85327b69)，许多模块尚未集成到 main。参见 [代码地图](docs/CODE_MAP.md) 与 [整合记录](docs/maintenance/2026-11-11/REPOSITORY_INTEGRATION_STATUS.md)。

保留 CORR2048、SFT2000、C500、BF16 REJECT、原 DEV／TEST 及 Teacher24 负结果。公开源的未来实测 EEF／宽度标签与目标已消费 Teacher 名义 H40 明确分开。原始预测与解码后执行结果分别报告。

## 阅读与贡献

先读 [AGENTS.md](AGENTS.md)、[START_HERE.md](START_HERE.md) 和 [仓库工作流](docs/REPOSITORY_WORKFLOW.md)。详细历史证据保留在 Git 与原报告路径；入口不维护旧 PID 或把旧队列当实时状态。

- [实验路线图](docs/EXPERIMENT_ROADMAP.md)
- [Goal V7 历史总纲](docs/protocols/GR00T_WBC_SCIENCE_GOAL_V7_20261031.md)
- [整理前 README](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/README.md)
