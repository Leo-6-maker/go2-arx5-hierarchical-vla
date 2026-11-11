# 本轮入口：原生相机＋公开 ARX5 迁移

本分支交付实验协议、声明性规格、预注册条件顺序与实现交接。**未启动远程训练或仿真；JSON不是旧launcher可直接接受的运行配置。**

- [完整实验协议](docs/protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md)
- [实验规格 JSON](experiments/native_arx5_transfer/native_camera_public_arx5_transfer_v1.json)
- [20 布局／80 条件槽 JSON](experiments/native_arx5_transfer/dev_panel_v1.json)
- [实现与执行交接](docs/handover/2026-11-11/NATIVE_ARX5_TRANSFER_EXECUTION_HANDOFF.md)
- [固定来源审计与待补证据](docs/reviews/2026-11-11_NATIVE_ARX5_TRANSFER_SOURCE_AUDIT.md)
- [完整报告模板](docs/templates/NATIVE_ARX5_TRANSFER_REPORT_TEMPLATE.md)

main入口整理已通过 [PR #374](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/374)。参考执行代码SHA为 `f0e19c8812896f98053e74b165d6c52e85327b69`；main没有自动集成整条旧执行栈。

下一项实际工作是核对现场现役状态，建立新body相机的全TRAIN100覆盖报告与实际状态重渲染版本，同时准备Table30 ARX5的有限获取清单及source时钟／反馈语义。相机通过后优先四TRAIN布局的SFT2000诊断；按协议完成新起点与cache／恢复资格，再推进三阶段训练。

执行时沿用既有授权与现场所有权约束，不因新增文档要求每个常规修复再次审批；也不因文档发布就宣称实际准入通过。保留原负结果、旧DEV/TEST、活动胶囊和全部正式失败。
