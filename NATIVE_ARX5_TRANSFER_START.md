# 本轮入口：原生相机＋公开 ARX5 迁移

本分支交付实验协议、声明性规格、预注册条件顺序与实现交接。文档发布不启动远程作业；JSON不是旧launcher可直接接受的运行配置。

## 执行接续实际进度 · 2026-11-12

- [原SFT2000 TRAIN4完整终态](reports/WM02/native-arx5-transfer-20261112/T0_SFT2000_NATIVE_TRAIN4_FINAL_ACTUAL.json)：4/4成功、60查询；全部八个原生／GPU子进程已退出，原条件没有重跑。它是TRAIN组合策略诊断，不是独立DEV。
- [官方父模型目标资格](reports/WM02/native-arx5-transfer-20261112/T1_OFFICIAL_PARENT_CUDA_CACHE_RESTORE_RGB_ACTUAL.json)：1,450新图像骨干编码、10次非候选更新／320访问、完整batch32对照、新进程4对2＋2恢复及普通RGB重建均通过；不继承资格权重为候选。
- A_TARGET250已完成250更新／8,000访问，完整保存和普通RGB重建通过；按原完整状态继续固定500终点。A500四TRAIN交付诊断使用独立声明，未使用锁定DEV／TEST。后继实际计数以新ACTUAL回执为准。
- [native／source TCP链审计](reports/WM02/native-arx5-transfer-20261112/SOURCE_NATIVE_TCP_CHAIN_AUDIT_ACTUAL.json)只比较12个封存状态，没有新积分；source采集来源／时钟仍未准入，只阻塞B。

## 2026-11-12 进度与下一步

- [最新进度与 Table30 采集来源审查](docs/reviews/2026-11-12_NATIVE_ARX5_PROGRESS_AND_SOURCE_PROVENANCE.md)
- [目标侧优先的后续实验计划](docs/handover/2026-11-12/NATIVE_ARX5_TARGET_FIRST_NEXT_V1.md)

执行 [PR #376](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/376) 在固定提交 `41111540d33398e916224dd07cd2638db2a5675f` 已报告：新相机1,450/1,450监督窗口通过，SFT2000同卡推理完成，四TRAIN诊断已读回一个成功终态。最后远端组读取为2026-11-12 01:11:30.844 SGT；另外三个终态未读回，不代表失败或仍运行。新官方父模型实际资格与A/B训练在该回执中均为0更新。

**下一主线：读回原组状态 → 官方父模型真实目标训练资格 → A_TARGET500 → 固定TRAIN4交付诊断。** Table30来源／时钟证据并行处理，只阻塞B_SOURCE。新DEV继续封存至两端点固定；本次补充不改变V1配方和正式A/B扩样门。

## 原 V1 协议与注册

- [完整实验协议](docs/protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md)
- [实验规格 JSON](experiments/native_arx5_transfer/native_camera_public_arx5_transfer_v1.json)
- [20 布局／80 条件槽 JSON](experiments/native_arx5_transfer/dev_panel_v1.json)
- [实现与执行交接](docs/handover/2026-11-11/NATIVE_ARX5_TRANSFER_EXECUTION_HANDOFF.md)
- [固定来源审计与待补证据](docs/reviews/2026-11-11_NATIVE_ARX5_TRANSFER_SOURCE_AUDIT.md)
- [完整报告模板](docs/templates/NATIVE_ARX5_TRANSFER_REPORT_TEMPLATE.md)

main入口整理已通过 [PR #374](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/374)。参考执行代码SHA为 `f0e19c8812896f98053e74b165d6c52e85327b69`；main没有自动集成整条旧执行栈。

原协议中的相机覆盖与数据版本检查已有上述ACTUAL回执；复用其身份。执行实现继续由独立分支管理，本计划分支没有自动部署或合并执行栈。

执行时沿用既有授权与现场所有权约束，不因新增文档要求每个常规修复再次审批；也不因文档发布就宣称实际准入通过。保留原负结果、旧DEV/TEST、活动胶囊和全部正式失败。
