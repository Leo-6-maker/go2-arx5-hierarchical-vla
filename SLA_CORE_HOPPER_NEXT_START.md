# 下一步：Corelab 数据采集 × Hopper H200 长程后训练

**最新用户决策：两条服务器任务链独立推进，不等 A40 或 Corelab 的旧训练；VANDA↔Hopper 可直接传输。** 本次提交只完成计划，新的服务器作业尚未由此发布会话提交。

- [完整计划](docs/protocols/SLA_CORE_HOPPER_PARALLEL_V1_20261121.md)
- [机器预算与依赖](docs/protocols/SLA_CORE_HOPPER_PARALLEL_V1_20261121.json)
- [TRAIN／VAL／DEV／CONF／TECH 逻辑人口](docs/protocols/SLA_CORE_HOPPER_PARALLEL_V1_POPULATIONS_20261121.json)
- [完整 Codex 启动 prompt](docs/handover/2026-11-21/START_SLA_CORE_HOPPER_PARALLEL_V1.md)
- [源证据与实现缺口](reports/WM02/sla-core-hopper-parallel-20261121/AUDIT_AND_PLAN.md)

## 第一次挂载应做什么

1. 同时准备 Core 新 scope/人口和 VANDA→Hopper 完整 ADAPT2K＋D2/R200/LIBERO 直传。父端点固定HF revision `5c460f195f6509f2f4da55dea32c356b45e00d37`、`ADAPT5K/step-02000-2bff547b6f2c`，完整hash从该seal取得。
2. Core逐条运行新TRAIN240/VAL24，H200有限工程资格后重新恢复父状态跑新增20K。不是等待所有Teacher通过；不是用29数组推理包恢复训练。
3. H200只在2K/5K/10K/15K边界接入已经READY的新数据；没有就继续。首次切换做一次条件2K控制，主臂不等它。
4. 固定端点Eval与训练并行；Core严格RGB16后跑新学生DEV/最终CONF，单world交替。
5. 回读真实PBS／持久服务与唯一owner、T0/deadline、实际代码／父／数据hash及断开交互后存活，才叫“后台已挂载”。

## 边界

从首次实际挂载起14天，H200总320GPU小时。主臂20K＋条件控制2K＋工程112＋提交重放256；最多22,368实际更新、2,863,104优化访问、125,224模型生成尝试。原生科学456＋TECH2，总458请求、20,911,974tick。未触发角色不借预算，原科学失败不重跑择优，健康旧作业不改动。WBC、原H40/接缝/STOP/评分/旧TEST保持。

当前main文档较早，执行模块有固定外部分支来源。按计划选择性整合依赖并封存实际运行树，不盲合整个活动实验栈；计划commit与执行commit必须分列。用户空间普通修复自主完成，真实资源授权缺项单独报告，不能伪造授权或让一个受阻角色卡住全部工作。
