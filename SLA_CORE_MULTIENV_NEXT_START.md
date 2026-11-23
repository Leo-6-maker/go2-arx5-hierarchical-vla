# 当前任务入口：Corelab多环境采集与固定学生评价

2026-11-22（SGT）。先读[完整计划](docs/protocols/SLA_CORE_MULTIENV_COLLECTION_V1_20261122.md)、[机读配置](docs/protocols/SLA_CORE_MULTIENV_COLLECTION_V1_20261122.json)、[逻辑人口](docs/protocols/SLA_CORE_MULTIENV_COLLECTION_V1_POPULATIONS_20261122.json)和[Codex启动任务](docs/handover/2026-11-22/START_SLA_CORE_MULTIENV_COLLECTION_V1.md)。

主线：恢复真实后台交接与Core canonical Eval；Teacher接触修复和新几何与之并行；单world持续采集，N2合格后扩容，N4条件启用。H200继续原20K训练，不等评价或采集全通过。

本次发布只提供计划。服务器新增作业、原生请求、模型生成和训练更新均0。最新审计代码为`1fee437d12d9a79deff8e2be0175da3f148dd48e`，活动服务器先核实，不按文档假设实时状态。原训练/旧人口保留，新TRAIN480＋VAL48及工程/接口分母单列。

保留WBC、物理、安全门、评分与历史失败。执行只在已授权实际资源内，活动capsule不热改，不删除数据或抢占未知资源。根历史入口未覆盖；本任务的未来工作排序优先于旧任务冲突条款。详见[审计与优先级](reports/WM02/sla-core-multienv-20261122/AUDIT_AND_RESOURCE_PRIORITIES.md)。
