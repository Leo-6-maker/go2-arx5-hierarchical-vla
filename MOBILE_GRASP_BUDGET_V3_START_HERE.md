# 移动抓持完整资格 V3

先读[F2碰桌只读归因](reports/WM02/mobile-grasp-budget-v3-20261115/PAD_TABLE_CAUSE_20261115.md)与[固定新版本JSON](docs/protocols/MOBILE_GRASP_BUDGET_V3_20261115.json)。继续PR #387，独立worktree来自c3748fe4，旧V2原运行源码cb24a1c1及全部失败保持。

新任务最多60秒，五阶段上限20/20/6/10/3秒；不强制阶段转换，不改4mm判据、H40/P10、WBC或原生安全。前三个条件只改阶段预算；最后的F2/M_FREE另有8mm高层抓点净空干预，结果单列，原M_FREE失败隔离不抹除。

最多四个新原生仿真条件，SLA调用/训练均0。当前本地28项定向检查通过，动态资格待后台真实结果；完整原生抓持、抬升20mm并保持1秒通过前，不扩采或训练。
