# 现有Codex接续：唯一入口与同步规范

采用#47本次交接指定的固定release，读取根AGENTS.md、docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md、docs/operations/WORKSPACE_ENTRY_ENFORCEMENT_V1.md、infrastructure/workspace/ENTRY_POLICY.json，以及已有reports/maintenance/2026-10-13-unified-entry-standard/GO_TASK_CARD.json。

保留现有单一规范和enter.sh/doctor.py/profile命名，不创建第二套gw或另一层顶级工作区。先刷新真实#11/#47、PI0292/PBS/lease与上一轮23项保留清单；33个已退役路径不重做。

实施E0–E4并落实补充要求：入口源和测试纳入同一共享library，部署固定entry release；新collect/train/eval/audit/pack都prepare/validate run manifest，检查host/authority/lease/版本/输出根后才由原approved launcher执行；新输出不得回Home和旧Projects。sync明确区分remote、管理checkout、已部署与运行SHA；活动source不pull，冲突不force。

PI0292只登记legacy-pinned，不重启/热切换；真正作业边界由原owner明确采用下一阶段入口。原code/data/stats/截止/预算不改。剩余源码先切消费者再在既有维护GO内退役；受保护资产只索引，不为ls整齐搬空。独立候选继续，不因一个lease阻塞全部工程。

沿用主规范同一次任务T0/ledger和4h/8CPU-core-hours/8GiB新增峰值，不因本补充重开或叠加；最后30min收尾。无新物理/模型/PBS/WBC/硬件或健康作业终止。执行已有CPU-only CI契约，不在Actions跑仿真/模型/SSH。分别报告唯一入口、新任务接入、Git同步、物理顶层收敛的实际验收。沿已有unified-entry-standard报告路径发布到#47并回读；#11仅做科学依赖协调。文档发布不是服务器已经部署。
