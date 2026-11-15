> 2026-11-15最新：[连续专家V2原生终态](MOBILE_CONTINUOUS_TEACHER_START_HERE.md)。旧F4–F6已自然CLOSED；旧移动共39请求、558,389积分、425次SLA预测，全部历史失败保留。另四条新专家已结束，3次出现抓持、完整任务0/4，侧向M_FREE安全停止1；未训练、未重投、未改WBC或安全门。下文为V1历史入口，后台运行描述不是当前状态。

# SLA 移动底座能力边界 · 冻结WBC
2026-11-15 · **首批原生移动与冻结SLA闭环已有现场结果；后续L1后台运行**

[执行实现 PR #386](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/386) · [首批实际报告](reports/WM02/mobile-wbc-sla-boundary-v1-20261115/FIRST_ACTUAL_20261115.md) · [逐端点分母与哈希](reports/WM02/mobile-wbc-sla-boundary-v1-20261115/FIRST_ACTUAL_20261115.json)

已封存M0／F1／F2共21次物理请求、247,735个原生积分、187次SLA生成；新增训练更新／访问均为0。F1专家0/2、学生0/3，F2专家0/2、学生0/4完成任务。F1另1个学生单元保持安全隔离；F4／F5／F6剩余18个条件已独立挂后台。不自动启动完整228探索。

[F2最新端点与后台余项](reports/WM02/mobile-wbc-sla-boundary-v1-20261115/F2_ACTUAL_20261115.md)

**主线决定：直接测试Go2＋ARX5完整移动WBC中的SLA能力边界；不要求固定底座raw、RP8、S16或JOIN10先成功。** 现役PR #384与所有历史结果原样保留，不热改、不并入主线运行。

阅读顺序：

1. [详细协议：任务、执行接口、安全与后训练分流](docs/protocols/MOBILE_WBC_SLA_BOUNDARY_V1_20261115.md)
2. [机读矩阵：预算／状态／冻结项](docs/protocols/MOBILE_WBC_SLA_BOUNDARY_V1_20261115.json)
3. [简短执行启动prompt](docs/handover/2026-11-15/START_MOBILE_WBC_SLA_BOUNDARY_V1.txt)

核心冻结：现用MuJoCo/OCS2 WBC/MPC、腿策略、臂PD、物理和原生2ms；两个PD4-500端点、各自统计与codec。首轮0模型更新。M_FREE只发EEF（真正省略BASE）；M_ASSIST发送EEF＋独立预定BASE软参考，不宣称SLA学会导航。固定root IK只作shadow，**必须先建成移动仿真实际安全profile，不能直接删除保护**。

首波最多48次物理episode（M0接口12＋L1专家12＋冻结学生24），完整探索上限228次包含首波。首次交付应包含真实底座运动、EEF/物体结果、原始请求/完成/失败分母及冻结版本证据。移动后训练取决于这些结果，不自动排满。
