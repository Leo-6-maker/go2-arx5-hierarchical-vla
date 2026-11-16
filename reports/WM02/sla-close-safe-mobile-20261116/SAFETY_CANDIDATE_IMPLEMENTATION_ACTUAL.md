# 接触前监测候选：实现与数值核对，仍未准入

PR392固定计划`74264d5b`，执行PR #393。本文及软件检查均不是安全重新准入。F1/F2共同隔离、学生及M_ASSIST隔离保持；新增物理请求和积分仍0。

候选源固定`660616cc0b3186872fd1cca1b5275a97c2c57ea4`，数值核对源`1846e28ae8ea50c2676a09fd225bc72cc401fe74`。在Core原MuJoCo3.2.7编译独立库，库SHA256为`859ac806261e2c21bded94ef44d69bf50f2f47c16e961f9a26e87386f8c1a240`。**没有安装到原生进程、没有连接现役boundary或启动任何新native请求。** 当前G2仍运行独立固定source`392b7894`，不消费本候选。

监测合同保持已分析的条件：垫与桌面XY重叠、box-corner净空≤3mm、实际TCP z减最近精确receiver z再加50mm/s×参考年龄≤−5mm；参考年龄不超过50ms。参考是可由现有部署边界提供的receiver见证，不是未导出的controller-selected policy。数据只供独立保护，不能进入SLA输入或任务决策。

预拟实现于`mj_step`调用原链之前，在私有mjData副本做被动FK；条件触发则锁存SIGINT停止、随后不调用原链。参考过期／未来／非有限、布防后标记消失时按无效参考停止，不静默解除。该停止行为目前来自实现设计，**没有原生hook时序、动态预防或延迟资格证据**。原WBC/MPC/腿/臂PD/solver和所有原安全保护未改。

已实际核对固定四个2ms窗口共1204保存状态：C++与原Python结果一致，最大几何／参考差6.9389e−17；输入qpos/qvel/ctrl未变，三种无效参考均拒绝。使用RTLD_LOCAL直接调用只读ABI，没有LD_PRELOAD或mj_step调用；0新模型生成／优化／动力学积分。wall8.50s／CPU1.26s（仅核对调用进程；编译器子进程CPU未单独采集，不算神经训练成本）。生产witness的单项测试拒绝错时钟和非精确参考，原文件保留。

| 保存窗口 | 状态数 | 告警状态 | 首次告警tick |
|---|---:|---:|---:|
| 原成功F2-T3-r2 |301|25|19000|
| F2专家碰撞 |301|19|19136|
| F1专家碰撞 |301|16|19187|
| F1学生碰撞 |301|8|19481|

这次C++数值核对只覆盖四个固定窗口，**不冒称C++重扫了全部38个成功任务**。原完整Python分析覆盖41任务／693670状态，告警在三碰撞前34/316/12ms出现，同时会阻断1/38成功专家；预声明的零成功误拦筛选仍失败。原成功仍成功，原碰撞仍失败；没有阈值搜索、动态反事实或新native成功。

当前状态为`UNINSTALLED_NOT_ADMITTED_FAILED_ZERO_FALSE_BLOCK_SCREEN`。该候选尚不能按原筛选宣称可准入。若责任方拟采用不同的新预防合同，必须独立明确记录可接受的停止代价、代码、scene、冻结profile、有限面板和停止规则，之后才可讨论至多8次专家安全验证（F1/F2×两个审核布局×两reset，首次安全停止立即按审定范围隔离）；不会自动提出或启动学生pilot、DEV或TEST。没有这样的明确记录时，物理预算保持0。

[预拟合同](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/660616cc0b3186872fd1cca1b5275a97c2c57ea4/docs/goals/2026-11-16/CONTACT_FENCE_CANDIDATE_REVIEW_V1.json)与[实际JSON／source、scene及证据SHA](SAFETY_CANDIDATE_IMPLEMENTATION_ACTUAL.json)供独立审核；不构成解除隔离的文件。第一对G2还需读回完整500端点，任何离线合法率也不等于移动抓持成功。
