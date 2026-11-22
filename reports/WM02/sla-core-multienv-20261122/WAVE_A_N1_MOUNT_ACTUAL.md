# Wave A 单环境采集与现场观测

Corelab 独立回读时间 2026-11-22T03:42:55.491+08:00。固定计划 `4fe2c8cf1b69c168b2e3079e061bb7c702089cd3`。

## Wave A

- 96 TRAIN＋24 VAL 的 120 张原登记条件卡全部通过预检。实际已启动 **1/120**，终态 **0**，准入 **0**。第一条 `MEV1-TRAIN-F1-NEAR_NEG-00-HOLD-blue` 已记录至 **22,951 tick**，子进程 `2643358` 的独占 world lease 仍有效。实际请求使用 MuJoCo、冻结 WBC、完整安全门与原评分。
- 后台单元 `wm02-sla-core-multienv-wave-a-20261122-02.service`，PID `2643171`，调用 `818d3acfd2044d1f8edd4e5475d223b9`，cgroup `/user.slice/user-2009.slice/user@2009.service/app.slice/wm02-sla-core-multienv-wave-a-20261122-02.service`。源码 `eccd8e2e2089bb2505c50043f959959d07094d56`，spec `45d949a883c7c10ec31c0caef9a40bdd0a4c51225d5417dac68c7b05432e870b`，人口 `d8762c633e7897ade790ecc736704e5304e01ba8c0a1853f1cd87054ad012774`，期限 2026-11-29T02:52:28.000+08:00。已验证 `Linger=yes`，并从另一个连接回读进程及持续增长的积分。停止入口为运行目录的 `STOP.json`，已启动的请求按原合同收尾。
- 首个准备版本在导入原部署源码未提供的可选 Teacher 修复函数时退出，产生 **0** 原生请求。现役版本使用延迟导入并保留原采集路径，未重派任何原生条件。
- 本轮由 120 张卡合计限制在 **5,484,360 tick**。现场系统盘约 301.8 GiB 可用，原门约 274.2 GiB，额外预留 3.5 GiB 给在途轨迹、快照和部分传输。现有另一块已批准目录所在 HDD 的空闲空间约 542.8 GiB，同口径的空间门约 566.9 GiB。两处均需要继续按实测核算，已挂载的条件数仅是预算上限。

## 资源与接触诊断

- 两条历史成功轨迹实际占用约 0.57 GiB（HOLD、22,453 tick）和 0.96 GiB（PLACE、36,703 tick）。对应墙钟 305.33 秒和 535.56 秒。原日志的 165.17 秒和 297.11 秒为父进程 CPU。
- 独立旁路观测单元 `wm02-sla-core-multienv-wave-a-profile-20261122-02.service`、PID `2647649` 已读完整生产 cgroup，当前内存峰值 4.07 GiB。该值属于正在进行的单条生产观测。预登记 N1 短卡与完整卡矩阵仍为 **NOT_RUN**。
- Core 用冻结 MuJoCo 和记录的 qpos 对旧 TRAIN 三条 F1 蓝块右放失败复算。最后查询的 link2／桌面距离各为 3.156、1.799、2.856 mm，84–198 ms 后冻结保护记录接触并停止，停止后的积分均为 0。该格历史成功对照为 0。下一候选需以两次查询间的整臂与持物扫掠为依据。本诊断新增仿真积分、原生请求、训练和生成均为 0。
- 现役进程实际映射了 Fast DDS 的 `/dev/shm/fastrtps_port7000` 和 LTTng 的共享名；仿真与 Xvfb 同时使用 SysV 段。两种站点现有普通用户 IPC 命名空间验证均被权限策略拒绝。实际 selected reference 的控制消费身份仍缺失。**N2、N4 均为 NOT_RUN**，新增多环境请求为 0。

## 独立评价与训练

- Core cold、IPC、原 DEV6 服务都在运行。Core 快照可用主存为 20.54 GiB，原 cold 资源门为 36 GiB，GPU 空闲 15838 MiB。两端点新生成 **0**，phase0 DEV6 与 MAIN2K DEV6 原生请求各 **0**。DEV 服务等待实际 IPC。
- Hopper `728156.hopper-m-02` 已实际提交 MAIN **2,124**，最近封存 **2,100**。同一后继从 2K 以后累计访问为 Go2 7,936、RoboDojo 3,968、LIBERO 3,968。`TRAIN-059` 已就绪，实际优化访问 0。当前作业之后的下一作业尚未取得 JobID。

[机读回执](WAVE_A_N1_MOUNT_ACTUAL.json)
