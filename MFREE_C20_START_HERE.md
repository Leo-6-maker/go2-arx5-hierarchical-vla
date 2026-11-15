# M_FREE：从 V3 专家到一个 SLA 策略

本轮使用 [用户完整计划](docs/protocols/MFREE_C20_USER_PLAN_20261115.md) 和
[固定条件与配方](docs/protocols/MFREE_C20_FINAL_PLAN_V1_20261115.json)。
基于 PR #387 的报告头 `ecc1942a95de9a70cec9ad2883c7354258b05346`；
旧 V3 实际执行源码仍是 `094231c55abbafd18a7aa70e9e37e4bd52edc5c1`。
本入口与软件检查均不证明新增数据、训练或学生成功。

仅采 F1/M_FREE 与 F2/M_FREE（专家高层抓点 +8 mm），C4 是 C20 的子集。
每族五个预登记 XY 布局，每布局目标两次完整成功；最多四次独立尝试，
合计上限 40。安全事件隔离受影响族；技术错误停止新派发，不能自动重投。
先一个世界确认同步实时 RGB 与命令接缝，再两个世界独立生产。
另冻结 DEV16、旧学生 60 秒基线 8 条，以及候选 TRAIN8。

`MOBILE_ACCEPTED_BOUNDARY_050_V1` 读取原 receiver 的实际绝对时间 knot，
核对原 solver 的相同 target identity。下一 query 参考是实际 t+0.5 s，
不是 t+0.45 s 的发布行。原系统结果不回填；teacher 滤波内存只记审计。
实时 body/wrist 图像以有界 writer 封存，不从旧状态事后渲染作为原始观察。
当前 query 最终 H40 的计划监督与 P10 实际消费分别记录，未消费后缀不称执行。

一个候选 `MFREE_C20_FINAL_FM1000`：原 SLA_base／公共 adapter 初始化，
成功 TRAIN 新统计，FM-only，1000 更新／32000 访问，micro8×4，既有阶段采样。
500 仅 TRAIN 诊断；1000 普通 RGB 恢复后 TRAIN8 与 DEV16。
可选匹配后继不自动启动。旧模型保留其原统计，不静默替换。

M_ASSIST 整个模式继续隔离；WBC、MPC、腿、臂 PD、solver、物理和安全门不改。
旧 RP8、JOIN10、历史移动负结果、旧 DEV/TEST 保留。不用真机或 RL。
