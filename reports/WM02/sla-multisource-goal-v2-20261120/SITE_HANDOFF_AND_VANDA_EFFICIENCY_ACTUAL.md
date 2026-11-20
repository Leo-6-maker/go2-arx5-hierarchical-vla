# VANDA 高吞吐训练与 Core 后台接续实绩

Core 与 VANDA 截面为 **2026-11-20 16:28–16:29 SGT**。固定计划为 `1b2a25b7c1720051a2627d30831fb18130462374`。 [机读实绩](SITE_HANDOFF_AND_VANDA_EFFICIENCY_ACTUAL.json)

## VANDA

- 正式 ADAPT `1594211.stdct-mgmt-02` 仍在一张 A40 上运行。已封存 1,000 步，实际账本到 **1,131 步**。累计 Go2／RoboDojo／LIBERO 访问 **72,384／36,192／36,192**，最近五步中位 **44.10 秒／步**。D2 在精确 2k 处接续，2k 与 5k 独立 Eval 处于 PBS 依赖等待。
- 工程 PBS `1597888.stdct-mgmt-02` 已在原基线使用的同一张 A40 启动，父检查点为同一封存 ADAPT 1,000 步。源码 `f86955bab424f5500b20bbdaaead3596b24d3593`，执行 spec `fd9e9a05ef413b3a83b4aa5b1367687c3f11dce3202d70029edfc57103ba99f0`，主管 PID **883669**，站点分配截至 **17:51:19 SGT**。作业根目录 `STOP.json` 为停止入口。预取分支已记账 **5 步**，完成 **20 条 H40 辅助生成**。两条预取路径最多各 8 次更新。
- 串行原 PBS `1597847` 保留 F/1。3 次暖态准备和 5 次计时更新，计时 p50 **60.53 秒／步**。可选 CUDA trace 报 `CUDA_ERROR_LAUNCH_FAILED`，串行 140 个状态数组的指纹缺失。预取已有 2 次计时样本，中位 **56.13 秒／步**，两条候选比较尚待终态。工程权重禁止进入正式父链。
- [PR #419](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/419) 已加入 mb1／mb2／mb4、单线程有界预取、延迟 loss 回读与编译内梯度累积。全局仍是 64／32／32 个 FM 窗口和 4 条完整 H40，经同一域权重、裁剪与单次 AdamW 更新。提交 `6338c28619d23584b2d9b690fa0ecb4682400a72` 的 26 项 CPU 测试通过。批处理版本尚待有额度的真实 GPU 对照及状态、显存和吞吐核验，当前正式 lineage 使用原 mb1。

## Corelab

ADAPT0 前六个固定条件 6/6 实际执行并封存，抓持、抬升 20 mm、任务成功各 **0/6**。四个 PLACE 入区和稳定释放各 **0/4**。所有条件首个原生 STOP 原因均为非法接触，STOP 后新增 0 tick。总计 **132,555 新 tick** 和 **396 次生成**。外层终态含两条已存档接口证据，分开计入。终态证据 SHA `31c94319f629edbceb83f6040586075ca64f2ccc6b2ab9e63229fc6a49c4213e`。

| 条件 | 新原生 tick | 真实生成 | 首个停止原因 |
| --- | ---: | ---: | --- |
| HOLD-red | 20,719 | 61 | ILLEGAL_CONTACT |
| HOLD-blue | 18,374 | 51 | ILLEGAL_CONTACT |
| PLACE-red-left | 22,673 | 68 | ILLEGAL_CONTACT |
| PLACE-red-right | 23,355 | 71 | ILLEGAL_CONTACT |
| PLACE-blue-left | 23,195 | 70 | ILLEGAL_CONTACT |
| PLACE-blue-right | 24,239 | 75 | ILLEGAL_CONTACT |

MIX1001 已从 1,000 步封存点启动，服务 `wm02-sla-scale20k-mix-core-after1000-20261120-02.service.service`，PID **665212**，执行 spec `2fefed7bfb600f77ba1ecdeb3e1fb8f3e42bf9f9944457d7096b7a584116d4be`。截面仍在载入基础模型，新增优化更新 **0**。分段上限 2,000，实际分配和保存门生效，停止入口为作业根目录 `STOP.json`。TGT5K RGB16 与 ADAPT0 原顺序第 7–12 条分别由两个独立 systemd 观察者等待交接，截面新增 0 生成、0 原生请求。

Corelab 的持久 systemd timer 与 VANDA CPU PBS `1597930.stdct-mgmt-02` 已按 14 天截止登记。固定期限为 **2026-12-04 15:29:58.885714 SGT**，PBS 取次秒 **15:29:59 SGT**。Core timer 已核对 7 个登记角色，PBS 已核对 21 个登记角色，两者目前均为待执行的收尾机制。

与 16:02 截面相比，正式新增 **36** 次 GPU 更新，工程新增 **5** 次 GPU 更新，训练内新增 **164** 条 H40。原生新增启动 **4** 次，新增记账 **104,633 tick** 和 **328** 次模型生成。历史失败、固定评分和物理保护照原规则执行。
