# CLOCK500 Corelab 接收与配对顺序核验

Corelab 已落地接收和严格 RGB16 源码 `f9dbbedf8d046d5538027d2b4d7d1eeac11a01d2`。CPU 回读逐一核对 **8 个既有场景文件哈希**和冻结执行顺序。原 DEV 24 个条件各配 A/B，共 48 条，随后固定 B 确认 24 条，合计 **72 条、3,290,616 tick 上限、11,520 次原生模型生成上限**。原 D1 的 16 个 RGB 序号匹配。 [机读回执](CLOCK500_CORE_INTAKE_ACTUAL.json)

接收端要求整包和各个成员哈希一致，分别绑定臂、父检查点、比较、D2 和冻结人口。补充测试核对同一份 A 包以 B 身份提交时被拒绝。现场核对原 ADAPT0 既有资格仍能读取，并验证 ADAPT0 包不可用于 CLOCK。实际 RGB16 仍待两个真实 CLOCK 端点，每臂严格复验上限 16。Core 原生派发取决于现场 GPU 资源和原 world 资格。

截至 **19:09 SGT**，VANDA 两个导出 `1598593／1598594` 和原触发 `1598095` 仍等待前驱。两个 CLOCK 包尚未封存或到达 Corelab，未启动本轮 RGB16 或原生。此阶段新增站点作业、训练、访问、模型生成、原生请求和积分均为 **0**。 [原 PBS 回执](CLOCK500_ENDPOINT_EXPORT_PBS_ACTUAL.md)

同期 MIX 实际 **1,262** 步，访问 **40,384／60,576／60,576**。ADAPT `1594212` 为 R，实际 **1,329** 步，访问 **85,056／42,528／42,528**。较上一实绩新增 **20 步、2,560 次访问、80 条训练内 H40**。TGT5k 仍等原 RGB16。另在 PR #421 修正了 ADAPT2k／5k 原生 launcher 的回执读取。现场用已封存的 ADAPT0 验证了原入口，2k／5k 自身的实际 GPU 资格仍待端点。 [PR #421](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/421)
