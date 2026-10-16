# Vanda 闭环建设：实际执行进展

状态快照：2026-10-16 21:22:57 Asia/Singapore。**建设已获当前用户明确 GO 并实际执行；完整模型＋WBC闭环仍未运行。**

采用依据是用户在现有 local-PC 对话提出“可以按照计划开始推进vanda端的实验闭环”。[执行协议](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/09cbbe69ed706bd091251399f5ee9f045d8d1322/docs/protocols/PI0294_VANDA_CLOSED_LOOP_BRINGUP_V1.md)登记同一 owner、固定输入及阶段验收，普通同范围部署修复不另问 routine GO。本轮不迁移或重复 Hopper E0、不重开旧 CoreLab Teacher、不给旧失败改判、不训练。

## 实际完成与工程失败

- 用户指定的 WBC 包已上传项目资产区，CoreLab 源/已有 local-PC 副本/Vanda 实际回读 SHA256 一致：`6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`，215194129 bytes。人类入口通过标准工作区 assets 导航；私有 registry 保存固定原件身份。
- CoreLab 已有 clock 构建收据与该包的六项原始输入哈希相同：WBC task 配置、浮动 URDF、机器人 XML、场景 XML、腿部 policy 和 policy 配置。原收据另外三个 compiled binary 不在源码压缩包中，需要本机重新编译与独立资格；六项相同不能宣传全平台数值等价。
- 已从 CoreLab 只读保存并私有转交8个 clock/startup 参考成员，source seal `da8091c0a350a4c619a45aff31dece7db0a5f32728a9c96cfaa4b0639468ff9e`。它们只作迁移参考，尚未在 Vanda 编译或物理资格；未复制/启动旧关闭 lease。
- 新增并部署 `vanda-native-build` CPU bootstrap adapter，绑定 manifest/lease/output registry/实际 PBS host，登录节点拒绝。Docker ROS Jazzy 基底按 linux/amd64 digest 固定；单次4核/32GB/最长3h，编译并行2，0 GPU，源码校验和命令日志留私有运行根。
- 首作业 `1462481.stdct-mgmt-02` 在合法 CPU 节点实际运行：完整归档解包、`SHA256SUMS` 校验通过；container bootstrap 成功获取/解包 OCI image，到 `%post` 进入前因站点 `/app1` bind 的目标目录不存在而停止。WBC 编译、策略 forward 和物理均未开始。实际 walltime215s、进程树 CPU48.324495s；该工程失败与关闭 lease 原样保留。
- 修订 definition，在 `%setup` 仅为该临时 container rootfs 创建站点挂载目标；没有改宿主系统或 WBC 源。修订 source `f5be6f40c9a57f8053d278dff66b437004fa3383`，新独立 run `20261016-vanda-native-build-02`、job `1462595.stdct-mgmt-02` 已实际提交，快照仍 Q。这是修订后待验证，**不是已经修复验收 PASS**；沿第一次 GO 的原 T0，不重置窗口。

## 当前实际作业

| Job | 当前状态 | 工作 |
|---|---|---|
| `1462438.stdct-mgmt-02` | Q，node_pool 资源不足 | 原1卡15分钟硬件/CUDA检查 |
| `1462451.stdct-mgmt-02` | H，等待前项 afterok | 原1卡30分钟固定观测M2048推理检查 |
| `1462481.stdct-mgmt-02` | F / exit1，已封存 | 首次CPU容器构建；站点挂载失败 |
| `1462595.stdct-mgmt-02` | Q | 修订后的容器＋原生WBC编译 |

两项原 GPU job 未重复 qsub。为三个仍排队的本任务 job 显式记录其原队列默认 `node_pool`（GPU=`gpu_node`，CPU=`cpu_node`），未改变项目、CPU/GPU数量、walltime、科学源码、模型/统计、query预算或依赖；前后 selector 私有记录保存。未改他人作业、原有其他 Vanda 作业或 Hopper E0。

## 运行证据边界与接续

目前实际 Vanda GR00T加载/查询、WBC启动、机器人积分、任务episode、optimizer update 均0。此前 M2048/Cosmos固定字节和CPU processor PASS继续有效；不能用排队、源码存在、容器构建尝试或六项源输入匹配替代 GPU、EGL、WBC/时钟、任务结果或完整闭环验收。

先取得实际容器与编译结果，保留原生限位/碰撞/finite/停止机制，继而进行有界原生启动、时钟/参考消费/观测资格，再接同节点隔离模型进程。真实新观测→H40→SafetyGate→prefix8→实际ACK→新观测才算闭环；任务成功另判。软件路径/构建问题自主修，真实安全失败停止受影响物理并保留证据，禁止调WBC/原安全阈值追通过。

新原生 run02 manifest SHA256：`fd00a6e4cef09c964df60791ea922b038d1ae5ff471aea0708bfdde073c0584e`。首 run01 manifest SHA256：`58e7dbc27053dcbb956a215fc1047b137d96c51d7a46d39377114787e9346dc4`。源码与部署记录分开，私有 WBC/vendor/路径/原调度输出/RGB/权重/凭据没有进入 GitHub。

构建依据：[Apptainer定义文件阶段及APPTAINER_ROOTFS](https://apptainer.org/docs/user/1.4/definition_files.html)、[容器构建](https://apptainer.org/docs/user/1.4/build_a_container.html)、用户指定交付包清单、当前项目原生收据。后续结果必须在其实际阶段另作回读，不能把本快照改写为未来已完成。
