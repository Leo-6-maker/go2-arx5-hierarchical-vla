# PI0293 发布前状态刷新与计算节点准入修复 — 必须首先读取

本文件是同一 PI0293 GO 的最新采用差异，优先于 OVERNIGHT_GOAL.md 第1节引用43784131时的较早状态；不形成第二个goal，不新增另一套预算。

## 1. 最新实际起点

发布前重新读取 GitHub HEAD ab3050c8cffeccd60e4ea362d8d7793b7ca929da 的 PHASE_STATUS 与 S3_SMOKE03_TERMINAL_RECEIPT：
- CFP04-CF-007 的登录/当前成员、有效日期及余额查询已成功；不能继续把之前的登录失效当唯一当前阻断。
- PI0292 S3-smoke-03 已真实提交为664308.hopper-m-02，并以F/exit1终止。失败发生在 `QUALIFIED_NATIVE_AMGR_ABSOLUTE_EXECUTABLE_REQUIRED`，即计算节点的原生amgr可执行性/字节身份检查，原模型smoke wrapper尚未启动。日志尚不能区分“路径不存在、不可执行、hash不匹配”。
- 该次GPU allocated5秒、PI0292累计17秒；model loads/updates/generation仍0。旧PBS已用3/4、尚余1次，但没有真实smoke门，旧main不能自动提交。不能把这个错误写成模型训练质量FAIL。
- 当前peer664176.hopper-m-02属于CFP04-CF-009、R/1GPU；不取消、不借用、不把其资源计为本任务。
- 本人quota最近used761426853888B/hard1099511627776B，约314.9GiB可用，已不是下午“差16.67GiB”的状态；在真正提交和allocation仍现场核对。

以上为读取时点的收据；采用时若又有新合法作业，以新事实握手，不覆盖或重复启动。

## 2. 新goal的第一项Hopper工作必须对准这个具体故障

先保存旧失败日志/manifest及精确退出位置。对计算节点正式支持的可执行文件路径、实际realpath、文件类型、执行权限、解释器/module环境和hash分别给可读诊断，不再合并成一个无法定位的REQUIRED错误；CPU测试应分别覆盖这些分支。

**不能假定login节点的绝对amgr路径和二进制字节必然在compute节点完全相同。** 调查集群原生支持的account/project查询方式。若确实不同，使用明确记录的compute侧合法路径/module/身份；不要把login-node脚本或凭据擅自复制过去，不注释掉全部allocation权限检查，也不安装系统级工具。

若compute节点本来就不提供amgr，仅在核实集群允许的流程后，将本项目自建guard中的这项路径假设改为：提交前fresh且成功的本人project授权证明 + 调度器接纳的exact job owner/project/resources + allocation内真实job/node/GPU绑定。需要记录为什么这些证据足以替代“同一个amgr二进制必须存在”的假设。官方调度hook、项目权限、本人资源上限和数据/模型身份仍不可绕过；有真实权限否决则停止GPU线。

这属于本次明确新goal授权下的**工程准入实现修复**，不是放宽机器人安全/任务质量门。新guard在新PI0293快照封存并CPU验证。下一次自己的小型合法allocation应在同一次作业内先打印非敏感逐项计算节点探针、通过后马上进入真实原生model smoke，避免再用整轮PBS只发现一个模糊路径错误。所有新提交计入本次最多4次qsub；不篡改旧PI0292的3/4计数。

## 3. 与多任务主线并行，不能继续只写guard报告

Hopper工程负责人处理上述单点故障；CoreLab负责人同时实现并行live worker及两技能三任务场景。按照已有数据、认证与GPU实际状态调度，Drop-A就绪就微调。全局14h墙钟/12H200h/3h评价预留等仍使用同一PI0293T0。

若PI0292仍无健康神经作业，新goal优先多任务，不用原本仅余1次main提交额度把新研究永久锁死；本次是新的用户授权和多任务数据/模型问题，旧结果永久保留为未完成。若已出现健康旧作业，保持运行并遵守本团队single-H200槽及账号two-allocation上限。

这次应力争取得真正模型load/update/生成/闭环的视频和证据，不能把更多CPU准备测试作为主目标；确实仍被平台权限阻断则继续数据线并如实交付DATA_READY_POLICY_NOT_RUN。

固定依据：
- [最新阶段状态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/ab3050c8cffeccd60e4ea362d8d7793b7ca929da/reports/PI0292/20261013T025824Z-localpc-pi0292/PHASE_STATUS.json)
- [真实smoke03失败收据](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/ab3050c8cffeccd60e4ea362d8d7793b7ca929da/reports/PI0292/20261013T025824Z-localpc-pi0292/S3_SMOKE03_TERMINAL_RECEIPT.json)

本次ChatGPT只发布计划和源文件；没有在CoreLab/Hopper调用模型、仿真或qsub。scene generator尚未在本会话运行，需owner实际运行/校验/封存，更不能据此声称场景已经物理可达。
