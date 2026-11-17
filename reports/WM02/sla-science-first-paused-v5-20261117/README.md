# V5：科学优先的暂停仿真合同——当前实际结果

执行固定[PR399 Goal V5](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4ba0c36918f570309558ba1c6283fb0b6d0438d8/docs/goals/2026-11-17/MFREE_SCIENCE_FIRST_CLOSED_LOOP_V5_20261117.md)。**G0与G1的非物理工程已实现并在Core验证；S0独立准入未签，新增原生0，仍无SHIFT7真实抓持成绩。** 严格350ms实时优化是[PR400并行支线](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/400)，不自动阻塞未来独立获准的暂停仿真研究。

## 当前真实分母与代价

| 工作 | 实际执行 | 不可外推 |
|---|---|---|
| G0 | R1/R3主unit inactive、PID0/cgroup空，三lease CLOSED，原owner一致；原F1/F2隔离保留。回读235候选源码、单模型包30成员（212,883,052B）、18历史scene/MJB及冻结二进制/配置 | 新GPU/site授权、scene准入、现役原生部署 |
| G1 Core缓存接缝 | F1/F2各1个已保存R3输入/ACTION，2/2原完整H40编译；每条3点部分消费后补7点，合计20个mock ACK；0实际相机采集、0新生成 | 学生on-policy、真实WBC消费、动态安全、抓持或任务成功 |
| 成本 | 成功CPU程序wall **1.911376s**、process CPU **0.094479s**；49项本机定向CPU回归；Core运行后独立读回235文件及20个mock receipt、原R3 hash不变 | wall耗时含存储，既非推理延迟也非native停止延迟；未采集本轮host RAM峰值、G0累计CPU时间或首个失败调用CPU成本 |
| 全部新增科学效果 | 模型生成 **0**；优化更新 **0**；优化访问 **0**；原生请求 **0**；2ms积分 **0** | 原生抓持/20mm抬升/1s保持均UNKNOWN，不能写0/2任务成绩 |
| 工程失败保留 | 首个CPU设置在创建任何receiver前遇到ACTION SHA不匹配；1次失败调用、0 mock请求/积分。原失败空目录及960951fe源码保留 | 不是旧R3重新执行；未替换有效科学失败 |

成功候选执行source **`980933e0896c247edeb7d9faf66c11fb987ce4b0`**；235文件seal **`31ee0ca7c67611cc01e24de035daafbd68f73e07a506e79128938fcaffe98c93`**。Core结果SHA **`b4c812673ef3b9907cd84b6760b464a16943dd13e545504d8e383861551ed68f`**。完整结果见[Core缓存接缝](CORE_CPU_SAVED_PAIR_RESULT.json)、[退出后独立核对](CORE_POST_EXIT_VERIFY.json)、[绑定](G0_CURRENT_BINDINGS.json)、[成员回读](SHIFT7_MEMBERS_READBACK.json)、[source seal](CPU_CANDIDATE_SOURCE_SEAL.json)、[效果账](EFFECTS.json)。本次是有限CPU调用，**没有挂载新GPU或原生后台unit**。

## 实现与原合同边界

新增`sim_paused_contract_v5.py`复用原`hybrid_nonphysical_v3.Receiver`与`compile_proposal`，没有改它们。新会话只接受固定SHIFT7单模型包与已记录初始命令证据，强制配置外部STOP入口和有界墙钟服务等待。request绑定profile hash、10s服务限时和模型身份；观测可用时刻与请求创建时刻独立记录，过老观测不能通过重置创建时间继续。

生命周期为QUERY_BOUNDARY→PAUSED_WAIT→PAUSED_ACKED→MOCK_RESUME_PREPARED→MOCK_CONSUMING→QUERY_BOUNDARY。native时间在query等待、编译、ACK和恢复前验证期间必须保持原观测边界；STOP、取消、进程generation、episode/request/binding或动作hash变化即失效。准备与真正mock恢复之间再次检查，token只可用一次。取消/重启清空可执行历史但保留已消费点和不可变回执。后续上一夹爪参考仅来自原接收端已确认的实际mock原子，不读取未消费H40后缀。部分P10不能finish成10。

保持原40×50ms完整编译、原EEF/夹爪映射一次、无BASE oracle、400ms native TTL、严格完整50ms余量及相同后缀续期。新profile**不把native TTL后移到ACK**；暂停改变的是受审时间域，非原门限。WALL_CLOCK_STRICT原R3仍失败。原WBC/MPC/腿/臂PD/solver/2ms/碰撞保护均未修改。代码只有mock token，**没有MuJoCo、ROS、actuator、native resume或hook安装接口**；因此CPU通过无法自行恢复隔离。

首个Core CPU设置将包含normalized等字段的`SHIFT7_OUTPUT.npz`误当接收端`ACTION.npz`，却使用后者SHA。身份门正确拒绝；最小修复绑定原ACTION，加了“Provider归档与接收端动作不同”的回归，以新source及新输出目录执行。没有改旧ACTION、原R3 records、判定门或封存源码。

定向验证：`PYTHONPATH=.:src python3 -m pytest -q tests/test_sim_paused_contract_v5.py tests/test_hybrid_nonphysical_v3.py tests/test_hybrid_request_v1.py`，**49 passed**。覆盖STOP等待/ACK后/部分消费，时钟前移、有限超时含观测年龄、旧响应/绑定、完整H40第31点非法、ACK/action/compiled/wire篡改、原续期与部分历史、进程重启。现有pytest `pythonpath`配置提示不影响结果。这些不是真实native hook资格。

## 需要的一次实质安全决定

已完成[单页决定请求](READMISSION_DECISION_REQUEST.md)，不自签。最小缺口是实际MPC选中reference身份、拟议native STOP hook动态最坏响应/暂停期间wall-STOP，以及最终native adapter/scene/site的获准绑定。三碰撞、连续告警12/20/34ms、原监测器1/38误拦、RP8/JOIN10和旧学生0/13结果全部保留。原生请求继续0。

责任人若认为接触证据不足，可以明确拒绝并给最小缺口，或只签独立预算的非接触STOP/reference资格；不能默认启动8+8+32+40。未来科学GO仍由真实原生grasp、≥20mm抬升、≥1s保持与独立≥8cm任务方向底座位移证据判定。旧GPU212生成额度已经用尽；本轮没有新授权、没有自动延长旧11月19日allocation，没有训练/C60/C100/RL。
