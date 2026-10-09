# PI0287 完整目标核验：未完成

依据原canonical #6084617266、release `7a9f0c02b55222646601307ccd7922e9297e2ab5` 和用户增加3次启动的明确回复。上一goal turn已取得六项真实结果、CPU复算和HF回读，属于进展；不能据此把完整目标改写成“环境能显示”或“失败报告已交付”。

| 原要求 | 当前可检查证据 | 结论 |
|---|---|---|
| 私有WBCv1与合法Isaac Gym身份 | SOURCE_IDENTITY、原RUNTIME_MANIFEST；本轮33文件当前SHA复核一致 | 历史完整Stage2/Dog已绑定；新版包不含完整Stage2，不能混称新版 |
| 多actor root/DOF/body/contact索引与真实刷新 | environment.py、indexing.py、n=1/2实际RESULT、CPU索引检查 | 已实现并有部分实证；完整奖励/终止/Jacobian/last-state使用链未逐项实证 |
| env0 reset不改变env1 root/DOF/object/reference/history | pinned n=2实际paused reset和已检查history | root/DOF/object及所检查history通过；native reference/history全链未完成 |
| 独立真实夹爪及物体接触 | 两命名关节实际开闭、对称性和物体落桌 | 空载通过；双侧物体接触/抓握判据未实际校准 |
| 同状态真实双96RGB、相机矩阵/安装/时钟 | 保存NPZ与camera_trace、实际视频 | 真实RGB存在；不是32秒任务完整20Hz同步数据 |
| 世界EEF→原Gamma/TimeLaw/native preview/progress，首个actor实际读取新参考 | bridge.py、实际canonical/installed数组及CPU原生模块检查 | 安装路径存在；CPU复现derived actor目标缓存未刷新缺口，真实首点缓存未保存 |
| 六项各10s、无abort/硬限穿透及末2s跟踪 | 修复版六JSON/NPZ、95完成tick、CPU重算 | 0/6；全数早停，末2s指标未观测；hotswap尚未到首个替换时刻 |
| 有限开发、冻结教师后pilot8≥7/8 | teacher.py/scorer.py CPU；实际gate | NOT_RUN；G2未满足，不能执行 |
| TRAIN64≥56/64、DEV16≥14/16与全部失败分母 | 固定协议/任务卡、DATA_MANIFEST | NOT_RUN；实际任务数据0，不能拿参考诊断冒充专家数据 |
| 实际四通道命令/反馈/RGB/诊断，H40未来实际命令与边界 | recorder.py CPU、dataset.py CPU | 模块存在；无真实任务流，因此实际H40数据证明缺失 |
| TRAIN-only统计、物理codec往返、官方processor/collator真实读取 | prepare_finetune.py与TRAINING_READINESS | NOT_READY；未拟合/未读取官方任务输入，不启动GR00T |
| 可执行doctor/scene/reference/teacher-pilot/collect入口 | doctor.py/run.py及实际运行收据 | 前三类部分实证；teacher-pilot/collect仍只有guard，未完整接线 |
| 代码、命令、真实视频、原始证据、HF回读 | CLOSEOUT、SOURCE_IDENTITY、72文件7,956,019B固定HF回读 | 已交参考阶段证据和真实短前缀视频；没有任务视频/专家数据/训练就绪证明 |
| 全预算累计、GR00T/WBC更新0、无真机/旧训练重启 | LAUNCHES/BUDGET、四RUN、当前无自有SIM PID | 4启动/21env/784physics/205预扣/38RGB；完整CPU总量UNKNOWN，不能宣称全cap已证明PASS |

## 新的零模型接入证据

`audit_install_cache.py`复用已有实际私有Gamma/TimeLaw CPU setup；原生源码AST核对actor读取ee_pos_err、ee_rot_err_axis_angle、arm_target_pos_body、arm_target_quat_body。安装新的[2,3,4]世界目标后，这四类预置缓存不变，原生`_update_ee_task_space_error`没有被调用。结果为 `CPU_REPRODUCED_INSTALL_READOUT_CACHE_GAP`。

这证明安装/readout接口存在缺口，**不能证明六项实际FAIL由此造成**：实际首个actor这些缓存、完整obs/history/action未保存。硬限穿透和native abort事实及原STOP保留；独立CPU重算通过仅证明限位算术，不能补成整个执行链的正确性证明。

最小修复方向是在首个actor读取前更新原生派生目标状态，并保存实际新参考与读取缓存的一致性；不得借机更新progress/gain/weight或重写失败。当前没有把此改动应用到remote runtime，也没有再起GPU。原协议两轮停机接入修复已消费，用户只额外增加了启动数；此外现有G2失败门仍约束依赖链，不能把剩余两次启动当成质量重试授权。

当前仍有明确未完成要求；goal不能标COMPLETE。科学链停止，CPU证据可交审查；是否追加同语义接入修复和重新资格验证需后续明确决定。selected/deployment=NONE。
