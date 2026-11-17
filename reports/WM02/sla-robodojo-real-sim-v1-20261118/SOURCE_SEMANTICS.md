# SOURCE_SEMANTICS · first12 actual

**BLOCKED_LABEL_SEMANTICS.** 首12条在总60条候选内，12/12文件SHA与固定HF LFS字节一致；保留SOURCE_NATIVE，0条获准进入本次command混合。没有启动Real10/25训练、模型生成或新原生请求。

固定HF revision：`1127bda94f172d757386945af3eb3a6754d1a4e6`，仓库`RoboDojo-Benchmark/RoboDojo`。实际6个ARX X5任务各100条；元信息先固定每任务10个候选，再取其中前2个。60候选声明9.9795GiB，实际只下载12条2,021,030,623字节。未因结果更换episode，余48条未读。逐文件路径、SHA及暂定划分见SOURCE_INVENTORY.json与SPLIT_MANIFEST.json。

## 动作和反馈的实际关系

12条共12,585帧。对左右臂各三种字段（arm_joint_states、ee_poses、ee_joint_states），每条均有`action[t] == state[t+1]`，覆盖75,438个字段行；最后一个action也与最后state相等。这里是浮点逐值精确相等，未把NumPy比较称为原始字节逐位证明。文件名叫action及action与同帧state不同，都不足以证明它是控制器发出的命令。

这与“下一帧反馈序列＋尾部重复”的构造完全一致；尚未取得实际采集器对这些字段的生产者/dispatch定义，故不把该数值证据提升成已执行命令证明，也不伪造ACK、P10、5703初始化或上一实际命令时钟。固定计划要求future-achieved源标签另立source-only研究，不能偷偷进入本轮command混合。

## Frame / TCP / 时钟 / 夹爪

官方字段转换按每臂xyz+wxyz四元数+gripper拼接，未提供本批真实采集的frame/TCP标定。已查真实文件没有标定、时间戳、session/operator或scene簇字段；只有标量frequency=30，不能据此声称测得硬件延迟或完整相机/命令同步。当前只使用显式名义网格作数值诊断，不建立可训练canonical reader。

上游policy对real arm-base坐标作假设，并可加双臂base间距偏移；官方仿真双臂配置的±0.3m不是这批真实数据到Go2的标定，未复制。SO(3)数值与单位四元数检查不能解决未知TCP绑定。

源gripper是每臂一维记录标量，不能根据0–1数值范围就宣称已测毫米宽度、开合方向或已发命令。CODEC_COMPATIBILITY只作条件性的`.002+.040*g`等效宽度诊断，未clip、放慢、平滑、改decoder或调用硬设2秒的旧labeler。

## RGB、双臂与划分

检查head和左右wrist三路长度/尺寸；每条解码head首/中/末帧及双腕中帧，共60张，60/60含标准XPL-RGB1标记，均480×640×3。原压缩JPEG字节保留；未再编码训练数据，也未构造未来观测输入。

稀疏图像可见双臂工作区及部分双手活动，但不能证明完整窗口内另一臂没有支撑/搬运依赖。本轮未把静止另一臂当作独立性资格，未生成未经核对的单臂子任务文本。12条raw与完整numeric指纹互不相同；这只排除完全重复，不能证明session/scene独立。6/2/2仍是候选槽位，SOURCE_DEV/TEST尚未读，所有合格split计数均0。

## 来源与边界

- [固定HF数据卡](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/blob/1127bda94f172d757386945af3eb3a6754d1a4e6/README.md)：Apache-2.0声明；代码许可分别记录。
- [官方真实任务目录](https://robodojo-benchmark.com/doc/real-tasks/)：遥操作及任务定义，不能代替本批命令生产者/标定证据。
- [固定上游converter](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/scripts/convert_robodojo_eef.py)、[pose policy](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/policy/SLA_Memory/openpi/src/openpi/policies/robodojo_policy.py)、[RGB编码/解码](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/utils/process_data.py)。
- [上游real入口](https://github.com/thriving191228/XPolicyLab/blob/c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454/utils/run_real_env_client.sh)明确公开版不支持real-world evaluation；所指真实client/采集器不在已审阅公开树中。未执行该入口、未连接硬件。

下一项需要命令/反馈生产者、frame/TCP/同步及单臂独立性证据；若确认future-achieved，则SOURCE_NATIVE预适配或通用夹爪表示须另立科学版本。当前0/10/25配比不变更变量、不启动。
