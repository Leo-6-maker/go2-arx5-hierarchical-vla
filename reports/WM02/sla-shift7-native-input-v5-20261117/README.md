# SHIFT7移动原生输入接入候选 V5

**实际状态：SOURCE_CANDIDATE_NOT_DEPLOYED；新增模型生成、优化更新/访问、相机采集、原生请求和2ms积分均0。** 用户“授权推进”已记录并用于继续工程工作；独立签发人的身份、角色及具体READMISSION决定尚未给出，不由执行Codex代填。真实移动抓持、抬升20mm和保持1秒目标仍未完成。

本次针对一个影响实际接入的缺口：旧移动`consume()`虽调用`close_reference_input_v1.remember`，该函数只记录`CLOSE_TIME500`，旧预测帮助函数也只为该方法传递`previous_reference`。直接把方法名换成`SHIFT7_SINGLE`，不会得到SHIFT7时间解码器必需的上一条实际发出夹爪命令。R1/R3的保存状态输入不能证明这段移动消费→下一query路径可用。

## 改动与实际验证

新增版本`MFREE_SHIFT7_PAUSED_V5`、方法`SHIFT7_SINGLE`限定M_FREE。在原`consume()`确认完成25个原生tick后，沿既有`remember`调用记录实际发送命令宽度及来源时钟；拒绝部分、跳序或倒序原子。当前夹爪测量值可能受物体阻挡，继续作为可得状态输入，但不替换上一条实际发出命令。旧TIME及其他方法保留原行为。

原`TargetSequentialAdapter.cancel_reference/close`新增仅对SHIFT7生效的历史失效调用。取消在尝试发布前即失效，因此即使ACK缺失，也不能拿旧历史生成下一条SHIFT7请求；既有不可变消费记录保留。这不是新物理STOP，不改变CANCEL/HOLD ACK的原语义。

新增`shift7_native_input_v5.capture/verify_query_return`候选：调用原`_pause/_after_pause/_same_pause`，比较真实格式状态包、quota、fresh pause ACK和物理/FSM/腿历史；绑定实际scope spec、同一native输出目录、冻结scene相机与episode固定frame。样本完整qpos/qvel/ctrl必须与当前原生packet一致。文件仅含普通body/wrist RGB、可得EEF状态、固定frame、原编译器参考及上一实际命令时钟；教师stage、object truth、未来H40均不进入模型输入。

候选要求spec显式声明总`max_ticks≤30000`，初始化也计入；记录实际剩余可消费原子上限，部分P10不会标成10点。没有修改旧120-query循环、初始化、物理dt或历史评分。全H40编译与WBC消费仍需由获准的实际consumer完成；不能把剩余消费上限当成截短H40的许可。

定向CPU验证共**78 passed / 0.98s**，使用明确的非积分fake adapter、原二进制packet解析及原pause ABI。覆盖上一发出命令与测量开度不同、truth字段排除、取消/换episode、状态/ACK/quota/腿历史/输入hash变化、STOP、取消ACK失败后的历史失效、部分原子、总tick边界及旧TIME兼容。**没有新Core运行、真实相机渲染、GPU模型调用或native动态STOP资格。**

候选源码`c8272b8b94067ade9bca78bed9708c44a4bb049e`；236文件源码seal见[SOURCE_SEAL.json](SOURCE_SEAL.json)，seal SHA及全部实际效果见[EFFECTS.json](EFFECTS.json)。仅新增接入代码和历史管理；原模型/编译器/夹爪映射/H40/速度加速度门、WBC/MPC/腿/臂PD/solver/碰撞保护未改。没有热改PR401已在Core验证的980933e0候选或其它胶囊。

## 接下来需要的实际结果

这份候选没有native launcher、模型调用或恢复积分API；旧resident close-pair没有因此获得SHIFT7调度能力。它修复了后续原生consumer必经的输入及命令历史接缝，**不等于整个闭环已经接通**。

下一项实验仍是独立决定范围内的动态STOP/reference资格，之后才可能进入专家安全8与学生pilot。依据固定[PR399 V5](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/4ba0c36918f570309558ba1c6283fb0b6d0438d8/docs/goals/2026-11-17/MFREE_SCIENCE_FIRST_CLOSED_LOOP_V5_20261117.md)，签发者需明确风险接受、profile/source/scene/WBC/guard及有限范围。现有[一次性决定材料](../sla-science-first-paused-v5-20261117/READMISSION_DECISION_REQUEST.md)中的actual selected-reference和动态STOP仍未知；当前签发source若纳入本候选需绑定c8272b8b及新seal，而不是把旧980933e0的CPU结果当成本候选现场资格。

“授权推进”使工程继续，但未说明由谁以何种实际角色签发、是否仅限安全资格、或是否接受完整接触科学面板风险。当前不将其改写成LIMITED_SCIENCE_READMISSION，也不据此填造GPU/site grant。训练、C60/C100、RL、旧失败重投和原生隔离均保持原约束。
