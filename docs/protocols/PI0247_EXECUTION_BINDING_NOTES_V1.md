# PI0247 — 执行绑定说明 V1

本说明与 `PI0247_GENERATION_CONSISTENCY_TO_SELECTED_SIM_V1.md`（固定commit `1ce12dcfefa37c31eb6228f19bafbeed0e706df3`）一并读取。Authorization保持 `PI0247_GENERATION_TO_SELECTED_SIM_GO`。它明确原方案的实现含义，不增加模型、loss、训练步数、seed、诊断或验收范围。

## 1. M干预不直接覆盖active，但可以影响其网络预测

主协议启动正文中的“不影响active输出”，以第4.2节精确定义为准，严格解释为：**不直接覆盖、裁剪、平滑、重定时或用专家替换active0:10的值**。inactive状态改变之后，网络重新计算的active预测可能变化，这正是H1要测量的因果通道；不要求两条路径的active输出相等，也不能把这种差异当成工程错误。

M在每个Euler点对inactive执行已定义的解析路径，active仍由该次head输出更新。诊断和U梯度实现均用out-of-place的mask组合，避免in-place修改autograd已保存状态；M路径只含噪声/tau/静态padding信息。

## 2. 更大支持集的两组噪声如何固定

D1和D2沿主协议复用原实际packet。

M64新增48个source的B0，在该stage首个模型查询前，由独立CPU `torch.Generator` seed22301生成一份[1,40,132]标准正态float32 tensor并封存；对新增source使用同一份tensor，明确其跨source共享。原16的B0仍是原S0逐source实际packet，不覆盖。B1对全部64继续使用已封存P1 N1。保存源float32与R*实际入口转换后的tensor身份。

FULL固定64 TRAIN screen继续使用M64同一组packet，不根据checkpoint重新抽取。CALIB保持主协议的baseline逐source首个实际noise捕获/封存和四候选复用；不根据baseline成绩重取noise。两噪声组不当成独立数据集或32/128个独立样本。

本规则是确定性的prospective binding。新增的B0构造不称旧screen exact replay；历史16对应关系和源hash保留。

## 3. 模型/运行身份与旧结果

D1固定P-final2000与R0-final10000，模型身份由原receipt/manifest的完整内容确定，不能只认HF前缀或目录名。P不是PT，原误抄摘要的hash应由独立closeout及原结果校正。

S4作为runtime ID表示原四步stock；selection的S4表示独立的正式checkpoint选择阶段，两者不要在manifest中共用字段。建议使用 `runtime_id` 与 `selection_stage` 分开记录。

即使D2中原P权重配合新runtime通过，也只标 `PI0247_M16_RUNTIME_QUALIFIED`；不撤销PI0246旧stock FAIL，不把P写成FULL selected_checkpoint。正式S4选择前selected_checkpoint保持NONE。

## 4. U不是标签条件推理，也不是DAgger

U中的完整可微rollout从纯噪声和label-free观测条件开始。a_star只用于原native训练目标和最终监督loss，不能出现在U的状态初始化、动作前缀、RTC、状态替换或中途校正中。

U是离线可微生成训练，不运行WBC、仿真或采集on-policy expert标签，因此不是DAgger。训练时eval mode与autograd分别控制：U关闭随机dropout但不能关闭梯度；native分支原mode保持，退出时恢复。旧S0、P1、C/T/P/PT不因新增U再次运行。

## 5. 执行纪律

主协议阶段PASS后的条件推进已经授权。真实输入/标签/decoder冲突、不可容纳的完整梯度图或原sim参数无法唯一恢复时，按对应typed blocker停止受影响阶段；不为补历史日志增加模型试验，不把普通科学FAIL称工程故障重试。

两文件发布均不等于远端实验已启动。Issue #11是canonical，PR #77只pointer，不merge。没有新monitor。
