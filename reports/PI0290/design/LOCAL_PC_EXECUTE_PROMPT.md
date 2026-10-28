# 给现有local-pc Codex的PI0290执行指令

**这是用户批准的新有限科学任务；请做实际代码/CPU/MuJoCo物理/Teacher pilot，不要只写计划。**

仓库：`Leo-6-maker/go2-arx5-hierarchical-vla`；现役分支：`pi0267/executor-decoupled-recovery`。
先刷新Issue #11，并以最新明确发表的PI0290 canonical SHA固定本次执行，不从滚动branch HEAD隐含继承实验配方。
必须逐字阅读：
1. `docs/protocols/PI0290_INDEPENDENT_MUJOCO_TASK_TEACHER_AND_DATA_PILOT_V1.md`
2. `reports/PI0290/design/GO_TASK_CARD.json`
3. `docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md`
4. `reports/handover/2026-10-12/local-pc-2317-sgt/ADOPTION_REPORT.md`与`NEXT_DATA_GOAL_DRAFT.md`
5. `docs/coordination/2026-10-12_TASKANCHOR_UNITAS_DATA_AND_FINETUNING_REVIEW.md`、PI0286终局和PI0289终局。

## 目标

在Corelab搭建**独立于WBC**的MuJoCo 3.2.7动态工具/双指夹爪、桌面4cm cube→16cm zone真实物体任务；从Panda式Teacher生成有接触因果证据的EEF H40＋gripper监督，并从第一条episode保存完整20Hz双RGB、实测TCP/gripper、command、历史、事件、三维标定、物体运动与结果；做4个commissioning和8个固定pilot、独立CPU再评分，交付机器可读的实验数据质量证明。Teacher/工具不得瞬移物体、焊接/磁吸抓取，失败必须保留。

**WBC包只读参考：**私有`wbc_mujoco_minimal.tar.gz`的SHA256 `6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`；可复用几何/TCP/MuJoCo设置，但不运行或修改OCS2/MPC/腿RL/臂PD/控制接口；也不需要旧PI0289 G1 PASS来生成独立任务数据。数据source必须写`INDEPENDENT_DYNAMIC_TOOL_SIM_ONLY`，不可标为Go2全身WBC成功。

## 执行顺序

- D0：先在Corelab有限检查真实独立MuJoCo Python import、3.2.7模型/渲染后端、项目隔离目录、合法资源/个人quota；记录WBC只读资产hash；旧PI0289证据不改。缺独立runtime时可在私有隔离环境安装/复用，不sudo不干扰公共系统。
- D1：实现`task_data`新模块与独立物理工具，满足finger双侧实测接触、真实对象运动和双RGB同tick；注册场景和控制参数。CPU测试加最多4 commissioning，保留所有失败和预算。选择模型声明作为SIM-only，不冒充硬件规格。
- D2：pilot前冻结scene/teacher/driver/scorer/layout/seed/hash；固定P2900...P2907的4见过布局族+4未见布局族，8次first-valid实际episode。不重试有效质量失败、不看成绩后换seed。
- D3：保存每次原始物理/视觉/控制完整证据，复算完整32s与28-30s连续41点的抓握-抬升-搬运-释放-撤离-保持，以及真实接触和安全门；逐条验证实际20Hz、H40、rot6d、gripper极性、历史不泄漏和privileged字段隔离。统计8/8完整和≥7/8成功（heldout≥3/4）作为PILOT_READY资格；否则准确报告有效失败。
- D4：GitHub提交可公开源代码、单元测试、D0–D4 receipts、固定manifest、实际结果、短视频/原始证据位置。大原始数据只有在HF新repo许可/隐私/可见性和write scope确认后上传private/approved repo，READY最后写并逐文件hash回读；不能上传私有WBC/模型/mesh。HF不通时留Corelab私有数据并精准报告。
- 附带写下一PI0291官方预训练GR00T B0微调**设计草案**；本轮不运行H200/GR00T神经网络。

## 预算和禁区

最多12次物理episode start（4 commissioning＋8 pilot）、每次<=32s；2ms步长总<=200000物理积分（含注册工程setup8000）；相机/状态成对样本<=7692；Corelab GPU union<=1h、CPU<=24core-h、输出<=10GiB。一个live physics owner实例，不抢占其他用户GPU。H200新PBS、GR00T load/forward/backward/eval、WBC调用/训练、真机全部0。没有无限物理debug或同布局质量重试，不能用旧PI余额或修改任务门限追分。

首次阶段成果要实际贴Issue #11：D0环境、D1物理/相机、D2 pilot seals、D3真实得分、D4终局；当第一条完整抓放成功时尽早回报原始物体结果/录像证据，不等所有pilot完成。任何阻断交实际发生的代码/命令、时间、完整原失败和最小待决事项，不准写虚构PASS。

没有授权启动GitHub cloud Codex；现有local-pc是唯一owner。结束时`selected=NONE, deployment=NONE`。正式TRAIN64/DEV16及H200 GR00T微调必须等本次数据资格通过后另订协议。

**用户要的是新的数据基础设施和可用的真实物体专家证据；请推进到实际成果或精确阻断，不要自动转去调师兄的WBC。**
