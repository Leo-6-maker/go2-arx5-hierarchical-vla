# Codex执行交接 — WM01

基线：`0a27753bbda8ed75ab6b74aeb64755e0427362c7`；新计划：[WM01 protocol](../../protocols/WM01_TASKSPACE_WORLD_MODEL_POLICY_V1.md)。本文件及所在提交是用户要求的GitHub交接，不启动新的Codex会话或服务器owner。

## 你要完成什么

在保持原生执行器与安全不变的情况下，实现一个小型结构化WM和task-space actor学习实验；用真实源域分叉/闭环证明或否定它的价值。**不是继续微调一个GR00T轨迹头，也不是重新训练WBC。**

先读根AGENTS、短START_HERE、WM01计划与`experiments/wm01/plan.json`。V7允许同范围工程自主修复；不要为普通路径、序列化或新CLI实现反复索要GO。实际站点许可、资源/owner缺失时报告一个具体缺口，不能伪造grant。

## 0. 接管与资源安排

取得本交接的完整SHA并建立独立WM01开发分支/source capsule。刷新相关#11/#47和实际M13/PB/M32/consumer状态；GitHub当时的1514400/PID不是当前真值。保存dirty/local-only工作，不reset或热pull活动路径。现役运行继续，不改shared59/DEV68/旧对照、不停止其他用户GPU。

沿现有entry/profile/registry/Journal/授权HF路由。不要新增scheduler、预算ledger、审批系统；不要逐轮扫描全部历史模型。管理端可做相关轻量检查，AMD不跑重型训练/仿真。目标S1/T2与实机仍不在本轮范围。

## 1. 当前真正存在的命令

```bash
python experiments/wm01/check_plan.py
python -m unittest discover -s experiments/wm01/tests -p 'test_*.py' -v
```

它们只验证计划一致性，不做native/model资格。本提交没有WM训练器或采集CLI；下面是要实现的接口，不能直接宣布已执行。

## 2. 按工作包连续推进

### W0：复用原生源链，先拿到正确转移

定位当前T1源任务card、Teacher和score，锁完整SHA；不要将`night_score`的整机标记用在固定底座。复用`pi0291_arx5_data.engine.Engine`及当前源adapter，新增WM薄wrapper，不修改IK/力矩/gain或安全阈值。

实现不同state白名单、50ms真实区间记录、0.4秒macro拼接、8×7→EEF10/H40 compiler；验证anchor、row-rot6d、夹爪范围、prefix8消费，区分raw/installed/consumed/achieved。清点已有TRAIN记录；缺驱动器状态不能填零。

完整save/restore包括native积分状态、目标/缓存/RNG/监测记忆。用相同状态/命令恢复两次，验证后续trajectory/contacts/terminal一致；若只有数值容限等价，记录容限和误差，不谎称bitwise。4个TRAIN canary使用可信原Teacher和原评分。

完成`RESOLVED_SPEC.json`与数据审计后，在实际profile内采集初始64条有界尝试，包含失败；不足可读转移则报告不足，不用反复补成功掩盖。先固定TRAIN/CALIB/DEV/TEST及parent分组。必要native原件在私有存储，Git只放脱敏摘要。

### W1：单卡WM实际训练与恢复

实现残差ensemble、SO(3)/事件头、多步loss、replay与恢复；先CPU合成合同/梯度测试，再一张已获许可GPU的200-step smoke及完整checkpoint恢复，然后固定5000-step首模型。不要先解冻VLM或新增视频生成器。

固定持恒/action-shuffle对照；报告连续物理单位误差与接触/失败分层。单步MSE好不等于可用于规划。

### W2：真实分叉判定模型是否有用

固定64 CALIB实际状态、五种动作方案、最多4macro展开。使用完整native恢复，真实执行梯度建议/反向/随机等对照；所有分叉计成本。按协议报告排序、regret、梯度方向和真实改善率。未满足决策有效性条件则不开始actor BPTT；给负结果及一个有证据的改进方向。

### W3：共同初始化与两臂

在相同有效BC标签上生成共同初始化（无法表示时两臂同随机，优化前固定）。实现共同PPO循环；B仅增加协议定义的WM/想象更新。模型参数freeze≠`torch.no_grad`，必须测试action gradient非零且模型参数不更新；真实terminal和time-limit bootstrap分开。

先seed1103两臂各5000真实macro，按端点评DEV；合格且无工程阻断再同源续到20000，并完成另外两个seed。所有臂记录物理互动、优化次数、总时间与设备；不按成绩换seed/调reward。不得在TEST选checkpoint。

### W4：交付科研结论

固定20000端点，在50个新TEST布局、3seed、两臂做原生完整评价；报告未运行、拒绝、硬安全与成功各自分母。统计按scene/seed而非窗口；同时给等交互和实际计算成本。正负结果都形成`reports/WM01/<run_id>/FINAL_REPORT.md`与机器可读results。

W0/W1完成但W2失败仍是有效阶段交付；不要用“尚未达所有关卡”抹去已经产生的数据/模型，也不要无界加训练。W5视觉蒸馏、DOOR扩展与目标迁移在首轮结论后给独立配方，不自动用旧权限跑真机。

## 3. 建议实现接口（NOT_IMPLEMENTED）

源码职责见[CODE_MAP](../../CODE_MAP.md)。实现后才登记以下薄CLI的真实命令：

```text
collect.py    --spec RESOLVED_SPEC.json --run-id ...
fit.py        --spec ... --data ... --resume ...
diagnose.py   --spec ... --checkpoint ... --split CALIB
train.py      --spec ... --arm ppo|ppo_wm --seed ... --endpoint ...
evaluate.py   --spec ... --checkpoint ... --split DEV|TEST
```

所有CLI接受明确run/output/profile绑定；import无仿真/模型/网络副作用；`--help`不能构造场景或下载模型。复用原launcher，不创建自己的常驻服务。若已有同职责模块，优先adapt并测试，不重复实现。

## 4. 必要测试，不扩成全历史仪式

- 普通Python和`python -O`均拒绝跨episode/reset和错误time/frame；生产检查不用assert。
- consumer实际执行prefix与标签一致；H40未消费后缀和future-achieved不能做命令标签。
- 同parent分叉/恢复始终同split，TEST不可进入归一化/replay/调参。
- primitive边界、SO(3)、第一anchor、夹爪；两臂compiler一致。
- 完整native恢复；训练resume含model/actor/critic/optimizers/RNG/normalizer/replay cursor。
- WM parameter freeze与actor input-gradient；真实reward与模型reward实现逐值一致；terminal/truncated处理。
- 无Teacher真值泄漏到视觉学生；无target WBC mutation；新文件不import时调度。

只改薄wrapper时跑新测试和相关旧seam；影响共用模块才扩大回归。fixture通过≠native执行；发布≠部署。

## 5. 一次回报应包含

写清当前工作包、实际命令/full SHA、数据/模型身份、真实转移/更新/完整任务数量、正负结果与首失败机制、资源和下一条可执行命令。常规来源与恢复信息用一份manifest，不为每项函数造一份权限收据。

长证据写reports，根入口只保留摘要链接。正常feature branch/PR，不force、不删除历史。用户将本交接交给现有执行agent后，它可在实际权限内连续推进；本文不承诺后台实验会自行启动或在某个时间必然完成。
