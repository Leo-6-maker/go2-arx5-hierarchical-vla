# WM02 NEXT1 · 已实际执行并结算（2026-11-05）

实际执行接续固定`a673d4b0d361c6fea9fcd5d5b9cbd28f58461b4f`，原机制登记的新NEXT1窗口现已CLOSED。40次原生安全PASS、174920 tick；新共享replay28窗口/672转移；三组WM各200+1，Actor0。稳定前缀与真实动作辨识通过；神经M0科学Gate未通过，C1仅短暂预接触、尚无物体接触。**不复用CLOSED lease或把原生PASS当任务成功。** [实际成果/完整SHA/成本/接续点](reports/WM02/next1-20261105/NEXT1_FINAL.md)，[实际命令](reports/WM02/next1-20261105/EXECUTED_COMMANDS.md)。以下保留原计划说明作为历史上下文。

基线：[PR #356](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/356) 完整NIGHT1收尾 `3d62dc098eadf789aa92f962484e1bdc442c7a99`。NIGHT1全部动态worker/lease CLOSED、原截止已过，**不得重开或派发**。本文件与同分支的[详细方案](docs/handover/2026-11-05/WM02_NEXT_WINDOW_ACTION_EFFECT_AND_CONTACT_V1.md)和[机读Gates](experiments/wm02/next_window_action_effect_v1.json)只提出后续研究，尚未取得新scope。

## Codex阅读与动作

先读`AGENTS.md`和真实NIGHT1原件`reports/WM02/night1-20261105/NIGHT1_FINAL_CLOSEOUT_07.md`，再读本NEXT1详细方案/JSON。用现有原站点owner/registry/Journal/锁核对实际当前身份和权限；**没有新有效finite simulation-only scope时，只做R0只读诊断、静态接触复核与具体scope申请，不启动MuJoCo、WBC、GPU训练或旧命令重放。**

取得合法新scope后，优先`R0→S0→S1→D0→M0`：查C08起点42mm运动共模、设计并真实验证稳定决策前缀、采四父×四动作（HOLD/+X/−X/+Y 1mm）的原生动作差、冻结三种匹配RAW/SCALED/NO_ACTION 200+1模型对照。若模型有真实动作辨识，再在冻结后用预留P09–P16前瞻；接触线从当前静态IK再资格的单环境分级原生实验推进。每步来源/真实消费/动作/状态/安全/资源均留原Journal与独立不可变run。不要仅按最终42mm与1mm比值判定跟踪错误；分别统计自然settling、BASE运动、相对BASE TCP及参考跟踪误差。

预留20分钟结算；新scope未批准、模型/接触FAIL或物理STOP均保留实际PARTIAL/负结果，不替换原数据和WBC；S0不通过可继续离线，不得为提高通过率更改原生安全。**不运行正式Actor/PPO/GR00T、不扩4slot/不迁MJWarp/不自动合并main。**

## 本次交付效果

纯文档与配置：新增原生积分0、WM更新0、Actor更新0、worker0；下一步由实际Codex执行者在新准入下展开。