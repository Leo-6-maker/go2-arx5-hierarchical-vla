# PI0246 — Gate-Pass Champion 长程自主训练计划

日期：2026-09-24（Asia/Singapore）  
Decision：`GO_BOUNDED_GATE_PASS_CAMPAIGN`  
Authorization：`PI0246_GATE_PASS_CAMPAIGN_GO`  
发布状态：`AUTHORIZED_NOT_EXECUTED`  
协议基线：PI0245 commit `f33f345a2419b2eb59a542c2117834a799a2d567`  
当前执行证据：Issue #11 comment `5800283247`  
Canonical ledger：Issue #11；PR #77 仅 pointer，不 merge。

## 0. 总目标

PC agent 的首要长程目标是：

> **训练并封存至少一版在原 fixed16 TRAIN cohort 上通过原联合门的模型权重与训练配方。**

成功状态定义为：

`M16_GATE_PASS_MODEL_FROZEN — STOP_FOR_PI_SCALE_CONFIRMATION`

“Pass gate model”必须同时满足：

1. 原 SafetyGate：`REJECT = 0`，WARN 保留并报告；
2. median per-window XYZ RMSE ≤ 0.020 m；
3. p95 per-window XYZ RMSE ≤ 0.035 m；
4. median per-window SO(3) RMSE ≤ 0.050 rad；
5. 完整固定16窗、身份、输入、模型、噪声、指标与训练更新证据有效。

这不是“找到一个看起来最好”的checkpoint，也不是泛化、WBC policy selection 或 sim 成功。即便取得 gate-pass champion，`selected_checkpoint`仍为`NONE`；最终policy selection仍属于后续FULL-family selection。

本章程允许 agent 在下面明确列出的有限科学树内连续推进，不需要每个普通分支再次等待确认。未列明的loss、lambda、LR、seed、sampler、表示、gate或数据修改不得自动加入。

## 1. 当前状态与已闭合事项

### 1.1 冻结历史结论

PI0244 R0与R2结果保持不变：

- R0 final10000：5/16 REJECT，XYZ median 47.100 mm，p95 54.982 mm，SO3 median 0.03745 rad。
- R2 XYZ4 final10000：15/16 REJECT，XYZ median 48.746 mm，p95 59.655 mm，SO3 median 0.03623 rad。
- R2全部15个REJECT包含LINEAR_SPEED_REJECT；物理峰值集中于segment37→38。
- XYZ4未取得位置拟合收益，不能继续自动扫XYZ2/8/16。
- P32不适用的旧证据保留。
- `selected_checkpoint=NONE`。

不得重训R0/R2、改旧FAIL、挑中间checkpoint或以新实验覆盖旧证据。

### 1.2 PI0245 P0已完成

Issue #11 comment `5800283247`记录：

- TRAIN-only时序/decoder绑定与CPU数学检查PASS；
- 有效relative action norm_params为(40,9)，XYZ scale/offset为(40,3)，逐时间尺度真实存在；
- 原32条R0/R2 decoded XYZ可由官方逆处理重现；
- R2物理峰值16/16在segment37，而normalized raw peak主要更早；
- 35..39五点贡献较高SSE，但排除尾部后位置误差仍远高于准入线；
- 新物理delta loss的CPU fixture/decoder parity通过；
- 固定P1 noise bank已经生成并封存；
- P1 stock queries=0，P2 optimizer updates=0。

因此接手agent**不得重做P0**。先读取并验证P0小型manifest/hash，然后从P1继续。

## 2. 科学树总览

```
P0 已完成
  ↓
P1 固定32-query noise/scale diagnostic
  ↓
PI0245 P2:
  C = native 2K continuation
  T = native + physical temporal delta 2K
  ↓
PI0245 P3 paired screen
  ├─ C/T 任一联合PASS → 冻结 gate-pass champion → SUCCESS
  └─ 两臂均FAIL
        ↓
      若XYZ median或p95仍FAIL
        ↓
      P = native + physical position 2K
        ├─ PASS → champion → SUCCESS
        └─ FAIL
             ↓
      PT = native + physical position + physical temporal 2K
        ├─ PASS → champion → SUCCESS
        └─ FAIL → TREE_EXHAUSTED / STOP_FOR_PI_NEW_HYPOTHESIS

若PI0245两臂均FAIL，但两臂XYZ门都已经通过、只剩其他失败，
则P/PT不具备本章程的自动准入依据，直接STOP_FOR_PI_NEW_HYPOTHESIS。
```

最多新增的自动科学训练：
- 已授权PI0245：C 2000 + T 2000 = 4000 updates；
- 本章程新增：P 2000 + PT 2000 = 最多4000 updates；
- 总计从当前时点最多8000 canonical updates；
- 首次联合PASS后关闭后续未执行arm。

不授权20K/50K、seed sweep、lambda sweep、sampler step sweep或动态按成绩调参。

## 3. Goal A — 完成PI0245，不修改其定义

### A1. P1固定32-query

严格执行现有PI0245协议：

- 原bridge4 A/B/C/D各一窗；
- R0-final10000与R2-final10000；
- N0(seed22301)、N1(22302)、N2(22303)、N0_roll11；
- 32个唯一stock queries；
- 保存实际完整initial noise、condition、每个native denoising step的action state与predicted velocity；
- 不把相同seed称为历史screen exact noise replay；
- 不按结果替换noise或追加seed。

P1只用于机制证据。只要没有真实数据/decoder/ABI/identity冲突，不要求先证明唯一根因，直接进入PI0245 P2。

### A2. PI0245 C/T双臂

沿PI0245固定协议：

- 两臂均从相同R0-final10000权重开始；
- fresh AdamW/scheduler；
- 原16 TRAIN；
- 相同2000项曝光表，每窗125次；
- seed22301；
- LR 1e-5，WD 1e-5，maxgrad1；
- cosine2000，warmup10；
- batch1/accum1/workers0；
- C：`L_native`；
- T：`L_native + 0.1 L_phys_delta`；
- checkpoint1000/2000；
- C→T串行；
- 两臂均完成，不因为C先PASS而隐藏T结果。

实际sample/tau/noise/augmentation顺序应尽可能配对并保存；实现observer不能消耗额外随机数。普通有限control表现差不是工程错误。

### A3. PI0245 P3判定

主screen：
- shared step0 ×16；
- C1000/C2000/T1000/T2000 ×16；
- 80逻辑cells；
- 共享实际condition/noise packet，不能用seed代替张量证据。

仅final2000判各臂原联合门。

若一个臂PASS，另一个FAIL：PASS臂为champion。
若C和T都PASS，确定性选择：
1. 较低p95 XYZ RMSE；
2. 若数值相等到1e-6 m，较低median per-trajectory max linear speed；
3. 再平手，较低median XYZ RMSE；
4. 再平手选C（更简单的loss）。

保存另一通过臂但不继续训练。

若C/T都FAIL，进入Goal B的准入判断。

## 4. Goal B — Physical-position rescue arm P

### B0. 准入

只有当：
- PI0245 C/T均完成且均未联合PASS；
- 且至少一个final arm的median XYZ或p95 XYZ仍超原门槛；
- P0的decoder/scale binding仍有效；
- 没有新的标签、frame、ABI或统计身份冲突；

才允许P。

### B1. 物理位置目标

沿PI0245同一次native forward构造：
- `x_tau=(1-tau)epsilon+tau*a_star`
- signed `v_hat`
- `a_hat=x_tau+(1-tau)v_hat`

使用已验证的可微官方XYZ decoder镜像：
`E_k = D_xyz(a_hat)_k - D_xyz(a_star)_k`

固定：
- `p_ref = 0.05 m`
- 点mask：三个XYZ槽均为原有效mask；
- 三维L2平方求和，不额外除3。

`L_phys_pos = mean_valid_points ||E_k / p_ref||_2^2`

P arm唯一新loss：
`L_P = L_native + 0.1 * L_phys_pos`

这是物理单位尺度的绝对位置残差项；它与PI0245 temporal项不同，专门补偿temporal loss对恒定物理偏差不敏感的问题。

不得：
- 重新计算或统一TRAIN statistics；
- 改decoder；
- 去除clip；
- 加XYZ4；
- 加gate hinge；
- 加endpoint特殊权重；
- 按时间位置37单独加权。

### B2. P训练配置

Family：`PI0246_P_PHYSPOS_M16_2K_V1`

- initializer：同一sealed R0-final10000；
- fresh optimizer/scheduler/RNG；
- 原16；
- 与PI0245相同2000项schedule；
- LR1e-5、WD1e-5、maxgrad1、cosine2000、warmup10；
- 其他R0实际resolved配置、trainable/frozen、BF16 autocast、augmentation、noise/tau、H40、stats全部不变；
- save1000/2000 full Trainer states；
- 不warm-start C/T失败终点。

CPU/synthetic必须先验证：
- lambda_pos=0时native value/gradient完全一致；
- oracle a_hat=a_star时phys-pos=0；
- constant physical bias产生非零loss；
- variable scale、relative reference、clip/zero-range parity；
- mask/finite；
- 不第二次forward，不额外采noise。

### B3. P评价

优先复用PI0245 P3封存的shared condition/noise packet。

逻辑cells：
- shared step0不重复；
- P1000/P2000 ×16 = 32新增main cells。

final2000用原四项联合门。
若P PASS：冻结champion，停止PT。
若P FAIL：进入Goal C。

## 5. Goal C — Combined physical position + temporal arm PT

Family：`PI0246_PT_PHYSPOS_PHYSDELTA_M16_2K_V1`

从**同一R0-final10000**重新开始，不从P-final继续。

唯一loss：
`L_PT = L_native + 0.1*L_phys_pos + 0.1*L_phys_delta`

其中两项必须使用同一次native forward、同一a_hat、同一decoder镜像与同一原mask；不能重复sample noise/tau。

训练配置、2000 schedule、LR/optimizer/save点与P完全相同。

main screen新增：
- PT1000/PT2000 ×16；
- 使用与P/C/T相同的shared condition/noise packet。

final2000只按原四项门。

若PT PASS：冻结champion。
若PT FAIL：
`PI0246_GATE_PASS_TREE_EXHAUSTED — STOP_FOR_PI_NEW_HYPOTHESIS`

此时禁止自动：
- 把lambda改为0.05/0.2/1；
- 加速/jerk/endpoint loss；
- 20K/50K；
- 换seed；
- 改sampler steps；
- 改gate；
- 截尾或retime；
- 组合XYZ4/P32；
- 解冻backbone。

## 6. Gate-pass champion如何冻结

首次得到合法PASS后，写：
`fit_gate_champion_manifest.json`

至少包含：
- arm与recipe；
- parent/initializer；
- final checkpoint tree/hash/HF fixed revision；
- exact processor/stats/decoder；
- loss完整公式与lambda；
- optimizer/schedule；
- code/source/runtime；
- main16 condition/noise identity；
- 四项准入指标；
- gate issue分布；
- per-window结果；
- clipping/saturation；
- P1/robustness诊断状态；
- trainable/frozen/update证据；
- `target_xyz_in_model_input=false`；
- `selected_checkpoint=NONE`。

终态：
`M16_GATE_PASS_MODEL_FROZEN — STOP_FOR_PI_SCALE_CONFIRMATION`

注意：
- “gate-pass champion”是TRAIN-support fit checkpoint；
- 不是泛化checkpoint；
- 不是最终selected checkpoint；
- 不自动叫general model；
- 不直接送WBC。

## 7. 噪声稳健性报告

PI0245原final C/T ×bridge4×4noise诊断照常执行。

若P或PT成为champion，再对**champion-final**运行同一bridge4×原4noise = 16个固定robustness cells；同一noise bank，不新增seed。

这些结果不改变主16 PASS定义，但必须与champion一起报告：
- 若主16 PASS但多noise频繁REJECT，仍记录为M16_GATE_PASS_MODEL，但标记`NOISE_ROBUSTNESS_WEAK`；
- 不因此偷偷换主seed或重新训练到多noise都过；
- 后续scale PI据此决定是否需要sampler/noise robustness研究。

## 8. 工程、资源、恢复规则

- 本地PC：git、CPU tests、SSH、artifact hashing；
- Corelab：Linux integration、P0/P1及能容纳的screen；
- H100/H200：只有实际模型容量或正式训练需要时使用；
- 不在H200反复pytest/path debug；
- 一个arm一个active owner，C→T→P→PT串行；
- first-valid scientific outputs冻结；
- 工程失败可无限次针对性修复，但不能新增science cell/seed/step；
- exact resume仅在同一arm完整Trainer checkpoint基础上；
- report/HF/publisher错误只修报告/上传，不重训；
- 独立family/output/HF prefix，不覆盖R0/R2/PI0245；
- 不删除initializer、历史checkpoint、当前reader依赖和唯一durable副本；
- token不入Git/日志；
- 不新建monitor。

## 9. 排队期间允许做什么

可并行CPU准备后续P/PT：
- loss overlay；
- decoder parity tests；
- run-card schema；
- schedule/hash；
- storage estimate；
- screen reader；
- champion manifest schema。

状态只能是`READY_NOT_STARTED`。

在C/T final verdict前：
- P/PT不得backward；
- 不加载P/PT项目模型做额外真实science forward；
- 不提前按中间指标决定lambda；
- 不运行M64/FULL。

## 10. 本章程不授权的后续

取得M16 champion后，本章程**停止**。

M64/FULL/CALIB selector/WBC/sim/DEV-E/F/DAgger/hardware/merge均不在本章程内。可以只做CPU配置草稿，不得继续backward或sim。

原因：本次用户目标是“训出一版pass gate model”。先取得第一版可信的M16 gate-pass champion，再根据其noise稳健性、训练recipe和资源成本决定是否扩64/full，避免把“能拟合16窗”和“可扩展/可部署”混在同一个自动实验树里。

## 11. PC Agent 启动正文

执行 `PI0246_GATE_PASS_CAMPAIGN_GO`。

先完整读取：
- PI0246本章程；
- PI0245协议commit `f33f345a2419b2eb59a542c2117834a799a2d567`；
- Issue #11 `5800283247` P0证据；
- PI0244 R0/R2 canonical FAIL。

当前P0已完成，noise bank已封存；不要重做。
从P1 32-query开始，随后严格完成PI0245 C/T 2K双臂与P3 paired screen。

C/T任一final2000联合PASS时，按本章程确定性规则冻结gate-pass champion；
若两臂都FAIL且XYZ门仍失败，自动运行P physical-position 2K；
P FAIL后自动运行PT combined 2K。
P/PT都从同一R0-final10000重新开始，不能warm-start失败arm。

所有arm保持原16、同2000 schedule、seed22301、LR1e-5、
原R0精度/augmentation/model/freezing/sampler/H40/stats。
P只增加native+0.1 physical-position；
PT只增加native+0.1 position+0.1 physical-delta。
物理loss使用同一次native forward和已验证的scale-aware decoder镜像；
禁止第二次noise/tau采样、XYZ4、gate hinge、尾部专门权重或统计修改。

每臂只判final2000：
REJECT0；
median XYZ<=0.020m；
p95 XYZ<=0.035m；
median SO3<=0.050rad。
中间checkpoint只报告，不替代final。

首次PASS后关闭所有未执行arm，封存
`M16_GATE_PASS_MODEL_FROZEN`和fit_gate_champion_manifest；
selected_checkpoint仍NONE。
若C/T/P/PT全部合法FAIL，停止
`PI0246_GATE_PASS_TREE_EXHAUSTED — STOP_FOR_PI_NEW_HYPOTHESIS`，
不自动新增配方或步数。

工程问题自主修复不限次数；科学预算、seed、loss和门槛不能扩大。
本地/Corelab优先工程和小任务，正式容量需求才用H100/H200。
Issue #11先canonical，PR #77只pointer；不merge、不新建monitor。
不要仅回报compile/排队；目标是实际final PASS模型或树耗尽后的完整机制证据。
