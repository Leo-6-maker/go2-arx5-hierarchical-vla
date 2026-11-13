# Codex 启动 prompt：SLA＋冻结 WBC 长程推进 V5

日期：2026-11-13（Asia/Singapore）  
[主计划](../../protocols/SLA_LONG_HORIZON_EVIDENCE_V5_20261113.md) · [机读矩阵](../../protocols/SLA_LONG_HORIZON_EVIDENCE_V5_20261113.json) · [入口](../../../SLA_SOURCE_TARGET_MOBILE_START.md)

## 直接交给执行端

你正在继续 `Leo-6-maker/go2-arx5-hierarchical-vla`。目标是交付可测量的 SLA 上层策略，而不是继续扩写计划：先打通目标数据与原生执行，再检验空间监督、条件化、恢复、源混合，最后共享四技能和移动复用。

### 1. 读取与实际状态

读取本V5主计划、JSON、V4.1操作说明、仓库工作流及AGENTS；后续明确授权优先于旧首轮范围。记录实际阅读的文档SHA和运行SHA，不默认main含最新runtime。

本轮已读执行PR #384为 `cb09e71441b2f02307317b60d059ea06c38cb55b`，回执时间2026-11-13 23:35:31 SGT。PR #383是文档；#384及后继是执行。先查最新提交、原生owner/PBS/unit/manifest，不把该快照当现场状态。

重要快照：
- R1_500已完成训练与H200/Core RGB恢复，原生请求仍0。
- `T_PICK_HOLD_TRAIN01_P10_2EP_V4_1`已有2条/122窗口；只属一个布局、一个物体、PICK_HOLD。
- H200 job716146当时正在从干净父模型训练具名pilot，513/3000更新、16,416访问，已有完整step500；不重新投递。
- 稠密空间缓存122条已读回，但空间梯度和匹配空间臂未完成。
- 固定专家panel16/16已关闭；P10还改变了分段consumer，不能称纯频率消融。
- D100/D300/PICK_PLACE生产准入0。
- 五任务默认公开包实际每任务50条、共250；已查20条，S_cmd仍0。不要再声称已获得五任务×200条。

### 2. 权限与保护

沿用用户明确的Corelab独立多世界授权，每世界独立控制owner；同一共享checkout只一个writer。H200使用现场有效的单卡allocation，无需重新问2/4/8actor相同许可。保留正常运行中的capsule/作业，不热改、不中断、不盲重投UNKNOWN。用独立worktree和正常PR集成，不force、不自动合并、不清理用户权重/数据。

WBC/腿策略/增益/cost/solver/2ms动力学/原安全保持冻结。真机、在线RL、低层重训、系统权限和他人任务不在本次执行范围。不存在可用工具/权限时明确阻断，不声称已执行。普通用户态工程修复自主推进，针对变化接缝验证，不反复跑全部旧资格。

### 3. 第一优先：完成正在进行的策略交付

核实job716146的真实进度/终态；健康则按原3K终点继续，结束后核对完整checkpoint、normalizer、codec、consumer和普通RGB新进程重建。旧R1既定R0/R1诊断与新pilot的TRAIN8分开，不把两者成绩混起来。

实现/复用对应原生桥，让模型raw H40真实安装并消费，报告原始合法率、实际消费数、抓持/抬升/保持及第一失败。若模型动作被拒绝，定位具体通道/表示/时间，不用更多同分布训练掩盖。教师替换仅作标明的诊断，不算学生成功。

旧S1最后未启动条件只在精确初始化缺陷修复后冷接续；保留已发生的15条终态及原S1→S2→GR00T交付。不把历史收尾变成新数据和新技能的全局锁。

### 4. 同时准备空间后训练，不等待模型先取得高分

复用122条稠密缓存，先做alpha0公共梯度/更新一致性，再验证alpha0.5对部署VLM LoRA有梯度、teacher冻结。只新增必要的有限CUDA接缝，不重复完整旧恢复资格。

同起点/相同数据/同visit与随机流下，FM与空间臂独立从SLA_base开始；不从FM终点继续伪装匹配对照。小pilot空间臂3K，仅用于拟合/机制；若D100先ready，可直接做D100匹配两臂并明确小pilot空间未运行。

### 5. 数据和效率是持续主线

复用已通过的dense reader、recipe_v4、provider_v4、分片、末尾两秒真实记录及expert16。先从就绪任务生产具名20–50完整episode，不等五物体/两任务齐全；正式D100/D300仍按10姿态层×5物体×2任务，数量和身份不能冒称。

PICK_PLACE必须含转移、释放、撤离、稳定终态。RGB/state10Hz、anchor2Hz、命令20Hz/H40；实际频率与native step对应，不能重复最后一帧或合成尾标签。相对动作锚点不被state dropout清零。输入不含物体真值、expert阶段或source_id。

Corelab用持久CPU专家/物理/相机，采专家不加载SLA；有界writer与共享buffer双消费者生命周期正确。H200读冻结snapshot，参数/optimizer/梯度常驻GPU，不在每microbatch/layer/flow/physics step搬大张量；必要观察H2D、小动作D2H、日志/checkpoint分别计量。

测同8条件×2重启的1/2/4actor有效完整示范/小时和失败率；有收益再扩8/16。GPU物理迁移不是开始生产的条件，EGL读回也不冒称零拷贝。

### 6. 源数据按最新审计分流，不再等待虚构S1000

五任务清单沿V4.1，不重下已验证归档、不重做已完成20条检查。只补采集clock/code、desired command、FK/TCP、当前夹爪四个缺口。15Hz序列字段不证明物理均匀采样，achieved endpose不变成desired命令。

若公开command最终可用，最多250条池按每任务40TRAIN/10源DEV（已查episode放TRAIN），名为S_PUBLIC_CMD200。不能恢复则保留S_aux，改在相同开源任务中仪器化源专家，先S_GEN_CMD100再有证据扩1000，另外留源DEV。

源无需Go2执行成功才纳入，但必须有源语义/时间/工具证明。先T-only与M75（24T+8S/B32），数据名带实际Ns；不等待Ns1000。源准备不阻塞目标policy。其他比例、顺序、等目标曝光对照按V4/V5条件性进入。

### 7. 成功后按失败机制推进，不展开所有组合

在D100/D300 FM/SF共同端点上完成固定C24：8位置对、8指令对、8实际EEF偏差对，每方法48条件。需要真实一致场景，不拼假图/错状态。它是DEV机制测试，不回灌原轨迹。

首波至多400 TRAIN学生rollout采200可信纠正片段；同父两臂各1K/B32、相同新optimizer/固定统计，normal100与normal80+correction20。学生错误前缀不作BC。state dropout、adaptive alignment、history一次只改变一个因素，满足具体疑点才启动。

柜门和桌面取书插槽各有10–30完整种子就可训练首版，不等待抓放100%或D1000；曲线路径需要学生consumer真实保留。随后一个共享权重/EEF接口/语言条件的四技能模型，加入新任务后复测旧技能，不用多个head或checkpoint冒充共享。

### 8. 移动与后果模型属于后继分支

先实际BASE/frame/mode能力，再新站位静止操作，再导航稳定接管、持物运输，最后连续移动。不能靠reset算导航，也不靠删站立安全锁算移动；固定臂缺base标签mask，不补假零。推理使用当前实测base，不读未来base真值。暂停仿真不等于2Hz实时控制。

仅当可执行策略存在明确接触/迟滞或观测歧义时，先history，再研究显式动作条件后果模型。实际安装完整预览、消费prefix与控制历史要保留；反事实需要可恢复完整状态，不把相关预测叫因果。世界模型、力触觉和RL不是当前主线前置；本次不自动启动RL或实机。

### 9. 每次回报和提交

使用现有记录机制，一份当前状态加固定证据即可：
1) 实际checkpoint/更新/各域visits及恢复；2) 完整示范/全部尝试/覆盖/纠正数量；
3) raw→installed→consumed及任务结果；4) 首失败与具体修复；
5) 吞吐、H2D/D2H/同步和真实成本；6) 下个由证据决定的实验。
未运行、技术失败、策略失败分别报告；历史负例不重试筛选。实际运行身份去重，继承预算不重复累计。

下一次交付应至少包含当前候选终态或真实进度、一个原生策略固定诊断的结果/明确阻断、空间链进展和新增数据版本。不能只提交更多说明、测试计数或新计划。

## 本地静态验证（不是训练启动命令）

```bash
python -m json.tool docs/protocols/SLA_LONG_HORIZON_EVIDENCE_V5_20261113.json >/dev/null
git diff --check
```

实际CLI、站点路径、allocation和capsule参数必须从当前运行源码/已有profile读取，本prompt不编造已经实现的launcher。正式启动前固定数据/模型/consumer/seed/有限walltime，作业健康时不再不断重审批。
