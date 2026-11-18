# Codex启动prompt：SLA D2完整Teacher与并行主线

把以下整段交给具备本项目仓库与已授权Core访问的Codex。文档发布本身没有启动任何作业。

---

继续GR00T+WBC项目SLA移动底座后训练主线。请直接实施并自主推进本轮有界D2工作，完成实际可验证的结果；不要停在再次写计划，也不需要逐个工程修复再次问GO。

先读本PR同commit下：
1. SLA_D2_PARALLEL_START.md
2. docs/protocols/SLA_D2_TEACHER_AND_PARALLEL_20261118.md
3. docs/protocols/SLA_D2_TEACHER_AND_PARALLEL_20261118.json
4. reports/WM02/sla-d2-audit-20261118/ANALYSIS.md

固定报告/可复用源码基线：PR409 b29bd2badc976e2288f9d009e4bf14cc41b769af。第17/18条实际运行source为a83803fb71c76937020e88f210a592fd1524bf80，不能混同报告head。先回读GitHub和现场最新状态；如有并发进展，列差异并保留既有真实效应。文档PR基于main，不代表main已含runtime；从已核实执行栈在独立worktree接续，正常feature PR，不reset/force-push、不热改活跃胶囊。

已知终态：旧Teacher18/48，9完整成功、8接触STOP、1任务耗尽、0技术失败、526198科学tick；接口2/11906tick单列。F1 PLACE-red-left 0/3，只余1，资格已数学失败。旧剩余30（含24号）禁止继续，原M-BASE/M-SRC保持未启动。不要将旧9成功或新开发成功回填旧/新资格分母。

主任务：
A. 用已有12/13/15/18封存轨迹与D0核对nominal/submitted/accepted/selected或UNKNOWN/actual，修Teacher抓取—搬运—下放—打开—撤离完整路径。第18条已真实转腕和入左区，失败在LOWER；当前LOWER仅入口>0检查，之后没有逐query剩余路径复查。优先修实际TCP/object变换、完整SO(3)/全臂几何、提前余量与完整90s时长。accepted≠selected；9.77mm为几何净空差；D0的0/34不支持立即加previous-command学生输入。不能调WBC补偿。
B. 加入阶段profile，统计完整进程树。当前每25tick后spin(.15)处理满150ms，第18条至少274.5s在这些窗口；先测再改，不能全算空闲或直接删除。复用现有事件就绪、增量日志、常驻相机与异步图像写入能力；新mobile合同与旧资格隔离。event-ready无实际消费证据就保留legacy，不阻塞Teacher或legacy双world。
C. 建新移动多world协议：完整独立OS进程树、domain/CPU/owner/目录/tmpfs/cache/quota/ACK/STOP/证据。复用旧双world基础，不绕过锁。先N1对照N2；N4只条件工程资格，正式科学最多2在途。按机器计划的8卡/4profile上限及2组双worldSTOP隔离（共4请求）验证，总工程含全部接口/技术成本≤64请求。量吞吐时隔离其他工作负载。实测CPU/cgroup/容量，不假设机器空闲核数。
D. 新TEACHER_DEVELOPMENT24：两组各12新条件，最多两实质候选版本，组间冷态修改；同一最终Teacher/backend在12格各≥1完整成功，不拼接不同版本。第一组已全过可提前冻结；否则第二组按已证实机制推进，未通过则真实交付，不重投旧失败/加第三组。开发24不入首轮训练。
E. 新TEACHER_TRAIN_QUAL48：8新scene×6指令，Teacher/backend/条件先冻结，每格≥2/4、总至少24合格全流程正例。保留名义H40计划与真正消费P10的区别，TRAIN-only共同统计一次拟合。旧18、开发24和模型DEV/TEST不混用。每次派发原子检查s+r≥2，r含在途；数学不可能后不再派发，已合法在途按原cap/STOP封存。隔离或STOP后积分故障立即停止受影响消费者。
F. 仅门通过后启动M2-BASE/M2-SRC，唯一迁移原M未用预算（两臂各1000/32000），不新添重复预算。源S1000复用，只移植20共享LoRA，其他67目标数组同初值、同数据/统计/噪声/loss/fresh optimizer。C20访问20%＋新目标80%，不是sim/real比例。screen48→唯一候选confirm24→DEV60→一次TEST60，沿原V6门。两臂平衡world/CPU和队列。
G. 学生多world前修REQUEST.json单槽为有界身份队列；先单常驻GPU、batch1顺序推理，每world独立暂停等待，身份含environment/episode/capture/native_step/request/model/package。取消、旧响应拒绝、超时、去重都可复核；不复制模型占满GPU，不启动态batch。检查生成计入原M离线1600。

保持WBC/legs/MPC/solver/gains、2ms/50ms/H40/P10/TTL/STOP/评分，HOLD60s、PLACE90s，历史TEST与失败、RealMix BLOCKED_LABEL_SEMANTICS、无真机、无输出clip/smooth/retime。原selected UNKNOWN不永久阻止几何开发；未知owner/真实资源不足不能用计划文件替代实际准入。

先核实实际allocation；已知旧截止2026-11-19 02:51:23.855488 SGT，服务≤12h、campaign≤24h，取实际最早截止，不自动延长。当前已知速度不能保证一次allocation跑完所有条件；按Teacher修复/双world→开发→QUAL48→M2顺序完成能合法完成的阶段，N4和源模块诊断低优先级。派发前预留实际收尾时间，关闭自有服务并封存未启动后缀。

请自主完成可用实现、针对改动的有效检查、实际有界原生验证、计量和GitHub PR回读。每阶段发布ACTUAL md/json，写明真实更新/访问/生成/请求/tick、完整成功与首失败、G7、资源/并行度、lease和后台终态。fixture/静态路径/短probe不冒充完整任务成功；暂停吞吐不冒充连续实时资格。最终给出结果与明确的下一决定，所有没启动的阶段写0，不把计划描述成实绩。

---

