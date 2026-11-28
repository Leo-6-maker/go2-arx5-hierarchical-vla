# Codex 启动 Prompt · SLA V7.1-A 无人值守自主执行
请接管 Leo-6-maker/go2-arx5-hierarchical-vla 当前 GR00T+WBC / SLA 实验，按照 PR #529（https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/529）最新计划提交中的 docs/protocols/SLA_V71_UNATTENDED_DAGGER_PLAN_20261129.md、docs/protocols/SLA_V71_UNATTENDED_DAGGER_REGISTRY_20261129.json 和原V7.1三份文件执行。先读取PR实际head并固定SHA，读取 experiment/sla-v7-control-context 最新执行分支以及必要的Core/V65增量；原V7.1基准 c4364780327924a0db71f84082b7c6b4a54f3bb8，本轮发布时最新执行 07c433dd24ad359933a0b4c7381a10e12373abf2，之后真实回执优先。不要把文档分支当运行源码。

目标是长程、有界、无人值守科研推进，不是回复一个计划就停止：Frozen Expert真实能力保持→C1匹配早期纠正训练→A2控制状态→按证据选择C2/S2/B2/B4→有效因素组合与完整自主PLACE独立确认。Frozen Expert是参数更新方式，不是DAgger；D1/C1纠正数据必须是真实学生状态上的教师实际消费，后续以合格新学生再采集并冷聚合，形成真实迭代。R4仍为当前部署/回归基线，原WBC/OCS2/PD/腿策略/L1冻结，H40/P10、2ms/50ms、40mm/s G7、原接触/支撑足及硬STOP后零积分/安装/放行不变；禁止GT学生、保护集训练、失败prefix当专家、把teacher-assisted当student成功。

启动先对账真实现场job、owner、租约、checkpoint、实际预算、个人quota和原期限；健康或未知job不重复派发，不因轮询超时重启。先前V65、V7-A、B6、S1以及37格回置诊断、96条标签复演、8245窗因果context已完成，不重复。保持原科学门及历史SHA。立即并行：C1组合真实RGB/state/action/prompt/mask reader与94共同槽/aux核对；冻结候选的原37格和12状态×3噪声的独立TRAIN名册；A2/S2零初始化入口与三路径；C2首接管时间分层；红15/蓝17/红19全身执行归因；L0真实批次。使用现有协调者和独立工作包，Hopper只做实际训练/资格，Corelab做数据、固定权重、恢复和原生，Vanda暂停。不要创建重复调度器、无限轮询或抢其他owner。

真实资格和运行严格按登记：Frozen Expert至少2步真实H200资格、共享权重/AdamW decay/moments/count不变、活动IO clip记录、完整Core冷恢复；通过后250正式步（完整匹配才能复用原Dense零250），原37格+额外TRAIN多噪声+原源域门通过后才登记少量原生。C1两臂同R4父/相同已验证保持scope/各250，原bank仅末2差异槽普通Teacher vs 新早期纠正；不复用旧bank Control，不混A2/S2。A2同结构中性vs真实6D residual，各250，原token/action统计不变。根据实际结果决定下一项，不自动增步、改λ或同时改多因素。迭代DAgger保留2个Recovery名额给新合格学生，另1个仅有明确新前瞻依据才用；每次教师接管必须在合法未STOP完整P10边界，真实教师后缀通过实际消费、原安全、成功与reader准入再冷封存，不热换活动bank。

当前已知正式5100/8000、工程24/24、学生46/50、Recovery17/20，原320GPUh与2026-12-05 12:41:52 SGT截止；现场实际账本更严格者优先。当前工程余额0，先核已有明确授权，**本prompt默认不增加额度**。无新增工程授权不得派发真实资格或绕过；在等待时把所有合法可独立推进工作做完，不反复请示。若用户另有明确新增内部上限且集群award/owner/时间/容量均通过，则按实际合法分配自动执行，不逐步再询问；没有实际授权时一次性报告具体缺口，不能伪造授权。E2现有最多4卡，仅方向筛选，不宣称完整保持；原完整十格/臂及独立确认是最终晋级要求。

持续以READY/RUNNING/BLOCKED/CLOSED的有界任务循环推进：核依赖→合法派发→真实终态→封存和远端回读→判定→下一READY。每次科学结果/异常和最长约12小时有实际变化时提交实际回执到执行分支、更新PR #529进度，报告代码/父/数据/bank/noise/receipt SHA、真实更新/积分/前向/attempt及剩余预算、完整负结果和下一动作。技术中断保留实耗，恢复完整Adam/RNG/cursor/下一行，不重跑已完成实验；硬STOP卡绝不重开。不要仅输出阶段报告就退出；若一条主线阻塞继续其他工作，直到目标达成、期限/预算/授权边界到达、用户撤销或没有合法READY工作。到停止条件时安全收尾、释放自己owner/租约、保存完整状态、列明已达/未达目标与下一冷接力入口。不得擅自merge main、force push、删checkpoint、改凭证/安全/基础设施。

现在开始实际接力，先报告已核对的最新SHA、现场任务/余额和READY工作，然后继续执行，不要重新向我询问已经明确的计划和常规实现细节。
