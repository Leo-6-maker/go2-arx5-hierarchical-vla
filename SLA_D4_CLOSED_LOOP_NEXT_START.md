# SLA 当前入口：多源长程训练与 Go2+ARX5 本体适配

2026-11-19，Asia/Singapore。最新用户要求：制定详细长程自主推进计划、充分利用Corelab/VANDA A40并发布GitHub。

## 当前计划

- [完整自主推进V1：两站调度、数据、训练与闭环](docs/protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.md)
- [机读任务、课程与精确预算](docs/protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.json)
- [Codex完整启动prompt](docs/handover/2026-11-19/START_SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1.md)
- [证据与发布核验](reports/WM02/sla-autonomous-longrun-plan-20261119/REVIEW_AND_VALIDATION.md)

继续现役MIX20k/TGT5k；当前有限world段结束后优先新增目标TRAIN采集；新数据可用即从封存MIX端点fork一个64T/32R/32L的连续ADAPT5K。Teacher全格、D5成功、J全48和N2都不是扩训练前置。两站GPU、CPU数据准备和world轮转并行，普通工程自主修复，实际allocation和owner继续核实。

本次固定依据：PR418 39d8f6e3、PR417 4c9ccb9c。已发布观察为MIX452、TGT898（两者均22:36:12观察），TGT500离线288+48完成；J10/48、D5 0/24抓持；R200已Core核验，首6采集守候已运行、新目标示范仍0。它们不是当前机器实时状态。PR正文的旧“7更新/GRASP0/J1未启动”不可作接续依据。

## 现役合同与历史

- [PR418仍执行的92622691 SCALE合同](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/926226914a0c08ee1ebb143e0edea1434728107b/docs/protocols/SLA_SCALE20K_DIRECTION_20261119.md)
- [PR417仍执行的cf9d1acb D4/D5/J合同](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/cf9d1acbf72e96b1bd158322ff8cbaf8fe28fabb/docs/protocols/SLA_D4_CLOSED_LOOP_AND_GRASP_20261119.md)
- [上一版官方配置复核](docs/protocols/SLA_SCALE20K_OFFICIAL_REVIEW_AND_TARGET_ADAPTATION_20261119.md)
- [本入口更早版本全文，Git中完整保留](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/cc96f1709170e47298463337d050c79a24b7c7d9/SLA_D4_CLOSED_LOOP_NEXT_START.md)

计划PR以main为集成目标；执行栈仍单独固定，不认为main已经部署runtime。新reader/统计/loss/数据版本只进入新谱系或预登记冷态课程边界，活动胶囊不热改。原失败、旧TEST、WBC、H40/P10/50ms/2ms、安全和评分保留。本次发布新增训练/生成/原生请求/服务器作业均0，真实硬件请求0。
