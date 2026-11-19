# 第二批 Go2 TRAIN 原生采集与 ADAPT 资格入口

2026-11-20 01:25:49 SGT。首组六条已冷态关闭并对账，第二批复用唯一 MuJoCo world，采用六个新场景和新的 reset／seed。注册顺序沿用 F1 红左、F1 蓝右、F2 红左、F2 蓝右、F2 蓝左、F2 红右。场景编译逐个与 208 个历史场景哈希比较，机器人指纹一致。旧失败仍属于各自的原条件。 [机器可读回执](TARGET_SECOND6_NATIVE_AND_ADAPT_PBS_ACTUAL.json)

| 服务 | Core 单元 | PID | 现场状态 |
| --- | --- | ---: | --- |
| 第二批专家采集 | `wm02-sla-scale-target-adaptation-second6-native-20261120-01.service` | 4080484 | active，顺序执行 |
| 批次封存后的 G7 标签准入 | `wm02-sla-scale-target-adaptation-second6-admission-20261120-01.service` | 4084487 | active，等待前驱退出 |
| 现役 MIX20K | `wm02-sla-scale20k-mix-core-after500-20261119-01.service` | 3957773 | active |

采集以 `7a77fc9817fed86424863df247aa03ca02b018bc` 执行，spec SHA256 为 `67cf8c3dd3eeefadc1e66e544f1e073ad57c6d6e5b8cea4707907a2fe9b0a5f1`，最晚 **04:58:28 SGT**。独立连线确认实际进程、lease、原生首积分回执和状态包。当前实测 **已开始 1/6 请求，终态 0/6，累计观测 32,420 tick**。F1 红左在 tick 21,017 达成正确抓持，在 tick 23,388 抬升达到 20 mm。入区、释放、撤离与任务成功仍待该条终态。真实机器人请求、学生模型生成和采集侧优化更新均为 0。

第二批上限 6 请求、304,218 tick。冻结 WBC、安全门、2 ms 时钟与原评分继续使用。准入服务在原生 owner 和 cgroup 冷态后读取封存结果，并校验完整 H40、真实 P10 消费和 RGB 时钟。准入产物写入独立目录。D1 固定成员保持由原封存清单定义。

## ADAPT 实际资格入口

新增单卡 PBS 入口先核验实际 D1、完整 R200、跨站输入哈希与原 MIX500 检查点，然后运行真实 128 窗口梯度资格和 ADAPT0 独立恢复。另有一段独立的 phase25 入口，须读取真实 PBS 冷态与 ADAPT0 恢复结果才能执行。Core 的阶段化实读复核了三域 22 个样本，哈希仍为 `1812283b099ffa1db4f70b519a497f678b80fa7aa651eecfab3a7482b0107615`。本批次尚无新的 ADAPT PBS JobID，ADAPT 梯度、优化、预测及原生执行均为 0。原始 R200 传输与 VANDA 匹配实读完成后可进入此入口。 [PBS 入口源码](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/commit/15aea3b2f7fa078ed18685d2545c60211a4ccfe6)
