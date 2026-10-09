# PI0287 固定参考结果：0/6；整体 PARTIAL

授权：原 release `7a9f0c02b55222646601307ccd7922e9297e2ab5`、canonical #6084617266；用户随后明确允许额外最多3次串行启动，记录在 `SERIAL_LAUNCH_EXTENSION.json`。第四次启动 PID3252669 已完成；不继续消耗剩余两次启动追成绩。

## 原始结果及独立复算

| 固定条件 | 已保存控制tick | 最后保存时间 | 停止原因 | x5_joint5硬限穿透 rad |
|---|---:|---:|---|---:|
| hold | 13 | 0.260s | 硬限越界 | 0.002899647 |
| horizontal_5cm | 14 | 0.280s | 硬限越界 | 0.000017881 |
| vertical_5cm | 13 | 0.260s | 硬限越界 | 0.000001907 |
| pure_rotation_0p10rad | 25 | 0.500s | NATIVE_ABORT | 最后已保存点无越界 |
| vertical_return_5cm | 15 | 0.300s | 硬限越界 | 0.000124097 |
| hotswap_0p4s | 15 | 0.300s | 硬限越界 | 0.000000119 |

五个硬限判定直接用加载actor properties的原始范围，保存实际DOF与限位数组。CPU从数组独立重算全部95个已完成tick，与记录一致。`x5_joint5`下限为 -1.6710000038146973 rad；hold实际达到 -1.6738996505737305 rad。小于微弧度的越界与较大越界分开列明；没有放宽原“无实际硬限穿透”门。

纯旋转在随后一次原生step内触发安全终止；具体终止分项及该终止step的最终状态没有保存，记 UNOBSERVED，不猜为接触或底盘高度原因。所有条件都未完成10秒，末两秒RMS/max均 UNOBSERVED。热替换条件未到0.4秒，不能声称热替换执行成功。

作为前缀观察，hold停止点实际TCP相对安装目标的位置差约0.08366m，纯旋转最后保存点约0.24333m；这些不是协议要求的末两秒指标。保存的canonical采样位置最大偏差不超过约1.79e-7m，旋转采样最大偏差约2.56e-4rad。该采样对照不能证明完整actor输入、native preview和物理执行链均无问题。根因仍未闭合，不把本目标接入FAIL推广为冻结WBC普遍失败。

旧第三次启动的六项soft-limit结果保持工程无效，原JSON/NPZ/hash不变。第四次结果保存在独立 `reference-fixtures-hard-limits-v2`；不覆盖旧收据。

## 停止及交付范围

终点为 `REFERENCE_QUALITY_FAIL`，整体 `PARTIAL`。已完成实际资产、索引、夹爪、双RGB、FK及CPU模块；G1仍仅有已列出的子项证据。按原门停止教师开发、pilot8、TRAIN64/DEV16与官方processor读入，真实任务数据为0，不拟合新statistics。后续native teacher-pilot/collect接线仍未完整实现，未把guard计为完成交付。

视频只编码六项保存的真实body/wrist RGB的共同前三帧；明确慢放、短前缀和失败标签。后续帧仍保留在原NPZ，没有补帧、循环假装32秒任务或抓取成功。

累计4次串行启动；21 env starts、784 physics integrations、205 actor预算预扣（204实际forward、1旧观测错误预扣）、38 RGB pairs、teacher0。四次仿真进程service合计113.578461s；self CPU合计29.081630s，不含全部compiler/子进程/本机CPU，完整CPU总量仍 UNKNOWN。没有GR00T forward、优化更新、WBC更新或硬件操作。

下一步需以这些原始前缀和历史执行器身份向师兄核对匹配的Stage2运行契约；本轮不换checkpoint、seed、gain、progress、阈值或几何继续尝试。selected/deployment=NONE。
