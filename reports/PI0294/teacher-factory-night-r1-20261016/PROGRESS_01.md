# TF-NIGHT R1 实际进展 01

**尚未跑通 Teacher：一次完整安全启动，第二次发生夹爪原始行程越界；启动 release 未取得重复性资格。完整成功 seed / TRAIN / DEV / Teacher 提交 / 任务 RGB 均 0。** 本报告是中途进展，不是目标完成。

## 用户规则已采用

用户明确更新“允许不断重试 debug 直到 teacher 跑通”。固定 `d046e174a100b623d1b7d485b7c399f9488034d6` 登记该直接授权，取消任意工程重试次数及工程接入重资格次数上限。每轮保留实际失败、成本、受影响 CPU 检查与新 source；WBC/腿策略/MPC/增益/安全/评分、旧 T0、原总资源与 09:00/10:00 截止不变。真实安全事件终止当前 episode，失败 startup module 隔离；不能改个报告 SHA 原样重启。

## A：真实启动

| 尝试 | 实际结果 | 原生积分 |
|---|---|---:|
| 00 | DDS domain 不支持，native 子进程启动前退出；原预留原样记账 | 0 |
| 01 | 原生 GetUp 仅凭可见目标提前切换，物理尚未开始 | 0 |
| 02 | 在 sim=0 等完 GetUp 后再 ENTER；FL calf 原安全 FAIL | 26 |
| 03 | GetUp 在真实物理中完成，随后原腿策略阶段 FL calf 原安全 FAIL | 618 |
| 04 | 使用原生 ENTER 将 mode/InitRL/history 初始化置于暂停事务；完整 5s，独立全 2500 包复核 PASS，最终 WBC | 2500 |
| 05 | 同 source 独立重复启动，右指原始 44mm 行程越界；未取得第二个 PASS | 617 |

04/05 runtime source `770c559864fdd6b258be50904c3ac2291d81e4e5`。模式初始化的实际 native step/qpos 不变，未改 native 实现。04 全 20 个 limited joints 合法、非法接触 0；04 的成功不能覆盖 05。

05 在 step616/617 有右指原始越界，峰值 **0.04417412886975903m**，原上限 **0.044m**；首次 high-rate 原件与 native stop 分开保留。未忽略小越界、未调整夹爪阻尼/控制参数、未更改原安全范围。当前该 startup module 已隔离。

这证明暂停切换可取得一个真实完整启动，尚不证明其充分消除共享启动问题。真实加载日志确认新旧均为同一份已冻结腿策略；没有证明“加载错模型”或唯一根因。当前只继续有证据的允许范围 debug，不以同 source 反复启动追第二个成功。

## B：参考、配额与数据工作继续

- 原保存 V3 query14 的 **4.012846783rad/s²** 拒绝精确复现。两条超限 stencil 都在连接末端之后，并未直接跨连接。
- 固定四档局部时间候选保留全部结果：1/1.25 档峰值 1.33795，1.5 档 1.00349，2 档 **0.91714**。只延长涉及的三个区间，原连续空间路径不改；此为 **saved joint witness** 检查，不是完整路径/碰撞/物理资格。
- 原真实已执行 ref13 prefix6/7（native step5275/5300）与下一窗口对齐。两个窗口 root 绑定不同，禁止直接差分其局部关节见证并宣称实际命令跳变；世界 EEF seam 明确保留。原执行参考、未执行预览和 actual qpos 三者分开。
- 15 项参考 CPU fixture 契约通过；13 项变长数据/真实提交 prefix/19D 测量状态契约通过。官方固定 GR00T source 的三 action 模态构造通过，BASE 为独立 absolute NON_EEF，未二次相对转换。
- 已实现变长 EEF13/state19 导出、TRAIN-only 一次统计冻结及独立官方 CPU consumer 入口。**没有合格 bank，因此未拟合 stats、未实际消费数据，不能称数据闭环 PASS。** 原生 simulation profile 缺 scipy 的尝试已记工程失败；官方完整 processor/collator 实跑仍 NOT_RUN，后续复用已有 CPU-data profile。
- 长时 adapter 的 23 配额正反例和 40 状态控制等价 CPU 收据此前已完成；原控制公式和 native observer 不变，不代表 128 秒物理已执行。

## 原件、预算与下一步

六次尝试 159 个原件、469520083 字节已复制到独立私有证据位置并逐文件 full readback；tree `80b55f44adecd333d69437799581c8cba7e82767674851bd6f175d988d7fdd4b`。未公开 raw/WBC、未删除源、未重传旧夜间档案。

累计账见 ACTUAL_PROGRESS_01.json；原夜间 99946.07684278488 CPU-core-seconds 上界和旧 1 次安全事件保留。R1 新增 3 次真实安全事件，总计 4；用户 debug 接续不清零。无 GPU/PBS/GR00T/训练/真机。下一步定位 05 的实际夹爪/启动调用问题；只有新的可证实自有 wrapper 修复才重新资格。两次同域完整安全启动及完整参考资格后自动继续任务，不等待例行 GO。

`selected=NONE; deployment=NONE`。
