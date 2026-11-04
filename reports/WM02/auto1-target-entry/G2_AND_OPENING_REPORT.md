# AUTO1续轮：原参考入口通过，夹爪小模型完成

2026-11-04。继G1六层初始化通过后，继续推进原WBC参考链及最小夹爪响应修复。没有扩大S域采样、加载GR00T、训练WBC或启动Actor。

## 原生参考入口的实际结果

成功运行源码 **`7939f7caf6a443a65faa21b60e0b7eef5d51c0cb`**，exit0。原native二进制仍为 **`e9aab70b6c1d2af76c6e178fed1fd8e86299dffb9624d15a85e58ba4d3f426ff`**。实际执行的原clock controller是 **`885042aee96eec2761e5cc9c1598194c94181bebe0de0548eb48b1110ddaa186`**，solver是 **`4a5f5aa6e0b9d4bde62f4d1da73135ef97e5268945a0cecf114d9190d3e04fd0`**。初轮成本表中的canonical controller `fc985496715995778e1ba1580fddfd3f3922fd4b5c7e0954b66cef7b6ecd128f` 是冻结构建输入，不能与实际执行的clock controller混称。

首次资格源码 `f7ddfdfee730d6e50b3fab95bf0cec3562b3cd53` 在2500个安全预热tick后等待WBC反馈超时，exit1，四个进程正常关闭。随后源码 `ddcf459d95925b1d6df9921481b4383466aa58b2` 的零积分诊断确认WBC ACK、五个节点及话题存在，但EE／MPC观测为空；该诊断未执行预定原生阶段，CLI exit1原件保留。

已从实际运行binary及其原源码定位：controller保留 `source_tick == pi0294_last_tick_` 时立即返回的保护。暂停后才请求WBC，模式ACK不能推进其控制tick；这不是话题拼写错误或MPC算法崩溃。原MRT reset调用是异步请求，也没有证据支持此前怀疑的同步future死锁。

修复只调整薄adapter的初始化握手：先精确暂停2500tick，收到WBC模式ACK后，再推进**恰好3个有界初始化tick**，保留原controller的重复时钟保护、WBC数学及所有安全。原R1 quota validator允许5秒预热后的+3tick边界；直接编译原header的测试确认2503允许、2504和warmup6拒绝。

本次实际暂停在2503tick／5.006s，原EE／MPC来源时间5.004s，2ms源龄符合原25ms门。原EEF10 compiler及receiver真实完成：

`measured state → proposed H40/41点wire → accepted → installed ACK → cancel → fresh HOLD ACK`

wire SHA256 **`43a9006861b03263dc8eaa1de817861899a864833e95bac3f123b0ea00e397e4`**。install/cancel事务期间额外积分0，四个子进程全部exit0，trace审计PASS，原生安全未触发。3个tick单列partial，没有补为50ms样本。**安装后未执行任务tick，实际MPC policy/reference消费仍NOT_TESTED，G2八项动态诊断未完成。**

累计新增目标积分 **12353tick = 494个完整50ms初始化区间＋3个partial tick**，含所有原失败及本轮预热。有效目标WM学习行、WM优化更新、Actor更新和新完整任务均0。[原始退出码、三次身份及完整SHA](G2_REFERENCE_ACTUAL.json)。

## 不依赖未来真值的夹爪响应模型

实际拟合源码 **`c8a85bbba2d0aff114c181490e7deb4df989b4f1`**，exit0；模型文件SHA256 **`3719639c4153f2a01885223d2e1fca05f3a59851e46b2353bbc5264fe7348a9b`**。仍使用原数据 **`c12293bc877758254a14d19ec112f147fe19c93ffacf4c8399ae8ebfcb42c808`**，没有新物理或神经推理。

本次不是扩大整网：用完整已执行命令前缀从已知reset重建limiter目标，用当前观测开度作为此固定释放诊断的初始接触下界，再按**事先保存的候选8×7命令及其第二段保持**推算目标曲线。仅在原FIT24的120个释放候选上拟合一个滞后参数，480个预定运动样本的中位数为 **0.004911770981569752**（归一化开度）。未来实际开度只作FIT标签或评价真值，未来controller memory／接触标签没有进入预测。40个评价候选的初始开度与决策前已执行前缀完全一致，误差0。

在**已经打开的**40–47这8个父场景、40个释放候选上：

| 指标（归一化开度） | 冻结M4-B | 因果小响应模型 |
|---|---:|---:|
| H1 RMSE | 0.00420689 | 0.000547983 |
| H8 RMSE | 0.0479313 | 0.00567775 |
| H16 RMSE | 0.0694714 | 0.000380199 |
| 原评分下正regret父场景 | 3/8 | 0/8 |

抓持概率仍使用M4-B在实际分叉前保存的原预测；Q公式、真实结果、候选集合与规则均未改。模型JSON重载预测误差0。实际是**一次闭式拟合、1个参数、零WM优化器更新**，不冒充200步WM训练或完整优化器checkpoint恢复。

这是定位误选机制后的**事后诊断**，不是盲测或新在线策略成功；原规则本就达到所采oracle，不能声称控制收益超过原规则。小模型的初始开度下界适用于此静止释放探针，不能推广为闭爪、滑移或一般接触动力学。它要求已知reset的完整因果命令历史，没有把未知内部目标补0；没有接入M4-B或Actor。M4-B原checkpoint `9676312c5b810a757c465174ddc4302fc67b1f6896f8ba66dbed5242dee77f7b` 仍冻结。[模型、逐父结果与因果检查](OPENING_RESPONSE_ACTUAL.json)。

## 命令、检查和成本

以下入口均已实现并实际运行；变量绑定原私有输入／独立输出，完整解析argv保留在运行记录中：

```bash
PYTHONPATH=src:. python -m scripts.wm02.commission_target \
  --spec "$RUN/EXECUTION_SPEC.json" --output "$RUN/native-01" --run-id native-01
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 60 "$PC_NATIVE_PYTHON" -B -m scripts.wm02.fit_saved_opening_response \
  --fit-root "$OLD_DATA/collected-01" --replay "$V5/PAIRED.npz" --audit "$V5/AUDIT.json" \
  --new-root "$NEW8/validation-01" --manifest "$PRIVATE/audit-new8-01/STREAM_MANIFEST.private.json" \
  --output "$PRIVATE/source-opening-fit-01/fit-01" \
  --source-sha c8a85bbba2d0aff114c181490e7deb4df989b4f1
```

现有world-model suite **76普通＋76 `-O`全通过、无跳过**，原receiver／EEF／gripper契约另6＋6通过。source-map含41个精确原始commit对象，公共完整tree与实际源码tree相同；未force-push或合并main。

新增三个G2 worker计CPU **80.037717s**，累计Core worker CPU **263.241507s**；PC夹爪审计1.662672s、小模型拟合5.850346s、因果anchor复核0.750589s。GPU新增时间0。Core campaign最终约1.339GB；原初轮PC命名产物约257.75MB，此后新capsule／输出额外保留，准备／测试／传输成本未完整逐项计量。失败同样入账，无虚构账单或能耗。

导出时三个G2 MainPID均0、lease CLOSED，campaign有限范围仍ACTIVE，**没有后台运行器**。下一科学步骤是实际动作消费与G2八项诊断、目标顺序Replay和200+1恢复；小响应模型需要新的前瞻执行检验。当前可立即复核本次交付：

```bash
PYTHONPATH=src:. python3 -m scripts.wm02.review_auto1_round \
  --report-dir reports/WM02/auto1-target-entry
```
