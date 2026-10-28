# PI0283 终局：执行完成，质量 VALID_QUALITY_FAIL

固定授权：Issue #11 canonical 6048886543；release `508e3db7b63d3c47758794f3ffbad87b577e6e9d`。用户已解除墙钟截止；所有科学与资源硬上限继续有效。

## 固定质量结果

| 终点 | TRAIN拟合 | VAL拟合 | TRAIN XYZ中位数 | SO3中位数 | 夹爪MAE中位数 / p95 | 原持续成功 | 首次成功条件 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MORE1024 | 1/24 | 1/24 | 22.882 mm | 0.014252 rad | 0.005615 / 0.294073 | 0/4 | 0/4 |
| MORE2048 | 2/24 | 2/24 | 15.244 mm | 0.012808 rad | 0.001099 / 0.552441 | 0/4 | 1/4 |
| EVENT1024 | 6/24 | 3/24 | 12.193 mm | 0.012378 rad | 0.002686 / 0.726587 | 0/4 | 1/4 |

原 ACTION2048 的1/24拟合、17.335 mm位置中位数和0/4持续成功原样复用，没有重跑。以上每点都是相同固定48个离线输入与4个完整32秒源任务；12个有效episode均保留首个有效结果、原噪声和原分母。CPU独立复现预测解码、输入/噪声字节、native任务谓词、安装参考与同控制后的实际TCP。

原资格仍要求28–30秒全部41个控制采样点同时满足on_top、static、not_grasped。三点所有条件在该窗口都是0/41成功，TRAIN均未达3/4。DEV和CONFIRM为NOT_EXECUTED，不当成已测FAIL。SOURCE_PROTOTYPE、source_candidate、selected、deployment均为NONE。

## 改善与剩余差距

- MORE2048条件10003在10.80秒仅成功一个控制采样点，10.85秒丢失；最终保持窗口仍被抓住。
- EVENT1024同条件从6.65秒开始有44个连续20Hz成功采样点（至8.80秒），8.85秒首次丢失。已保存数组显示首次及后续成功丢失对应static转为false；这不是已证明的唯一模型或控制器根因。最后一次成功在13.95秒；28–30秒物体静止且未被抓住，但on_top为0/41。
- EVENT把接近探针XYZ降到42.066/19.205/24.199/28.453 mm，仍4/4高于10 mm。首次释放夹爪MAE为0.864258/0.617432/0.745850/0.375488，仍4/4高于0.1，且尾部误差不能被小的总体中位数掩盖。
- EVENT的TRAIN拟合6/24均来自晚期静态输入；phase0/1/2仍分别0/4。phase3包含首次释放和后续保持，不能把整个phase3称为静态。
- EVENT的抓持3/4、曾on_top2/4。描述性“抬高>2cm且维持0.3秒”3/4；历史抬升谓词未记录，仍UNOBSERVED，不拿该描述性指标替代资格。

本轮支持有限剂量下的位置拟合与部分交互有所改善，同时关键转换拟合及终末任务保持仍不足。EVENT是沿MORE后的条件性累积续训，不能纯归因事件采样。当前证据不能简化为“数据不够”、单一IK/控制器原因或GR00T普遍不泛化。

## 参考、安装、跟踪分开

EVENT的reference→installed、installed→实际TCP均按同次控制的数组计算，单位mm：

| TRAIN条件 | reference→installed RMS | installed→实际TCP RMS | IK保持次数 |
|---|---:|---:|---:|
| 10000 | 0.067 | 7.663 | 0 |
| 10001 | 0.070 | 9.250 | 0 |
| 10002 | 7.969 | 17.788 | 103 |
| 10003 | 3.694 | 9.149 | 47 |

这不是两个minimum之差，也不把准确安装、低TCP RMS或一次任务成功当作完整闭环PASS。

## 实际训练与恢复

MORE2048与条件性EVENT1024均完成，三份14文件完整Trainer保存并独立CPU核验。MORE消耗4096微批/16384draws（12285唯一UID）；EVENT消耗2048微批/8192draws（6042唯一UID）。四phase曝光严格平衡，原标签、processor/statistics、native masked flow-matching、冻结范围和精度不变。Adam全局步分别3072/4096/5120；仅按授权重置分阶段cosine调度。两组实际LR峰为MORE5e-5/5e-6、EVENT2.5e-5/2.5e-6。原native loss/reduction/累积、输入/epsilon/time、模式、行10梯度与其他31行/冻结张量全部核验。

native loss的前/后16更新均值：MORE0.0362233→0.0333367；EVENT0.0364563→0.0231171。不同采样分布下的loss均值不是纯因果对照或成功证明。

实际连续8 vs4+保存+新进程恢复+4的MORE canary在模型/optimizer/scheduler/RNG以及输入、S4预测和gate上精确通过，16个canary更新已丢弃并计账。EVENT复用该真实通用carry机制及完整MORE2048父状态核验，没有伪造第二份EVENT canary。

## 预算与工程账本

| 项目 | 新实际 / 硬上限 |
|---|---:|
| 正式更新 | 3072/3072 |
| 全部完成更新 | 3088/3136（3072正式+16 canary；memory/rollback为0） |
| 质量查询 / S4 heads | 1108/3456 / 4424/13824 |
| 原固定离线 / 闭环 | 144行 / 12个完整episode |
| 物理starts / 技术替换 | 13/40 / 1/4 |
| controls / physics integrations / RGB pairs | 7940/30000 / 39700/160000 / 7941/30000 |
| GPU占用保守并集 | 8881.532s /14400s（2.467092 GPUh） |
| 技术训练恢复 | 2/2 |
| 有效质量重试 / 新teacher / WBC / target physics / 真机 | 全部0 |

旧PI0282的2080完成更新、874查询、3492 heads、16 starts永久保留；新旧实际总完成更新5168。GPU采用Hopper三个完整PBS分配的start→obit（6991秒）及Core同一CUDA0上模型/等待完成的source driver区间并集（1890.532秒），重叠只计一次；早期reservation和model-only账本保留供复核。CPU由PBS及已保存waited-child收据记录；短元数据命令未独立计时者明确UNOBSERVED，不伪造完整CPU利用率。

工程原件保留：653749缓存绑定失败0训练更新；653762完成MORE后在EVENT首个forward前被旧的额外canary检查拦住；同状态最小修复后的653917正常完成EVENT，退出码0。Core缺CUBLAS环境和合法NVMe路径绑定各造成一个已保留的失败查询；后者仅20个中性控制/100次物理积分/21 RGB，原输入与唯一替换首观测逐字节一致。所有有效结果均不按成绩重试。

## 模型与证据交付

HF PUBLIC仓库：`leoyuer/pi0246-gate-pass-campaign`。三点均已逐文件固定revision回读，model-only不冒充完整恢复：

| 完整Trainer | HF prefix | 固定revision |
|---|---|---|
| MORE1024 | `arms/PI0283/MORE/checkpoint-1024` | `54ba8f2ba4001ef587b58a5d9353b3b34947aae6` |
| MORE2048 | `arms/PI0283/MORE/checkpoint-2048` | `11e413b687bb101f5066bb3d8ccc3f97acf84e77` |
| EVENT1024 | `arms/PI0283/EVENT/checkpoint-1024` | `2a288c49d2d2aa7517da2103bf92911bade01947` |

`MODEL_LOAD_RECIPE.json`提供全部14成员SHA/尺寸、恢复语义、原codec与BF16推理边界。三份完整Trainer仍在Hopper；Core只保存推理工作集，不转存整套optimizer。已消费的导出/推理重复副本有精确退役inventory；完整恢复和唯一证据没有删除。scratch quota仍UNKNOWN，df不冒充个人quota。新目录的逻辑/实际占用与80GiB界限另有存储收据。

三段双相机视频直接来自保存的实际RGB，带source Panda/native PD/non-realtime/not WBC及真实失败水印；无新渲染或物理调用。compact evidence包含输入/输出/targets/noise、安装/物理/接触/预算、代码/封存及小型收据。完整20Hz RGB observation数组仍在Core，compact包明确不包含它们，并保存其SHA索引；完整Trainer在独立HF前缀。各主机compact固定revision及全部文件回读收据列于PUBLICATION_RECEIPT，科学终局不因此触发重训。

原WBC、SafetyGate、decoder/ABI/dt不变；邮件/monitor、目标机器人/真机、TEST/DEV-F、merge均未开放。停止于本固定假设集的有效质量FAIL，剩余预算不授权追成绩或另开训练。
