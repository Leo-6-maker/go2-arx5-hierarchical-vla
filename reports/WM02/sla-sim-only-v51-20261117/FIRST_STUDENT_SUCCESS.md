# 首个真实SHIFT7移动抓持成功

冻结SHIFT7单模型通过普通body/wrist RGB、自身EEF状态和上一实际命令，在真实MuJoCo/冻结WBC中完成了一个原定F1/M_FREE条件：**native grasp、抬升26.406mm、连续保持1.300s，任务方向底座净移动102.950mm**，原综合评分通过，无硬STOP。没有专家接管、教师阶段输入或BASE软参考，没有新增训练。

| 真实端点 | 分母与结果 |
|---|---|
| 原生 | 1请求，23,203个2ms tick＝5703初始化＋17,500正式任务；正式任务35s |
| 模型 | 70次真实新生成，完整H40编译70/70 |
| 实际消费 | 140原生安装ACK、70完整P10、700确认原子；exact solver target确认700/700 |
| 抓持 | 120个native grasp原子；抬升≥20mm连续保持1.300s |
| 底座 | 任务方向102.950mm；实际路径277.030mm |
| 跟踪 | EEF平移RMS18.185mm、旋转RMS0.07882rad；这是跟踪误差，不能替代任务评分 |
| 原始trace | RECORDED_TRACE_CHECKS_PASS，errors=[] |

实际source `c2b245cb6da251833cfc9cf9813f4e01e9da2d85`，模型package `ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb`；spec/result等完整SHA见[机读结果](FIRST_STUDENT_SUCCESS.json)。发表提交与实际执行source分别固定，现场原始证据保存在Core私有scope。

这是暂停仿真profile、既定已知匹配场景的一次真实成功，**不是严格350ms连续实时资格，也不是独立DEV/TEST泛化结果**。仍未导出controller-selected reference，未伪填。原生WBC/MPC/腿策略/臂PD/求解器/2ms及全部原门保持。

同轮另外两个已完成科学条件保持失败：F2首次接触为`x5_link6`腕部碰目标方块（geom73/78），原ILLEGAL_CONTACT在tick16,817中止，STOP后积分0；另一F2在query2的第三次生成被原完整H40速度/加速度门拒绝，未安装。另有旧F1在23个P10后因IPC读取ENOENT技术中断，不抹去或重放。此时学生为1/3有效科学成功，另1技术中断，共4个学生尝试；其余4个原定条件继续，DEV/TEST与训练未自动启动。
