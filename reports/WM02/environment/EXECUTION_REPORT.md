# WM02 P1.5实际执行：环境、GPU200与H16

2026-11-04，Asia/Singapore。**E0环境核验、E1独立GPU神经烟测完成；E2离线状态容器／评分定义与反例测试完成；目标原生导出和动态资格仍未完成。** 没有重新采集已通过的872条S域诊断，没有构造目标执行器，没有提交新PBS。

| 用途 | 完整源码SHA |
|---|---|
| 本轮交接／实际Core环境预检脚本 | `90c0507333d2a9c6b26ee9c22278b050a9249fad` |
| 首次打包导入失败，模型加载／更新0 | `45e83fd2d2e65645cc275eeedfd33a0697d378f6` |
| 实际成功GPU200＋H16执行 | `21f2811f00aedb37b47e7b52d827d4ba1e7cb38e` |
| 目标离线容器／站立评分及其CPU测试 | `36f6d287cee0ddb0abfefda1ce2b7de81f9c6491` |

独立checkout和新分支从固定交接建立；正常合并PR345后续停止回执，不重写其分支、main或旧capsule。源码包按实际Git对象生成，逐文件hash绑定到原registry／lease；运行期间不热改。私有checkpoint、数值数据、WBC资产及主机路径不进Git。

## 环境与旧队列

三套环境实际版本、冻结23文件和软件依赖见 [ENVIRONMENT_MATRIX.md](ENVIRONMENT_MATRIX.md)、[机器环境回执](ENVIRONMENT_ACTUAL.json)。Core实际运行交接的`env_preflight`，exit0，9核心＋14初始化成员全部匹配。裸shell缺库的原因是未应用原绑定环境；恢复原runtime_environment后三份冻结ELF均无未解析依赖、8模块导入成功。没有安装或重编译ROS2／OCS2／MuJoCo／WBC。

E-G从可信旧神经环境复制精确依赖，建立独立venv；49个顶层条目、28个依赖发行版，实际torch/numpy导入均来自新目录。没有共享可写site-packages或改动原训练／控制环境。Torch2.9.0+cu128、CUDA12.8、NumPy1.26.4、Python3.12.3，driver580.173.02。

按用户最新“关闭所有旧队列”指令，原Core relay派发器及其CPU子进程通过一次SIGTERM结束；此前短暂SIGSTOP已在关闭时解除，进程不是留在暂停态。旧队列未执行项记UNRUN，原lease CLOSED、canonical结算与Journal已独立核对。M13保持USER_STOPPED，原PB两个worker lease均CLOSED；四个已知BOOK native96 actor也已CLOSED，其原失败／成功数据均未改写。原BOOK结果不计作WM02结果。

本地尚未启动的WM01 W2 continuation已按原锁／Journal关闭，W2标UNRUN，不重开旧lease。Vanda旧1515716已F／Exit0、5000端点与lease CLOSED，未取消其已完成模型或重新提交。当前账号另有两个03:25后提交的其他项目作业；没有将其误认为旧GR00T-WBC队列或改动。Core其他用户GPU任务未终止。

发布前再次只读核验（2026-11-04T04:02:39.570453+08:00）：所有已知旧Core与本轮WM02 unit均MainPID=0、相关lease均CLOSED；原目标隔离SHA未变。GPU0空闲23740MiB且仍有其他用户进程，GPU1空闲15899MiB。即时空闲不作长期reservation。

旧控制收尾出现过部分元数据写入：空ControlGroup被误判为cgroup根、已退出unit的CPUUsageNSec为`[not set]`、以及OwnerLock不支持with语法。均在实际读回后修复：已有queue／lease关闭不重放，采用原journal中的90.550802 CPU秒完成canonical与原INTENT对账。保留原错误，不把这些工程问题称作学习失败。

## 实际模型、数据与更新

一个5×2048 SiLU／LayerNorm MLP＋GRU256结构化状态增量模型，实际 **19,113,552参数**，不是把设计中的约17M当实测。93维设计物理状态、400维H40条件、9维固定context；旋转通过SO(3)指数更新。该tensor绑定是合成工程版本，尚非原生传感schema资格。

数据为256个合成TRAIN序列、每序列8原子步burn-in＋8步自由展开；masked terminal之后不作监督，collector slice不冒充32秒timeout。数据sha256：

`97eede32ab18193db6577f3cca63463297b8613255efe3cfda247c746decfe95`

真实S/T转移均0，没有将固定底座数据补BASE零作为目标标签，没有读取CALIB／DEV／TEST或Teacher隐状态。归一化仅使用合成TRAIN有效状态。没有加载GR00T或WBC模型；旧WM01模型／训练器不作本轮WM模型。

| 实际GPU效果 | 数量／结果 |
|---|---|
| WM主优化 | 200 AdamW更新，FP32、lr2e-4、clip1、effective sequence batch256；microbatch32累积8次；51,200序列访问，不是独立真实episode |
| 微批8／16／32 | 均实际测量完整effective256梯度，不做optimizer.step；相对micro8最大梯度差3.73e-8；32档约0.303秒，8档约1.629秒 |
| WM恢复探针 | 额外1次：从checkpoint199加载model／optimizer／normalizer／RNG／cursor，再执行同一个第200步；参数与优化器最大误差0 |
| 主200端点重载 | 权重／optimizer最大误差0，cursor51200，normalizer与RNG恢复路径执行；CPU随机数恢复另有单测 |
| 固定32序列TRAIN切片loss | 同一切片从0.3518843651降至0.0444712974；不是DEV误差或原生能力结果 |
| Actor／Critic | 各1次真实优化；H16=0.8秒，只在原子0／8重规划；effective starts64、microbatch16累积4次 |
| WM冻结与输入梯度 | Actor更新前后WM参数bitwise相同、WM参数grad全部None；两次决策到预测TCP的输入梯度均非零，与直接动作代价梯度分开记录 |
| Actor／Critic完整重载 | 权重输出／optimizer比较最大误差0；保存RNG、replay cursor及父WM checkpoint身份 |
| 原生任务／目标分叉／新PBS | 全部0 |

GPU实际构造WM实例3个（主模型与两次重载），Actor／Critic各2个（含重载）；CPU微型模型单测不是科研训练更新。实际GPU WM optimizer.step共201次，不能漏报恢复探针。详见 [WM02_GPU200_ACTUAL.json](WM02_GPU200_ACTUAL.json) 与 [ACTOR_H16_ACTUAL.json](ACTOR_H16_ACTUAL.json)。

Actor工程目标是明确标记的合成物体到目标平方距离与动作代价，**不是**尚未绑定的原生reward，也不是P4正式优化配方。H1／H8／H16和start_batch8／16均测过梯度；这证明神经计算图与冻结／保存机制，不证明学会Go2抓放或校准了真实动力学。

另作CPU只读checkpoint字典检查，没有构造模型／初始化CUDA／增加更新：实际WM state全部float32、30个optimizer state项；normalizer含mean/std，RNG含Python／NumPy／Torch／1个CUDA设备状态。Actor／Critic工程optimizer分别为AdamW lr3e-5／3e-4、weight_decay默认.01；这些实际参数留在机器结果，不冒称完整P4配方。

完整checkpoint身份：

- WM200：`3e54ed8dabd24ac4003a4c4cdcd601d1728e5850f486f2c56ea93b926a81dd91`，229,415,061字节。
- Actor/Critic1：`2faeadfc5109c0e7e4d5f0cdf15d50bdc45c3d7fbf4020f69d5892d752393b39`，5,782,997字节。
- checkpoint199保留用于独立续训比较，未删除或覆盖。全部留在Core私有节点本地目录，以`weights_only=True`加载，不跨环境反序列化任意动态对象。

## 显存、成本与负结果

实际GPU0的NVML总量为32760MiB；不是GPU1的16376MiB。Torch分配上限13GiB。WM训练阶段峰值allocated527,145,472字节／reserved704,643,072字节；H16优化阶段565,406,208／675,282,944字节。整个进程NVML采样峰值1192MiB，包含CUDA上下文等；Torch值是分阶段统计，不冒称完整恢复阶段的连续全局峰值。

共享期间全卡采样峰值32085MiB，含其他用户负载；它不是WM显存需求。当前实验成功不代表长期有空闲reservation，也没有实际在GPU1或原生sim＋WBC＋渲染并驻条件下验收16GB。完整profile见 [LOCAL_GPU_PROFILE.json](LOCAL_GPU_PROFILE.json)。

成功unit实际CPU78.996058秒、墙钟76.553736秒（含普通／-O CPU单测与加载）；神经主进程CPU67.748073秒、烟测墙钟67.314996秒。WM训练及端点恢复段62.719457秒，Actor H16更新／保存重载段0.711130秒，均为上述包含关系，不能相加。环境复制setup的native CPU4.162455秒、worker墙钟11.448035秒另列。预检、源码调查、PC测试、SSH／发布未完整计时，不称完整总成本；旧队列90.550802 CPU秒仍归旧实验。

第一次GPU任务因执行包遗漏公共contracts依赖，在模型导入前exit1；没有进入测试、CUDA模型或optimizer。该lease已独立关闭，源码未热补；后继用完整src包、新run身份和固定SHA执行成功。另保留pip元数据不存在的只读探测错误与Torch2.9 TF32设置弃用warning；实际采用FP32／allow_tf32=false，未因此重跑有效200端点。

## E2与首个阻断

[状态／动作映射](TARGET_CONTROL_STATE_MAP.md)基于实际源码调查。新增数值容器拒绝缺MPC/WBC/receiver块、对象序列化、错误身份或篡改；它不实现live exporter，不声称qpos/qvel恢复等于完整控制恢复。[新站立T1评分](TARGET_SCORER_BINDING.md)复用原物体／安全阈值，显式新版本区分旧BASE前移任务。四项离线测试含五类评分反例，普通与`-O`均通过；原scorer／SO3／SafetyGate回归70项通过。总world_model套件30项通过；Core新venv神经6项普通／-O均通过。最后兼容性检查中，无Torch的E-S正确skip神经测试模块（exit0），独立Torch环境6项仍全部通过；未给源环境增加Torch。全仓库CI未跑。

**首个确定阻断仍为原S1/T2隔离没有有效解除／目标动态准入。** 完整控制状态导出或相同prefix重建资格、真实ACK/消费进度及站立T1全身安全传感绑定也仍缺。E3未启动，E4真实采样→WM→Actor→再次采样尚未执行；新合成模型不能替代这些证明。不会把permission=true写入spec、重训WBC或先投A40/H200长训。

## 真实命令与下一条可执行命令

下列为实际入口调用参数的脱敏表示。GPU有限worker实际通过`neural_smoke.main(args)`调用入口，其等价CLI列在下方，并非另启动过一份shell训练。完整OS启动argv、绝对路径及操作helper SHA保留在执行者私有`ACTUAL_COMMANDS.private.md`；已完成的Journal／onceIO不重放，旧CLOSED角色不重开。

```bash
PYTHONPATH=src "$WM02_TARGET_PYTHON" -m scripts.wm02.env_preflight --spec "$WM02_E0/RESOLVED_SPEC.private.json" --target-root "$WM02_FROZEN_TARGET_ROOT" --output "$WM02_E0/ENV_PREFLIGHT.private.json" --gpu-query
# exit0；实际9个核心字节匹配；14初始化成员由原完整path pin另查
PYTHONPATH=src "$WM02_GPU_PYTHON" -m scripts.wm02.neural_smoke --spec "$WM02_NEURAL_RUN/EXECUTION_SPEC.json" --output "$WM02_NEURAL_RUN/gpu200-h16-01" --run-id gpu200-h16-01 --profile wm02-gpu0-neural-smoke --microbatch 32
# exit0；200主更新＋1恢复探针，Actor/Critic各1
PYTHONPATH=src "$WM02_CPU_PYTHON" -m unittest discover -s tests/world_model -p test_target_interfaces.py -v
# exit0；4测试，含5类物理评分反例，仅CPU合成检查
```

下一条可执行只读复核为执行者的`final_readonly_verify.py`（经已有Core transport），检查旧owner与本轮lease终态、GPU即时状态及未解除的原目标隔离。它不加载新模型／物理、不重采872条、不提交队列。继续科学阶段前须绑定合法真实TRAIN数值replay和原native准入；目前尚无可冒充已实现的正式collect→fit→Actor→collect CLI。
