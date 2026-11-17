# R3_LATENCY_ROOT_CAUSE_V1 — 零生成只读复盘

固定PR398 `3497840724c6bfcf54ec4317a5ed605290310b14`、R1源码`fa1477b1`、R3源码`7a8633b7`和原失败结果。本轮新增训练更新／训练访问／模型生成／原生请求／积分均**0**，没有重投R3、加载SLA权重或改旧胶囊。**V5科学主线的非物理SIM_PAUSED合同与独立安全准入另行优先；本工程支线不自动阻塞经批准的暂停仿真研究。**

交付为[完整32条JSON](FULL32_ATTRIBUTION.json)、[逐条表](FULL32_TABLE.md)、[候选排序](CANDIDATE_RANKING.json)与可复核的纯CPU分析脚本`scripts/wm02/r3_latency_root_cause_v1.py`。缺失时间戳保留null，不以文件复制时间冒充执行时间。

## 1. Provider代码一致，四个既有匹配条件仍出现差异

两端SHIFT7包manifest均`ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb`，同原父权重／统计／codec；13个Provider、模型、tokenizer、预处理、采样和解码相关Python文件SHA相同。调用均为一个SHIFT7 Provider、原10步采样、同frame/state/RGB/语言与上一实际发出参考合同。R1和R3 Provider计时都包含预处理、token/设备传输、JAX调用及同步、NumPy回传与物理解码；计数持久化和输出NPZ落盘均在Provider计时之外。R3并未把第二个adapter或结果落盘塞进Provider计时。

从原始记录找到4个相同episode/anchor/noise条件，Provider输入数组、语言、上一参考以及normalized/physical_H40均逐字节一致，0新增推理：

| R3 index | R1 Provider | R3 Provider | 差值 |
| ---: | ---: | ---: | ---: |
| 7 | 179.363ms | 324.655ms | 145.292ms |
| 13 | 178.488ms | 334.727ms | 156.239ms |
| 23 | 175.848ms | 389.714ms | 213.866ms |
| 29 | 178.687ms | 355.745ms | 177.057ms |

这4组排除了“仅仅换了输入样本”对该子集的解释；它们不是随机化性能对照，不能独立确定CPU或GPU根因。冻结输出一致也不能证明kernel耗时一致。

有效声明的GPU设备、allocator fraction、数值XLA flags、OMP/OpenBLAS/MKL单线程设置和配置CPU集合一致；R3在同一配置的两个逻辑CPU上还运行controller、轮询和archive，R1是直接调用进程。额外竞争是值得优先验证的候选，**当时CPU调度/offCPU、GPU clocks/温度/利用率均未记录**。无法从事后静态配置证明实际资源干扰。

当前从实际sla-pt运行环境读取到JAX/JAXLIB0.5.3、NumPy1.26.4等版本；这属于事后只读观察，不能代替两次历史进程未保存的完整环境快照。分析桥接系统Python的metadata未作为运行环境版本。

## 2. 590ms派发周期并不全是模型

热态响应分母31；热态相邻派发有30个间隔，二者分开统计。按每条原始同主机monotonic时钟重建恒等式：

`dispatch_i→dispatch_(i+1) = dispatch_i→安装返回/拒绝 + 终态后串行尾部到下次capture + 下次capture→dispatch`

中位数分别为366.959ms、130.709ms、79.902ms；它们的分位数**不能直接相加**当作周期分位数。完整逐条求和与原589.892ms中位派发间隔一致。30个间隔中28个，在当前终态时已经晚于下一release，因此这些尾部没有名义网格等待。

尾部130.709ms包含消费/finish或cancel、诊断性原编译器重检、context写入、seal、累计records/goal_state持久写、下条Journal预留，以及可能的release等待。其中seal入队实测中位0.767ms；移除它后仍有129.942ms混合残差。**没有单独的P10与两次progress写入monotonic计时，无法把残差全部分摊成精确函数耗时。**

Journal保留了created/completed epoch：INTENT持久写＋空操作这一段中位25.387ms、p95 35.489ms。整个reserve还包含COMPLETE持久写和末次目录fsync；创建至最终rename的原元数据span中位36.749ms，属于带墙钟假设的旁证，不是完整reserve单调耗时。

原代码每次在主线程持久写4次：reservation的INTENT/COMPLETE两次、累计REQUEST_RECORDS和goal_state各一次；每个durable.write会fsync文件、rename、fsync父目录。异步archive仍与这些操作共用底层存储，队列没满不排除存储或CPU竞争，但容量等待确为0、峰值2/4，不能把1.69Hz归因于队列满。

原tmpfs文件仍在，468份内容与封存SHA相符。11条全P10的“install写入→最终live写入”包络中位2.717ms、max3.823ms；在原写时钟连续、写入顺序反映执行顺序的假设下，消费函数位于其中，finish末次fsync不在保证内。这支持先查持久进度/Journal等尾部工作，**不能将该旁证改写成精确P10函数时长或原子发布时间**。21条0原子条件也占有尾部工作，P10不是所有尾部的唯一原因。

## 3. 观测年龄不能从较晚的request创建重新开始

input-available→request创建：中位7.943ms、p95 32.578ms、max46.507ms。request创建→派发中位0.209ms；worker除Provider部分中位10.764ms；传输/发布/轮询混合残差中位13.711ms；响应读回→安装返回或拒绝中位4.048ms。这些为嵌套区间的逐行分解，内部H2D/kernel/sync/decode不能由它们进一步分开。

index5、6、7、8取得ACK时，input-available年龄已≥400ms，而Receiver仍使用更晚的request-created源检查原TTL。两个年龄都保留，原23 ACK／9安装前拒绝／12首点前拒绝／11完整mock P10及原R3失败不改写。

完整32表保留观测可用、request创建、派发、响应、安装函数返回的相对monotonic事件。**精确模型完成、安装原子发布、第一条消费的monotonic事件原来没有记录**；模型完成仅能限制在派发＋Provider耗时与响应读回之间，第一原子只保留“0点未发出”或“已发出但wall事件缺失”。存档副本的mtime一律不用作运行时刻。未来相机采集/传输年龄仍未测。

## 4. 一个已确认的小配置遗漏已修复，未部署

R1和worker的spec及site.validate声明/验证了compilation_cache_dir，但原entrypoint没有将它应用到jax.config。实际声明目录当前均为空；历史有效cache config、继承环境、cache miss/重编译事件没有保存，所以不能宣称“每条都重编译”或“cache关闭导致热态435ms”。进程内热JIT缓存和磁盘持久缓存也不同。

候选补丁只在原site.validate之后、Provider/JIT构造前显式绑定已有目录并回读记录，保留显式cache disable，不改模型/精度/10步/H40/时钟/安全。定向CPU检查验证配置应用、失败拒绝和数值选项不变；**未部署到Core，也没有新的GPU结果或延迟改善声明**。

## 5. 下一项与安全主线

候选验证优先级见JSON：共享CPU/服务并发和内部传输/同步是未确认候选；主线程持久工作、输入交付和年龄差是已知成本；cache遗漏是独立配置缺陷而非已证实的热态根因。此前图像预处理只读/copy的0生成小样本约1ms，不能解释全部差异，也不能替代加载状态下的内核profile。

若另获GPU授权，最小匹配设计是预先固定16个输入/noise，每个A直接调用、B实际worker各一次，总≤32；统一checkpoint、cache/precision、GPU/site/CPU资源，记录主机预处理、token、异步device enqueue、实际kernel/同步与解码以及IPC。必须先锁定双方同等的profiling同步策略，计入测量开销，不能凭此文启动。任何后续连续32资格另立source与有限授权，原R3负结果继续保留。

S0仍独立：controller-selected reference UNKNOWN、真实native STOP最坏响应未测、三碰桌/成功专家1/38误拦和隔离不改。V5可先完成非物理SIM_PAUSED合同与一次性独立准入准备；350ms失败不自动阻塞获批的暂停仿真研究，但本工作包不授予任何native/STOP hook安装/新GPU/训练许可。
