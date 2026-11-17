# RoboDojo Real/Sim V1 · first delivery actual

**首项交付已完成：现役V5.2完整终态已封存；RoboDojo首12条实读后，本次command混合为BLOCKED_LABEL_SEMANTICS。Real10/25和新24条件均未启动。** 不是策略成功声明，也不以CPU检查或发布代替训练/任务结果。

## 现役V5.2：完整24后才决定

[PR405终态](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/14886acdf739484cac914578e72736e63cedef62/reports/WM02/sla-mobile-f2-v52-20261117/E2_FINAL_ACTUAL.md)于2026-11-18 02:05:02 SGT封存。B0=5/8（F1 2、F2 3），B1=6/8（3、3），B2=6/8（2、4）；合计17/24综合成功、5原H40拒绝、2接触STOP、0技术失败。STOP后均0积分。B2对B1总体差0，F2收益伴随F1下降及STOP，未证明总体闭环收益。完整24后无合格候选；开发排序P=B1不等于部署胜者。

原E1两臂各500更新/16000访问和792离线生成复用，未重训；本轮旧队列完成1384次真实生成和481233原生积分，两个旧接口未重复计费。D0未重复。5/7失败仍为H40边界，P两次失败均接触前首端拒绝，E3覆盖假说缺少具体配对证据，不自动纠正扩量；确认/DEV/TEST均0。旧36布局人口未被本混合计划占用。

实际275源文件、C20统计及site binding SHA核验不变。旧unit自然exit0、MainPID0、followup/E2/worker leases CLOSED、世界/神经锁释放。没有热改、抢占、重投或续期。exit0只表示有界队列完整结束；RP8/JOIN10/R3/PR403及其它历史负结果保持。

## 源数据实读与阻断

固定HF `RoboDojo-Benchmark/RoboDojo@1127bda94f172d757386945af3eb3a6754d1a4e6`，6个ARX X5任务各100条的元信息先行封存，固定seed2026111801选总60；每任务前2条组成首12，包含于60中。已下载并核验12/12文件、2,021,030,623字节、12,585原始帧；48条未读，不更换不合格样本。[SOURCE_INVENTORY.json](SOURCE_INVENTORY.json)含逐文件真实LFS SHA，而非git指针或Xet ID。

[语义实测](SOURCE_SEMANTICS.md)：12/12中左右臂各三字段均有action[t]与state[t+1]逐值完全相等，75,438字段行；最终action重复最终state。它匹配下一帧反馈构造，未取得采集器dispatch/command定义，不能作为已发命令证明。只有nominal frequency=30Hz，无真实时间戳/同步、frame/TCP标定、session/operator证据；未伪造P10、上一实际命令或Go2时钟。全部原始双臂SOURCE_NATIVE保留，canonical合格0。

[表示实测](CODEC_COMPATIBILITY.md)：条件性把源标量映射为2–42mm等效width后，ZOH 522/1568完整双臂数值窗的必要误差下界已>0.2mm，最大20.38mm；线性敏感性506/1568，最大20.99mm。未超必要下界的窗口不视为通过逆编码、RMS、crossing或事件覆盖。没有调用旧2秒labeler、改decoder或只挑静止段凑覆盖。源EEF原目标包络违规单列而未当坏数据过滤。

解码检查60张head/双腕样本，60/60标准XPL-RGB1、480×640×3；原压缩JPEG保留。稀疏图像、另一臂静止或raw hash不同都不足以证明单臂独立/session独立；[SPLIT_MANIFEST.json](SPLIT_MANIFEST.json)明确6/2/2仅候选槽、所有合格TRAIN/DEV/TEST均0，未开启源TEST。

## 实际身份、成本与下一动作

计划be37da06fd007b24da324ae1cce6ba4ba441687e。审计源码d0371bfd9495652e853757534a8d837381ff461a；数值诊断源码09efbebbc4acb343578047d259aca65b23ea02c1；完整数值证据SHA8025c0fea6e9e7d578c302dba5257859493d74d788cb77b72aa71e396cce6e58。site绑定与原allocation截至Nov19 02:51:23.855488 SGT核验；旧服务期限没有被新CPU服务延长，也不据已关闭旧lease推断未来GPU许可。

独立CPU环境sla-source-audit：h5py3.16.0/NumPy1.26.4/HDF5 2.0.0，只读复用旧数值依赖，没有修改sla-pt/native环境。审计unit wm02-sla-robodojo-real-sim-v1-first12-20261118-01实际PID/cgroup已独立观测，linger=yes，Restart=no，2CPU/4GiB/1解码worker，有限2h且实际117.28s退出、lease CLOSED。首12profile CPU3.97s；成功数值检查31.70s wall/1.84s CPU；60图0.78s wall/0.27s CPU。全部新owned树逻辑字节2,048,840,790，低于64GiB；共享旧依赖不重复计。终态cgroup峰未恢复，仅保留观测下界1.51GB，不伪报完整峰。

数值检查两次技术失败分别为包装代码解析、NumPy bool序列化；均保留，新版本先过Core合成HDF5/JSON回归再封存，没有重下载或重跑科学实验。失败分析CPU未单独测得；成功CPU不是全阶段CPU。局部测试4通过/2因本机无h5py跳过，Core实际两个HDF5 smoke通过，未运行CI。检查数不算策略资格。

**本RealMix工作包新增训练更新0、优化访问0、模型生成0、原生请求/积分0、真机命令0。** SOURCE_NATIVE读取与诊断成本另计，不能把0优化访问写成0数据读取。原B2/Real0身份f95a5bafc2c67cae722c74544fd2c6b6aa1496805f04179dd13c134338081e25未变。

固定兼容性门未过，两个500端点和新24/确认8/新DEV32/TEST40不派发。下一科学动作需要解决action/feedback生产者、frame/TCP及可核验单臂/事件语义；若确认future-achieved，应另立SOURCE_NATIVE预适配或通用表示研究，不能把改decoder混进0/10/25配比。冻结WBC/安全门、C20统计/loss、旧失败与保留TEST人口全部保持。
