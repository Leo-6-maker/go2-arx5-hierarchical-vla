# Codex接手：E1 TRAIN Learning + Recovery GO

**当前用户明确要求以科学推进为第一目标并放行下一goal；E1已经给出完整、有界GO，不是待再次确认的路线草案。** 当前对话仅发布授权与交接，没有替你启动服务器。由现役唯一local-PC owner登记一次E1_ADOPTION_AND_BINDING，核对站点权限、实际任务/资源后直接实施，无需对下面每个正常工程步骤重复询问GO。

从本handoff的同一固定GitHub提交读取：
- docs/protocols/PI0294_E1_TRAIN_LEARNING_RECOVERY_GO_V1.md
- reports/PI0294/design/E1_TRAIN_LEARNING_RECOVERY_TASK_CARD_V1.json

先读AGENTS/START_HERE/已部署统一入口、最新#11/#47；保持单一发布者、每个共享checkout单一mutation owner，不force/reset/热改运行capsule。原E0 208175f9的12条件/11有效拒绝/1中断及所有终态不改。

## 你现在要交付的不是另一份准备报告

A. 用E0四条TRAIN原件，恢复三个S1拒绝的真实接触几何、prefix内首次违规时间及原模型/同状态专家/最多三种分量反事实。已有公开数据已经证明三个TRAIN拒绝offset为0.254/0.312/0.138秒，均在0.4秒前缀内；不要通过缩短远期预览或放宽S1追通过。具体geom pair未知，需要实际日志/影子复算，不猜。

B. 封存原v3.0的4组8条双任务TRAIN成功示范、原stats/processor和最多96个探测窗口。在Hopper从M2048权重开始新的1024-update small-set Flow Matching学习，按协议保留VLM冻结/原有效batch32，固定LR和scheduler，@256与@1024做真实自主TRAIN4评价。M0旧原件不改；新训练是warm-start新optimizer，不叫精确续训。Oracle暂未合格也不能挡住这项学习诊断。

C. CoreLab并行验证source Recovery Oracle。Q0–Q6原CPU准备只做到mock，正式persistent collector/实际state fork/真实expert H40仍需实现。选最多12个E0实际安全TRAIN状态（包括已中断T2的确定安全前缀），全状态/控制历史正确复制后，原S1逐prefix检查。最多12个2秒fork及4个到原32秒时钟终点的完整恢复。能够返回INVALID；某任务不合格不阻断另一任务或B。

D. 合格任务自动进入一轮干预式DAgger采集，最多每任务4条新episode。学生M1先执行合法prefix，第一次S1拒绝时由合格专家在同一状态通过原S1后接管余下任务。非法学生动作不执行。后续expert状态、最初student访问anchor、oracle未执行建议、实际安装和ACK分别记录。生成真实可加载的E1_SOURCE_CORRECTION_V1，不能把失败模型动作、合成状态或8点补零成H40当专家监督。

E. 合格纠正数据成立后，从同一M1@1024各做512更新：BC-only和50%BC＋50%correction，匹配优化步数/配方；各以固定末checkpoint做自主TRAIN4、无Teacher接管。纠正数据仅来自E0 fork也允许学习，但如实标E0_STATE_AGGREGATION_ONLY，不声称新M1在线DAgger已经执行。数据不足则E跳过，交真实B/C结果；不无限训练或扫描超参。

F. 输出第一屏就是：每任务真实自主成功/有效失败/工程不完整、TRAIN碰撞几何与offset、真实Oracle完整恢复分母、纠正数据的独立episode/anchor/window数、可消费数据版本、checkpoint及各端终态。不要用loss或tests代替行为，不自动selected/deployment。

## 必须守住的边界

- 本E1只用source固定底座D0/S1与EEF10，不运行目标整机WBC，不更改WBC/plant/安全/score，不加BASE，不混v3.1。DEV及最终确认不用于新模型查询、teacher标签或调参；旧DEV汇总可作为背景。
- 正常S1拒绝是有效结果而非整个goal永久停止；真实live/fork安全错误或关键状态不一致则停止本E1新source物理/纠正采集，独立合格旧数据上的离线训练/归档可继续。
- 原E0最后缺失TRAIN F0/000 T2不补填；E1同布局的新模型行为是新的学习诊断，不是重试旧负例。
- Vanda已采用的环境建设继续独立推进；本包不暂停它、不重启TF-R3旧安全停止，也不把source Oracle合格当target资格。

## 新资源一次绑定

新E1 T0；24h排队接纳、36h硬收尾。Hopper最多1H200同时、累计8GPUh、3PBS、每次<=4h；总optimizer updates<=2048，推理请求<=3200。CoreLab48coreh/Hopper活跃32coreh，各<=8活跃核；单source live worker。main启动<=32，隔离fork<=16；live/fork<=800000积分、影子<=1200000，总<=2000000。详见机器卡，包括存储上限和收尾预留。是新的研究包，不挪用原E0/其他在途任务额度；站点许可不足取小并一次报告。

GPU申请前修中继生命周期：原E0是relay进程消失后WAIT_ACTION，Hopper生成已完成但未提交，不能编造其他根因。先做进程退出/部分写入/断网fault injection，采用持久request/obs/action/ACK和原子落盘，read-only重取同ID允许，未知神经生成/物理动作不盲重试。继续使用既有安全时钟和权限机制，不存凭据。

## 一次接手后连续执行

按本GO先准备可复用输入和可信source服务，再按预算获得实际GPU。A/C的CPU工作与B学习并行；同scope路径/构建/事务/统计绑定问题自主修并记录，不反复为常规修复请求确认。只在真正越界、新科学因素、实际安全或资源不可用时停相关支路并报精确原因。不要在同一个blocked判断上连续三轮只做重复核实。

接手回读后在#11和#47登记E1来源/实际T0/新run/阶段资源；终报及代码固定SHA上传回读。原始RGB/权重/WBC/私有路径/凭据不入Git。授权状态与实际执行状态分开：本handoff为GO，真正运行开始和完成以你的原生收据为准。
