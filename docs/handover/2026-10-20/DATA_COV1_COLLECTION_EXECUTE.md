# Codex入口：DATA-COV1 详细长程数据采集计划

日期：2026-10-20。状态：**PLAN_PUBLISHED_AWAITING_EXPLICIT_DATA_GO_AND_NATIVE_ADOPTION**。
本次用户要求发布详细计划，不把文件发布或CI成功当成新物理GO。原D0已经关闭，不复活它的租约，也不把当前PV1剩余额度自动转入新采集。

## 1. 只读这些固定文件，不新建第二套计划

从同一完整固定提交读取：
- `docs/protocols/PI0294_DATA_COV1_LONG_COLLECTION_PLAN_20261020.md`
- `reports/PI0294/design/DATA_COV1_STAGE_GATES_AND_PROPOSED_BUDGET_20261020.json`
- 本入口。

然后读现役AGENTS、START_HERE、原PV1科学协议/资源卡、#11/#47和已部署统一入口/registry。现有唯一local-PC owner继续统筹PV1与未来数据，不新增调度owner/AI子代理，不更改排队或运行的B0。

当前证据基线为 `4b13c60922b1788b1691c73a059dfac4c628f2a5`，D0实际报告 `reports/PI0294/PV1/20261018T182956Z-localpc-pv1/FUTURE_DATA_D0_ACTUAL_SUMMARY.md`。43补审=32PASS+11codec拒绝，另1条原raw失败保留；其中29条原参考PASS形成的168条/93组只是未来资格并集，尚未封存，更没有改PV1139条银行。新增16组全为F2/+Y，方向缺口仍在。

最新已发布B0仍12/48、更新0，677540是最后观察的排队job；进入新窗口必须刷新真实job、CPU/渲染保留、owner及账本，不能据旧快照重投。

## 2. 需要执行的独立数据路线（获得明确批准后）

`新scope与资源绑定 → 冻结144个TRAIN候选槽和split → 原静态资格 → 先4个预声明任务 → P24全48次 → 数据门与预算 → P72累计144次 → 数据门与预算 → P144累计288次 → 分阶段固定银行及总收尾`。

每级的24/72/144是**累计不同几何组**，各组T1/T2；不把阶段数相加，不承诺全部成功。设计为3种独立外观profile AP0/AP1/AP2 ×2方向×2距离=12层，每层累计2/6/12组。优先补AP0/-Y及AP2/-Y；方向不从legacy F1派生。第一批4次来自两个缺口组的T1/T2，包含在48次内，先全部审计再继续。

保持原4mm任务Teacher、物体、相机、语言、机器人/夹爪物理、WBC、S1和score，原v3.0 codec不变。近距0.12–0.15m、远距0.19–0.22m和更广XY/yaw都只是拟验证范围；用预冻结有限静态候选，不能根据Teacher成功重抽。

每段都要全部计划/尝试/排除分母、完整安全任务、原raw/image/时序、601窗口codec、6真实reader/collator锚点、混batch/repeat，以及首次codec错误anchor与实际物理事件。只有全准入成功进入BC银行。失败记录单独留存，不伪造S1未执行后的状态，不把Teacher扩采叫DAgger。

## 3. 资源与阶段门

提议一次性完整批准上限：192 CoreLab CPU核时、12本地控制核时、96h日历，最多4个独立单核worker，288个总首尝试、9216000总live/shadow积分、36400总构造、256GiB峰值输出/暂存、128GiB传输。**这些为计划值，不是已有可用额度。**

Pilot-only上限32 CoreLab核时、4本地核时、24h、48次/1536000积分、6500构造、48GiB峰值、16GiB传输；中段及各停止时刻见机读卡。预算累计而非叠加；无额外qualification episode、benchmark episode或物理replacement。没有新的Hopper/A40训练推理GPU、PBS、神经query或optimizer。原renderer若使用Core图形GPU，按真实情况计量并服从PV1优先，不假称完全CPU软件渲染。

P24门：每任务20/24完整成功、18/24最终准入、16/24双任务配对组、每设计层每任务至少1/2准入，AP0/-Y和AP2/-Y分别每任务至少3/4准入；安全与未知副作用硬门保持。P72/P144的同比分层门在机器卡。门失败则停止扩采并交付报告，不改门追数。

若批准`PILOT_ONLY`，P24结束即关闭。若明确批准`FULL_CONDITIONAL_CAMPAIGN`并实际完成独立scope/lease/站点容量绑定，后续在原阈值与fresh预算全部PASS时自动推进，不逐episode再请示；否则仅记录计划。

## 4. 工程执行要点

原capture.py固定验证PV1命名空间和96组，**不能删guard或伪装旧ID承接144组**。用新增版本化scope/薄调度复用原Teacher、S1、episode、codec和audit。具体命令在实现和验收前标NOT_IMPLEMENTED，不猜远端路径。

最终binder写入完成后、任何worker派发前核对0400/owner/非symlink/所有成员哈希及真实RGB/numeric绑定；metadata副本不是完整银行。静态文件错误不要拖到获得计算资源后才发现。源修改须普通提交、精确SHA CI通过、冷态部署；不热切PV1。

worker按1→2→4增加，互斥CPU和独立model/renderer/output。PV1获卡或资源压力出现就停止新数据任务派发，当前数据episode按原安全边界完成或保存partial；不可让数据生产阻塞B0心跳、RPC、渲染或最终验收保留。原工作会话和单owner不替换。

## 5. 数据交付与下一轮训练

分级交付候选 `DATA_COV1_P24/P72/P144`：完整成员清单、raw/codec准入、12层覆盖、事件索引、失败、成本、固定revision及READY-last。未实际封存不能写READY。D0原参考168条资格并集可在新scope中组装成明确来源的可选union；历史24条12mm合格变体独立保存，不混入覆盖对照。

理论全合格上限：P24合并216条/117几何组，P72合并312条/165组，P144合并456条/237组。真实值按准入和排重重算；旧新并集仍可能方向不均。不同几何组、task episode、重复H40窗口分别计数。

每一级可独立交付，不等144组才产生数据价值。**PV1得到H200就跑原B0/A/B，不等待这个新银行，也不热注入它。** 未来同父/同codec/同更新的数据规模学习曲线，以及任何normalizer重估、事件采样、恢复或模型训练，需另行科学计划与GO。

首次回复：`DATA_COV1_PLAN_RECORDED_AWAITING_DATA_GO`，列明是否已另有新数据批准、scope/现场资源缺口和当前PV1状态。只有批准与现场绑定都成立后才写`DATA_COV1_NATIVE_ADOPTED`。科学报告#11、实际工程和CPU/渲染/存储/租约收据#47；GitHub发布本身没有启动任何采集、训练或删除。
