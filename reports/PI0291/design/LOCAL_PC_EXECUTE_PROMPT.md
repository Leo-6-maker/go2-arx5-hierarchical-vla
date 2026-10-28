# PI0291 — 给现有 local-pc Codex 的夜间接手 prompt

你继续作为 GR00T+WBC 的唯一 local-pc 执行 owner，接手用户新要求的“夜间推动真实 ARX5 场景、数据采集、训练数据集和 Corelab 清理路径整理”goal。请实际完成已获范围授权的工程与采集工作，不停在再拟一份计划。

## 先读固定材料

在 Issue #11 本次 PI0291 canonical 交接中取得固定 release SHA，完整读取：
1. `docs/protocols/PI0291_ARX5_REAL_ASSET_SCENE_DATASET_OVERNIGHT_V1.md`；
2. `reports/PI0291/design/GO_TASK_CARD.json`；
3. #11 最新 PI0290/D4 与现有 goal/lease 的真实状态；
4. #47 Phase-2 M0–M3 和 archive process addendum，读取本地四份真实 audit 原件再做相应维护动作。

以完整协议为准，本短prompt不缩减任何记录/验收要求。首次T0和run UUID持久化；换窗口/重连沿用原deadline和所有计数。最多6小时且不晚于2026-10-13 09:00 SGT，最后45分钟停止新增physics并收尾。

## 这晚按下面顺序执行

**N0 续接**：刷新现役 `pi0267/executor-decoupled-recovery`、固定运行SHA/预算/路径；读取PI0290 D4实际收据，只补缺失项。PI0290动态工具8/8成功与私有原始HF归档已经有证据，不能重采或改叫ARX5。

**N1 场景**：优先已验证MuJoCo3.2.7＋用户交付真实ARX5/Go2-X5模型。搭建明确固定底座/支撑的真实六轴手臂动力学、独立有限关节driver、对向夹爪joint/collision/actuator、桌面/物体与真实body/wrist RGB。核对x5_ee TCP、44mm每指stroke、归一化0闭1开、具名joint/actuator和同tick相机。不要用自由悬浮工具代替ARX5，也不要修改/运行师兄WBC。

**N2 教师资格**：在最多12次commissioning内自主修复独立场景/teacher，保留全部原始失败。真实contact抓放；保存第一条成功原视频和physics hash。首pilot前冻结全部身份/布局/语言/预算，8条各一次；完整证据且成功≥7/8、资格新布局≥3/4才扩采。封存后不允许混合不同物理/相机/动作语义版本拼通过率。

**N3 正式采集**：新namespace、预注册episode/layout-group split，固定64 TRAIN＋16 DEV首次尝试。BC成功准入门分别≥56/64、≥14/16；全部失败和分母留存，不能补seed或凭DEV再调teacher。新物理总starts≤100，积分与CPU/空间等遵守任务卡。

**N4 训练数据**：保存原双RGB、actual TCP/gripper、absolute committed EEF10、planned/committed/achieved区别、合法历史/边界和privileged独立字段。32秒完整episode有641观测、640命令，最多601个H40窗口；正常replan可跨，reset/abort不能跨。正式v1不足H40剔除，不补末帧命令。原生loader配置为relative时输入absolute命令，避免双重相对化。TRAIN-only统计＋真实官方Dataset/processor/collator CPU single/mixed/two-worker读取与数值审计。不能用NPZ存在或手写shape宣称训练链通过。

**维护并行**：接续已经授权的#47 M0–M3，不重置二期≤50GiB新增回收额度，已回收40.994GiB单列。读并hash四份原audit及checker，登记NFS/4080S SSD/home/HF/PI0291路径。24候选copy-only；≤9cache＋≤2派生视图按原门逐文件处理；两旧停止态归档进程按已授权严格身份门处理。缺原审计则维护delete0/进程退役pending，继续独立ARX5。PI0250/PI0276/PI0290依赖、唯一Trainer、原始数据/失败、private WBC全部保护。

**N5 交付**：私有HF按独立ARX5身份、READY-last与固定revision/hash回读；原始视频/数据/代码可定位；报告scene/teacher/dataset/loader/HF/maintenance分项状态、全部attempt和已用预算、真实可恢复下一命令。#11交科学终局，#47交维护终局。只发真实结果和sanitized小收据。

## 执行边界

本goal是新的有限ARX5数据任务。旧adoption禁止新physics不阻止本goal明示范围；旧PI0289 WBC FAIL继续保留，不放松其validator。Corelab负责环境和CPU数据链；Hopper新PBS、GR00T load/forward/optimizer、WBC、硬件均0。两服务器不直连，HF用作固定大资产交付。

在范围内自主修复和推进，前置门过了不再等新的阶段GO。单项阻断只停止依赖它的动作；继续有效场景/数据打包/路径整理。质量失败不重试冲分，工程动作不能重开预算。达到截止或资源上限后交已完成的真实产物，selected/deployment保持NONE。

最高完成标记仅在完整协议全部数据门通过时使用：
`ARX5_DATASET_READY_FOR_FIRST_TRAINING`。
否则给组件状态与确切缺项，不能把partial叫成功。

