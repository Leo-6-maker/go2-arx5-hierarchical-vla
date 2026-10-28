# GEN1 M0：已有组件、范围及候选接续限制

核对源码与计划固定到 `c6a0bf2ec403d99ad442120f62a4f95d9c3c1fd2`。本次是源码／已有元数据读取和原纯数值函数检查；没有构造 MuJoCo、渲染、加载模型、采集或新建服务器任务。现役 B24 的 source4940、模型、人口、评分和后台 owner 保持原样。

承接[详细实施计划](../MULTISCENE_ENVIRONMENT_CODEX_BUILD_PLAN.md)。M0 已完成复用位置核对；M1 尚未实现，M2–M5 没有原生执行。下文的资格限制不能被配置文件或 CPU 测试替代。

## 先处理的具体发现

已有 24 个 Teacher 候选是旧 COV1 V1 的 12 个 TRAIN 几何×两任务。实际逐条调用原 `data_cov1.plan.validate_group`，12 个都通过原几何验证；再调用原 `preflight_score_geometry`，6 个 NEAR 全部因距离小于 0.15m 被拒绝。它们距离为 0.126321–0.146634m；6 个 FAR 为 0.195507–0.217177m，通过的只是距离前置检查。

因此，**历史静态 PASS、合法原几何、当前完整任务入口可执行是三件事**。既有候选和静态原件全部保留，不改成失败示范；但不能直接把这 12 个 NEAR 任务条件排入后继动态采集。修订版 `NEAR_015_018_USER_AMENDMENT_V2` 的独立范围为 0.15–0.18m，不能按相同 slot ID 静默覆盖 V1 坐标、hash 或资格。未来若采用修订版，须显式绑定实际版本、来源和新资格；不能根据当前 DEV 失败选择替代几何。

这个问题通过原评分前置门发现，不是用 legacy validator 错误否定 COV1。原 24 候选报告从未声称动态 Teacher 或标签已合格，其历史正文不回写。

## 可复用函数及实际边界

| 能力 | 现有实现 | 输入→输出与接续边界 |
|---|---|---|
| 固定双任务文字 | `scripts/pi0294/scene_cards.py::INSTRUCTIONS, task_card` | group字典＋T1/T2→任务卡；红 cube／蓝 square语义固定。`initial_state_sha256` 摘要的是group设计，不能当实际重置状态 |
| 原布局生成 | `scene_cards.py::scene_group` | split/family/index/candidate→确定性PCG64卡；F1空间、F2视觉、F3相机联动，只用于原义回归，不冒充GEN独立因子 |
| 几何排重 | `scene_cards.py::normalise_layouts, near_duplicate_report` | 已保存card→source/goal/yaw几何；2mm/2°检查忽略外观。GEN仍需补全谱系连通分量与全历史覆盖，不能仅凭此函数证明独立性 |
| PV1冻结人口 | `pv1/plan.py::candidate_pool, evaluation_cards` | 原128组有限池；12 DEV几何×2任务×2noise=48，20 LOCKBOX×2×2=80；原白名单、顺序、总体保持 |
| COV1版本范围 | `data_cov1/plan.py::candidate, validate_group, amend_near_pool, preflight_score_geometry` | 独立namespace／revision；欧氏距离与Y gap不同。旧V1与NEAR修订分别保存 |
| 外观XML层 | `scripts/pi0294/scene.py::xml_scene` | card＋只读asset_root→XML；支持table/floor RGBA、light_scale、两相机平移/RPY。首版只允许桌面色变化，其余固定 |
| 原物理构建 | `scripts/pi0291_arx5_data/scene.py::xml_scene`；`pi0294/scene.py::make_engine` | 原固定基座ARX5资产派生，有限关节/夹爪；`make_engine`才导入MuJoCo、构造模型。M1不得调用 |
| 共同home接缝 | `scene.py::make_engine` 的 `initial_tcp` | 默认source XY＋table_top+0.10；已有override要求4×4 TCP变换。共同home尚未选择／静态验证，不得以配置常量宣称合格 |
| 原静态与可见性 | `scene.py::qualify_engine, rendered_field_of_view` | 已构造零步engine／真实双RGB＋投影→IK/contact sweep/FOV报告；含原Teacher阶段，是真实原生构建而非纯元数据检查 |
| 原双任务Teacher | `scripts/pi0294/teacher.py::make_teacher` | 实际card、物体状态、engine→原T1/T2教师；保留其动态及S1资格，M0/M1不实例化 |
| 实际已提交标签 | `pv1/export.py::committed_packet` | 实际raw samples→state10、640 committed命令、RGB身份、601合法H40窗口；调用者必须先证明scope，函数本身不验证GEN split |
| 原R0 codec | `pv1/export.py::codec_packet` | 已提交数组＋原processor→原物理误差检查，无静默裁剪或换stats；不是GEN元数据验收函数 |
| 完整成功导出 | `pv1/export.py::export_committed_case` | 完整32s成功、S1及raw独立审计→数据/官方CPU处理链。内部写TRAIN标签，未来GEN必须在外层先严格拒绝DEV/TEST，不能直接绕过旧 `packet` 白名单 |

共享export模块有训练适配器和dataset依赖；M1只复用纯元数据组件，不为了验证SceneSpec去导入export或官方GR00T链。M2以后的组件接入还需要原入口、合法隔离范围及准确真实资产绑定。

## 范围不是一个通用validator

| 来源 | 生成分布／验证范围 | GEN接续含义 |
|---|---|---|
| legacy F0/F2 | source x 0.442–0.458；Y gap 0.155–0.17；yaw ±0.08 | 与F1不同，不把所有legacy写成0.16–0.20 |
| legacy F1/F3 | source x 0.435–0.465；Y gap 0.16–0.20；yaw ±0.08 | GEN布局可从明确小范围单独定义；不改变旧family |
| legacy粗检查 | source/goal x 0.42–0.48；abs(y)≤0.14；欧氏距离0.15–0.22 | 粗数值门，不证明任意组合可达／可见／任务成功 |
| COV1 V1 | source x 0.425–0.475；goal dx ±0.012；center_y ±0.025；NEAR欧氏0.12–0.15、FAR0.19–0.22；yaw ±0.16 | 原候选按此来源解释；不能用legacy包络重判 |
| COV1 NEAR修订 | 独立revision的TRAIN NEAR欧氏0.15–0.18，其余按原实现 | 旧V1、旧DEV/LOCKBOX metadata和失败保持；不自动迁移 |

源与目标相机／cube红色／goal蓝色／原摩擦、质量、dt、solver和R0保持；源码中有背景与光照接口不代表首版允许同时变化。实际编译物理不变性和RGB目标识别尚未测试。

## 历史排除索引现况

已精确定位并记录hash：原PI0291布局封存、PI0292确认布局、原PV1完整人口、COV1 V1封存人口和24候选。另有已审计C1四学生父轨迹及实际训练曝光记录，可以作为后续谱系join来源。

**GEN1全历史排除状态仍为 `EXCLUSION_INDEX_INCOMPLETE`。** 这里指尚未构成并核验覆盖全集的统一索引，不表示文件不存在。M1需把实际训练/replay、C1父场景与分支、所有旧确认及LOCKBOX、COV1原版/修订及外观相机兄弟按来源hash合并，保留冲突和谱系连接。不能只排除上述12候选或新GEN目录，也不能只检查实际被采样过的几何而漏掉已打开的DEV。

本次没有生成G_DEV/G_TEST，没有打开新的测试结果，没有宣称零泄漏或未见组合成立。后续候选生成必须在完整索引可验证后才取得这类资格；fixture只能证明算法规则。

## 下一项最小工作

M1以独立、纯元数据的SceneSpec和谱系索引为出口：分离geometry/visual/reset seed；保存design_hash并把actual reset/RGB/static/dynamic留为NOT_RUN；先分区再派生外观与任务；严格拒绝未知字段、非有限数及模型可见的特权元数据。原评分前置距离检查作为独立必需项保留。旧12 TRAIN候选仍是历史来源，不自动变成新的TEST或已合格数据。

原B24先完成并给出配对结果。M2需新的合法隔离资源；M4/M5须在A/B审阅及有效后继工作包之后。无新增预算、PBS、模型、物理或训练授权，原停止时间不变，完整科学goal未完成。

[原函数行号、SHA及12个几何逐项数值回执](../../../../reports/PI0294/design/parallel-science-20261022/GEN1_M0_REUSE_AND_CANDIDATE_RANGE_AUDIT_20261025.json)
