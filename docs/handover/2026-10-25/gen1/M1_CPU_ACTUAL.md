# GEN1 M1：元数据实现与真实 CPU 检查

本次完成配置、确定性有限候选、谱系分区和 CPU 回归。没有修改现役 B24、旧 PV1 人口、R0、评分、安全门、WBC 或运行源码。没有新调度器、账本、原生登记、模型构造、渲染、采集或训练。

## 已实现

- `scripts/pi0294/generalization/spec.py` 是严格类型／字段验证器：有限米制 XY、弧度 yaw、原几何来源、独立种子、geometry/variant/lineage、明确 reset 版本。它相当于 SceneSpec 的可执行类型定义，额外字段和非法数值拒绝。原 `task_card` 和 T1/T2 指令直接复用。
- `population.py` 使用 SHA256 命名种子子流、每槽最多 32 个有限候选，枚举顺序与 Python hash seed 不影响名单。原 2mm/2° 规则用于几何排重；外观、任务、同父分支和经历史节点相连的传递谱系不能跨 split。报告保留冲突，消费方有显式拒绝函数。
- `validation.py` 将配置绑定到原 ARX5 参数；只允许桌面 V0/V1/V2 颜色、原双相机和原 R0。没有新增策略输入字段。新 GEN 记录暂不能调用训练导出：DEV/TEST 拒绝，TRAIN 也须待未来真实 raw/codec/split 资格，不能伪装旧 PV1 TRAIN。
- `artifacts.py` 接受已绑定的历史索引，输出元数据和预览设计；它没有执行器、自动下一阶段或资源准入功能。`configs/pi0294/generalization/gen1.yaml` 是 JSON 子集 YAML，无新依赖。

## 实际清单与验证

实际输出为 12 个原 COV1 TRAIN 几何、12 个新 G_DEV、12 个新 G_TEST，共 36 个 **V0 几何设计**；不是 36 次尝试或示范。新 DEV/TEST 四个布局层各 3 个，768 个有限候选中保留 1 个排重拒绝，未耗尽候选。没有按 Teacher 或学生结果筛选。

已读取的历史排除索引含 8,830 条保留身份的几何记录：原 382 条索引、COV1 原始完整候选池和修订候选池等。修订池通过原函数重建，完整序列化 SHA 与已保存的原生绑定回执一致；不是重新执行静态构造。原 88 个 legacy、8 个确认和 128 个 PV1 几何均已核对。158 条原 manifest 记录、139 条新增 manifest 记录、8 个 C1 导出及 4 个父场景都已关联；A/B 端点各 131,072 次真实保留训练曝光均关联到这些 manifest／几何。

真实清单无所供历史几何碰撞的新 DEV/TEST、无跨 split 连通谱系；配置中的提议组合未加入 TRAIN/DEV 的实际 V0 清单。**这证明已列明来源的元数据分离，不证明所有可能历史数据的普遍独立性，也不证明某种颜色组合从未被旧随机视觉训练见过。** 因此主张是“新布局下指定外观鲁棒性准备”，`new_TEST_qualified`、`unseen_combination_qualified` 仍为 false；未给出实际静态、重置或模型泛化 PASS。

完整 `GEN1_SCENE_MANIFEST.jsonl`、有限候选池、`SPLIT_REPORT.json`、`FACTOR_COVERAGE.csv`、组合表、历史索引和 reset 报告保存在已有私有研究目录；公开回执提供其全文 SHA 和字节数。新 TEST 坐标不放入 GitHub，不制作人工筛选 TEST 图片。六份确定性产物已在两次真实构建间逐字节一致。

本地 23 个 GEN1 用例和原 COV1、PV1 export/evaluation 相关回归合计 **38 passed**。独立子进程阻止 MuJoCo/Torch/ROS/OpenGL/engine 导入和网络／子进程调用；另外验证进程 hash seed 和枚举顺序无关。测试中的合成谱系只用于规则验证，没有冒充原生证据。现有 pytest 对 pythonpath 配置有一条提示，执行时已显式提供路径；没有升级环境。GitHub CI 未宣称通过。

## 范围差异、失败和独立 reset

M0 查明的六个 COV1 V1 NEAR 原评分距离拒绝保留。导入仍用原 COV1 范围和标签，既不改 legacy validator，也不静默替换为修订几何。GEN1 新几何单独用小范围 legacy profile，Y gap 0.18m 归 NEAR；历史 COV1 的欧氏距离分档保持原义。

共同 home 已准备三个固定 TRAIN calibration 候选，均使用原 downward rotation 和现有 `initial_tcp` 接缝；不依赖 source/goal/task。尚未选中任何候选，状态为 `COMMON_HOME_UNQUALIFIED`。当前 card 转换会拒绝把这个未验证 profile 交给执行端。配置 hash 不作为真实 reset state、相机标定或 RGB hash；这些测量仍为空。

准备过程中两次本地 join 假设错误已保留在回执：原排除 digest 绑定的是规范化几何而非原 wrapper；部分实际原始训练曝光行没有 `scene_group`，需要用对应已保存 manifest 的 `group` 关联。已按原语义修正、重新完整核对，无远端操作或未知写。

## 下一项最小阻断

六个 TRAIN 预览几何已在观察任何新图之前固定，层数为 +Y近2、+Y远1、−Y近1、−Y远2；预定 V0/V1 × 原双相机共 24 张图。目前实际图片、构造和积分均为 0。纯视觉预览是诊断，不是新增训练样本或未见组合证据。三个选中旧 NEAR 几何仍有原评分前置拒绝；以后不得把预览成功当成这些 Teacher 条件已能派发。

**M2 尚无已采用的 GEN1 隔离原生执行范围。** 原停新时刻已过，现有 B24 的特殊窗口、lease 和预留不能借给 GEN1。先按原入口核定有效范围、完整保守资源与原生绑定，再做构造／24 张图／零积分、T1/T2 实际初态和机械不变性检查；不能把零积分等同于无资源或自动授权。当前无需为只读 CPU 准备变更预算。

M4 的 12 次 Teacher pilot 仍须完整 A/B 结果审阅及有效执行范围；M5 和训练未启动。旧 NEAR 的动态前置问题须在其明确版本下解决，不能调评分、重抽容易场景或覆盖历史失败。完整 WDR1 目标保持 ACTIVE。

[实际机读回执](../../../../reports/PI0294/design/parallel-science-20261022/GEN1_M1_METADATA_LINEAGE_AND_CPU_ACTUAL_20261025.json)
