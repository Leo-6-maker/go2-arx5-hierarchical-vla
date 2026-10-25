# A4096 / B4096：实际模型与保留训练数据卡

本卡将原完整 checkpoint、已完成模型交付、Eval 实际 `MODEL_READY` 和已保存的训练消费记录关联。它是最终 A/B 结果包的身份附件，不是新的训练、选模或泛化结论。当前只运行原 N294101 首面板；正式 DEV48、B0 保持比较、独立确认和目标 core64 仍分别未完成。

## 两个实际加载端点

| 项目 | A_CONTROL4096 | B_COVERAGE4096 |
|---|---|---|
| 完整 checkpoint manifest SHA256 | `e0b9d8cb67efc82ae8daa67d71f95df8fb5c607e0f75e26d58cf01412a3e807a` | `be7ab71dee0fb68e17efae06a04ff8dde9cd776ff429c1d6c408ae01323a9c19` |
| 保留分支更新 / 实际 anchor 消费 | 4096 / 131072 | 4096 / 131072 |
| 实际 MODEL_READY 原件 SHA256 | `6625f00151c6adac35d1b46bba2d4363b1242142754a2e4007386be6e57664c5` | `94f08ac784398576689bcd889107419da1a3da81db3860a6c4596a21767973e1` |
| HF 交付固定 revision | `fbff9c0c3fd0cd5bfbadf345c88a20be00fd970c` | `45e586477de6139279834163bd3da4592a290d32` |
| Eval 固定源码 | `7cc751803e18b911fbe6ad896efd9a1c0ed936b0` | `4940cbbad36cfeb8a89aa694b795f77fbcba99a0` |

固定交付位置为同一 dataset 仓库 `leoyuer/pi0294-public-artifacts` 下 `PV1/V4A_SAVED4096_INFERENCE_20261024/<arm>`。每臂推理 payload 为 12,576,324,408 字节，另附原许可和说明。公共推理副本不包含 optimizer/RNG/scheduler；原 15 成员完整恢复状态仍保留在原授权存储。不能拿公共推理副本冒充完整续训状态。HF 交付 revision 与运行 relay 的 model revision 是不同身份字段，机读卡分别保留。

两个实际加载端点均保留原 R0：`fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`。本次逐项核对九个加载成员与原完整 manifest、历史交付清单：数值 tensor 原样，公开时声明的路径元数据在加载前恢复。

两份原 `config.json` 只有顶层 `model_name` 的私有路径绑定不同；原 processor config 字节相同。两份 `embodiment_id.json` 字节哈希不同，但固定 HF revision 的真实 615 字节文件分别匹配原加载哈希，解析后的全部键值相同，`new_embodiment=10`。这个差异属于序列化键顺序，不能当作 embodiment 映射变更。私有路径值不发布。

## 真正保留到端点的数据曝光

下表由原 worker trace、实际 forward microbatch 和完整 sampler cursor 的七段接续记录汇总；不是重新模拟采样得到的期望值。所有段连续覆盖每臂 cursor 0→131072。

| 数据通道 | 实际参与的 export episode ID 数 | A anchor visits | B anchor visits |
|---|---:|---:|---:|
| 原始 T1 | 96 | 32768 | 16384 |
| 原始 T2 | 30 | 65536 | 32768 |
| 共同 C1 T1 | 8 | 32768 | 32768 |
| 新覆盖 T1 | 75 | 0 | 16384 |
| 新覆盖 T2 | 64 | 0 | 32768 |
| 总计 | 不跨通道相加充当独立几何数 | 131072 | 131072 |

两臂 T1、T2 实际各 65536 次。B 用新覆盖数据替换了一半原 T1/T2 通道访问，总更新数和任务比例保持匹配；新库贡献 49152 次，占 B 保留消费的 37.5%。8 个 C1 导出并非 8 个独立父场景：原审计映射到 4 个父场景。两臂 C1 的完整 episode/anchor 直方图完全相同，不能仅凭 C1 重复曝光数量解释 B−A。

原 manifest 中记录数不等于本次消费支持数。旧 DATA_COV1 P24 的小规模资格记录也不等于这里实际注入的 139 个 PV1 新覆盖导出。窗口、外观兄弟、恢复分支和 export 数均不能直接解释为独立物理几何数。谱系及事件支持详见已有 [TRAIN 审计](V4A_FAILURE_ATLAS_TRAIN_COVERAGE_AND_CANDIDATES.md)、[C1 审计](V4A_C1_PARENT_FORK_EXPOSURE_AUDIT.md)和 [GEN1 历史分区核对](gen1/M1_CPU_ACTUAL.md)。

## 比较成立的范围与保留限制

实际 Eval 源码版本不同。本次从两个固定 Git 对象逐字节核对 16 个指定科学文件：原 scene/scene_cards、episode、S1、numerics、relay、E1 model/neural、PV1 core/neural/driver/evaluation_plan/model_inputs/offline_policy，以及原 ARX5 scene/config 均相同；整个 `src` Git tree 也相同。机读卡列出每个 SHA 和全部变化的 scripts 路径。账户、资源、共享 GPU、恢复、队列及汇总等运行层有版本变化，**不声称整个运行环境或资源竞争条件完全相同**，也不从这些核验推导吞吐可直接比较。

保留端点 4096 步不是历史累计计费更新数。原失败 owner、丢失的训练效果、零模型启动失败和全部费用继续保留；B 的已审计端点复用依赖此前明确批准的例外，不能把外层 Exit1 改写成成功。本卡无新增模型、查询、训练、物理或原生写。

完整 B24 尚需原后台完成后独立核验，最终 B−A 按原同几何配对方法计算；单噪声面板结果即使有效，也不是完整 DEV48 或独立泛化证据。不根据中途结果改权重、评分、数据或启动训练。

[逐项身份、数据曝光与固定源码核验机读卡](../../../reports/PI0294/design/parallel-science-20261022/V4A_AB4096_MODEL_DATA_IDENTITY_ACTUAL_20261025.json)

后续实际接续：[原A24+B24现已完整审计和关闭](V4A_A24_B24_COMPLETE_FIRST_PANEL_RESULTS.md)，首面板差值与成本已出具；以上等待B24的文字是本身份卡首次发布时的历史状态，正式DEV48／独立确认限制仍保留。
