# GEN1 就绪情况与下一最小阶段

2026-10-25；依据[V5](../../protocols/GR00T_WBC_POST_AB_LONG_HORIZON_GOAL_V5_20261025.md)。[M0/M1](gen1/M1_CPU_ACTUAL.md)已完成：复用ARX5源域/场景/Teacher/export，确定性种子子流、36几何设计及谱系分区、8,830条历史几何索引、38项CPU回归。沿用原实现，不重建框架或重跑已过探针。当前新增的是合法NEAR版本的六TRAIN预览元数据绑定；真实图像、模型构造、积分、Teacher和导出均未发生。

## 明确的版本与历史保留

旧DATA_COV1 V1的replicate1六NEAR距离依次为AP0/+Y .146634、AP0/−Y .132754、AP1/+Y .126321、AP1/−Y .129153、AP2/+Y .132246、AP2/−Y .143983m。它们在旧几何范围内的历史静态PASS不变，但均不满足原任务至少0.15m前置门，禁止据旧卡进入动态Teacher。不改坐标、ID、评分或旧失败，也不通过筛掉失败改变旧分母。

本阶段准备版本 **GEN1_TRAIN_PREVIEW_V2_NEAR_015_018**：NEAR选用已保存 `NEAR_015_018_USER_AMENDMENT_V2` 的精确坐标，经现有 `generalization.spec.import_cov1_train` 导入；不是重新抽样。旧V1人口SHA `cc9df4f973d45ce68094523ede75898d0f1737668850c124b451d36939a34739`；历史V2来源SHA `563ddbaddd8da9a55fdc4850dfb2e1294a5a4483e5cc4768c710cda347aaaba3`。新坐标产生新geometry ID，保留同一TRAIN lineage。原M1 DEV/TEST、原六预览设计和旧坐标均不改。

| 六预定TRAIN谱系(末段) | 版本 | 欧氏距离(m) | 本轮资格 |
|---|---|---:|---|
| AP0/+Y/NEAR/001 | V2新geometry | .154187013 | 元数据距离PASS |
| AP1/+Y/NEAR/001 | V2新geometry | .156320534 | 元数据距离PASS |
| AP0/+Y/FAR/001 | V1原geometry | .199526525 | 元数据距离PASS |
| AP0/−Y/NEAR/001 | V2新geometry | .162753851 | 元数据距离PASS |
| AP0/−Y/FAR/001 | V1原geometry | .217176504 | 元数据距离PASS |
| AP1/−Y/FAR/001 | V1原geometry | .195632965 | 元数据距离PASS |

[机读核验记录](../../../reports/PI0294/design/parallel-science-20261022/V5_AB_POSTMORTEM_AND_GEN1_READINESS_REVIEWED_EVIDENCE_20261025.json)逐项列旧/新geometry ID、来源group hash和两外观design hash；私有12份精确spec包SHA `952cca224ad3622872d118ab09d779301637991c94417977333aace00d041200`。六几何×V0/V1的schema和原距离门均通过；历史COV1静态结果不是GEN新外观的原生资格。B实际139条PV1新覆盖与这些COV1候选不同，不能互换解释。

## 接下来只准备24图和两种reset用途

固定geometry/visual/reset种子29420261025/29420261026/29420261027，沿原有限候选与SHA子流；六谱系层次仍为+Y近2远1、−Y近1远2。V0桌色[.69,.70,.66,1]，V1[.72,.65,.58,1]；本阶段不用V2颜色。红方块、蓝目标、原双相机、光照和全部物理参数固定，只有布局和单一桌色因素。两任务使用原指令，图像不计独立物理示范。

- **LEGACY_MATCHED**：原above-source初始化，用于旧域机制匹配和未来C/D；TCP随物体初始化不能证明视觉泛化。
- **GEN_COMMON_HOME**：独立未合格版本。原三个TRAIN候选TCP位置[.45,−.04,.54]、[.45,0,.54]、[.45,.04,.54]m，原downward旋转；尚未选择。需静态IK、碰撞、双相机视野、Teacher接近路径和R0 state通道/范围核对，再记录T1/T2实际相同reset状态/相机/RGB身份。配置hash不能替代实际状态；不悄悄更换normalizer。未来GEN双方共享同一已资格reset。

最小实际M2交付是6×2外观×2相机=24张初态图，并对每对外观保存构造次数、data.time=0、积分调用0、实际robot/object状态一致、相机标定一致、仅允许桌面材质不同的证据。通过现有标准entry、RUN_MANIFEST/注册根/原lease和资源准入后才执行。M2既不是Teacher数据，也不证明共同home可达或动态成功；当前图像0/24、构造0、积分0，全部native字段NOT_RUN。

## 下一关口与当前最小阻断

先取得**仅M2隔离零积分构建/渲染的有效执行范围和实际CPU/输出上界**，再使用现有入口；不因当前旧cap有余额就启动。原停新/停训已过，V5没有延长原时钟或批准新lease；本轮不申请12周预算、不把准备当已派发。之后M3验证共同home；若无合法范围则停在可审阅spec包，不拿CPU PASS冒充24图。

完整A/B现已可审阅，但M4还需后继有效范围、合法reset/任务前置与标签合同。小pilot固定六合法TRAIN几何×T1/T2各一次=12次首次尝试，不是12条成功；失败保留，不补抽凑数。仅当其数据能补充已识别支持缺口，才考虑M5原export及更多数据。旧已打开DEV不转TRAIN，新TEST不因Teacher成功筛选；G_DEV/G_TEST当前私有名单不为本次预览重排。

源域C/D混合研究可用现有银行与LEGACY_MATCHED独立推进，不等待GEN_COMMON_HOME，也不自动夹带新GEN数据。冻结WBC仍只补原CORR/R0的完整prefix、安装ACK、实际反馈和STOP，core64=0/64；目标NOT_RUN不当负迁移结果。当前全部原数据、R0、安全评分、WBC和运行源码保持原身份。

后续CPU增量：[V2加入历史分区后的检查](../../../reports/PI0294/design/parallel-science-20261022/V5_GEN1_V2_HISTORY_DELTA_CPU_ACTUAL_20261025.json)为45个唯一design、原8830条历史，无跨分区连接；未重排原DEV/TEST。真实24图仍NOT_RUN，本阶段不自动Teacher pilot。
