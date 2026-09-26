# PI0248 — M64姿态与保持修复：有对照的微调计划

日期：2026-09-26（Asia/Singapore）
授权标识：`PI0248_M64_POSE_REPAIR_GO`
发布状态：`AUTHORIZED_NOT_EXECUTED_BY_GPT`
审核源码：`9b766e9a6d28ab504a11b3283083bac364c6cbb8`
Issue #11为canonical；PR #77仅pointer，保持未合并。

## 1. 审核结论与本轮范围

接受PI0247终态 `SCIENCE_FAIL_M64_SUPPORT_EXPANSION`，旧结果不改。但必须更新里程碑：U-final2000已在原16 TRAIN窗口、B0/B1分别通过四项联合门。当前已有M16-qualified模型，尚无正式selected checkpoint或本轮WBC任务成功证据。

本轮授权：复用已有数据做窄CPU分解和新增旋转损失实现验证；无真实输入/标签/decoder冲突后，执行C/R两臂各2000更新及固定配对screen。目标是区分“再微调2000步”与“补齐自由生成姿态监督”的作用。不是保证通过，不重训旧L/U/M64，不延长旧协议预算。

两臂都从已完成M64-final5000权重开始，新建optimizer/scheduler。C保持原U目标；R仅增加第5节定义的旋转终点项。其他数据、采样、LR、模型和评价规则相同。

本轮完成后提交M64修复与保持判定。通过者形成新fit-recipe候选，进入FULL准入审查；本文件不启动FULL、CALIB、selector或WBC/sim。既有长期路线仍为合格M64配方→FULL完整family→原selector→winner TRAIN准入→原frozen-WBC闭环。不能把小样本或修复模型直接称selected。

## 2. 来源及证据边界

必读来源：
- Issue #11 terminal `5847647939`；L/U verdict `5846868900`；M64起点/1000 `5847239992`；2500 `5847395732`。
- PI0247 release `50e66efe6ccfbe814d04caf12499493821be7957`的主协议与execution notes；R2 `c0a1ca9d7b15f2b8383bb71fd333b8add39e5997`。
- 当前 `scripts/pi0247_lu_train.py`、`scripts/pi0247_u_rollout_overlay.py`、`scripts/pi0245_physical_delta.py`、`src/go2arx5_vla/trajectory/so3.py`以及实际screen/独立CPU验证器。
- pinned GR00T source `51d4c89f72fda44cbf77285c6a8114b52676b8a1`，实际冻结processor、TRAIN statistics、modality及decoder。
- HF repo `leoyuer/pi0246-gate-pass-campaign`，terminal evidence revision `fd9b33a4ab6b19502d739d744e13af00988084d3`，prefix `reports/PI0247/M64_U_5845272077/step5000_resume633819`。587文件回读PASS是执行侧已发布证据。

GPT本轮读取了GitHub账本、执行ref、上述相关源码与提交；没有SSH/live qstat、项目模型调用、训练或原NPZ复算。当前环境未成功取得HF原始publication文件，故未独立重验587文件。只做了不含项目数据的旋转弦距数学恒等式自检，不等于真实decoder或GPU梯度验证。

最后读取的terminal发布时间为2026-09-26 15:53:46 UTC；其中无活动作业是该次观察，不是实时scheduler状态。

### 已完成的主要结果

| 模型/集合 | noise | REJECT | XYZ median/p95 (m) | SO3 median (rad) |
|---|---|---:|---|---:|
| L-final2000，原16 | B0 | 1 | .037978 / .048852 | .035983 |
| L-final2000，原16 | B1 | 0 | .042444 / .055575 | .044354 |
| U-final2000，原16 | B0 | 0 | .013208 / .020167 | .019177 |
| U-final2000，原16 | B1 | 0 | .012841 / .019872 | .022006 |
| M64-final5000，全64 | B0 | 1 | .022735 / .076386 | .110892 |
| M64-final5000，全64 | B1 | 0 | .023066 / .078875 | .111746 |
| M64-final5000，原16 | B0 | 0 | .016699 / .039546 | .114383 |
| M64-final5000，原16 | B1 | 0 | .019702 / .039652 | .111746 |

U两组PASS；M64的全64与old16、两noise共四组均FAIL。全64位置median从起点.09103降至.02273m，说明扩量仍在学习；旧16姿态却在1000步已由.01918升至.09184rad，早于后续恢复训练，不能用resume/finalizer解释。

唯一M64 REJECT是 `pi0227_persistent_train_c_j02_xm015_s04_ym050_r03_r03_227419:obs-0033` 的B0，线速度2.0884175759101504m/s。它不能解释全体姿态与位置尾部失败，也不能通过换noise或删例解决。

## 3. 当前归因与待检验假设

A. U原16过门支持完整生成终点监督在所测条件有效；不是“GR00T始终无法拟合”。

B. old16保持实测退化，但旧16仍在64训练集里。应称retention regression/训练内干扰，不能说完全没有回放，也不能未经梯度测量断言唯一原因是梯度冲突。

C. 源码 `_physical_losses` 只处理generated[..., :3]：自由生成辅助项监督XYZ位置与增量，旋转和gripper仍由原native flow loss监督，但没有同类生成终点旋转项。R检验这一目标覆盖不对称是否影响扩量。

D. 暴露量可能影响新48尾部：M16阶段每窗125次，M64每窗78/79次；起点和难度不同，该比值不能单独证明预算不足。C提供额外训练量的同期对照。

E. 本次仍是同一视觉到达指令下的状态/目标支持扩展，64均进入训练。不是已经进行多技能泛化，也没有本轮WBC追踪失败证据。

不同时改变旧/新采样比例、学习率、骨干冻结范围或归一化。原16仍获得均匀训练；是否需要更高回放权重属于后续另一假设。

## 4. G0：复用已有artifact的窄分解与准备

零新模型查询，直接读取封存0/1000/2500/5000 records、final128 cells及真实训练日志。

输出一个 `pi0248_m64_breakdown.json`：
- 以完整source_episode_id拆old16/new48；各B0/B1、A-D、phase报告位置median/p95/max、SO3、gate及最差窗口。61个rollout-base与64个window不是64独立任务。
- old16相对原U结果作同输入/noise比较；身份不匹配明确披露，不重新抽样。
- 沿H40看姿态角误差、位置bias/尾部SSE、rot6d范数/共线风险、inverse-normalization裁剪及专家目标分布。旧R2第37段统计不能直接代替当前M64。
- 核对old16标签/frame/processor在扩量前后一致、actual64曝光表与已发布计数一致。已独立复核的完整性证据直接复用，不再做整库审计。
- 原loss/mode日志足够则分析；未保存的逐维native rotation loss或梯度冲突记UNOBSERVED，不新做真实模型诊断补历史。

只要没有真实语义冲突，G0不必确定唯一根因即可进入训练。日志缺项仅影响解释时如实记录；真正标签/decoder冲突则停受影响实验，不更改旧数据或验收标准。

## 5. 精确的新训练目标

保留原U：

`J_U = L_native + .1*ell_xyz(a_free) + .1*ell_delta_xyz(a_free)`。

XYZ按原.05m、增量按原.05s及1m/s归一化，原mask与所有10个有效维度的native loss不变。

- C：`J_C = J_U`。
- R：`J_R = J_U + .1*ell_rot(a_free)`。

令D_R表示原processor.decode_action以及当前native row-rot6d→SO3路径的可微镜像，包含实际反归一化、clip/zero-range、相对旋转组合及state anchor。定义Rhat_k=D_R(a_free)_k，Rstar_k=D_R(a_star)_k，m_k为原rotation slots3:9全部有效的mask，theta_ref=.05rad：

`ell_rot = [sum_k m_k * ||Rhat_k-Rstar_k||_F^2/(2*theta_ref^2)] / sum_k m_k`。

合法旋转相对角theta满足：

`||Rhat-Rstar||_F^2/(2*theta_ref^2) = 4*sin(theta/2)^2/theta_ref^2`。

它是小角度下近似角度平方的chordal surrogate，不是6个归一化数字的MSE。训练选择该形式避免新增acos端点导数问题；验收仍使用原geodesic SO3 RMSE。lambda_rot=.1和theta_ref=.05是本轮预注册值，不扫系数、不在线修改。

使用同一次原native forward和同一次完整S4纯噪声生成结果；姿态项不增加第二次rollout或随机抽样。四步全梯度保留。专家仅用于监督loss，不进入生成条件/前缀/RTC。不要增加旋转速度loss、teacher蒸馏、XYZ4或padding干预。

### CPU增量验证与真实首更新

复用已有U实现，只增加rotation decoder/loss和stage参数。

CPU验证D_R与原CPU decoder/scorer的旋转矩阵一致，测试非平凡anchor、行/列差异、各horizon实际参数与clip规则；有效矩阵元素max abs≤1e-5。测试预测等于专家时loss≈0、已知角度恒等式、mask、lambda_rot=0原值/梯度一致及非奇异点有限差分。

`rot6d_rows_to_matrix`与同文件列约定函数不是同一转换。近零/共线输入遵循原合法域检查，不能新增identity回退、任意epsilon或直通梯度。原部署decoder和scorer不变。

第一正式R更新在预算内记录全链有效梯度、mode恢复、同noise及loss分项。日志来自已有tensor，不增加forward/backward。每步记录native/XYZ/delta/rot/total、有效数及饱和/正交化状态。没有额外optimizer smoke。

已有同语义S4 parity可复用；如新入口确需模型parity，整个PI0248最多4个非选分完整S4 rollouts，单独记账。CPU数学验证不算项目模型查询。

## 6. C/R共同配置与固定起点

起点固定为PI0247 M64-final5000：
- repo `leoyuer/pi0246-gate-pass-campaign`
- revision `093c450fd49052c8fd3285bf38111a30a1f59009`
- prefix `arms/PI0247/r2-qualified-U-M64-20260926/recovery/633819.hopper-m-02/checkpoint-5000`
- tree SHA256 `1811bfa5c29fb7ce2e911ade0a0b476607e32ac1c1322cce68b8dff04449d776`
- full bytes 20,883,060,977。

C family `PI0248_C_U_CONTINUATION_M64_2K_V1`；R family `PI0248_R_ROTATION_ENDPOINT_M64_2K_V1`。

两臂各2000 updates、新AdamW/cosine2000/warmup10；LR1e-5、WD1e-5、betas .9/.999、epsilon1e-8、maxgrad1、batch1/accum1/workers0、seed22301。原实际trainable/frozen、参数精度、autocast与U重计算策略不变。

原64按既有(target_family,phase,source_episode_id)排序，以独立PCG64(22301)逐轮全排列拼接截到2000；16个窗口32次、48个窗口31次，具体ID由表决定，不能特意把32次分给old16。两臂使用同一表，不叠加shuffle。原observation index0、H40、dt=.05、S4、132维/active0:10、statistics和两路RGB不变。

C→R串行，均完成。R不从C终点开始。两臂各保存1000/2000完整Trainer状态。新增rotation项不得消费额外随机流；保存实际样本/noise身份，不仅依赖seed作配对声明。

新优化器与新的cosine时程属于本轮共同设计；不称旧M64 exact resume。旧U-M16合格权重及旧M64失败family保持不变。

## 7. 固定screen、保持判定与停止

旧M64-final5000的128 cells是共同起点，按身份复用。只新增：

`C1000/C2000/R1000/R2000 × 原64 × 原M64 B0/B1 =512 cells`。

使用原实际packet，不重造B0/B1。Corelab逐文件验证推理视图，完整独立CPU复算；合法REJECT/高误差不重试，checkpoint1000仅观察，不替代final2000。C结果不改变R配方。

每臂final2000分别验收全64 B0、全64 B1、old16 B0、old16 B1四组：
- 原窗口及40点完整、数值有效；
- 原SafetyGate REJECT=0，WARN保留；
- median per-window XYZ RMSE≤.020m；
- NumPy linear p95≤.035m；
- median per-window geodesic SO3 RMSE≤.050rad。

new48/A-D/phase只作分层，不新增隐藏硬门。两noise不池化，不当独立数据集。主报告比较C/R相对共同起点及R−C，说明姿态修复是否牺牲位置或速度。单训练seed不证明普遍显著性。

两臂有效完成后：仅一臂过门，记录该fit-recipe候选；两臂都过门，预先优先较简单C；两臂均FAIL，`PI0248_M64_REPAIR_HYPOTHESIS_EXHAUSTED`。不追加步数、采样比例、loss或solver。

通过终态：`PI0248_M64_REPAIR_QUALIFIED — STOP_FOR_PI_FULL_ADMISSION`。交付weights/recipe/runtime/原64和old16两组证据及FULL准入建议。selected_checkpoint仍NONE。下一阶段科学路线明确，但不把本轮未知结果预写成selected或sim成功。

## 8. 工程执行和交付

PC agent可自主修复保持科学语义的路径、导入、保存、恢复、归档和报告问题，做针对性回归后继续；没有任意次数上限。健康staged代码不改，旧card/artifact不覆盖。新增配方字段改变才是新科学决策，不能借工程修复改seed/LR/门。

本地PC控制与代码；Corelab做CPU/可容纳screen；四步可微训练使用已验证大显存路径，PBS auto合法调度，不绑定特定hostname，不重复已知4080S完整训练OOM。不再开启邮件/monitor或扩大home/scratch清理。

沿成功的持久保存与HF固定revision回读链；源和推理视图分开。真实路径、容量与owner接手复查，不能把整理完成当quota保证。保留旧U合格模型、M64失败模型和唯一恢复源。本文件不新增删除授权。

stage内恢复完整optimizer/scheduler/RNG/消费cursor，不能新optimizer冒充resume。完成训练后仅报告/上传失败，不重训。工程失败实际更新独立计入raw，合法负结果永久保留。

本轮新增canonical更新最多4000；新增scored queries最多512，额外非选分parity最多4。旧PI0247 canonical9000/raw10000、旧PI0246及全部历史计数不充值。rollout/head/recompute/optimizer分别记录。

最少交付：G0分解、rotation镜像与梯度测试、两sealed cards、实际训练日志与完整checkpoint收据、独立512-cell screen、四组最终判定和新fit-recipe或明确FAIL。Issue #11追加canonical，PR #77只pointer。没有FULL/CALIB/WBC/sim/DEV/F/DAgger/硬件/merge执行授权。

## 9. 启动正文

执行 `PI0248_M64_POSE_REPAIR_GO`。完整读取本文件和四条canonical。接受旧M64有效FAIL，同时保留U-M16两noise已PASS的里程碑。

先复用旧artifact拆old16/new48与姿态/位置尾部，不重跑旧实验。验证同官方decoder的可微rotation镜像及CPU梯度后，自动运行C/R各2000，均从固定M64-final5000开始，新optimizer、相同64/2000表、LR1e-5/S4。C=原U；R=原U+.1旋转终点弦距，按第5节单位和分母。

同一次完整纯噪声S4生成、四步全梯度；不额外采noise、不改old/new权重、不把专家当condition。C/R各完整1000/2000，原baseline128复用，新512cells使用原B0/B1。只final2000，全64/old16两noise四组原门；两臂都完成后判定，均过优先简单C。

工程最小修复和exact恢复自主，科学预算不变。报告/HF失败不重训，不开启邮件或monitor，不再做大清理。成功封存M64-repair候选并提交FULL准入；失败停止新假设。不提前选模或sim，selected_checkpoint=NONE，PR不merge。
