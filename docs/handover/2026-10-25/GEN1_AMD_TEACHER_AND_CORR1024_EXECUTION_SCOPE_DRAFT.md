# GEN1：AMD Teacher → Vanda 独立微调执行包（待上限采用）

2026-10-27，SGT。用户已要求延期Eval、生产多场景数据后微调，并直接选择共同CORR2048独立开始。本包落实该方向；下列新上限尚未批准/采用，不据此启动原生采集或训练。

Teacher不必等待4080 GPU。AMD实际有16逻辑CPU、约32.5GiB空闲内存，根盘约269GiB和共享Home约1.37TiB可用；Apptainer可用，但现有管理Python无MuJoCo，4080的node-local运行环境不存在于AMD。应复用已封存SIF/26source资产，注册真实AMD CPU角色；不修改管理环境或启动WBC。当前新构造/积分/数据均0。

执行规模固定为六个既有合法V2 TRAIN几何，V0/V1双相机24图；每变体重复T1并独立构造T2作reset/RGB/标定核验，加三个home有限初态检查和原counter，共最多44次静态构造。共同home不用于这轮训练。随后只对六V0几何各T1/T2一次，共12个首次Teacher尝试；合法失败保留，不重抽补成功。先只研究布局；暖色桌面仅预览，未伪称已作为新训练数据。

AMD复用原ARX5场景、Teacher、S1、完整32秒任务评分和raw audit；最多2个隔离worker，各任务3600CPU秒/32000live+shadow ticks/81constructor/768MiB。真正的安全故障或未知效果停止包。成功raw送到Vanda，一份CPU-only PBS使用现有官方processor/R0逐窗口审计和export，随后封存数据银行。每个任务的四个布局层均有至少一个真实成功、raw/codec审计合格的支持，才自动接一份A40训练；否则报告支持缺口并停止训练，不造标签。

训练为一个新GEN1_LAYOUT_V0_CORR1024分支：共同CORR2048、fresh optimizer、1024更新，沿原C50优化器与专家lane比例（新expert0.5/全分支0.375），用新GEN银行替换该lane。C1/旧任务replay、R0、VLM冻结、loss、H40/Prefix8、4步推理和科学评分不变。不接续未Eval的C50/D25，不选“最好”checkpoint，不重复已闭合训练。数据adapter/新host runtime尚未部署，CPU门不是native许可。

真实账本字节上的完整复算保留C/D两个原Eval和新端点的一份原24格CoreLab Eval、原matrix3和最终20GPUh/40GiB。原C未来IO按全额再计，不延用已消费旧request信用；AMD raw/export保留两份完整上界，不预设hardlink免费；Vanda银行另计10GiB。旧失败、所有checkpoint和metadata不删除、不退款。当前Eval继续延期，不能把尚无行为结果称为改善/泛化。

一次需要四项上限调整：global PBS34→36（正式export+train，另有一个严格零效果工程备用）；GPU站点输出1021→1176GiB；Core输出393→477GiB；总传输300→332GiB。GPU121h、site/Core/LOCAL CPU、更新16384、查询/物理上限不变。全未来GPU421663/435600秒、更新14786/16384，CPU也在旧cap内；所有逐维算术见机读草案。备用不是必须消费的额度，不允许重放有效Teacher失败或未知写。

沿原T_CD1792932749.8133287、stop-new-training1793451149.8133287、deadline1793537549.8133287，不重置。到期限仍不可Eval则记录NOT_RUN并保留已审计端点，不延时、不标科学目标完成。批准/实际采用后使用原entry/registry/owner/Journal/onceIO完成冷部署和fresh准入，普通工程故障在本包边界自主修复，不逐次问GO。唯一mutation owner、无新增agents/helpers、无新调度器/预算体系，凭据与私有资产不入Git。

[完整机读执行与逐维复算](../../../reports/PI0294/design/parallel-science-20261022/GEN1_AMD_TEACHER_CORR1024_CONDITIONAL_EXECUTION_AND_SCOPE_DRAFT_20261027.json)。
