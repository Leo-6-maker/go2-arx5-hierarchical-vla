# Codex启动prompt：SCALE20K多源本体适配主线

用户明确提出当前500步小样本训练可能不足以适配Go2+ARX5，需要综合多个数据集大规模后训练。按[新规模计划](../../protocols/SLA_SCALE20K_DIRECTION_20261119.md)及同名JSON推进，把SCALE-MIX20K作为下一研究主线。此prompt不热改现役作业，不将文档发布冒充资源授权。

## 立即可做的准备

- 读取PR417最新实际状态；保留其绑定的cf9d1acb计划、D4人口、D5-W/T和J0/J1固定500步作业。它们继续提供诊断，不是SCALE启动的高成功前置。
- 从当前可信执行栈建立冷态独立工作树，保留并发改动。检查已有SOURCE_NATIVE/H40/H50模块，但不通过改旧500/1000硬编码来冒充新scale合同。
- 准备C20+逐条合格历史PLACE的新目标TRAIN；扩展RoboDojo-Real TRAIN并接入LIBERO TRAIN。保留旧DEV/TEST，按episode/task/scene/lineage去重。源建设目标1000–5000独立episode、≥20任务；目标并行扩100–300，不等目标数量先齐。已审ARX X5合同只适用于其子集，新增RoboDojo本体/采集版本独立核对时钟与动作/状态关系，不统一假设30Hz。
- 数据各有native语义、时钟、normalizer和I/O，不能为混合方便把source未来记录参考改名为Go2命令。至少两公共源真正接入；如仅已有36源+20目标，明确仍是小数据技术前缀，不虚报规模。
- 各数据版本、访问银行、模型初始化、参数清单和20k LR在首更新前冻结。新数据只在明确checkpoint开新curriculum版本，不悄悄污染规模曲线。

## 实现新训练身份

共同初始化从已验证SLA_base开始，不默认继承B500或旧optimizer。两臂共享/目标初态一致。视觉编码器首版冻结、VLM LoRA、完整action expert和各域I/O可训练；WBC/腿控/硬门不变，不增加BASE head。

主线20000更新、有效batch128=32目标+48RoboDojo+48LIBERO；目标-only对照5000、batch32。目标loss两臂均原G7损失+1.0物理宽度项（D5-W定义，无delta），混源loss为目标均值+1.5源R均值+1.5源L均值。一份合并梯度、一次optimizer update。两臂前5000步目标manifest及版本完全一致，不只给混源臂换目标数据。共享/各域参数清单互斥，未使用域I/O与momentum/AdamW decay均不更新，不能只置零梯度。microbatch按真实显存确定，gradient accumulation不改变有效batch。

实现完整模型/Adam/RNG/global LR step/各域cursor恢复、新包schema及推理加载。公共冻结资产尽量共享，避免无意复制整套模型。参数更新mask按domain验证，不能把旧J只更新shared20的限制当作本版完整action expert训练。

技术200更新计入20k同一进程。真实batch的梯度、标签、恢复与有限站点资源成立即可开始，不要求Teacher/D4/D5/96全过/N2通过。使用一个20k学习率表，TGT5K复用其前5000项；每次allocation结束按原额度保存退出，后续只在新的实际授权资源上恢复，不重置Adam。

## 检查点与评价

MIX保存500/2k/5k/10k/20k，TGT保存500/2k/5k；离线跨域指标与恢复按JSON计数。5k成功率低本身不使训练停止，检查数值、拟合趋势、输出和数据/参数问题。不能无限重复同配方而不看学习曲线。

在新DEV24上评价TGT5K、MIX5K和MIX20K，DEV一直叫DEV；20k再新确认24。Core评价与训练可在独立资源并行，不阻塞健康优化。旧TEST保持封存。新scope逐查询仍原完整H40、真实前命令速度门、P10和2ms评分，STOP后0积分。新包接缝至多预留原两接口。

## 资源与交付

新计划上限25000更新、800000T、1920000S、100000训练完整生成、2760离线生成、98原生请求/4399394tick/15360原生模型生成；40k、第三公共源、真机、旧TEST均不自动启动。

先测真实prefix的秒/更新、峰值显存、data-wait和checkpoint耗时，提交具体GPU小时与有限allocation规格。旧D4约10.89s/update只能作旧配方参考，不能据此承诺大batch/全action expert耗时。不取消或抢占现役健康任务，不挪用旧M2预算。

每阶段发布真实ACTUAL：独立episode/任务/场景数、逐域实际访问、更新/完整生成/资源/恢复、曲线与闭环分母。数据或资源暂未具备时，把能完成的工程与manifest准备做好，说明具体缺口；不要再次要求用户批准“是否允许不等Teacher训练”这一已经明确的方向。
