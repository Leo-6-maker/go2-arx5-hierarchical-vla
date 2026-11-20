# Codex 接续：恢复MIX、完成D2课程发放、取得学生闭环

按[2026-11-20审核与调整](../../protocols/SLA_LONGRUN_ADJUSTMENT_20261120.md)和[机读清单](../../protocols/SLA_LONGRUN_ADJUSTMENT_20261120.json)执行。原[长程配方/预算](../../protocols/SLA_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1_20261119.md)及现役固定SCALE合同仍有效。用户授权服务器后台排队自主接续，不需要实时Codex轮询，Eval可独立并行。

## 目标与当前证据

任务是完成实际接续，不只停留在准备文档。先在有合法访问权限的执行环境做一次增量live核对，再把可执行的有限后继交给Core用户服务/VANDA PBS。不要把此prompt中的历史状态当作当前机器状态，不要因为某一站阻塞而停止另一站健康训练。

用户11月20日11:40 SGT报告：TGT2000和3500已封存，1593720从3500继续至5000，最近4324；TGT2000 Eval1593719完成288+48、48/48独立恢复；ADAPT750封存、至1000后继运行；MIX869训练侧保存但外层被oomd终止；两条新HOLD专家成功，共46156tick、139H40/1390P10并已到VANDA。两条私有回执basename为CORE_TARGET_HOLD_PAIR_EVIDENCE05.private.json、VANDA_ADAPT_D2_HOLD_CUSTODY01.private.json。本次ChatGPT审计没有读到私有内容，因此先回读并去敏公开摘要，不能写“已由ChatGPT核验服务器”。

公开报告head3670aaca与实际runtime不同：ADAPT训练worker1432c8dc88c57adf39e8fa363f5438c5e63aa8fd，handoff2d81436be0e4dc02bf67a180ea1cf00e818e29df，ADAPT0部署f0d3bc1a53bc5066d3590e6022ef2fc6c0cb9116。报告head缺部分核心代码，沿现场清单核对全部依赖，不盲checkout最新报告分支来启动，也不修改活动胶囊。

## 1. 保留健康队列，先对账再注册

读取TGT1593720/1593721、ADAPT当前后继及1594084/1594215的真实JobID/状态/owner/端点，不重投已排队或已完成的任务。TGT实际afterany+worker seal gate已经部署，部分ADAPT边afterok；不要为了改成同一种依赖而动健康作业。独立Eval只读封存checkpoint并用独立GPU/主存；无空闲卡就排队，不阻塞TrainNext。TGT3500不新增正式Eval，2000已有结果不重做。

## 2. MIX869异常恢复与主存修复

- 回读oomd事件、原服务/cgroup/ancestor、trainer RESULT/COMPLETE/UPDATES、STOP、owner和原worker清单，确认旧进程确已消失。旧07:14:32窗口已过，申请/绑定真实新有限资源，不延长旧卡。
- 原外层RESULT缺失时，不补造exit0。新增独立CRASH_RECONCILIATION，保留外层失败，核验140数组、模型/Adam/RNG、完整账本、LR/bank/cursor、实际未完成尾段消耗。
- 869完整且无更高不可恢复尾段时从870接续。若只剩较早完整端点，显式登记恢复分支和尾段损失，不称无损续训。所有恢复资格与尾段重放的实际更新/访问/生成计入对应原账，先扣剩余额度，不借其他谱系预算；若不足到原终点，封存可达前缀并明确未完成，后续版本再列必要增量，不自动超限。
- 修改最小用户态冷态编排分支，不改原训练recipe；验证坏数组/短账本/活owner/STOP/过期等实际风险。新有限后继要有真实unit/JobID、独立连接验证、870后的实际更新与下一seal。
- 查主存峰值及并发负载，不能把oomd直接叫CUDA OOM，不关闭主机保护、不删数据、不默认改microbatch4。Core主存不足时用同recipe、同完整状态和唯一owner做已合格A40冷迁移。恢复不依赖Eval分数。

## 3. 现在准备D2和2001–5000，不等2k才写后继

- D1已有旧23条+1新PLACE。若只加入新HOLD2条，则D2为26条/2031窗，E_new=3，新目标12、旧目标52，源32/32。episode与窗口分开。
- 扩展CPU预检，按每个新增episode的首/闭合前/P10到期闭合/末窗实读；原四点选择会漏中间episode。全139窗核标签、时钟、mask、P10接续、前命令、split。两站生产reader与三域bank hash相同后才标D2_READY；14文件SHA不能替代读取。
- 固定D1 state统计、C20动作codec/统计、R48/LIBERO统计、MIX500亲本、Adam/RNG/LR不变。现役worker不改，只提前实现阶段2编排、真实有限PBS依赖和Eval5000。
- phase2000封存后一遍决策：D2_READY则2001切D2；未就绪则D1成员进入第二阶段bank继续；若已做2001以后D1更新，不回滚热切。D2准备和2kEval都不变成训练成绩门。
- 未来parent SHA只能在实际seal后绑定。既有至2000作业保持，第二阶段后继按实测p90/save余量有限分段直到5000；每段沿实际s+1，不把排队当已更新。

## 4. 修复部署证据并立即排原学生面板

- 原ADAPT0跨卡严格0/16结果保留。先查是否有原数值数组；当前公开runner只保存比较布尔值，CPU注入同预测的解码差不是GPU证据。
- 按新计划第7节，在生成缺失差值前封存DEPLOY_COMPAT_V2：同权重/输入/噪声，XYZ≤1mm、SO3≤0.01rad、宽度≤0.5mm、crossing≤50ms；原guard_signature、crossing有无和P10内外身份一致，同Core独立恢复逐位一致；所有条件合取。不得按新结果改阈值，不改变H40/接缝/STOP/评分。
- 缺数组时最多使用现有336技术储备中的32生成（先扣已耗），保存原16固定输入输出及同Core独立恢复。旧16仍入原账，不重复计为新资格未执行。此Core双进程资格只对固定backend/worker/dtype/环境做一次；A2k/A5k核各自完整状态及A40恢复、各用原剩余16次比输出，并仅在上述环境完全相同时复用资格。环境变化先登记有界重新资格与真实剩余预算。
- 通过新部署资格后，按冻结SHA cdcd114a712739315d06ee923fe8221142d911ab6b60a64f30eac1b619eec2e0的DEV24原顺序派发ADAPT0，前6只是前缀，随后续18。不能按成功与否重选条件或端点。ADAPT2k/5k各24、固定5k另确认24保持，总96。
- TGT5k按旧SCALE人口/预算另行评价。原生只有一个合格world；推理与训练并行还须真实GPU和主存余量。已有两批专家采集已计入公平性规则，A0一旦就绪，下一个冷态world块先给原DEV前6，不从本版重新累计两批采集。

## 5. 数据采集与后续诊断

在原240尝试内准备最多12条新TRAIN（4HOLD优先F1蓝/F2红，6PLACE补缺失格并验证已修复的预抓取/携物/释放路径，2恢复仅在生产者真实可用时派发）。按原单次50703tick/180专家计划封存场景、seed、producer与顺序。未准备条目可保持未派发，不为凑数复制片段。两次同cell/producer同首因后，先改有证据的原因，不原样重试。

现役88窗阶段面板保持，汇报24个P10到期闭合正例的及时闭合、H40/接缝和联合通过；native另报实际消费、接近、接触、抓持、抬升、保持、PLACE与STOP后tick。D2新数据不替换旧评价题目。专家成功、学生结果、真机结果分开，硬件操作仍0。

## 6. 收尾与成果

只在REGISTERED、正式seal、Eval终态、异常或谱系终态发布简洁ACTUAL，服务器心跳/日志保留，不挂实时Codex/SSH轮询或新ChatGPT定时任务。使用既有owner/onceIO，未知提交结果先只读对账，不重投。不以更多检查/receipt代替真实更新和学生native。

下一次交付至少明确：MIX实际恢复到哪步；D2每episode读取是否通过；ADAPT是否真实排至5000、是否已到2000并切换；TGT5k是否完整封存；部署V2结果与原学生前6/24真实分母；采集新准入、失败与剩余额度。仍阻塞项给具体证据和下一可执行修复，其他任务继续。

原SCALE及ADAPT预算均不加：原SCALE25000更新；新增长程5120更新、655360访问、38390生成、362请求/16824986tick。历史D4 E0两接口11906tick是carried证据，不每批计新。保持WBC、H40/P10/50ms/2ms、STOP、评分、历史失败和旧TEST。正常工程在已授权范围内自主处理，真实资源/凭据/所有权限制如实记录。
