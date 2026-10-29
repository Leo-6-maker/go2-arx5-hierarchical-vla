# E2提交前退出的冷态接续

本轮实际原生回读仍为旧后继owner OS1、两个记录进程已退出、GLOBAL `7eb5802e`、used PBS41、所有已登记PBS终态，pending/unknown为空。后继02未登记native请求，无qsub intent/reply，但自身预备lease仍ACTIVE；不能称健康后台链、GPU作业或新训练。

旧保存错误是物理存储全扫描时，一个CPython原子写缓存临时文件在枚举与lstat之间消失。`minimal_retry_physical_guard`仅给当前E2输出目录增加显式选项：该精确`__pycache__/*.cpython-*.pyc.<数字>`缺失时，重新枚举并检查**整个目录一次**。不会跳过该文件而接受半份清单；第二次异常、普通文件缺失、旧历史扫描、symlink/owner/hardlink/空间问题仍立即失败。控制/临时空间reserve继续计入。

新接续不重启旧owner或改其sealed源码。先用原onceIO关闭其own lease，要求原固定manifest/source、原OS1/stop、当前内核进程终态、无登记行/提交intent和无模型原件。随后独立回读close intent/receipt和完整保留目录清单。旧02及其控制费用永久保留，不向PBS账补造一次提交。

`minimal_e2only_pre_native`只在原GLOBAL追加明确的“未登记预备已关闭”见证，停止重复预测其未用native hold；原已付控制、所有旧行/费用不变。允许唯一`attempt2/cold01`新角色，仍是原始父fresh512和原3个释放相位，不消耗新的科学臂或改变时钟。新角色必须把旧02完整目录、闭合lease和清单加入原144294967296字节gross pool；无删除/去重credit。

现有元数据CPU bank接近用尽，不能先做新预检再记费用。新控制通过同一GLOBAL先付3600site CPU/600 PC CPU，并用完整14维核算保留原matrix/final和唯一未来native hold；这沿旧实验原账，不使用夜间新增源域额度。准备付款只绑定源SHA、身份和准备计划，明确没有最终manifest和GPU许可；冷态预检真实通过后，才一次绑定实际manifest/输入审计，不重复付费。原Journal/有限计数器累加新付资源及旧费用，不清零。

检查：第一次45项通过、2项因选错本地解释器缺Torch而失败；换用现有CI环境后相关10项通过，合计48个不同检查通过。新付款/forecast检查使用真实旧GLOBAL和闭合原件，全部新请求仍为CPU fixture。两次本地回归分别保留240/220秒上界，原付费上限不变；首次未提前单列reserve的记账偏差已明确追加，未隐藏或返还金融费用。原有限管理程序编译通过。

此发布仍是冷代码与实际旧状态核实。旧lease关闭、GLOBAL退休/新控制采用、新角色预检和新PBS均需真实回执；不声称已经完成。新GR00T结果0，E2完整512未达成。若同一故障在一次修复后的接续再出现，依夜间授权停该分支，转其他独立工作。NFS/挂载/SSH/服务/权限/凭据修改0。

实际增量：[旧未提交预备已关闭并保留](../../../reports/PI0294/design/parallel-science-20261022/ARX5_E2_UNSUBMITTED_OWNER_CLOSED_RETAINED_ACTUAL_20261030.json)。own lease SHA800484fa、完整目录92,315,648字节，独立close intent/receipt与snapshot成立；原GLOBAL一次追加未派发关闭见证，SHA8438e681。一个更早stage因缺parent在首个文件效应前退出；其精确目录已实际确认不存在，旧Journal/完整费用和原计数器归档保留，未重发。新角色沿已付原输入资格复用，仅重核实际字节/物理gross/全14维；全代码与关键新依赖须封存。新控制付款、cold角色和GPU尚未启动。
