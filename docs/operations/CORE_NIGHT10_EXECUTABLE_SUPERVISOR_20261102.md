# CoreLab 10小时后台科研接续：可运行代码与实际部署边界

**状态：独立本地Python/CPU测试完成，CoreLab原生运行/驻留资格仍未取得。** 本协议是现有Goal V7 + Issue #299 的一个**薄任务依赖执行器**，不替换源码中的模型、Teacher、任务评价器、现有科学owner、canonical账本、onceIO或安全检查。源程序 `scripts/pi0294/night10/supervisor.py`，候选选择 `policy_review.py`，user级一次性挂载入口 `launch_core.sh`。只有当前获授权、真实接入CoreLab的Codex执行owner可以绑定和启动。

## 当前实际边界和科学目标

- 科学基线：Phase4096自主放书首次成功1/4有效、6未运行，所有原DEV记录与STOP保留；不能将0/1失败模型的其他未运行条件伪造为0/10。
- 截至2026-11-02 01:09 SGT的GitHub Issue #47原生回执：新增PORT08柜门4/4成功（合计5条完整柜门Teacher）；8次TRAIN持夹恢复7条合格、1条无事件（无专家标签），7条正在逐条导出；后继manager07活跃。其余6条新BOOK几何尚未执行。不要复跑已执行种子。
- GPU0 32760 MiB，主要承担当前单一源物理/双RGB和后继合格训练。GPU1 16376 MiB，已有10步训练和42项恢复证明，但最低只剩611123200字节<1GiB，**正式训练未准入**；可在PR #305定义的独立推理资格通过后运行不带优化器的saved-TRAIN推理。GPU1不是自动获得训练许可的卡。
- SSD此前现场余量约313GiB但实时变化；每项启动前原生检测至少30GiB余量+当前任务预计新增峰值。CPU能并行进行数据登记、CPU静态几何、质量审计、官网Processor、旧记录导出与统计/数据冻结；**没有已验证的CPU无GPU双RGB渲染路径时，CPU静态轨迹不是完整训练示范。**
- 夜间10小时是**最多36000秒的新工作派发窗口**；单项科学任务独立有限预算。若出现安全/UNKNOWN effect，停止相关物理/写入路径，不为了用满时长重复已运行条件。

## 首次部署由Codex执行（具体命令需真实绑定）

1. 在Core确认 `source manager07`、旧source用户级unit、canonical lease和全部已开始的pilot真实终态；**不得因为一个旧manager的字符串解析错误就重投已有实测成功的柜门或恢复episode**。旧链活跃期间只允许低负载CPU静态计算/资格，不抢GPU0物理owner。
2. 在现场从真实owner manifest生成**单独的BOUND_PLAN.json**，包含每个现成原生训练、导出、DEV、静态准备CLI的绝对exe、cwd、独立输出/receipt和科学身份SHA。不要把不可用的模块名猜成可执行CLI，更不可以用一段`python -c`生成假科学PASS。
3. 配置 `lane=gate` 的 `readiness` 指向**真实原owner的CLOSED回执**。`gpu_train` 的 `receipt` 必须是实际完整checkpoint/全态审计；`physics` 的 `readiness` 绑定旧actor终态，`preflight` 必须调用已验证的原始native独占owner/冷态安全检查。**`preflight`只读，不是解除原STOP的工具。** `gpu_inference` 仅限通过现场资格的数据与GPU；`gpu1_training_admitted`默认false，改true必须附上新原生训练准入receipt。
4. 当前确认的 `D0`=原15条BOOK+6条专家后缀；`D1`=D0+新增合格完整BOOK示范；`D2`=D0+7条中真正完成导出和官方资格的晚期专家后缀。若D1新BOOK仍0，禁止启动空D1；D2不能包含学生接管前缀。三个训练臂应固定同一b248父模型、处理器/统计、优化量和参数白名单；不同臂不互相接续失败权重。
5. 分支任务依赖推荐：

   | 任务 | 资源与依赖 | 失败后允许什么 |
   |---|---|---|
   | `wait_source` | CPU read-only gate，等待旧manager07闭态 | 未就绪不启动后继物理/数据冻结；已有CPU静态任务可继续 |
   | `cpu_scene_prep`, `cpu_static_audit` | 互不共享核心的CPU槽；可与GPU独立执行 | 不阻断已验证的旧数据基线 |
   | `cpu_export` | 等待原native source closed，再通过官方数据导出资格 | 如果失败，`D1/D2`不可准入；独立D0仍可通过本身的既存数据来源 |
   | `freeze_D0/D1/D2` | CPU实际原TRAIN-only统计、RGB/EEF10 SHA、数据谱系；各臂独立输出 | 缺失的新样本只挡对应扩充臂，不挡其他臂 |
   | `train_D0/D1/D2` | GPU0串行，若GPU1真满足1GiB额外余量且有正式证书才可能并行 | `train_D1`失败仅跳过`eval_D1`，D0/D2仍继续 |
   | `eval_D0/D1/D2` | 唯一 `lane=physics`，各自独立冷态native评分 | 单一DEV安全STOP使本面板截断，但不替代其他模型冷启动资格；其他可独立执行 |
   | `policy_review` | CPU软依赖：等待各评价终态，无论成功/失败 | 只给出开发候选，不打开确认TEST、不宣称WBC成功 |
   | `shared_pilot` | 仅四任务共同数据、DOOR≥5、同父零更新基线和新独立配方都合格时 | 未准入则保留未运行，不制造四任务模型 |

   对独立结束后仍需继续运行的任务，用 `after`（**soft dependency**）；只有同一模型的合法checkpoint→对应评价才用 `needs`（**hard dependency**）。政策上中间模型失败不能全局中止；本地安全未知效果仍必须阻断相冲突的物理/写入。
6. 先验证配置但不执行：

   ```bash
   python -m scripts.pi0294.night10.supervisor check --plan /ABS/BOUND_PLAN.json
   python -m unittest discover -s tests -p test_core_night10.py -v
   ```

   最终使用现有已启用linger的**用户级临时unit**（不修改系统服务配置）：

   ```bash
   bash scripts/pi0294/night10/launch_core.sh \
     /ABS/BOUND_PLAN.json /ABS/NEW_NODE_LOCAL_RUN_ROOT core-night10-20261102-01
   systemctl --user show core-night10-20261102-01.service -p MainPID -p InvocationID -p ActiveState
   python -m scripts.pi0294.night10.supervisor status --output /ABS/NEW_NODE_LOCAL_RUN_ROOT
   ```

   这只是**挂载命令**；Codex须在两个分离控制连接后核验同一PID/proc_start、STATE观察时刻前进、第一项native任务真实进程及lease/回执才称READY_FOR_DISCONNECT。GitHub的代码与CPU测试不足以代替该证明。
7. `supervisor`不读取或修改历史评分；`COMPLETE`仅表示原始命令退出0且带类型的native receipt验证通过，不等于task成功。`FAILED`记录真实模型/业务失败；`SKIPPED`是依赖或准入缺失；`UNRUN`是未在10小时启动窗口内获得完整资格；`UNKNOWN`必须独立核账，不自动重试。失败输出和成本保持原样。
8. `policy_review`只读原始DEV评分，门槛为固定同场景开发人口至少4项有效、至少2项完整成功、硬安全0；未达标就输出无合格policy，而不是伪造。若有开发候选，下一步仍需独立未暴露确认。使用一份真实预冻结的候选JSON卡调用 `python -m scripts.pi0294.night10.policy_review --card /ABS/POLICY_REVIEW.json --output /ABS/DEV_POLICY_REVIEW.json`。固定DEV不再假称盲测。

## 调度器JSON格式

`duration_s`固定36000，`max_cpu_parallel`1–8，`disk_margin_gib`≥30，`gpu1_training_admitted`默认false，`tasks`1–40个有向无环节点。task字段：`id`、`lane`（gate/cpu/gpu_train/gpu_inference/physics）、`after`（软依赖）、`needs`（硬依赖）、`readiness`（动态只读闭态门）、`admission`（固定数据/显存批准门）、`command`（绝对argv列表，不能为假科学命令）、`cwd`、`max_seconds`、`write_gib`、`cpus`（CPU亲和性，不允许重叠运行）、`receipt`（绝对JSON路径及预期status等）、GPU任务的`gpu`物理序号、`gpu_uuid`和`min_free_mib`，以及可选 `allow_shared_gpu`（必须符合现有共享许可）、`preflight`（原生只读安全/owner资格）和`env`（不覆写CUDA设备）。

`receipt`与`readiness`规则以JSON字段的点路径核验，例如 `{"path":"/actual/native/RESULT.json","equals":{"status":"CLOSED"},"minimum":{"valid":1}}`。**必须填真实原生结果路径；只读check通过不是科学资格或CUDA通过。**

## 具体验收

- `M0` 冻结旧完整成绩与新SOURCE当晚所有尝试，旧1/4和曾经STOP不改。
- `M1` CPU静态/实际导出和四任务统计、原始RGB，新增专家示范与学生前缀严格分开。
- `M2` 每一GPU独立显存/进程资格，GPU1训练若未获准，仍GPU0串行；不以用户许可绕过实际1GiB规则。
- `M3` 原生GPU0上的真实可恢复更新、一次独立DEV面板与完整失败表；一个训练臂失败不挡其他分支；不在DEV里用Teacher接管。
- `M4` 若真实数据和资源许可，四任务共享策略一版，独立任务分项评估；保持冻结WBC和目标S1/T2隔离。
- `M5` 10小时终态：SUMMARY.json列出每个节点的COMPLETE/FAILED/SKIPPED/UNRUN/UNKNOWN与真实回执，不用改科学分母；科学Issue #11/运维Issue #47发布真实原生结果。

**部署注意：**现有Core source manager07可能运行在Node-local私有代码树而非GitHub main工作树。Codex必须在原进程冷态完成后将本次程序发布到自己的新独立代码树；不准在现有sealed capsule原地git pull/hotpatch，也不准因GitHub新文件存在就假设夜间自动启动。不要访问维护中Vanda、不要删大文件/原始数据、不要改配额／NFS／系统服务／SSH／凭据。
