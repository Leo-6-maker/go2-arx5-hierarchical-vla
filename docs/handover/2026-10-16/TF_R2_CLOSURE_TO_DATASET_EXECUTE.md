# Codex接续入口：TF-R2闭爪动态到完整Teacher数据闭环

**PROPOSED / NOT ADOPTED / NOT LAUNCHED。** 本文是用户要求的下一步审核计划；发布本身不赋予新计算许可。先读同一固定提交下的协议和机器卡：
- `docs/protocols/PI0294_TF_R2_CLOSURE_TO_DATASET_20261016.md`
- `reports/PI0294/design/TF_R2_CLOSURE_TO_DATASET_20261016_TASK_CARD.json`

再读AGENTS、START_HERE、统一入口/冻结WBC规则和#11/#47最新事实。使用现有local-pc唯一owner和`~/Projects/GR00T-WBC`入口；不建第二调度者、不热改运行capsule。

## 必须先辨明的时间/权限

旧09:00停止新工作已过，旧10:00只允许收尾。#11 comment6203010585的单项草案不能在旧截止后自动使用。需要明确采纳本继任包的新T0+7小时停新工作/T0+8小时收尾窗口及余额转入；它不改旧T0/关闭记录，不提高原累计资源总上限，不假造人类采用。

未有采用收据时保持无物理dispatch，合并报告一次，别连续三轮重复相同blocked。采纳后登记一次`R2_ADOPTION_AND_BUDGET.json`，包括固定release、用户采用证据、原账快照/转出、新绝对截止、资源转入和唯一owner。进程重启不能重置窗口。

## 已经解决的部分不要重做

审核基线`9b8fb7862231b6dbdb443162ec584909723cdb09`。启动06/07已在H47/open单worker域各完成安全5秒；pause锁范围、接收端ready、原看门狗500ms余量及长任务quota有后续修订。复用匹配hash的资格，不重回旧pause/100ms额外门，不重复全历史审计。新episode仍正常启动并全程安全检查，不能复制已站立qpos/history。

最后PC2 E4真正卡在闭爪：96秒任务只到6.996秒，原右指0下界被穿越到-0.229538mm，尚无物体接触。原命令和PD复算一致；这是动态安全FAIL，不是上传/数据格式/缺包，也不是高层GR00T失败。T1此前开放手指接触上越界是另一问题，本轮不要同时修T1。

源回放teacher目前`stage_at`主要按时间推进，actual接触记录并不等于阶段反馈。这是后续风险；先保留它完成S7单因素试验，不能同时改FSM追分。

## 采纳后的连续执行

1. 两路CPU准备：S7闭合桥+失败回归；96秒导出/失败拒绝/独立processor。复用现有资产和任务配额，控制器/夹爪PD/physics不动。
2. **S7**：同一T2/PC2/H47/open、原TRAIN源/seed/2cm入口/96秒，仅把22到2mm的20Hzstep请求改为0.95秒五次曲线。时长来自原40mm/s上限；按本次actual entry保持原到达/ACK时刻。42到22mm及后续源技能不偷偷改变。采样请求、2ms slew和实际运动分层核验；连续C2不证明实际安全。
3. 在同一完整episode里记录启动、闭合、接触、推动、撤离与保持；不是每个阶段新开一次试验。闭合安全但任务不成功，才按协议条件进入S8（仅实际事件门控，等待最多12秒含在96秒）及S9（仅从actual TCP/root重接原技能入口，每episode最多2次）。这两项是新的、必须随完整R2采纳的科学范围；只采用旧+1草案不得启用。
4. S7/S8任一完整成功即跳过余下因素，立即审计该seed并冻结Teacher；加两个预定独立开发布局，合计至少2/3完整安全成功后生产。seed/资格不暗算进生产配额。
5. 先完成T2 **8 TRAIN+2 DEV成功**。最多20个预定生产group，固定16/4分集且每group一次；不按DEV改Teacher。不等待T1，不声称移动/双任务。先serial，隔离资格充分后才最多2worker；提速失败不是阻断serial的理由。
6. 真正导出EEF13/实测state19/双RGB/语言和TRAIN-only统计。每条96秒1920命令、1921观测、1881个完整H40；prefix8/20Hz时间一致，命令不被actual未来代替，特权物体/FSM不入学生。慢速hold窗口分布要披露，不能把重复窗口当独立示范。
7. 独立CPU进程读取全成员hash、numeric窗口及分阶段真实RGB，经官方processor/collator；仅schema、空bank或12条失败全部拒绝不算消费PASS。私有远端受阻先交本地bank+消费者，不转public、不删旧资产、不为报告重跑物理。

## 自主修复与停止

遵循已有“允许不断重试debug”的同语义工程权限，不增加无依据的第N次小门；所有工程尝试都计总预算，明确代码/原件证据和独立安全检查才能分类。无证据的未知或真实限位失败不能变成工程豁免。

S7若真实安全FAIL：停止新的任务物理，不换profile/终点、不进入S8/S9，输出夹爪动态边界缺陷包。S8/S9或生产中真实安全失败也停止该task新物理；共享startup/腿/身体安全异常停止全部。绝不调WBC、夹爪PD、机械范围、solver、摩擦、力限或评分追成功。预执行拒绝不是物理事故，但固定科学候选耗尽必须保留FAIL。

提议转入上限20coreh（至少6收尾）/8总并发核/32总启动/1616096积分/61472双RGB；同时受原R1和父夜间实际余额。新科学条件最多S7/S8/S9三项。原05:41快照不是最终余额，后续补证/未计量保守成本都扣。CPU包括solver/腿策略CPU/渲染/编译/审计；GR00T/GPU/PBS/真机/下载/删除0。

## 交付

先报告完整成功和实际物体行为、所有安全事件、TRAIN/DEV和消费者，再报告工程。新增证据一次校验，旧hash包复用，不重复上千文件归档。交必要增量报告、release/bank、资源终态及恢复入口到GitHub，#11/#47回读。

状态分清BLOCKED_NO_SEED、SAFE_CLOSURE_ONLY、TARGET_SEED_ONLY、SINGLE_TASK_PIPELINE_CLOSED。本轮只要T2生产和消费闭环成立就是该单任务工程交付；它不是最终双任务policy成功。selected/deployment保持NONE。
