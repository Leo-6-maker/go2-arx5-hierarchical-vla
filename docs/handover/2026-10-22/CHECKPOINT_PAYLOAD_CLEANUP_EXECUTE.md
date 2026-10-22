# 立即执行：失败 checkpoint 载荷清理与 PI05 有界轮转

日期：2026-10-22，Asia/Singapore。ID：CKPT_PAYLOAD_CLEANUP_20261022。
状态：USER_AUTHORIZED_EXECUTE_AFTER_NATIVE_ITEM_CHECKS；发布不是远端删除收据。
授权依据：用户明确要求「分析哪些checkpoint可以直接删除，对于失败的checkpoint只需要保留训练配置和记录即可，无需在空间占用，直接删除」。不再逐项询问是否删除；满足以下条件的对象由现有唯一执行 owner 直接处理，未知对象单独跳过。
审核基线：587a15f94dde787f423e46108a2c76ca0f6c7e58；#47/6334161032 表明新窗口已安全接手原 A1。本文是有限存储维护授权，不增加 GPU/PBS/训练/物理额度，不改变科学评分、WBC、现役源码或原截止。

## 1. 本次决策与范围

**失败且已明确弃用、无活动或未来绑定消费者、非唯一有效恢复点的 checkpoint 大载荷，可以直接删除；只保留配置、日志、统计和小型审计记录。无需把废弃权重再复制一份、压缩回同一 scratch 或重新上传大备份。**

本指令对符合条件的失败载荷明确补充了旧交接中的“本交接无删除授权”。不解除有效恢复状态、源父权重、基线/选中模型、数据集和环境的保护，也不以 Exit!=0、低任务成功率或目录旧作为自动删除判据。

执行顺序：
1. 精确核查用户点名 PI05 `vla_wbc/checkpoints/...s6000_alltrain...` 的 step1000/2000/3000/4000 轮转条件，合格即删；保留5000/5999以及另有有效引用的best/基线。
2. 清理已终态、未再引用的失败 run 内模型/optimizer 残片与确认弃用的失败候选载荷。
3. 仅核查 `Projects/go2-arx5/experiments` 中同类 checkpoint，不扩展为删除全部历史实验目录。

限定用户拥有的上述实验输出。PI05可能不是PV1项目，必须在其原注册/所属工作区确认所有权、作业和用途；不能因为路径同用户就把它当PV1资源或接管其任务。已获得的维护授权允许核验并轮转用户点名对象，但不允许终止其他项目工作或猜测其用途。

不删除：`runtime`/Conda/容器/模拟器、`data/Dataset`、PI0292原始/运行数据、模型公共缓存、WBC/vendor资产、源码、私有凭据。`Projects/GR00T-WBC` 不能整棵删。

## 2. 点名对象的判定

| 对象 | 处理决定 | 必须核实的条件 |
|---|---|---|
| PI05 step1000/2000/3000/4000 | 优先 DELETE_AFTER_CHECK | 精确匹配用户指定run；后继5000、5999完整且至少满足原恢复合同；相关作业已终态或其既有轮转合同明确允许；这四点无writer/reader/排队恢复输入/模型选中/评价或发布依赖 |
| PI05 step5000、5999 | KEEP | 当前作为两个恢复点；若实际另有新step，不能凭命名自行扩大本次删除集合 |
| `1474116`旧A失败run中临时/残缺训练载荷 | DELETE_AFTER_CHECK | 已真实终态关账、无消费者，保留1024真实更新与SAVE_FAILED收据；该run已证明没有完整1024 checkpoint，不虚构一个25GB可删对象或可恢复状态 |
| `1474352`的 `training/A_CONTROL/checkpoint-32` | KEEP：有效且在用 | 虽Exit1，15成员全状态已验收，complete SHA256 `1c50579c172367894fe83921d84e4053803dd451c1c3e693d7881f1ff673e6f8`；当前A1明确引用，绝不按“失败run”删除 |
| `1474599.stdct-mgmt-02`现役A1及其输入/后继产物 | KEEP | 最新公开为Q；删除前刷新真实状态，不能为清理取消/重投/热改 |
| 原CORR@2048及其引用的Cosmos/冻结分片 | KEEP | A、B共同父与DEC1预注册模型，不能因来自旧实验或通过alias引用就当垃圾 |
| 已确认放弃、无任何恢复/基线/选模/科研复现依赖的失败候选 | DELETE_AFTER_CHECK | 终局判定与原结果已归档；从可用模型索引退役，留配置、身份与失败指标，不保留大载荷 |
| 已有合格后继的冗余工程32/2048副本 | CONDITIONAL | 沿已有分段授权：后继完整独立验收、原消费者全部关闭、无未决副作用才清理；当前唯一state32不符合 |
| 受保护的科学1024/4096、唯一best、待评价模型 | KEEP | 未被明确废弃，不属于失败垃圾 |

当前step32位于Vanda运行链；用户报告主要针对Hopper scratch。必须按实际host/mount/quota分开核查。删除Vanda文件不等于Hopper释放同样容量，逻辑目录相似不证明共享filesystem。

## 3. 删除前只做必要核验，不再无限盘点

使用已部署host-aware入口/registry和现有维护工具，不在任意shell猜qstat路径。`qstat: command not found` 只表示当前shell缺入口，不是“没有作业”。在正确Hopper/Vanda角色读取完整本人队列，并结合producer终态、所有run manifest/lease、queued输入、checkpoint依赖/模型索引和同用户进程核对。

用户提供的 `...s6000_alltrain...` 是定位线索不是可直接执行的路径，禁止把它转为宽泛glob删除。先解析唯一目标run与四个精确step目录；有多匹配或不明所有权时跳过该对象，不猜一个。

按对象形成小型私有精确清单：host、挂载/配额域、realpath、owner UID、run/step、文件相对路径、device/inode/link count、size与allocated blocks、现有成员hash/manifest、修改时间、reader/writer与科学用途核查结果、决定和理由。已有固定hash可复用，确需补哈希时只对具体候选；不把“清理前必须反复扫整个Home/NFS”变成新前置。

删除资格必须同时成立：
- 目标为本次明确范围内checkpoint载荷，不是数据/环境/源模型；所有权与具体run可证。
- 原训练写入已结束；无当前/排队/已封存后继输入、eval、best/selected、rollback、复现或恢复引用。仅 lsof 无输出不足以排除远端消费者。
- 不是唯一有效恢复点；需要保留的后继已通过原完整恢复合同，而不只是目录存在或step更大。
- 不存在待核清提交/保存/复制副作用；删除对象身份稳定，取得与模型管理互斥的原锁。
- 配置及记录留存已完成且可回读。单个未知项标 HOLD，不阻止处理其他已证明安全项。

在每项unlink之前重新核对文件device/inode/size/mtime及引用，不follow符号链接、不跨mount、不chmod共享硬链接，不遍历到model_inputs alias的外部target。引用发生变化立即跳过。

## 4. 留什么、删什么

保留小文件：训练配置/命令/依赖版本、source与父模型/数据/统计量hash、seed、sample cursor与训练step的文字记录、trainer state JSON、loss/梯度/任务指标、stdout/stderr、错误栈、原CHECKPOINT_COMPLETE和成员manifest、PBS/租约/费用终局、引用/退役清单以及删除前后收据。原历史文件不改写。

可删除精确白名单中的载荷：候选模型/adapter权重、optimizer大tensor、已弃用的分布式训练分片、完整性失败的临时checkpoint文件，以及已验证其内容只为这些弃用checkpoint的冗余归档。扩展名 `.pt/.bin/.safetensors/.tar.gz` 不是充分条件；按具体manifest与用途确定。

保留原CHECKPOINT_COMPLETE作为历史证据时，必须另写明确 `PAYLOAD_RETIRED` tombstone/可用性记录，并从未来loader可选集合退役。不可留下一个会被误读为“权重仍完整存在”的READY入口；不篡改旧成功/失败历史。

仅删除获准的具体普通文件/其目录项；空checkpoint目录可在确认不含配置日志后清除。禁止对run根或Projects根执行宽泛recursive delete。部分执行失败准确记录已unlink集合；恢复时不得盲目重放已变更路径。

用户已允许废弃失败权重不可恢复，故这类对象不再要求大模型独立备份。有效恢复/父模型/受保护科学状态仍不在删除范围内。

## 5. 释放量的诚实计算

用户2026-10-22 14:03的数字为待现场刷新：scratch821.1GB/1.0TB，PI05六个step共194108293120B。按六个近似等大小估计，1000–4000四份约129.4GB（120.5GiB）；这是候选逻辑量，不是承诺实际释放量。

先按唯一inode和allocated blocks计算，不重复加共享硬链接。删除一个nlink>1的副本通常不会立刻释放其共享数据块；不可通过删除保留分片的其他link来强行凑释放目标。reflink/文件系统压缩/快照以及打开的已删除文件也可能使逻辑量与配额变化不同。

记录：删除逻辑字节、候选独占allocated blocks、实际unlink成功字节、同一host/同一配额域的before/after权威quota与块统计、测量时刻、并发写入或配额延迟。不能仅凭全文件系统df变动认定本任务释放量。quota未刷新写PENDING，不编造129.4GB实际释放。

删除只影响真实当前存储占用；不得退款历史GPU/CPU消耗、旧失败保守计费、高水位或未执行的新作业预约。原预约只有满足既有终局协议才释放，不用文件删除代替关账。

## 6. 执行与回执

由已经接手的唯一Codex owner在现有合法维护CPU/IO范围内执行，不再创建第二owner/永久清理服务，不阻塞当前A1心跳或认证放行。复用已有工具和锁；不为清理新增GPU、PBS、物理或模型加载。采用当下已批准维护权限与实际剩余资源；权限/挂载缺失时只报告精准阻断，不申请同一批候选的重复删除确认。

给#47一份合并实际回执：`CHECKPOINT_PAYLOAD_CLEANUP_ACTUAL`，逐项 DELETE/KEEP/HOLD、脱敏相对路径、原run/step、保留记录身份、实际unlink结果、前后空间与未知项；涉及科学恢复点保护在#11登记简短索引。只报告已执行的动作，不把本文件发布说成已清理。

本次发布端只有GitHub读写能力，未直接访问或删除Hopper/Vanda文件。执行端在核验后直接清理合格项；无资格的state32等继续保护，不因为用户要求“直接删除”而忽略用途与引用。
