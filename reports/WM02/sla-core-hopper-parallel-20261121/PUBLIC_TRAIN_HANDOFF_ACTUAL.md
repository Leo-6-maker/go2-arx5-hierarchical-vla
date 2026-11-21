# Core TRAIN 已经由公开 HF 数据集直达 Hopper

2026-11-21 SGT。项目负责人明确授权将本 Goal 新采集、已准入的 TRAIN 示范保存到 HF public。Core 服务器已发布前三个累计快照，VANDA 在 PBS 内读取固定 revision、校验并原子交给原收件目录，现役中继再直接传至 Hopper。Hopper 当前训练版本的课程目录函数已实际读到全部三份 seal。本地电脑只发起部署与读取回执，未中转样本。[公开数据集](https://huggingface.co/datasets/leoyuer/sla-core-hopper-train-v1) · [机器回执](PUBLIC_TRAIN_HANDOFF_ACTUAL.json)

| 累计快照 | 独立示范 | 窗口 | 文件 | 逻辑字节 | 原始发布 revision |
|---|---:|---:|---:|---:|---|
| TRAIN-004 | 4 | 333 | 30 | 132,926,268 | `13239cea6334a2cf90bc03373a61559944347df8` |
| TRAIN-008 | 8 | 661 | 56 | 263,855,137 | `c2f3483a86627fb50164edee81db07bf84c098f2` |
| TRAIN-012 | 12 | 994 | 83 | 396,780,276 | `e17ec4825889fe712c6616e6cd756dbbb19d0863` |

三份快照为累计成员。最新 12 条示范来自 7 个数值场景，3 条 PLACE 全是 F1 红块右放。HF 按 SHA256 共享同一 RGB 或标签文件。在 Core、VANDA 和 Hopper 三台机器上用新连接重新读取全部 83 个文件，分别计算文件 SHA256 和排序后的清单摘要。三站与匿名取得的公开索引均为 `469359cca48a585c845a1d0f8eb979b458e8e4d8dbeca11c6a02e4edf8077d3f`。原训练 seal 为 `04b0e4bbe182d438b06940caacc91a24bcd280e8205aff54f9efdbfc3bcb9a27`。[固定版本的窗口和文件索引](https://huggingface.co/datasets/leoyuer/sla-core-hopper-train-v1/blob/e17ec4825889fe712c6616e6cd756dbbb19d0863/snapshots/TRAIN-012/INDEX.json)

## 后台任务与停止入口

- Core 发布 unit 为 `wm02-sla-core-hopper-v1-public-train-uplink-20261121-01.service`，PID 2112058，用户 linger 已启用。独立连接读到相同 PID、已封存的三个发布回执及 `WAIT_SEAL` 心跳。读取范围由冻结人口、逐条 G7 准入及快照文件清单限定。服务没有 GPU 配额，CPU 上限一核，内存上限 4 GB。
- VANDA 入站 `1602334.stdct-mgmt-02` 从独立连接读回 R，PID 4107601，项目 `CFP04-CF-007`，1 CPU、0 GPU。原直传作业 `1601552.stdct-mgmt-02` 写出三份 Hopper 接收回执；其登记后继为 `1602101.stdct-mgmt-02`，依赖前段 afterok。新入站的后继由本段在到期前登记。传输源码为 `0501dbefb11fd3fbbd7098ab2856a7f13000d6f1`。
- 两个新增角色分别在其运行目录接受 `STOP`。Core unit 可独立停止，VANDA 日志与分段状态可按 JobID 查阅。到期沿用 12 月 5 日 12:41:52 SGT。该传输涉及 TRAIN 的 RGB、G7 标签及相应清单；模型发布沿用既有模型仓库。
- 当前 12 条均由原专家产生。后续若首次出现含修复版专家的快照，HF 可发布，VANDA 可校验并单独保管；旧训练读取器只接受原生产者格式。该格式需在冷态新版本取得兼容证据后再向训练端发布 READY，其余已验证快照仍可用于预登记课程。

## 训练和原生分母

19:22:42 SGT，Hopper 正式作业 `727394.hopper-m-02` 已实际提交 501/20,000 次新更新，Go2 / RoboDojo / LIBERO 分别访问 32,064 / 16,032 / 16,032 次，完整训练 H40 共 2,004 次。第 500 步已于 Hopper 封存，140 个状态数组对应 checkpoint `fecd2d02be4e214900e9acea2fb32c9bedf1e5048e1cdc41197893588bfb3330`。在该截面，VANDA 上一次已验证的完整模型备份为第 400 步。本次数据传输本身消耗 0 GPU、0 更新、0 模型生成、0 原生请求；新上传示范的实际优化访问仍为 0。

Core 专家 TRAIN 已启动 35/240，34 条终态，其中抓持及抬升 20 mm 各 24 条、14 条完整成功，18 次非法接触、2 次搬运超时，终态共 970,375 tick。固定 VAL 24/24 终态、8 条完整成功，678,919 tick。33 次硬 STOP 后积分均为 0。正在执行 F1 红块右放 `-03`。两条 F2 修复 pilot 等待同一 world，实际请求均为 0。phase 0 学生沿用严格 RGB16 终态，DEV/CONF 原生请求与积分均为 0。

在已核对的 CPU 读取中，994 个样本包含 39,760 个监督动作点及 9,940 个实际消费动作点。若 phase 2K 时采用这份快照，原配额会给每步 Go2 的 64 个访问分配 16 个原回放和 48 个非初始示范槽。正式的分支决定依据当时的 seal，当前课程切换与 CONTROL2000 触发均为 0。[原生起点、场景覆盖和 F2 接近诊断](TRAIN012_RESET_AND_APPROACH_ACTUAL.md)
