# Hopper Scratch 科学模型保留、旧副本退役与 HF Public 交付 V1

**2026-10-13。用户已明确授权本次 Hopper 个人 scratch 清理。发布此文件不是执行收据。**

目标：明确失败且不再需要的历史 checkpoint 和临时缓存 **核验后直接删除**；现役科学运行保留；有实际正面科研价值且可合法公开的推理模型发布 Hugging Face Public，并在匿名固定版本读回通过后退役本地冗余副本。现有 local-pc Codex 为唯一维护 owner；不新开科学 owner。此任务独立于 CoreLab AMD/4080S 既有删除额度。

## A. 已核对现状

- PI0292 的 64 TRAIN/16 DEV、原始数据和官方 CPU 数据链已经通过，私有 HF bank v2 固定 revision 为 **6075aac0cdc44b8d252d7540efbac208701c30b3**。截至最近 S3 实收据，当前 PI0292 模型 load/forward/main updates 为 0；pristine 预训练模型、全部数据、运行源、processor、stats、未来恢复点、任务 lease 需要保护。
- Hopper 的 hpc quota 在本日 14:13 报告 scratch **940.7 GB used / 1.0 TB hard**；必须以新的现场查询重新确立字节边界。PI0292 原 smoke 规划需要新增80 GiB峰值+20GiB余量；清理目标是满足实时 preflight，而非凑一个逻辑目录大小。
- 旧 PBS qsub 已真实拒绝：Project has already ended。**清盘不会恢复过期计算项目**。仍须用户在 Hopper 自己 amgr login 并确认新有效项目，禁止借用他人项目或绕过调度。
- PI0286 两条正式训练均未取得原资格：C0 TRAIN 2/8；C1 TRAIN 8/8、DEV 5/16；selected NONE。C0/C1 各自完整 Trainer 已在 HF Public **leoyuer/pi0246-gate-pass-campaign** 有独立全成员验证：C0 revision b0b8c65de8d5bbdf0946ccdda9ecb23348cf16c9；C1 revision d35dcda18def3d0691b55957ce461fe6b8449437。当时 C0 本地大文件已退役，仅 metadata；C1 本地仍有约20.47GB完整 Trainer。**先检查 C1 的现役消费者，再退役其不需要的 Hopper 原副本；不要重复上传或重复计 C0 释放。**
- PI0276 历史 full Trainer 4096/8192 有归档回读身份，但 selected NONE 不能单独判定质量FAIL。按照原实验科学门、唯一对照依赖及已验证恢复来源逐项分类，不能凭目录名或未选中直接删。

## B. 统一路径和资源保护

现场用 hostname / id / realpath / findmnt / hpc quota / df -hT / 本人 qstat、作业及 FD/lease 核查，找出 **本人个人 Hopper scratch 的真实 UID/挂载/根路径**。旧报告中的历史 /scratch 路径只用于定位，不作硬编码删除根。不要处理其他用户、CoreLab NFS/4080S、Home 的虚拟环境、私有 WBC、第三方资产、当前模型/数据源或任何健康作业。

新维护 run UUID/T0 及独立账本，最长4小时墙钟、CPU合计12 core-hours、并发4 CPU、增量小证据≤2GiB。新 HF Public 模型最多2个、合计新增上传≤30GiB。没有新的 GPU/PBS/model load/optimizer/physics/WBC 权限，断连不重置 T0。清理目标个人 quota used ≤850GB 属于软目标；绝不为达成目标删除唯一受保护资产。

## C. 七类候选的实际处置

| 分类 | 动作 |
|---|---|
| ACTIVE_PROTECT | 当前 PI0292 数据/统计、官方 N1.7/Cosmos 缓存、adapter、运行源、所有需要的恢复点、活动进程/lease 及父路径：不移动不删 |
| UNIQUE_REFERENCE_PROTECT | 唯一基线、WBC、原始科学数据或无独立恢复证明的完整 Trainer：保留 |
| FAILED_MODEL_DELETE | 固定质量门明确 FAIL、未被选择、无活动依赖且不是唯一必需对照：**留小型失败评分/身份收据，直接删除精确本地大文件，不再额外备份全权重** |
| TEMP_DELETE | 可重建且无使用者的 tmp/part、重复解压视图、中间临时权重、过时缓存：直接删除精确文件 |
| REMOTE_VERIFIED_LOCAL_DUPLICATE | HF fixed revision 的完整身份及成员 hash、实际恢复读回已验证，本地不被继续使用：无需重复上传，删除本地重复件 |
| HF_PUBLIC_CANDIDATE | 经过科学证据、许可和可公开来源核验的有用模型：上传仅推理必需文件，匿名固定revision读回，后再删本地不必留的重复件 |
| UNKNOWN_HOLD | 权限、UID、hardlink、symlink、运行FD、科学资格或远端完整性不确定：保持不动，继续处理其它合格项 |

**selected NONE、checkpoint 编号或文件夹 PI 名本身不等于 FAIL。** 对真正 FAIL 模型保存完整已发表的失败结论、score、数据/模型/processor/stats SHA、配置和小日志；无需把几十GB科学负权重永久留在用户scratch。如果它仍是预注册比较唯一必需模型，改为 UNIQUE_REFERENCE_PROTECT。

## D. 每个删除候选的硬性资格

1. 从现场 inventory 固定真实路径及全部待退役成员。记录 st_dev/st_ino/UID/nlink/size/mtime、allocated blocks、来源 PI、质量结论、HF完整读回或可重建 provenance。对大型模型优先用既有 hash/远端清单核对，只对精确候选补 full hash，避免扫描百万历史文件。
2. 删除前重新核查 inode/size/mtime/所有权、符号链接逃逸、未知 hardlink、已运行的 PBS/FD/cwd/lease、模型上传读取进程、保护后代。不能验证则 HOLD；已被更新则重审候选。
3. 可删除时直接按 **exact manifest** 逐文件 unlink 和空目录 rmdir，不使用按 PI 名的 rm -rf / 通配删除 / find -delete / 清整个 HF cache / git clean -fdx。scratch 内 .trash 重命名不算释放。既有 symlink 或 nlink>1 不能凭逻辑字节声称真实回收。
4. 删除后一项一项保存实际成员回执、allocated bytes 与 hpc quota 前后快照；不得强杀别人的进程或强删 WBC。其它候选可继续执行，不因一个不明引用全局停止。

## E. HF Public 发布门

用户已确认 Hopper HF key 存在。仅调用 hf auth whoami 验证身份/写权限，不打印 token，不复制凭据到 GitHub。公开前检查完整 N1.7 基座固定 revision 权重许可证；**权重适用 NVIDIA Open Model License，源码 Apache-2.0，不得混淆**。遵循权重许可证 NOTICE/署名及第三方限制，排除私人 WBC、mesh、原始视频/图像、私有训练数据、机密和本地绝对路径。

公开repo 类型必须是 model、visibility=public；显式文件白名单仅 safetensors 权重分片/index、config、已授权的 processor/stats/action schema、README/model card 和许可/NOTICE。不要上传 optimizer.pt、scheduler.pt、rng_state.pth、完整 Trainer 或任意隐藏文件。发布说明如实写明任务数据、EEF10/H40、训练范围、TRAIN/DEV/闭环实际成绩、限制和非安全部署状态；失败臂不得改写成成功模型。

上传成功之后使用 **匿名**固定HF commit revision 校验成员大小/hash及 safetensors tensor key/shape，并试读推理必需配置。只有全部回读通过，且本地无继续依赖，才删除本地冗余副本。没有合适模型时 HF_PUBLIC_RELEASE=NONE 也是合格输出，不为了公开数量发布有许可/隐私/科研问题的模型。

## F. 真实容量回读、成果及状态

结束时重新测量真实个人 quota 和 filesystem，区分按stat累计释放、allocated bytes 和个人额度实降；核对 PI0292 数据/模型/处理器/统计、活跃作业无变化。调用已存在的 PI0292 容量计算验证能否满足后续 S3/S4 的现实峰值与余量；**此维护不代替有效PBS项目授权，不自动提交训练**。

在 reports/maintenance/2026-10-13-hopper-scratch/<run_uuid>/ 仅提交脱敏 **ADOPTION.json、CLASSIFICATION_SUMMARY.json、HOPPER_RETENTION_MANIFEST.json、DELETE_ACTUAL_RECEIPTS.jsonl、HF_PUBLIC_MODEL_RELEASE.json、RESOURCE_BEFORE_AFTER.json、HOPPER_CAPACITY_READBACK.json、PROTECTED_HASH_READBACK.json、FINAL_REPORT.md**；真正路径/凭据/全哈希列表保留在私人记录。#47 发实际Hopper维护结果，若 quota 真正解封 PI0292 则 #11 仅登记新容量条件，不自行重启或重置科学任务。

**GitHub GO/评论本身不是 Hopper 已删除或 HF 已上传的证据。实际终局可为 PARTIAL_WITH_EXACT_HOLDS，并明确本地已删除/已释放/当前保留/新 public repo 数字。**

### 固定来源
- [PI0292 S3 调度与存储阻断](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6139319043)
- [PI0286 两臂科学失败终局](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/7d0c4af6fa4a055a72939c203481584a84b2552a/reports/PI0286/final/FINAL_SCIENTIFIC_REPORT.md)
- [PI0286 已有HF全量远端回读身份](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/7d0c4af6fa4a055a72939c203481584a84b2552a/reports/PI0286/final/FINAL_MODEL_INVENTORY.json)
- [NVIDIA N1.7 权重许可说明](https://huggingface.co/nvidia/GR00T-N1.7-3B)
