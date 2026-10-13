# Codex：Hopper scratch 直接清理／有价值模型 HF Public GO

用户新授权：现役实验保留、旧临时结果清理、冻结质量FAIL的旧模型不强制大备份而直接删除、真正有科研价值且可合法公开的模型可推HF Public。你是唯一现有 local-pc 维护owner，Hopper shell才是实际操作位置。

先读取本release的 docs/maintenance/2026-10-13_HOPPER_SCRATCH_RETENTION_AND_HF_PUBLIC_GO_V1.md 与 reports/maintenance/2026-10-13-hopper-scratch/GO_TASK_CARD.json；同步读取Issue #11 的PI0292当前科学lease/项目过期/ quota与 #47 维护成果。

执行 S0 现场用户scratch/容量/own jobs/inode盘点 → S1精确分类 → S2对现役无依赖的失败、tmp和HF完整重复本地副本执行**立即精准unlink** → S3至多2个合规且有价值模型单独HF Public推理包、匿名SHA回读后再删本地冗余 → S4核查hpc quota实降、保护资产无变化和容量门。

优先核验PI0286 C1：DEV5/16失败且完整Trainer已HF Public固定revision/全成员PASS，若当前没有任何consumer则精确删除Hopper本地约20.47GB大副本，不再重复上传。C0此前大文件已退役，不能重复算。PI0276 selected NONE不能独立证明质量FAIL，核验原科学报告/使用者后再分类。

失败与过时临时文件不必逐个再请示，也不必再备份一份同样大的科学负权重；保留GitHub可审计的小型负结论和sha即可。安全审核不通过标UNKNOWN_HOLD跳过，不能阻断其他安全候选。不得跨UID、删除活跃PI0292的原始bank/stats/cache/模型、WBC、Home、CoreLab或其它用户数据，不杀健康作业。

HF公开前核对NVIDIA模型权重许可证及NOTICE，匿名固定revision实际回读；严禁公开私有WBC/mesh/raw/凭据/optimizer/pickle完整Trainer，模型卡如实说明有限证据和非部署。HF Public没有合格模型则计0，仍继续清理。

本轮维护上限4h wall、12 CPU core-hours、4并发CPU、新证据峰值2GiB、公发≤2模型≤30GiB；不授权GPU/PBS/physics/WBC。最后#47更新真实删除文件/bytes/quota、HF公共版本和hold原因；如确实为PI0292解除存储容量阻断，仅在#11登记事实。项目CFP03-CF-004过期仍需用户amgr login取得有效项目，清理不会自动解决。

完整 GitHub 协议的发布绝不视为 Hopper 已执行成功。
