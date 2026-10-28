# 2026-10-28仓库整合记录

用户请求：整理代码库、删除不必要分支、合并到main，并写清最新实验路线。本次是仓库管理，不是新的实验、远端目录清理或运行部署。

## 已实际完成

最初共7个分支。旧main为 `ff50eeb9b9635b2932c765168d55fe0077880d04`；现役分支为 `b6b5a28a7e75be889c898f7f116cabce174f7785`。两者分叉，实验分支ahead2574/behind27，不能直接快进或以一方覆盖另一方。

| PR | 保存内容 | 实际merge commit |
|---|---|---|
| [#81](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/81) | 完整实验历史＋main独有维护／CI历史 | `ba122176420f19f33e0bbc7dea414d2a8f308fe4` |
| [#82](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/82) | Hopper历史维护的9份独有回执 | `d4996eb058e7076ebd36402932fad77357c0dc20` |
| [#83](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/83) | PI0257视频证据协议 | `2e033b2cc53e31784758ce93d11da5c71e19481d` |
| [#84](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/84) | PI0263两份历史协议 | `158083a23242f046e57319b4299ae36834ef01b2` |
| [#85](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/85) | PI0264最终账本 | `2080c725cc810f6fe1423cda90076206bf089641` |

全部为普通merge，保留双方parent；没有squash、rebase、force或改写旧commit。ops分支 `1279ecca`经GitHub compare确认是整合后main祖先，ahead0。四个其他旧分支各有独有历史，已先保留再列入退役候选。

两处共同改动冲突为README和CPU workflow。两份README原blob及较完整的实验CI均已归档；合并时先对齐main以完成正常merge，导航提交恢复原实验CI完整blob（含PI0294步骤）。pyproject、motion_api qualification和仓库回归文件已读取，双方对应blob一致，没有借整理改变科学数值源码。

## 导航与研究路线整理

main成为唯一集成发布入口。README、START_HERE、CURRENT_SCIENCE_STATUS、AGENTS、PROJECT_STATUS、RESEARCH_INDEX和REPOSITORY_WORKFLOW更新为当前路线；新增EXPERIMENT_ROADMAP和CODE_MAP。

原START_HERE约475KB、状态页约148KB不是删除：它们连同旧入口按完全相同Git blob保存于 [pre-consolidation](../../archive/2026-10-28/pre-consolidation/)。原详细进度日志、报告、失败、所有实验模块和固定SHA仍在原路径。新入口不再承载完整逐次运行日志。

当前路线：先收齐D25／GEN1行为 → 完成E0诊断与E1/E2同父同12示范512对照 → 独立泛化验证 → 有必要时开展独立整机标定／少样本目标适配；原DEC1无目标任务训练分支独立保留。平台固定Vanda训练、4080S评估，两站运行不实时互相依赖。

## 尚未执行：远端分支删除

**实际删除数为0。** 当前GitHub连接器支持合并／文件和ref更新，但未提供delete-ref接口；没有用不受支持参数或其他账户绕过。仓库也未开启merge后自动删分支。

当前有8个分支（原7个＋本次已合并维护分支）。[精确退役清单](BRANCH_RETIREMENT.json)列6个已合并候选；辅助脚本先dry-run，创建并回读同名恢复tag，再用精确SHA lease原子删除。**脚本未执行，tag未创建，不得把清单或脚本存在写成已清理。**

保留main及pi0267兼容ref。后者只为旧publisher/branch-bound tooling迁移保留；今后新发布走main。若旧分支继续前进，先正常整合新增commit，不能按旧HEAD删除。分支删除不清理任何本机worktree或服务器目录。

```bash
python scripts/maintenance/retire_merged_branches_20261028.py
python scripts/maintenance/retire_merged_branches_20261028.py --apply
```

执行者应保存命令输出，回读实际远端heads/tags后将真实结果记入#47。本记录继续保留为本次未删除快照，不追改历史事实。

## 验证边界与限制

- GitHub实际PR merge结果、完整parent关系、精确ref和关键blob可回读。
- 本次临时只读snapshot workflow没有产生可见运行；已从分支和最终tree移除，不留下后台任务。
- 本会话container与Python执行返回ClientError；因此未克隆完整仓库、未运行pytest、未执行分支脚本。没有新CI PASS或全科学代码审计声明。已存在的旧CPU结果不能算本次测试。
- 不修改任何GPU/PBS作业、服务器源码capsule、模型／数据／统计、WBC、安全门、预算和截止；不恢复暂停自动化或子代理。
- 合并是源码与历史收敛，不是所有旧模块的生产准入。部署仍须固定版本和既有检查。

最终维护发布SHA及文件回读由本次GitHub commit与Issue #47留言提供；无需把未产生的SHA写入本文件。
