# 2026-10-12 分支维护执行手册

状态：AUDITED_NOT_EXECUTED。ChatGPT本轮没有删除任何remote branch。当前GitHub连接没有删除refs/heads动作；local-pc用用户已认证的Git/gh执行这项已请求的维护，不要求粘贴token。

## 窄allowlist

见[完整JSON](2026-10-12_BRANCH_RETIREMENT_PLAN.json)。仅这四项可以进入live复核：

| Branch | Expected HEAD | Keeper |
|---|---|---|
| pi0263/train-template-counterbalancing | 1ea9a1ac7bc5f28f360f32c86fa958f266b56c4b | pi0267/executor-decoupled-recovery |
| pi0265/new-scene-feedback | 1ea7c2ecb593ce81d361c25da91386a5d27b708d | pi0267/executor-decoupled-recovery |
| pi0266/state-conditioned-feedback | 48048ae37b76534bbbe8a0ba76a9c84ed793e7ea | pi0267/executor-decoupled-recovery |
| repo-maintenance/consolidation-20261001 | f9546251ebf67d26dbe8b1d70e4c70584c821faf | main |

main、pi0267现役分支永久排除本次删除。pi0257/corelab-video-evidence-workflow-v1、pi0263/storage-subagent-model-audit-go、pi0264/language-diversity-transfer含现役分支之外的1/2/1个独立提交，本次保留。即便内容看起来被复制，也不能在没有证据时认为提交无用。

## 每项的安全流程

1. 确认origin仓库，记录dirty/worktree，不reset/clean它们。刷新default branch、保护、开放PR head/base。无法核实保护/权限时SKIP，不猜测。
2. 检查自己的本机/两端已知运行manifest或活动lease是否依赖该branch名称；发现活动使用就保留。
3. git ls-remote读取refs/heads/<candidate>必须等于expected；不同就SKIP且报告，不能更新expected后盲重试。
4. fetch需要的历史并用git merge-base --is-ancestor <expected> refs/remotes/origin/<keeper>确认可达。浅clone信息不足不是PASS。
5. 先检查同名tag及peeled commit。已有tag不同则SKIP，不overwrite。没有才建立同名compatibility tag（沿仓库2026-10-01惯例保留旧ref链接），回读指向expected后才允许删除head。tag仅为历史归档，不是质量release。
6. 再检查HEAD、保护、PR/active lease，执行单个明确ref的dry-run，然后精确lease删除。

```bash
# 模板，不可批量套用到allowlist以外。tag已经正确存在时不再推送。
git push origin "<expected_sha>:refs/tags/<candidate>"
git ls-remote origin "refs/tags/<candidate>" "refs/tags/<candidate>^{}"
git push --dry-run --force-with-lease=refs/heads/<candidate>:<expected_sha> origin :refs/heads/<candidate>
git push --force-with-lease=refs/heads/<candidate>:<expected_sha> origin :refs/heads/<candidate>
git ls-remote --heads origin "refs/heads/<candidate>"
```

7. 只有head已不存在、tag peeled commit仍为expected才记录DELETED。网络响应不明时先查，不盲重试。保存before/tag/commands/after与SKIP原因。
8. #47发布sanitized小收据并回读。禁止git push --mirror/--all/--tags、无expected的--force，禁止改写keeper、删除历史原件或HF资产。

四项全部成功时head数可能9→5；这是计划，不是已执行结果。原提交在keeper和兼容tag中保留，branch删除不是代码/数据销毁。需要恢复时先确认不存在同名新branch，再另行决定git push origin <preserved_sha>:refs/heads/<candidate>。

官方语义：https://git-scm.com/docs/git-push 。本手册只覆盖仓库维护，不能触发任何训练、仿真或模型实验。
