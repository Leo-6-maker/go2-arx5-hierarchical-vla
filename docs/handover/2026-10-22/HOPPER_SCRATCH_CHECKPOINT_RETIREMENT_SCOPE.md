# Hopper scratch checkpoint 可删判定与原生清理门

日期：2026-10-22 Asia/Singapore。状态：USER_DIRECTED_FAILED_PAYLOAD_CLEANUP_PENDING_NATIVE_VALIDATION。审核基线：587a15f94dde787f423e46108a2c76ca0f6c7e58。

## 0. 授权精确含义

用户提供14:03的Hopper盘点：scratch 821.1GB/1TB，Home 20.51GB/40GB；PI05的s6000_alltrain六份step1000/2000/3000/4000/5000/5999共194108293120B。用户要求确证无法恢复、没有独立用途的失败checkpoint **大payload直接删除，只留训练配置、日志、原始失败和成本记录**。成功的早期checkpoint如经现场证明冗余，也可作一次精确轮转；不能将退出码Exit1一律等同于checkpoint无效。

本仓库没找到s6000_alltrain的PI05原生job/引用证据，不能凭盘点视为已经退出或同一项目；本文件没有远端SSH与删除效果。执行仅限对目标Hopper scratch、PI05独立项目及PI0294分别**真正有权限的原维护owner**；不能让新Work窗口把PI05作业接管进GR00T科学账。任何活动VAP1 watcher/release优先，不可启动第二个科学owner。

## 1. 精确候选和保护清单

| 路径/状态 | 判断 | 前置准入 |
|---|---|---|
| PI05 vla_wbc/checkpoints/s6000_alltrain 的1000/2000/3000/4000 | 优先轮转候选，非立即无条件删除 | 现场证实PI05 train/eval等消费者终态，最新5000和5999各自完整可恢复，且1000-4000均非best/selected/后续resume/论文消融唯一引用；保留任何有特殊科学用途的早期点。每个候选单独列明exact files。 |
| PI05的5000、5999 | 保留 | 只有验证完整全状态、单独读取与选模身份后，才能以后重新评议。文件mtime不证明5999完整。 |
| PI0294失败A job1474116 / 1024更新 | 只删确证的无恢复价值残留大payload | 已有原生审计无CHECKPOINT_COMPLETE、无有效1024 full state；如运行根存在中断写入片段，且无consumer、无selected推理用途，可保留config/recipe/log/FINAL/RESULT/费用/失败收据后定点unlink。不存在的完整checkpoint不能当成可腾出32GB。 |
| PI0294旧premodel失败job1473877/1471903 | 只删确证无用的孤立大payload | 旧作业已terminal/settled无模型效果的原生证明、准确inode与引用确认；通常只有日志，不能虚报大量释放。 |
| **PI0294失败后完整checkpoint32（job1474352）** | **绝对保护：不因Exit1删除** | 15成员/25.54GB、CHECKPOINT_COMPLETE SHA1c50579c172367894fe83921d84e4053803dd451c1c3e693d7881f1ff673e6f8；它是A_CONTROL_SAVE_RECOVERY1 的当前唯一可信全状态恢复源，step32、sampler cursor1024。只有真正独立后继全状态验收、无活跃consumer、已有单项清退权限才能以后评议冗余副本。 |
| PI0294 A1 job1474599、CORR@2048父、未来A/B 1024/2048/4096科学状态 | **保护，不在本轮清理名单** | 14:12 SGT新owner已安全恢复现有watch/release；1474599最后Q，随时可能R，禁止清理在途输出、原父文件及当期resume。 |
| Projects/go2-arx5/experiments 历史目录 | 另列候选，不进入本轮删除 | 旧目录统计非今日现场，逐实验确认manifest、独立模型保留和先前退役记录。 |
| runtime环境、data/Dataset、PI0292、WBC/vendor资产、其他项目 | **不得因体积清理** | 运行与历史科学保护继续有效。 |

失败运行的保存规则：RUN_MANIFEST、source/model/data/stats身份、训练config、optimizer超参数及seed、log/loss/梯度、失败STACK、STOP/FINAL/RESULT、PBS原生终态、真实更新与资源花费、文件清单和旧hash一律保留；没有恢复/评估用途的模型与optimizer片段不用占磁盘。绝不把失败记录从账本抹掉。

## 2. 每个真实删除对象必须通过的原生门

1. 验证mount、真实项目owner/授权、目录realpath、PBS job终态、run/lease/registry和选模/consumer引用图。当前shell缺qstat不能推出job没运行；从合法登录或原owner已有官方PBS/qselect/trace工具读回。PI05与GR00T项目身份未经证实时只做盘点。
2. 读取checkpoint completion标记及关键成员；确认保留的最后两个checkpoint实际可用。若中间step曾为best或唯一评估/论文引用点，则从删除allowlist剔除。失败作业若先成功save又在后续sharing失败（如PI0294 step32），**绝不视为无效checkpoint**。
3. 精确记录文件及已存在hash/manifest、大小、uid、inode、nlink、symlink target、当前用户和打开进程/实际恢复引用。避免全树du/全文件重复hash引发NFS阻断；一个路径大小不等于能释放的物理块，nlink>1需保留其他消费者。
4. 先保留小型配置/日志/失败/成本记录及私有精确PREDELETE清单；只对预先列出的PROVEN_SAFE绝对路径执行一次unlink。不得用glob、mtime、整树rm -rf/find-delete、跨路径link-follow、杀健康job或为了达到空间数字而顺带删其他目录。未知效果先读回，绝不盲重复删除。
5. 独立验证post-delete仍存在有效checkpoint、所需train/eval消费链、原科学账/lease未变；记录真实du allocated、项目quota、df前后差值及保留状态。若unlink未释放空间（硬链接/打开文件），记录并停止，不追杀消费者。

## 3. 预期空间和执行顺序

PI05 6份194108293120B，假设相似且前4份均不被引用，四份逻辑可删约129405528747B =129.4 GB ≈120.5GiB。scratch 821.1→约691.7GB（约69.2%）只是假设所有逻辑空间均实际释放的估算。不能把这些数字写成今日现场du/quota结果。

先确认PI05训练是否结束、谁是best、5000和5999是否均完整；同步定点识别已确认失败但无有效状态的残片。优先释放最大、确定无用的PI05历史点；不要清理正在等GPU的A1或其唯一checkpoint32。现有主线VLA训练/WBC接口进度优先于清理。

## 4. 回执与停机界限

原生owner完成时产出私有HOPPER_CHECKPOINT_PREDELETE_ACTUAL（每path/step/引用/逻辑与实际块），然后只清理PROVEN_SAFE并产出HOPPER_CHECKPOINT_DELETION_ACTUAL（删前后配额、路径/文件/字节、保留和终态），向#47只发布脱敏小报告。无法确认作业/权限/引用/最后两个有效状态时，报告HELD_WITH_EXACT_BLOCKER而非冒险清理。本文件GitHub发布不等于已经删除或释放空间。

当前GR00T事实见#47/6334161032：新Work已真实安全接续同一owner，14:12:08恢复watch/release，1474599仍Q、step32完整，所有账本不重置；不能因此停止它来腾空间。