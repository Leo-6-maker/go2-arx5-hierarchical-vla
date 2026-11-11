# main 入口整理与执行来源 · 2026-11-11

## QUESTION

让仓库入口准确反映当前原生相机＋公开 ARX5 迁移首轮计划，同时保留既有执行来源、历史负结果与并发工作。

## CHANGED

仅整理 README、START_HERE、CURRENT_SCIENCE_STATUS、路线图、代码地图与 AGENTS 的最新首轮范围。历史长页用其完整 Git SHA 链接追溯。新增本记录。执行代码、实验结果、工作流、依赖、WBC 及运行胶囊未改动。

## ACTUAL

审计起点：

- main：`0a27753bbda8ed75ab6b74aeb64755e0427362c7`；tree `90baeb7eb2e46c5227993769905211d1d2896100`；2026-11-03 合并 #341。
- #371：`f0e19c8812896f98053e74b165d6c52e85327b69`；base `cad80fbc2a9e3f70288b56967f196731612c452d`，分支 `research/wm02-fusion-night1-execution-20261106`；仍为 draft。
- #371 自身：217 commits／358 files；其 head 相对 main：ahead730、behind0。
- 完整两端 recursive git tree：1,522 个 blob 变更，无删除；GitHub compare 的前300文件不是完整差异。
- #372：`fa322698226fdea878494a5ea3b7605c9d1a61a1`；#373：`885d8b48a47e3f7b45bd0c9ecd791df2a1e3cfd9`；均为 docs→main draft。

此整理通过普通分支／PR 集成；不把 #371 对其 feature base 的 clean 状态当作整条执行栈已审核。未合并执行分支栈、改运行 checkout、启动远程作业、下载大模型／视频或删除分支／历史数据。

## NEGATIVE

main 缺少多个当前训练与整机评价模块。新文档分支因此只交付协议与声明性实验规格，不能声称已经具备可运行的新 pipeline。

Teacher24 最终19/24来自用户本轮明确提供的结果；核验到的执行 head 仍只有 first10／first11 的较早部分回执。保留19/24的负结论，并补完整终态与哈希；不把部分9/10、10/11当最终成绩。

## DECISION

main 入口与执行实现身份分别标明。首轮实验在独立研究分支 `research/native-camera-arx5-transfer-v1-20261111` 展开，以既有执行 SHA 为复用依据；若后来集成执行栈，使用正常 ancestry／PR，单独审查实际影响边界，不能以整理文档为由整栈直合。

本轮最新指令只替代后续首轮优先级；旧协议、旧人口、旧负结果、健康在途作业仍保留。main 文档合并不表示部署或运行许可已由现场条件满足。

## NEXT

[新协议](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/research/native-camera-arx5-transfer-v1-20261111/docs/protocols/NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1_20261111.md) 完成相机、源数据、官方起点、缓存、完整恢复与新 DEV 身份的实际准入。现场既有工具可解决的工程缺口由执行方自主完成；不得用尚未生成的 hash／回执占位声称通过。

## 历史入口

- [原 README](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/README.md)
- [原 START_HERE](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/START_HERE.md)
- [原 CURRENT_SCIENCE_STATUS](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/CURRENT_SCIENCE_STATUS.md)
- [原 AGENTS](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/AGENTS.md)
- [执行快照](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/f0e19c8812896f98053e74b165d6c52e85327b69)
