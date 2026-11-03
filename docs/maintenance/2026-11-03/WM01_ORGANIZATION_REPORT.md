# WM01仓库整理与发布范围

日期：2026-11-03。来源基线：`0a27753bbda8ed75ab6b74aeb64755e0427362c7`；tree `90baeb7eb2e46c5227993769905211d1d2896100`。

## 发现与处理

1. 旧 `pi0267/executor-decoupled-recovery` 停在10月28日；最新科学/运维更新已随PR341合入main。新工作基于main，不以旧分支导航覆盖新事实。
2. 原README仍写D25/E1/E2是新技能前置，并把AMD列作数据准备计算端，与当前V7不一致。README与路线改为现役V7边界＋WM01新研究，AMD明确为共享存储/登录，不跑重型实验。
3. START_HERE和CURRENT_SCIENCE_STATUS分别29638/86866字节，重复完整日志。改成短入口与事实摘要；原全文通过下面固定SHA可读，原reports不移动、不删除。
4. CODE_MAP区分既有可复用实现、当前新提供的计划检查器、尚待实现的WM代码。原生Engine/scorer只复用，不为整理目录移动科学Python模块。
5. 新增WM01协议、Codex交接、机器可读计划和独立实验索引；不把新增训练程序写成已实现，不将CPU检查当native资格。

## 历史保全

- [旧README](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/README.md)
- [旧START_HERE](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/START_HERE.md)
- [旧CURRENT_SCIENCE_STATUS](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/CURRENT_SCIENCE_STATUS.md)
- [旧CODE_MAP](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/docs/CODE_MAP.md)
- [旧EXPERIMENT_ROADMAP](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0a27753bbda8ed75ab6b74aeb64755e0427362c7/docs/EXPERIMENT_ROADMAP.md)

此次没有删除Git分支、源文件、报告、数据、权重或环境；没有更改根AGENTS、WBC、driver、scorer、现役配置和工作区控制。旧source在Git中逐字节保留；不是把历史失败改写成新成功。

## 检查范围

本次在隔离本地目录仅对新计划包运行：JSON解析／静态合同检查、17项标准库单测（普通与`-O`各一次）、Python语法编译、变更文档内部新链接核对。计划检查器显式报告runtime/native/GPU qualification NOT_RUN。链接到既有文件的存在性依据本轮GitHub读取；没有用空壳文件假装完成全库回归。

全库CPU suite／GitHub CI、实际MuJoCo、模型训练、GPU profiling、PBS、SSH与服务器部署均未在本次执行。容器无法DNS访问GitHub，源码读取/发布走已连接GitHub工具，不把本地新文件目录冒充完整clone。

科学快照依据#11 comment6601059002、#47 comment6601052516和PR341实际报告；接管agent仍须刷新自己的作业/owner状态。发表分支/PR与固定SHA在GitHub发布回执中给出，不在此预造。

## 回退

本变更只涉及导航与独立WM计划包。需要回退时对本次提交作普通revert；不要reset整个main或移动活动capsule。历史入口可直接读上述固定SHA，不依赖新增副本或本地下载。
