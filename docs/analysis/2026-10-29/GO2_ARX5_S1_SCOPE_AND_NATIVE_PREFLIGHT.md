# S1批准、十天计划衔接与实际冷态核对

**后继实际：** [完整14维重算与原入口封存方案](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_ENTRY_AND_FULL14_PREPARATION_20261029.json)已用新原生全文hash/大小核对复用的完整GLOBAL/LOCAL字节，再调用原`first512_forecast`保留所有旧工作及matrix/final。原传输池仅余26,335字节；用户批准的是新S1最多8GiB传输，作为仅S1可用的追加范围单独记账，不用它偿还旧工作或扩大十天计划。旧14维义务均在原cap内，加入该S1范围后无预测缺口，尚未实际预约。总4CPU核时分为Core最多13,800核秒＋PC最多600核秒，未双重增加PC额度。

已定位同字节的本地XML候选。原`enter.sh`、`doctor.py`和完整profile配置可原样放进新run的`entry/infrastructure`，其既有相对路径规则自然把工作目录解析到`entry/code/vla`；无需重写入口或改变全局profile。新增组件只准备这一封存布局、验证原registry精确readback及真实挂载类型。有限入口资格程序仅调用原入口的解释器/工作目录探针，通过原onceIO登记原canonical registry并关闭自己的资格lease；不能启动Teacher、solver、simulator或模型。实际执行和完整原生worker资格另行核证。

两次完整原件只读请求因回复上限终止，未重发原操作；原CPU/传输上界全额保留，只有这两条已知只读操作的状态经原onceIO核账，其他UNKNOWN未动。随后用小型全文身份读数核对已有完整缓存，避免再次传回大原件。

用户已直接回复“批准该 S1 有限开发范围”。[授权原件](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_HUMAN_APPROVAL_20261029.json)固定原PR #143提案及全部上界，不重复申请同一许可。登记采用为2026-10-29 19:31:39.836045 SGT，24小时窗口于10月30日同一时刻结束；原stop-new、硬截止、历史费用和保护资源保持。

PR #144的十天计划已从main `672e1858369ee3b4ca9ebd725800a4dd3f15ef67`正常合入本工作分支。S1批准的4次上界只是TD06的一部分，最多2次完整确认不等于其8条件资格。十天新增训练、查询、容量和11月延期仍未采用。TD00/TD06标RUNNING仅表示本页的实际只读核对，未声称已启动物理。

## 当前原生事实

[SCOPE_AND_CAPACITY.json](../../../reports/PI0294/design/ten-day-policy-20261029/SCOPE_AND_CAPACITY.json)记录各自观察时刻与原件hash。Core当时没有匹配的目标物理进程或GPU计算进程，但旧task lease仍写ACTIVE且已过旧截止；未重开、删除或把它当新权限。D25独立读数同样显示服务不存在/主PID为0，而原lease仍ACTIVE；不能仅凭GPU空闲宣布原owner资源已释放。

原代码、解释器、模型、native库、clock/pause二进制均能在节点本地定位。旧配置的入口、registry、lease和XML资产仍指向共享存储。用已有ROS/ELF绑定顺序重新做只读解析后，三项冻结程序全部无缺库，107个实际解析库位于ext4；九个Python模块实际导入成功，CPU 0.196秒。默认未绑定环境下的ldd缺库不等于需要安装软件。本次没有安装、编译、基础设施修改、运行迁移或原生账本写。

下一步是冷态绑定新scope的节点本地入口/profile、已登记输出与本地控制/lease/onceIO，以及精确原资产。继续复用原registry和owner；不新建竞争登记系统。完整14维余额及保护资源尚未在本收据闭合，不能声明native admission PASS。

## 更晚Teacher结果必须覆盖陈旧“正在运行”摘要

前面的12次清单只覆盖三个早期系列，不是全部目标历史。已复用的[SDR3独立接触复算](../../../reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/source-demo-access-v3/TASK_CONTACT_RECONCILIATION.json)确认285个受力tick、累计0.57秒、只有右pad；没有抓持或成功，原关节加速度失败保留。因此不能概括为所有版本“从未接触”。

通过原registry定位TF-R3并读回四份原RESULT：midpoint、reactive V1、E1均为规划失败；E2运行源码`6af24da77cc11951366acd07c75939c9648a7fdb`在18,847积分处原ILLEGAL_CONTACT安全停止，654观测/82安装intent。旧REACTIVE_PROGRESS中的“E1正在运行”已被其真实终态覆盖。原安全失败版本不重跑，完整任务成功仍0；不把这四条与早期不同人口合并算策略成功率。

## 四段新参考的保存状态复核

复用已有TF-R3的500Hz Cartesian插值、最多5步局部IK及原0.7rad/s、1rad/s²界限，检查四个已经封存的候选，不新增规划或物理。使用独立MjData，base固定在保存构型，仅用于运动学复核。

| 参考 | 时长s | 500Hz点数 | 最大关节速度rad/s | 最大关节加速度rad/s² |
| --- | ---: | ---: | ---: | ---: |
| T1接近 | 6.15 | 3076 | 0.22435 | 0.31497 |
| T2接近 | 11.20 | 5601 | 0.19987 | 0.27555 |
| T1下降 | 2.30 | 1151 | 0.22802 | 0.46707 |
| T2下降 | 2.75 | 1376 | 0.27209 | 0.46269 |

四段均通过保存构型下的全模型接触/限位复核，CPU共5.936秒；[逐段首结果](../../../reports/PI0294/design/parallel-science-20261022/GO2_ARX5_DENSE_SAVED_REFERENCE_CHECK_20261029.json)保留。两任务接近都使用原V2 W0/T1实际首态；下降起点是明确的未来计划构型，不是新反馈。此结果排除了这些固定参考在该保存构型下的原速率/局部IK问题，不能证明自由机身动态跟踪、原生安装消费、真实接触或成功。新物理启动、GR00T调用、训练和PBS均0，任务资格仍待实际验证。
