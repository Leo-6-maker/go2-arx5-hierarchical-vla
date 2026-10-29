# D25原固定评价：节点本地输入准备

原D25@1024、24个N294101条件、请求/nonce、源运行`2a23eeb6`及费用/时钟保留。当前有效评价仍0/24。[只读实测](../../../reports/PI0294/design/parallel-science-20261022/D25_NODE_LOCAL_INPUT_PREPARATION_READONLY_ACTUAL_20261030.json)确认模型、两peer/owner、解释器和数据在本地盘；入口、登记/租约、部分资产及官方源码仍引用共享Home。原owner/neural unit没有运行。本次不把未知写或观察超时当作重启依据。

已逐字节核对26个原固定资产与现有节点本地副本，全部相同，无需再复制机器人资产。原GR00T固定commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`的`gr00t/`包、LICENSE和pyproject.toml共80文件/753587字节。仅此小包待暂存，不带原checkout的Git配置、凭据、私有修改、notebook或模型；既有环境不安装/升级。

计划及本次实际：在现有已登记且未启动的owner输出下，新增独立冷输入子目录，用原Journal和有限控制计数器记录一次写入。先验证原owner manifest、ACTIVE lease、原LOCAL未启动行、两个unit MainPID0、原entry三文件hash和固定Git对象清单，再创建目录；若目录已存在或写入结果未知，只独立读取，不重发。固定Git对象由一次`cat-file --batch`取出，全部验证header、长度和Git blob SHA，再O_EXCL写入并同步。`scripts/pi0294/pv1/local_runtime_inputs.py`只验证字节，不执行Git、第三方导入或启动；10项CPU检查覆盖篡改、截断、多余内容、私有路径和symlink拒绝。

本次小暂存上限：官方包≤1MiB、三个原entry文件≤32KiB、见证/目录≤1MiB；Core准备CPU≤10秒（写入和独立回读各≤5秒）、PC控制≤20秒、SSH封装总传输≤32KiB。使用原已付64MiB控制及128MiB Core暂存范围，执行前核实际剩余各bank与CPU；终态和收尾bank均不动。两条后继只读失败的状态和完整费用上界保留，仅以保存的只读intent/transport证明无写效应。此前管理安装UNKNOWN已有精确无目标文件/无原actor的独立证据，旧Journal不改为成功；新的小暂存必须使用不同明确操作和Journal，仍接同一原计数器，不重发旧管理安装。

暂存后独立读取每个文件的完整字节和SHA、原entry、写入intent/见证以及两个unit，确认旧manifest/LOCAL不变。准备尚不接入运行：entry的代码工作目录、源/资产绑定、原登记的节点本地custody与可变lease、全部库/缓存/temp/输出及完整剩余资源必须在后续冷边界明确采用。不能复制一份registry冒充原权威，不能热改sealed04，不能在依赖未闭合时先开模型。

本步骤新增模型/查询/训练/物理/PBS均0，不重传D25，不改冻结数值/评分、NFS/Home/挂载/SSH/网络/系统服务。此准备计划没有新增运行额度。目标仍是原完整行为评价，暂存和CPU检查都不是策略结果。

实际已完成：[原版输入暂存与独立回读](../../../reports/PI0294/design/parallel-science-20261022/D25_ORIGINAL_CODE_AND_ENTRY_NODE_LOCAL_INPUT_STAGE_ACTUAL_20261030.json)。80个原官方文件/753587字节和3个原entry文件已写入节点本地，83成员全文SHA一致；intent、见证和结果原件已独立核对。Core实际CPU0.060秒、PC控制CPU0.034秒、封装传输26702字节；原计数器两次操作CONFIRMED，终态/收尾bank仍0。两个原unit MainPID0/not-found，旧配置和LOCAL字节不变。准备阶段目录创建及回包预测的本地拒绝均先于远端派发，修正后才执行一次新stage。精确管理程序已保留在`local_runtime_input_management.py`，本机私有控制参数未发布。该准备不代表D25已运行，原0/24保持。
