# D25本地官方代码：保留原Git检查

前次[80文件暂存](D25_NODE_LOCAL_INPUT_PREPARATION.md)只有工作文件。原冻结的`train_adapter.official_import`仍要求官方HEAD、clean diff和四个指定文件的`git show`字节检查；因此该暂存尚不能作为原入口的完整官方源码。不会修改这项检查或任何15个冻结数值/物理/评分文件。

已只读取得原固定`51d4c89f72fda44cbf77285c6a8114b52676b8a1`的commit和全部tree对象，共141对象/27271原始字节。重算每个Git对象SHA并沿commit→根tree→全部子tree验证：原树440条受跟踪路径，完整`gr00t/`包加LICENSE/pyproject共80文件，与前次暂存blob身份完全一致；另360条文档/示例/子模块等路径不检出。四个gitlink只保留原目录项，不初始化/下载子模块。原读取首次把gitlink误当tree而停止，第二次返回包超过32KiB上界；两次只读失败与完整费用保留，后继在明确有限回包上限内完成。

`pinned_sparse_source.py`从这些真实原对象建立**仅该既有代码包目录**的Git元数据，用原`read-tree`索引及`skip-worktree`标记未检出路径。全部80个运行成员必须为H且无assume-unchanged；360条省略成员为S。80成员完整SHA和Git mode均取自同一原提交；其中14文件恢复原100755的执行位，内容保持不变。不会在整个人类工作区初始化仓库，也不复制原Git配置、remote、认证或notebook修改。

这使用Git的[稀疏工作树语义](https://git-scm.com/docs/git-update-index#_skip_worktree_bit)和[原tree读取](https://git-scm.com/docs/git-read-tree)。真实临时仓库检查已确认clean diff，并证明修改一个已包含文件会被原Git diff检出；篡改对象、漏tree、换路径和非法tree条目均被拒绝。合计5项新检查与10项既有包字节检查通过，尚不等于Core原入口资格。

下一次一次性准备只在原未启动owner的既有`cold-inputs/official`下添加Git元数据。沿原Journal/同一有限计数器，先验证原manifest/LOCAL/请求、原ACTIVE lease、两个unit MainPID0、前次83成员完整字节，再写独立intent。Git元数据及见证峰值≤3MiB，准备+独立验证Core CPU≤10秒，PC控制≤20秒，控制传输≤32KiB；均须先核原未用bank，终态/收尾保护不动。目录已有或效果未知时只读取，不再初始化或重放。所有Git配置只作用于这个新代码包元数据；现役环境、旧capsule、原官方checkout及系统配置均不动。

独立验证必须调用**原冻结源码**中的`official_import`，验证实际HEAD、clean diff、四文件`git show`及80成员原blob、全部141个元数据对象，随后核旧LOCAL和两个unit仍未启动。该调用只做CPU源码身份检查，零GR00T模型/查询/训练/物理；不把成功导入资格写成运行或策略结果。登记custody、本地lease、全依赖冷绑定与原24评价仍是后续工作。

实际一次准备及独立验证已完成：[原冻结函数实际PASS](../../../reports/PI0294/design/parallel-science-20261022/D25_FROZEN_OFFICIAL_LOCAL_GIT_NATIVE_PASS_ACTUAL_20261030.json)。80个代码文件全部H、360省略路径全部S，214个真实Git对象重新hash，15个冻结文件原字节一致。Git元数据实际1,675,264字节；两次调用实际Core CPU 0.298005864秒、PC CPU 0.028172181秒、控制传输27,550字节。原LOCAL/配置及两个未启动unit保持，模型/查询/物理均0，D25仍0/24。登记与完整冷绑定尚未采用。精确管理程序见`sparse_source_management.py`，私有参数和原始输出不公开。
