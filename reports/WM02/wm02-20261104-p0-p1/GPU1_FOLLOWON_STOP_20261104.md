# GPU1 后续任务停止纠正

2026-11-04，Asia/Singapore。用户反馈GPU1仍占用后，确认原727048已退出，但原M13管理器仍在运行；它在GPU1空闲稳定120秒后自动派发了PB317 boundary后续任务1425203。此前空闲显存是时点观测，未同时关闭后续派发，这是上一轮准备工作遗漏。未把后续任务误称为727048重启。

本轮先核对管理器3892803／proc_start339165562和worker1425203／proc_start350076447的uid、完整worker路径、unit及manifest；两者均属于原项目PB链。管理器无STOP文件检查，Restart=no，因此通过现有Journal记录后发送一次SIGTERM；独立确认管理器终态后，再用worker原STOP.json＋一次SIGTERM。没有SIGKILL、源码热改、检查点删除或WBC改动。

随后只调用原M13的`close_known_failure`收尾函数，通过已有onceIO关闭boundary原生lease并核对canonical成本；没有运行其`run()`或重新派发。原worker状态名保留`FAILED_BOOK_FINITE_OPTIMIZATION / TimeoutError`，外部原因明确为用户停止的partial，不能计作新的算法失败。管理器原STATE／SUMMARY在Journal保留前值后记为`USER_STOPPED`，不是科学完成。未触碰独立consumer08或他人GPU0工作。

02:57:25 SGT独立复核：

- 727048、1425203、3892803均不存在；三个unit均MainPID0、Restart=no。
- control与boundary两个worker lease均CLOSED；管理器STATE／SUMMARY均USER_STOPPED。
- GPU1空闲15899/16376MiB、利用率0%、无GPU1计算进程。距离管理器停止约167秒、worker停止约141秒，超过原管理器120秒空闲稳定窗口。
- 原control的1024／2048完整checkpoint摘要未变。boundary停在23更新／736次TRAIN访问，没有新的完整checkpoint；原父模型和训练原件均保留。

boundary执行源码为`ccfa2f2e87b86886030c6dd282dbfece5dfe4746`，manifest为`d0b30eafd593db973c8a8441bd18d5f0b28e9897c11294ca48cc70bef6463363`。worker报告CPU989.027621秒／墙钟438.298139秒；原native canonical结算CPU996.466470秒、GPU占用上界449秒，包含worker启动等服务开销，两组不相加。均归原PB实验，不挪作WM02成本。本次新增WM02转移／模型更新／任务／PBS均0。

真实操作命令（脱敏私有根别名，完整绝对argv在执行者私有命令表）：

```bash
python3 "$WM01_PRIVATE/core_control.py" "$WM02_PRIVATE/stop_pb_manager_once.py" 40
python3 "$WM01_PRIVATE/core_control.py" "$WM02_PRIVATE/stop_pb_boundary_once.py" 40
python3 "$WM01_PRIVATE/core_control.py" "$WM02_PRIVATE/close_pb_boundary_original.py" 55
python3 "$WM01_PRIVATE/core_control.py" "$WM02_PRIVATE/finalize_pb_manager_user_stop.py" 40
python3 "$WM01_PRIVATE/core_control.py" "$WM02_PRIVATE/verify_pb_chain_terminal.py" 40
```

以上均实际exit0。前四项已有完成的Journal／onceIO，不重放停止或关闭；最后一项是下一条可执行的只读复核命令。证据摘要、模型／数据身份和helper SHA256见 [GPU1_FOLLOWON_STOP_20261104.json](GPU1_FOLLOWON_STOP_20261104.json)。GPU1空闲仍不授予目标动态许可；既有S1/T2隔离不变，未宣称长期独占reservation。
