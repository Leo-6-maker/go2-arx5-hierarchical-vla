# WM02 本体场景实时显示

实时入口已实现：`scripts.wm02.watch_native_live`。它通过**既有 SSH 路线**在 Core 启动有限、只读的状态观察进程，在本机打开 MuJoCo 原生窗口。采集器、原 owner、lease、ROS、WBC 和物理时钟均不由显示端启动或改变。旧回放入口继续只接受历史记录。

## 绑定一个实际批次

使用现有私有 SSH argv JSON；它的远端命令应为 Core 的 `python3 -`。沿用已核验的转发入口和 host key，不更改 SSH 配置、不安装远程桌面服务、不开放新端口。下面变量由原执行 owner 从实际登记的批次解析，不能用计划中的虚构路径。

```bash
PYTHONPATH=src:. "$WM02_VIEWER_PYTHON" -B -m scripts.wm02.prepare_native_live \
  --command-json "$WM02_OBSERVER_COMMAND" \
  --remote-spec "$WM02_ACTIVE_EXECUTION_SPEC" \
  --cache "$WM02_PRIVATE/live-assets" \
  --output "$WM02_PRIVATE/live-config-NEW.json"

PYTHONPATH=src:. "$WM02_VIEWER_PYTHON" -B -m scripts.wm02.watch_native_live \
  --config "$WM02_PRIVATE/live-config-NEW.json" \
  --output "$WM02_PRIVATE/live-view-NEW" --max-wall 3600 --fps 15
```

第一条只读核对原 spec、冻结模型/执行器 SHA，并将原场景 MJB 缓存在本机；同 SHA 复用缓存。每个批次单独生成配置，不自动追随“最新目录”。两个原有采集 slot 可各开一个窗口，身份不能混流。配置、模型和截图属于私有运行产物，不上传 GitHub。

第二条持续接收 `native-01/state.bin`，本地仅保留最新快照。窗口支持原生旋转/平移/缩放，旁边显示状态、仿真时间、步数、实际 TCP、原生左右 pad 接触和阶段。若日志中存在实际安装确认，另显示已安装 H40 的终点；它不是当前执行 atom 的目标。关闭任一显示窗口只结束观察。无需显示器时加 `--headless`，输出 `latest.private.png` 与有限运行收据。

## 如何识别当前画面

| 状态 | 含义 |
| --- | --- |
| WAITING / WAITING_FOR_ADVANCE | 等待首包，或尚未看到当前连接期间的物理步前进 |
| LIVE | 已看到实际步数前进、数据新鲜，原 worker lease 为 ACTIVE；不代表任务或安全资格 PASS |
| PAUSED | 连接仍有心跳，但原生状态超过 1 秒未前进，或源文件已旧 |
| SEALED | 原 worker lease 已 CLOSED；只展示已封存的最后状态 |
| DISCONNECTED / STREAM_ENDED | 超过 1 秒未收到心跳／观察流结束；保留最后姿态并明确标注 |
| DATA_ERROR / OWNER_UNKNOWN / OWNER_EXPIRED | 数据、身份或 owner 状态异常，不标为 LIVE |

“source age”是源主机自身时钟计算的数据年龄，“receive age”是本机距收到上一包的时间；二者不冒充时钟同步后的网络单向延迟。500Hz 原生物理和显示帧率独立，显示可以跳过中间状态，不插值或外推。原采集的仿真暂停仍如实显示，不能据此宣称真机实时控制。

## 实现与边界

- 标准库观察进程只以只读方式打开原 spec、lease、模型、执行器、状态和命令记录；不导入 MuJoCo/ROS，不持有控制写入口。通过 stdout 发包，SSH 断开或有限端点到达后退出。
- 本机检查 episode/source/spec/scene/executor 身份、消息序列、原生 2ms 时间、完整包、有限数值及接触 geom 范围。步数倒退、同 tick 状态突变或身份串流均失败，不自动换源。
- 使用独立 `MjData`、`mj_forward` 与被动窗口同步；禁用 Python 积分入口并统计任何调用尝试。模型插件和控制 callback 必须为空。显示不恢复 MPC/WBC 隐状态，也不重新计算接触资格；接触显示来自收到的原生记录。
- 本次不启动新的目标物理、WM、Actor、ROS 控制节点或常驻服务。需要新采集时，仍由原 owner 机制登记新的有效有限 scope；不能复用 CLOSED lease。

验收包括暂停/断流、错源/倒退/损坏拒绝、只读文件保持、实际子进程传输、场景缓存复用，以及 Core 已封存原生状态的完整模型渲染。活动目标采集中持续运动的原生验收待下一次获准采集执行，不将历史末态或合成测试当作该项 PASS。
