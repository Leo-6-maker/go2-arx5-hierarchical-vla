# WM02 只读实时接入：实际交付

日期：2026-11-05，Asia/Singapore。基线为未合并 PR #358 的公开提交 `40e2aee587a63822c78c46ffa841224abc6f9192`（与本地 `c08699c4058c357fb9d6ae2e931acc47118005a1` 同树）。本次独立源码 checkout，不热改现役 capsule，不启动目标物理或训练。

## 已实现和实际验证

新增 `prepare_native_live` 绑定实际批次并缓存原模型，`watch_native_live` 经既有 SSH 路线接收 Core 当前 `state.bin`，在本机 MuJoCo 原生窗口显示完整 Go2＋ARX5、桌子、物体、实际 TCP、原生 pad 接触和当前阶段。仅保留最新快照，显示暂停、断流、流结束和封存状态。实现与使用命令见 [操作说明](../../docs/operations/WM02_LIVE_VIEWER.md)。

实际实现提交：`3e3d3379506db24019d1a5eae2d6e2def49b7c18`。有限观察进程退出修复及最终测试提交：`ee553937845528e92372e5c5ee18d7fe346770b8`；本次没有神经 checkpoint。

| 实际检查 | 结果 |
| --- | --- |
| 初次 Core→本机离屏显示，12s 上限/10 FPS | 退出0；107快照、90渲染帧，正确标 SEALED |
| 冻结实现离屏检查，10s/10 FPS | 退出0；94快照、57渲染帧，SEALED |
| 本机桌面原生窗口，10s/15 FPS | 退出0；136快照、138显示帧，SEALED |
| 最终有限源自然退出检查，6s/10 FPS | CLI与SSH均退出0；60快照、35渲染帧，0强制停止 |
| 实际批次绑定及场景缓存复用 | 退出0；第二次模型传输0字节，原文件不变 |
| 最终相关测试 | 普通 Python、`-O` 各13项通过；普通模式将 ResourceWarning 作为错误 |

四次实际显示合计397条快照、320显示/渲染帧，均来自同一个真实末态 step12503/t25.006s；**实际新转移0、状态前进0、新物理积分0、控制启动/发送0、WM/Actor更新0、完整任务0**。合成测试验证了递增状态的 LIVE、暂停、断流、身份/序列/数据损坏拒绝；它们不是活动目标原生验收。

当前实验已封存，因此尚未在新活动采集期间验收持续运动显示。下一次获准采集只需绑定它自己的实际 EXECUTION_SPEC，不能复用旧 CLOSED lease 或把这次静态显示写成采集运动 PASS。

## 原始身份与保持核验

- 被观察原生代码：`b18573a67bec91c1edb1d5b93a09dbb138cea3ed`，episode `next2-c2-grasp25-supported-01`。
- 原执行 spec SHA256：`bad50e1f11e525dcfd06ca05b7127ee294c96a94432c014544b6b006ac6dd5b3`。
- 同批次场景 MJB SHA256：`55d95256f4523d0dac24c6af5517723afe372659aa724608f38a9648e394f921`；74,468,695字节，首次压缩传输26,355,523字节。
- 原生执行器 SHA256：`29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2`。
- 实际末态包 SHA256：`ef868bdaa7a469776d7e103cc367018c498990eb0b0acf133dab68afd5ba0e77`；前后独立读取一致。
- 后续只读检查确认原 worker lease 仍 CLOSED、spec 不变、三条登记观察 PID 均不存在。未修改 WBC、原 native binaries、ROS、安全、物理步长、原有数据或 lease；未新增服务/网络端口。

模型必须使用原同批次编译 MJB，MuJoCo3.2.7、ABI34/32/20、无插件/控制 callback。显示端只在独立 MjData 上 forward/sync，三种 Python 积分入口被禁止，本轮调用尝试0。接触信息取原生包；重建几何不构成完整控制状态恢复或新的接触资格。

## 命令与退出

完整展开路径、私有配置、SSH argv、收据和截图保存在本机私有交付目录；公开仅使用已绑定变量，不公开主机入口、供应商资产和原状态包。

```bash
PYTHONPATH=src:. "$WM02_VIEWER_PYTHON" -W error::ResourceWarning -m unittest \
  tests.world_model.test_native_live tests.world_model.test_native_replay tests.world_model.test_replay_feed
PYTHONPATH=src:. "$WM02_VIEWER_PYTHON" -O -m unittest \
  tests.world_model.test_native_live tests.world_model.test_native_replay tests.world_model.test_replay_feed

PYTHONPATH=src:. "$WM02_VIEWER_PYTHON" -B -m scripts.wm02.prepare_native_live \
  --command-json "$WM02_OBSERVER_COMMAND" --remote-spec "$WM02_OBSERVED_SPEC" \
  --cache "$WM02_PRIVATE/assets" --output "$WM02_PRIVATE/READY_CONFIG.private.json"

PYTHONPATH=src:. MUJOCO_GL=egl "$WM02_VIEWER_PYTHON" -B -m scripts.wm02.watch_native_live \
  --config "$WM02_PRIVATE/READY_CONFIG.private.json" \
  --output "$WM02_PRIVATE/core-clean-exit-01" --max-wall 6 --fps 10 --headless
```

以上最终命令均退出0。桌面检查使用相同入口，`MUJOCO_GL=glfw`、不带 `--headless`、`--max-wall 10 --fps 15`。桌面和离屏帧数是各自显示计数，不是独立物理样本或吞吐资格。

## 保留的工程现象与成本

早期测试曾提示两个未关闭的子进程输出管道，已显式关闭；最终 ResourceWarning 严格测试通过。早期有限端点在 SSH 启动延迟后主动结束传输，SSH返回255，CLI返回0，原运行未受影响；原收据保留。最后增加3s自然退出宽限，实际自然退出检查 SSH/CLI均为0、无需发终止信号。桌面出现 Wayland 不提供窗口位置的非致命 GLFW 警告，实际显示完成，未改系统显示设置。首次本地远端 ref 名称缺失导致只读 Git 比较失败，显式 fetch tracking ref 后解决，未 reset 或改写历史。

四次显示本机进程 CPU 合计15.233750s，进程 wall 相加41.407996s（其中两次并行，不等于实际经过时间）。三次有计量源端的最后心跳 CPU 合计0.163381s，是源观察进程的采样计量，非全站成本。首次模型压缩/绑定、开发测试、显示 GPU/驱动和功耗成本未完整计量，未编造。训练 GPU作业0；不声称图形渲染仅使用CPU。

下一条可执行命令，在私有变量已绑定且使用新输出目录时打开当前真实封存场景：

```bash
PYTHONPATH=src:. MUJOCO_GL=glfw "$WM02_VIEWER_PYTHON" -B -m scripts.wm02.watch_native_live \
  --config "$WM02_PRIVATE/READY_CONFIG.private.json" \
  --output "$WM02_PRIVATE/desktop-review-NEW" --max-wall 300 --fps 15
```
