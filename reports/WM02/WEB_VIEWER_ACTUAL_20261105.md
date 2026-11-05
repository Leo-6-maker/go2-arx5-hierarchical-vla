# WM02 浏览器显示：实际交付

日期：2026-11-05，Asia/Singapore。基线 PR #359 / `a304566255f1211fb366ee38a381acaa157bd09b` 尚未合并，独立源码 checkout 从该固定提交建立，没有假设 main 包含计划或热切现役 source。

实现与实测代码：`452bd03288ee3f1dff9c8bcfb9342d79fc2be622`。新增本机网页 `http://127.0.0.1:8766/`，复用已有只读 SSH 通道与实际 MJB，服务端使用 MuJoCo 原生完整模型渲染，浏览器显示 JPEG、状态及相机交互。[完整使用方式](../../docs/operations/WM02_WEB_VIEWER.md)。没有新模型/checkpoint、原始数据集或动态训练批次。

后续显示身份修复：`90d3e085cddd41f42384d5d1be06130b0a477057`。网页元数据与JPEG响应携带同一观察会话ID；服务重启时清除旧图，拒绝晚到的旧会话帧，避免不同批次重复frame_id导致沿用旧图。最终相关17项测试在普通/-O各再次通过。此修复仅涉及显示身份，不修改采集数据。

## 实测结果

| 项目 | 实际结果 |
| --- | --- |
| HTTP首页、状态、画面 | 均200；实际 JPEG 130,308字节 |
| 可见浏览器 | 本机in-app浏览器实际打开；显示完整Go2＋ARX5、桌子与物体 |
| 视角交互 | 浏览器点击侧面→默认视角，camera实际90°→135°，渲染frame_id 1→3；物理步始终12503 |
| 状态和数值 | SEALED；t25.006s；TCP约(+417.6,−67.2,+576.5)mm；双pad未接触；实际阶段WITHDRAW/撤离 |
| 浏览器错误 | QA检查时脚本error日志0 |
| 断连检查 | 主动停止本机QA服务后，页面显示“数据连接中断”，保留最后真实画面 |
| 相关测试 | 普通Python与-O各17项通过，包含只读通道及HTTP相机/源信息/错误来源/无控制与资产路由检查 |

有限QA进程实际接收 **2781条快照、渲染3帧、实际状态前进0**；帧复用是因为绑定的真实批次已封存，相机只发生两次变动。新物理积分/真实转移/控制启动或发送/WM与Actor更新/完整任务均0。不是2781条新转移，也未测得12 FPS活动目标运动吞吐。

QA服务用Ctrl+C等效SIGINT主动结束，CLI退出0；只读SSH因本机主动结束返回255，`observer_stop_requested=true`，保留在原收据中。该退出不是目标仿真或WBC失败。页面通过实际断连检查后，将另起一个有限1小时的用户查看进程；后者为在途观察应用，其未来计数/成本不混入上述已结算QA。

## 源码、场景与数据身份

- 原生被观察源码：`b18573a67bec91c1edb1d5b93a09dbb138cea3ed`。
- Episode：`next2-c2-grasp25-supported-01`，worker lease保持CLOSED。
- 场景MJB SHA256：`55d95256f4523d0dac24c6af5517723afe372659aa724608f38a9648e394f921`。
- 原spec SHA256：`bad50e1f11e525dcfd06ca05b7127ee294c96a94432c014544b6b006ac6dd5b3`。
- 原状态包 SHA256：`ef868bdaa7a469776d7e103cc367018c498990eb0b0acf133dab68afd5ba0e77`。
- 原冻结执行器 SHA256：`29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2`。

QA结束后的独立Core只读复核：原状态/spec SHA与上述值一致，worker仍CLOSED，QA观察PID已不存在。

浏览器仅改变本机显示相机，不改模型/物理/控制。原始包、模型、供应商资产、主机入口及私有配置没有新增公开上传。界面截图和展开后的真实命令留在本机私有交付目录。

## 命令、成本与接续

实际QA命令（私有已绑定变量，未虚构不存在的接口）：

```bash
PYTHONPATH=src:. "$WM02_VIEWER_PYTHON" -W error::ResourceWarning -m unittest \
  tests.world_model.test_native_web tests.world_model.test_native_live \
  tests.world_model.test_native_replay tests.world_model.test_replay_feed
PYTHONPATH=src:. "$WM02_VIEWER_PYTHON" -O -m unittest \
  tests.world_model.test_native_web tests.world_model.test_native_live \
  tests.world_model.test_native_replay tests.world_model.test_replay_feed
PYTHONPATH=src:. MUJOCO_GL=egl "$WM02_VIEWER_PYTHON" -B -m scripts.wm02.serve_native_live \
  --config "$WM02_PRIVATE/READY_CONFIG.private.json" \
  --output "$WM02_WEB_PRIVATE/browser-qa-01" --port 8766 --max-wall 300 --fps 12
```

上述命令退出0。QA实际wall233.625007s、本机进程CPU4.225579s；源端最后心跳CPU采样0.741987s（非完整全站成本）。渲染GPU、浏览器自身、开发与测试总CPU/功耗未完整计量。未新增训练GPU作业或安装依赖。实现复用了现有MuJoCo/Pillow/标准库HTTP，前端没有外部CDN或额外框架。

下一条可执行命令为相同服务入口，使用新输出目录和`--max-wall 3600`。现有CLOSED批次将继续显示封存末态；新活动批次须先绑定其实际spec。该本机网址供当前电脑访问，远端访问与云端发布不在本次交付中。
