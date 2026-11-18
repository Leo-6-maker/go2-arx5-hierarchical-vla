# D3 并行工程 E1 实绩

2026-11-18 19:36 SGT 独立冷态读回。E1 的 LEGACY_N1 **8/8** 短卡均已完整封存，共 **85,624 原生 tick**，0 接触 STOP、0 技术失败。每条含 5,703 初始化 tick，后续 5,000 tick 使用原固定 150 ms 处理窗。实测管理服务墙钟 1,136.09 秒，使用 CPU 10–11、软件渲染。关闭时 PID 0、cgroup 空、lease CLOSED。

来源 `49750913631781deaf8432d968ebae6e1bdc99fa`，spec `1d02cc27be0c54027dc3f6fd1de58062b413ffca114fcc480aff1fd86901cb88`。旧 E 与科学条件均未重投。E2 复用原登记容差，E3 检查双方在 7,203 tick 真正到达屏障后才发单 world STOP。E2／E3／E4 当前派发均为 0，N2 科学后端资格尚未建立。

[机读实绩](D3_PARALLEL_ACTUAL.json)
