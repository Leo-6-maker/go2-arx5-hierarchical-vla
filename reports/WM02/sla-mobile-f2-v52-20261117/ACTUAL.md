# V5.2 actual — full24 terminal, no promotion

终态 `2026-11-18T02:05:02.569483+08:00`：**24/24真实MuJoCo请求完成，17次移动抓持综合成功、5次完整H40拒绝、2次接触STOP、0技术失败。无合格晋级候选。** 完整逐条件、抓持/抬升/保持/移动、成本及固定决定见 [E2_FINAL_ACTUAL.md](E2_FINAL_ACTUAL.md) / [JSON](E2_FINAL_ACTUAL.json)。两个STOP后均0积分。

| Arm | F1 | F2 | 综合 | H40拒绝 | STOP | 晋级阻断 |
|---|---:|---:|---:|---:|---:|---|
| B0冻结SHIFT7 |2/4|3/4|5/8|2|1|总数/F1/STOP/旧拒绝 |
| B1普通FM500 |3/4|3/4|6/8|2|0|旧3拒绝仅修复1/3 |
| B2 endpoint500 |2/4|4/4|6/8|1|1|F1及STOP |

B2对B1总体成功差0，F2+1但F1−1且STOP+1，没有证明总体闭环收益。B1对B0成功+1、F2差0；n=8仅开发比较。冻结B0也在本轮F2成功3/4，旧PR403 F2 0/4独立保留，不能将新条件成功全归因于额外训练。

E2真实新增481233个2ms积分、1384次模型生成、1379个合法H40、1377完整P10。积分守恒 `24×5703+13773×25+36=481233`。旧接口11906tick只复用，未重复计费；旧结果误扫queries路径的0生成字段未改写，实际分母逐项对账shift7-queries的请求/返回/动作SHA。5个拒绝未安装，首/尾/内部量化见 [E2_FINAL_REJECTIONS.json](E2_FINAL_REJECTIONS.json)。

| 阶段 | 已执行 |
|---|---|
| D0 |320输入/319动作、1376 C20标签、8专家550计划；0新生成/积分 |
| 训练前独立人口 |36布局簇DEV32/TEST40、205308次被动FK；0新动力学 |
| B1/B2 |各500真实更新/16000访问；87状态实际GPU重载逐位一致、28训练数组变化 |
| 独立普通RGB恢复 |三臂各16/16逐位复现，不加载优化器 |
| TRAIN96 / CN38 |三臂各96/96与180/180；每臂20+36 CLOSE无遗漏/超100ms |
| ROLL96 / 旧3拒绝 |93/96、93/96、96/96；修复0/3、1/3、3/3 |
| E2 |完整24及固定否定决定已封存 |
| 确认 / E3 / DEV / TEST |均0；原V5.1 pilot_GO=false保持 |

E1合计新增1000更新/32000优化访问、另2次诊断反传/8窗口、792离线生成；端点身份、匹配与费用见 [E1_PAIR_ACTUAL](E1_PAIR_ACTUAL.md)、[离线门](E1_OFFLINE_ACTUAL.md)。没有重复D0或重训端点。

完整24后开发排序P=B1，只是开发身份，不是部署许可。7次失败中5次为H40拒绝，P的两次失败均为接触前首端拒绝；没有同状态专家证据证明P的接触时机/覆盖缺口。[F1查询诊断](QUERY_DIAGNOSTICS_PREFIX_ACTUAL.md)也未发现反复将CLOSE推向未来。E3证据触发未成立，不借扩量掩盖主要边界问题，C0/C1不启动。无确认候选，DEV/TEST保持未打开。

受管服务 `wm02-sla-mobile-f2-v52-e1-e2-followup-20261117-02` 自然exit0；MainPID0、followup/E2/worker lease均CLOSED、世界与神经锁均释放。exit0表示固定队列结束，不是策略过门。实际275个源文件、冻结C20统计及site binding SHA重新核验；旧期限1794967938.2895093未延长。退出后没有任务依赖前台继续，也没有重启或抢占其它任务。

固定计划 `90d39196bdc345b977ee54ed64e750cf3561aa72`；审阅PR403 `dbf222b41cb666bff4dd5412b864d023aec69399`；实际训练源 `8ebdcbb7ab39c0f707c5b2ba94b0b097abc937e5`；实际推理/native源 `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`。SHIFT7父包 `ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb`；统计 `f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e`。E2 sealed result `8d7a0fcd2500a8b4fe85d9ec877295f8300e24af77835213449b8891cafe99d9`。

历史前缀 [F1_ACTUAL](F1_ACTUAL.md)、[E2_PREFIX](E2_PREFIX_ACTUAL.md)保留各自时间，不再代表现役状态。两个零更新环境错误及 [native解释器冷修复](E2_PROFILE_COLD_REPAIR_ACTUAL.md)均保留；条件评价代码仅CPU检查，未部署。WBC/腿/MPC/solver/臂PD、完整H40/P10/TTL/2ms、碰撞/STOP、原评分和全部旧RP8/JOIN10/R3/DEV/TEST失败保持。无真机、RL或动作加工；暂停仿真成绩不等于连续墙钟实时资格。
