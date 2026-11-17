# V5.2 complete24 native result

**17/24真实移动抓持综合成功；无合格晋级候选。** 终态`2026-11-18T02:05:02.569483+08:00`。全部事前固定24条件完成，未按前缀淘汰、未换seed重投、未重训已完成B1/B2。暂停仿真成绩不等于连续350ms实时或真机资格。

| Arm | 综合 | F1 | F2 | H40拒绝 | 接触STOP | 技术失败 | 晋级 |
|---|---:|---:|---:|---:|---:|---:|---|
| B0 | 5/8 | 2/4 | 3/4 | 2 | 1 | 0 | FAIL |
| B1 | 6/8 | 3/4 | 3/4 | 2 | 0 | 0 | FAIL |
| B2 | 6/8 | 2/4 | 4/4 | 1 | 1 | 0 | FAIL |

| 条件 | B0 | B1 | B2 | B2−B1 | B1−B0 |
|---|---|---|---|---:|---:|
| F1/T0 | HARD_STOP | COMPOSITE_SUCCESS | COMPOSITE_SUCCESS | +0 | +1 |
| F1/T1 | H40_REJECT | COMPOSITE_SUCCESS | COMPOSITE_SUCCESS | +0 | +1 |
| F1/T2 | COMPOSITE_SUCCESS | COMPOSITE_SUCCESS | H40_REJECT | -1 | +0 |
| F1/T3 | COMPOSITE_SUCCESS | H40_REJECT | HARD_STOP | +0 | -1 |
| F2/T0 | COMPOSITE_SUCCESS | COMPOSITE_SUCCESS | COMPOSITE_SUCCESS | +0 | +0 |
| F2/T1 | COMPOSITE_SUCCESS | COMPOSITE_SUCCESS | COMPOSITE_SUCCESS | +0 | +0 |
| F2/T2 | COMPOSITE_SUCCESS | COMPOSITE_SUCCESS | COMPOSITE_SUCCESS | +0 | +0 |
| F2/T3 | H40_REJECT | H40_REJECT | COMPOSITE_SUCCESS | +1 | +0 |

B2−B1总成功差0，F2+1、F1−1，hardSTOP+1；B1−B0总成功+1且F2差0。n=8开发比较，不作显著性结论。冻结B0已在本轮F2取得3/4，故旧PR403的F2 0/4保留且不能把新F2改善全归因于新增训练。

## 实际消费、成本和拒绝

24个新请求共**481233真实2ms积分**，守恒`24×5703 + 13773×25 + 36 = 481233`。两个旧接口11906tick仅复用，原E2胶囊的493139总计包含它们，本报告未再次计费。1384次真实模型生成与全部保存request/return/action身份对账；1379完整H40通过、1377完整P10，5个拒绝H40未安装。原结果误扫旧queries路径的0生成字段保留，真实分母来自shift7-queries，不热改原胶囊。

两个接触STOP（B0/F1T0、B2/F1T3）均0后续积分、0 trace错误。5个原H40拒绝：B0/F1T1、B1/F1T3、B0/F2T3、B1/F2T3为首端；B2/F1T2为内部index34=0.261575>0.25且首尾合法。F2/T3两个首端拒绝均在query16/8s，未接触即终止；保存源预测与margin见E2_FINAL_REJECTIONS.json。没有放宽完整H40、P10、TTL、STOP或评分。

各episode的抓持、抬升、保持、方向位移、EEF RMS、ticks/partial、首失败及result SHA均在E2_FINAL_ACTUAL.json。hardSTOP的旧detail未封存lift/hold，保持null，不补造0；其已记录完整原子中的抓持数为0。实际post-init差异按qpos/qvel/ctrl保留；相同scene/reset/noise不代表完整WBC动态状态相同。

控制器wall 8180.76s、模型worker驻留wall 8173.20s，24case CPU合计4041.07s。worker实际JAX live峰7,739,976,960字节；宿主受管服务最后观测峰下界20,475,625,472字节，退出后终态cgroup峰未恢复。未测GPU busy seconds，不把驻留wall称为纯GPU计算时长。E1两个500/16000端点及792离线生成成本见E1_PAIR_ACTUAL，不重复训练或重复计费。

## 固定决定与条件后继

B0仅5/8、F1仅2/4且有STOP；B1虽6/8、两族各3/4且0STOP，但固定旧拒绝仅修复1/3；B2虽6/8、F2 4/4且旧3拒绝全修复，但F1 2/4及STOP阻断。唯一确认候选为null，确认/DEV/TEST均0，V5.1旧pilot_GO=false不回填。

完整24后按固定开发排序得到P=B1（开发身份，不是部署资格）。本轮7次失败中5次为H40拒绝；P的2次失败都为接触前首端拒绝，尚无同完整状态专家证据证明P存在接触时机或学生覆盖缺口。E3触发未成立：不以扩纠正数据掩盖主要边界问题，E3/C0/C1均不启动。若后继改变这一研究假设，必须有新的具体状态诊断；不是重复本面板或自动追加训练。

原服务自然exit0，MainPID0、followup/E2/worker leases均CLOSED，世界及神经锁均释放，Restart=no且0重启。这里exit0只表示有界队列完整结束，不是策略过门。275个实际运行源码文件与C20统计重新核验不变，site binding SHA匹配，原期限未续。无前台依赖、无活动任务需要抢占。

实际native源码`2797d0c2c458667e241aa0d24d6d0b005f3caf1a`，E2结果SHA`8d7a0fcd2500a8b4fe85d9ec877295f8300e24af77835213449b8891cafe99d9`；full-state训练源8ebdcbb7与两个模型SHA见JSON。WBC/腿/MPC/solver/臂PD及历史RP8/JOIN10/R3/旧DEV/TEST、PR403原失败完整保留。
