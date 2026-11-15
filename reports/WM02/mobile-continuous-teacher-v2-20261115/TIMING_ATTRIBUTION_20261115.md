# 移动抓持专家的H40／P10时序归因

Core原始封存记录只读分析：29个请求、1,265条逐query记录；9.80 CPU秒／30.01 wall秒。0新增物理、0积分、0模型预测、0训练、0认证器调用，未重算接触或修改原结果。

[逐query表](QUERY_TIMING_20261115.csv) · [结构化证据及原结果SHA](TIMING_ATTRIBUTION_20261115.json)

旧专家每次H40终点最多15mm，P9仅消费8.4281%（1.2642mm），P10窗口末端为11.0392%（1.6559mm）。60次P9增量和最多75.853mm；P10增量和99.353mm不能当作连续推进，因为下一query又使用上一row9参考，丢回约0.392mm。它们均不是实际TCP运动上限。

| 原专家条件 | 阶段查询 | 原生跟踪RMS | 终态当前阶段误差 | 原始左右pad接触 |
|---|---|---:|---:|---:|
| F1 M_ASSIST EXPERT | {'PREGRASP': 60} | 20.95mm | 23.58mm | 0/0 tick |
| F1 M_FREE EXPERT | {'PREGRASP': 60} | 25.50mm | 24.46mm | 0/0 tick |
| F2 M_ASSIST EXPERT | {'PREGRASP': 60} | 22.10mm | 16.00mm | 0/0 tick |
| F2 M_FREE EXPERT | {'PREGRASP': 60} | 25.02mm | 26.00mm | 0/0 tick |
| F4 M_ASSIST EXPERT_PREFIX_INITIALIZATION_ONLY | {'PREGRASP': 19, 'DESCEND': 41} | 21.35mm | 32.56mm | 0/0 tick |
| F4 M_FREE EXPERT_PREFIX_INITIALIZATION_ONLY | {'PREGRASP': 19, 'DESCEND': 41} | 21.51mm | 32.52mm | 0/0 tick |
| F4 M_ASSIST EXPERT_PREFIX_INITIALIZATION_ONLY | {'PREGRASP': 19, 'DESCEND': 41} | 21.51mm | 32.42mm | 0/0 tick |
| F4 M_FREE EXPERT_PREFIX_INITIALIZATION_ONLY | {'PREGRASP': 18, 'DESCEND': 42} | 20.68mm | 31.92mm | 0/0 tick |
| F4 M_ASSIST EXPERT_PREFIX_INITIALIZATION_ONLY | {'PREGRASP': 19, 'DESCEND': 41} | 21.30mm | 32.61mm | 0/0 tick |
| F4 M_FREE EXPERT_PREFIX_INITIALIZATION_ONLY | {'PREGRASP': 19, 'DESCEND': 41} | 21.43mm | 32.48mm | 0/0 tick |
| F5 M_ASSIST EXPERT | {'PREGRASP': 21, 'DESCEND': 39} | 21.24mm | 35.15mm | 0/0 tick |
| F5 M_FREE EXPERT | {'PREGRASP': 23, 'DESCEND': 37} | 16.96mm | 36.98mm | 0/0 tick |
| F6 M_ASSIST EXPERT | {'PREGRASP': 21, 'DESCEND': 39} | 21.26mm | 35.13mm | 0/0 tick |
| F6 M_FREE EXPERT | {'PREGRASP': 23, 'DESCEND': 37} | 18.43mm | 36.74mm | 0/0 tick |

F1两条专家均60/60触及15mm上限，实际初始目标距离98.53/98.69mm，30秒后仍差23.58/24.46mm；从未进入4mm实际范围。F2两条也未进入该范围。F4六条在18–19次PREGRASP后真实转入DESCEND，剩余41–42次仍无法到达CLOSE，终态误差31.92–32.61mm。因此阶段转换并非完全不工作；缺失的是足够、连续且能跟踪的接近/下降。

原生接触直接来自2ms journal保存的pad/cube力，沿用0.05N阈值；没有使用forward重算力。真实native_grasp沿用原50ms观测。专家实际参考跟踪误差同时存在，不能据此宣称WBC无法跟踪，也不能认为改时间参数化一定能抓持。

据此条件性执行新MOBILE_CONTINUOUS_TEACHER_V2，最多四条F1/F2专家；只干预专家滚动时间与连续参考边界，原场景、BASE路线、30秒预算、4mm判据、夹爪语义、安全、WBC不变。新F4观测/AFTER_GRASP语义另立版本且不在此四条中启动。旧JOIN10、RP8及F1–F6结果保持。
