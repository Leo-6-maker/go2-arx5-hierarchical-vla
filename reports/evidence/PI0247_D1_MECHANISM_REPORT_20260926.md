# PI0247 D1 机制对照报告

**状态：D1矩阵完成，按预注册规则冻结 `R*=S4`。** 没有候选runtime满足全部选择门槛；这不是selected checkpoint，也不是sim成功。

## 固定设计与有效性

- 固定矩阵：R0-final10000 / P-final2000 × 原bridge4 A–D × P1噪声N0/N1 × S4/M4/S16/M16，共64个logical cells。
- 56个新查询，8个精确复用的R0/S4基线；所有64个cell均有效，未发生算术无效或新cell重复。新查询共608次action-head去噪forward。
- M仅在原16个TRAIN窗口逐一证明inactive 10:132的mask全零、编码后全零后启用；物理H40、dt=0.05、原decoder及SafetyGate保持不变。
- 旧PI0246 P2000/S4结果没有替代新cell：旧噪声不等于N0或N1，旧restore manifest与本轮绑定不同，旧记录也没有设备绑定。详见[重复身份审计](PI0247_D1_DUPLICATE_AUDIT_20260926.json)。

## P-final2000的R*门槛

`S4`基线XYZ中位数为43.498 mm，因此候选的90%上限为39.148 mm。

| 候选 | 无效 / REJECT | XYZ中位数 | 配对低于S4 | XYZ p95 | SO(3)中位数 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| M4 | 0 / 0 | 41.172 mm | 8/8 | 60.538 mm（S4: 62.798） | 0.03949 rad | 未达标：XYZ中位数高于39.148 mm门槛 |
| S16 | 0 / 0 | 65.053 mm | 0/8 | 81.298 mm | 0.05812 rad | 未达标 |
| M16 | 0 / 0 | 64.364 mm | 0/8 | 80.749 mm | 0.05736 rad | 未达标 |

M4通过配对方向、p95、SO(3)和REJECT门，但未达到预注册的XYZ中位数改善幅度；因此不能挑M4作为winner。S16/M16都比S4差。唯一冻结结果为`R*=S4`。

## 机制对照

下表为8组source×noise配对XYZ RMSE差值的中位数；负值表示候选误差较低。

| 对照 | R0 | P-final2000 |
|---|---:|---:|
| M4 − S4 | −2.037 mm | −1.481 mm |
| M16 − S16 | −1.229 mm | −0.455 mm |
| S16 − S4 | +15.082 mm | +20.819 mm |
| M16 − M4 | +15.459 mm | +21.880 mm |

在这组固定窗口和噪声中，M路径于相同步数下小幅降低XYZ误差；增加到16步则明显变差。该结果只描述本矩阵，不外推为一般因果或泛化结论。

P2000四种runtime的SafetyGate结果均为8 WARN、0 REJECT；SafetyGate状态与D2四项质量门是不同判据。

## 下一阶段

按协议进入原16窗口D2：P-final2000、`S4=R*`去重、B0使用原S0逐source实际noise，B1使用P1 N1；两个噪声组分别验收，不池化。任一组未过门则自动执行同一起点的L/U各2000步。

## 证据

- 机器可读完整报告：[PI0247_D1_MECHANISM_REPORT_20260926.json](PI0247_D1_MECHANISM_REPORT_20260926.json)，SHA-256 `a7e5eb96a93e8726577b740286ab6115d01938cb5f1219941fd2cf103cd1b6b3`。
- 冻结运行卡：[R2运行卡](PI0247_D1_RUN_CARD_20260926_R2.json)，SHA-256 `91c40aab890dfd1380f0fc12e506f123479bcab6c668b60548d10266f4ccf018`。
- 静态结构零证据：[PI0247_D1_STATIC_PREFLIGHT_20260926.json](PI0247_D1_STATIC_PREFLIGHT_20260926.json)。
- 输入身份审计：[PI0247_D1_DUPLICATE_AUDIT_20260926.json](PI0247_D1_DUPLICATE_AUDIT_20260926.json)。
