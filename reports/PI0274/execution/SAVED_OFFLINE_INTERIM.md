# PI0274 原968项离线面板：保存指标中间观察

[来源与每项record/tensor hash](SAVED_OFFLINE_INTERIM.json)。全部968项原注册输出齐备、身份及hash核对通过；Euler/解码/gate/指标独立CPU全量复算仍由最终postprocess执行。本表不是总体资格裁决。

| 原面板 | 行数 | BROAD→EQUIL XYZ中位RMSE（mm） | BROAD→EQUIL SO3中位RMSE（rad） |
|---|---:|---:|---:|
| fit | 224 | 6.064→5.993 | .044526→.038572 |
| factor | 304 | 4.601→5.079 | .039307→.032620 |
| legacy | 440 | 12.758→12.342 | .024347→.026262 |

各面板REJECT=0、指标缺失=0。fit两项改善；factor位置误差增大、旋转误差降低；legacy位置改善、旋转误差增大。所有原分母保留，未按成绩重试。

192早期、592晚期、64主闭环和32component仍按原冻结计划执行。离线几何误差不证明40秒可靠性、物体任务成功或WBC迁移。新研究方向按章程以物体操作为主线，PI0274原授权收口不变；selected/deployment=NONE。
