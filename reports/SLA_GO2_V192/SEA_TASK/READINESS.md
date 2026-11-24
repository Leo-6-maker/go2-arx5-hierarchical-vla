# SEA-TASK 首个水果抓取与抬升

固定 RoboDojo-Real 版本包含 18 个任务目录，其中 ARX X5 适用的是 cover_blocks、insert_tubes、make_bread、make_food、pack_and_pour_fruit 和 store_in_safe。按场地已核验录像优先，当前选定 `pack_and_pour_fruit` 的首个局部水果抓取和抬升。 [官方固定数据目录](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/1127bda94f172d757386945af3eb3a6754d1a4e6/data/RoboDojo_real)

| 源录像 | 先调整碗的闭爪帧 | 首次抓取水果闭爪／释放帧 | 期间右 link6 峰值上升 mm | T0／T1 输入起始帧 |
| --- | ---: | ---: | ---: | --- |
| episode 0000000 | 139 | 272／337 | 126.04 | 252、266 |
| episode 0000001 | 121 | 282／310 | 127.86 | 263、276 |

头部和右腕画面显示两次水果由右手夹起并搬运，选中 50 行窗口的左 link6 峰值移动在 0.064–0.326 mm。结论限于所选局部阶段，完整果盘倾倒使用双臂。上升量为右 link6 的源坐标读数。切片同时核验 30 Hz 因果历史及 `action[t] == state[t+1]`。两个已有冻结模型返回在帧 266 和 276 的 50 行误差为 6.12、11.01 mm，对应保持基线 52.61、145.53 mm，给出局部源能力证据。 [机器可读名册](REGISTRATION.json)

T0 原指令和 T1 保持两阶段语义的改写已封存。两个种子共 16 格，历史返回留作参照。T2 缺少已封存的同语义 Go2 观测，T3 只复用旧 Reach。新增模型前向与积分均为 0。
