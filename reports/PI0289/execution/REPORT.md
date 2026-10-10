# PI0289 P0/P1 实际接续

P0 完成；V1 CPU_PASS，原生10秒资格尚未运行。

- 原PI0288 42步与初始化等7个本地原件逐文件SHA复核一致，旧FAIL未改判；wbc.log仍保留远端。
- Passive源码强制arm DAMPING：kp/kd/tau=0；模型joint2/3 damping=0、limit margin=0、solref=(.02,1)。原物理线程早于控制接口运行。
- 实际旧arm ctrl42/42均0。旧qfrc未保存，为UNOBSERVED；CPU对43个已保存状态执行mj_forward，无积分，重建值不冒充原始力或唯一因果证明。
- 唯一V1采用交付task_floating initialState.arm的0/.9/.9/0/0/0。根与腿保持home；通过18关节/脚地/有限数/精确reset检查。
- 原生暂停→GetUp→HOLD实际锁存及增益检查→ENTER→原locomotion→WBC。仅启动时序与初态overlay，未改MPC/RL/增益/normalizer/模型动力学。原安全从首次积分启用；新增只读力项与控制时序，native warning及非法接触更严格终止。
- G1必须待实际安全启动通过；夹爪部署规格仍为独立G2阻断；教师/GR00T均0，selected/deployment=NONE。
