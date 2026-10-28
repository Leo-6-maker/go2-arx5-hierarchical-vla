# PI0294 R3：接触归因与真实恢复进度

状态：进行中，非终局、非模型资格 PASS。旧 P1、R2 有效失败和 D1=NONE 保留。

## 已取得的科学证据

- A/B 新日志分别与对应旧 R2 原件逐行相同，额外动力学/forward 调用为 0。
- A 的右指接触广义力为 +4.91619 N，关节限位约束为 −4.10507 N，净约束 +0.81112 N；不能把净值称为某一个接触力。
- B 在 2.236 s 的原 snapshot 中发生左夹垫—桌面接触，穿透约 4.254 μm、法向力约 4.702 N。原安全失败有效，未放宽门。
- 原生步返回阶段的求解器量与之后原 snapshot 的接触量分别标注，未混成同一时刻/阶段。
- S1 对两个已知危险前缀均在第六块安装前拒绝；实际只执行前 1000 个安全 tick，拒绝块的 live tick 为 0。

## 学习与教师线

M512 的 1030 个存储张量逐项相同；额外 lm_head 键与 embed_tokens 是同 storage、offset、shape、stride、dtype 的真实共享别名。
真实首更新前的 optimizer、scheduler、Python/NumPy/Torch RNG 与 sampler 均相同。
第一次 PBS 因未处理该别名的审计计数检查停止，新增更新和 forward 为 0，213 allocated GPU 秒记账；失败证据保留。
第二次 PBS 666047 已进入原配方续训。该 allocation 随后复用作评价，不新增第三次提交。

当前教师的完成数及真实更新以旁边带来源时刻的 EXECUTION_PROGRESS_2204.json 为准。
S1 未完成全部四个教师及八类上下文资格前，不执行模型 live episode。
S1 是使用完整模拟器状态的拒绝预检；不能称为未经筛选策略成功、WBC 成功或硬件安全保证。

后续保持 M2048、匹配 TRAIN4/DEV8 与原预算；selected/deployment=NONE。原始数组/视频/WBC 私有。
