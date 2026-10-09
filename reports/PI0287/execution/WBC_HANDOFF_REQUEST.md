# WBCv1 Stage2交接核对（一次性清单）

WBCv1.zip身份与授权一致：159225884 bytes、687条目，SHA256 `3c59a6d5f8f924df0b838ad2d501d3a853274d741db191a9de453f58212063ee`。包内实际只有Stage1 dog权重及其参数；不能把Stage1目录中的arm JIT称为完整Stage2。

本机及Corelab另有历史完整 `2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018` 的039999配对。实际CPU核验了arm/dog全部推理布局、参数快照、自由基座Go2-X5资产、end_to_end模式、201/12060 arm与90/2700 dog观测及15/12输出。该组合明确标 `HISTORICAL_EXECUTOR_BASELINE`；不是新版WBCv1权重。

若需要新版WBCv1执行器，请一次性交付或指定已有受控路径：

1. 同一Stage2版本的完整arm和配套冻结Stage1/dog权重，指定唯一step、文件SHA256。
2. 实际parameters/config快照、使用的normalizer（若没有独立normalizer请明确）、观测/历史/动作布局与执行频率。
3. 与权重匹配的源码版本和Go2-X5 URDF/mesh、实际TCP link与局部偏移、action_mode。
4. body/wrist安装外参，以及真实夹爪开度、两关节驱动/mimic/力速度限制和接触判据说明。
5. 实际冻结推理启动命令及合法原生运行环境。不要提供随机初始化arm或Stage1-only arm JIT代替。

现有历史组合已可恢复和CPU读取，轨道A未等待新权重。当前GPU继续点是工程启动次数授权边界，并非声称服务器不存在Stage2。
