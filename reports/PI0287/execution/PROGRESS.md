# PI0287实际接入进度（未完成）

## 后续更新：第四次启动已收口

用户明确授权额外最多3次启动后，已运行修复后的六项固定参考。实际结果0/6：5项加载硬限越界、1项原生abort。独立CPU重算一致；没有任何10秒完整fixture。按原门停止teacher/pilot/TRAIN/DEV，不再追分启动。完整结果见同目录 `CLOSEOUT.md`、`REFERENCE_CPU_RECOMPUTATION.json` 与实际失败RGB视频。累计4次启动、21 env starts、784 physics integrations、205 actor预算预扣、38 RGB pairs；GR00T/优化更新/teacher均0。下文保留前三次历史进度，旧soft-limit FAIL仍是工程无效，不改判为新结果。

固定授权：release `7a9f0c02b55222646601307ccd7922e9297e2ab5`，Issue11 #6084617266。隔离本机PI0287 worktree及Corelab `/home/liuyu-nfs/Projects/go2-arx5/PI0287`；原PI0286无重启。

## 已实际完成

- 固定协议、机器卡和canonical全部读取；WBCv1压缩包hash匹配。原Isaac Gym/Python3.8.20/torch2.3.1/CUDA12.1实际import，未升级全局torch。
- 历史完整Stage2/Dog039999真实布局核验；控制周期0.02s、physics dt0.005s、decimation4，原WBC safety配置保留。原件/权重/资产不上传Git。
- 原创multi-actor Gym边界处理全部root/DOF/body/contact gather、root/DOF scatter和外力tensor展开；实际n=2 robot sim actor为[0,4]。暂停时只reset env0，env1 root/DOF/object及已检查的wrapper history不变。
- 真实桌面、4cm物体、16cm蓝区、body/wrist 96×96真实RGB。两侧夹爪按名称驱动；开闭方向、对称性及物体落桌检查通过。此部分是明确标注的NO_WBC/pinned资产诊断，不是抓取成功。
- 自由基座初始状态下独立URDF FK比对：XYZ约3.45e-7m，SO3约3.38e-7rad。双RGB真实画面已实际查看。
- 实际私有Gamma/TimeLaw的6项CPU桥测试通过；教师/评分/记录器/数据检查共63项CPU通过；soft/hard限位独立回归1项通过。共70项，不等于70个物理条件。
- 外部EEF世界系参考绕过bank/肩部重锚定，原native preview/progress及arm→plan→dog链保留；实际六次首tick调用产生12次WBC forward。未修改WBC权重、gains或progress。

## 工程结果及裁决更正

三次串行启动：scene-smoke成功；reference第一次在首个策略调用前缺roll初始化，原件保留，修复实际pose bookkeeping；第三次六fixture各一tick停止。

第三次自动写出的 `REFERENCE_QUALITY_FAIL` **撤回为工程无效判定**：代码使用原生奖励soft limits冒充硬限。实际源码证明该字段会缩窄URDF范围；仅保存了boolean，不能独立证明实际硬限穿透。原六份JSON/NPZ和RUN_3永久保留，erratum单独追加。

修复已落实并通过CPU反例：从加载的actor DOF properties读取原硬限，逐tick保存DOF/上下限/穿透量和关节名。视觉蓝标采用独立collision group，使其保持声明的视觉标记作用；未改变robot/table/object物理碰撞配对。该修复尚未再次GPU执行。

G1生成的PASS收据只证明其中列明的接入子项；完整reference/history reset覆盖、物体接触负例校准及后续实际时钟/安全链仍需继续实证。**G2未取得有效质量结果，pilot8、TRAIN64/DEV16、G5官方processor读取全部未运行。** Teacher与dataset模块已有实际CPU可调用实现；run.py后续teacher-pilot/collect尚未完成原生接线，不把guard当完成。

## 当前预算

真实累计14 env starts、393 physics integrations、13 actor调用预算预扣（其中12实际forward，1在观测阶段失败）、17 RGB pairs、teacher0、GR00T forward0、GR00T/WBC updates0。三次sim进程服务时间合计58.561815s；进程self CPU合计17.812693s，未包含全部编译/本机/子进程CPU，不能据此宣称完整CPU cap PASS。其余原件和预算不重置。

原协议单列最多3次串行启动，已经消费完；已在停机边界完成两项接入修复，尚未启动第四次。需要对额外串行启动作具体授权澄清后才继续GPU；不会把这个工程错误写成控制器科学FAIL。

## 运行命令

Corelab GPU节点使用成熟环境脚本：

```bash
bash /home/liuyu-nfs/Projects/go2-arx5/PI0287/project-src/scripts/pi0287_target_sim/corelab_environment.sh \
 /home/liuyu-nfs/Projects/go2-arx5/PI0287/project-src/scripts/pi0287_target_sim/doctor.py \
 --config /home/liuyu-nfs/Projects/go2-arx5/PI0287/private/config.json \
 --output /home/liuyu-nfs/Projects/go2-arx5/PI0287/evidence/RUNTIME_MANIFEST.json
```

同脚本运行 `run.py --mode reference-fixtures --config ... --output-root ...` 是下一已修复入口，但现有3次cap会拒绝额外启动；不会删除LAUNCHES或预算绕过。新checkpoint、随机策略、改gain或seed追成功均未进行。

selected/deployment=NONE。目标是继续取得可信目标数据，不将当前资产/CPU进度冒充完整实验交付。
