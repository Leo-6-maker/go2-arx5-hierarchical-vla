# 成功示范来源与整机适配审计（2026-10-15）

**结论：优先复用自己的TRAIN技能片段，以公开小样本提供候选接近/接触轨迹和MimicGen接口参考；源任务成功不自动取得Go2＋ARX5资格。** 本次完整读取用户附件，核验官方来源、固定revision、文件大小/hash、动作元数据和关键源码。没有下载HDF5/ZIP/视频、加载外部序列化内容、开展新IK/物理/模型/训练。新来源实验尚未获本文件授权。

## 1. 自有资产资格必须分开

固定[PI0294数据卡](../../reports/PI0294/20261013T142457Z-localpc-pi0294/DATASET_CARD.md)与[manifest](../../reports/PI0294/20261013T142457Z-localpc-pi0294/DATASET_MANIFEST.json)：v3.0为158条，v3.1为295条且包含v3.0，不能相加为453条。v3.1的295条包含DEV及legacy；不能把全部295作为TRAIN。按manifest，TRAIN为184条新增准入＋64条legacy，DEV为31＋16；未来抽取必须逐episode核对原split，不跨入DEV/confirmation。

v3.0固定私有HF payload revision7554da77f7ff0e7e1682e3c1f71ed33680039335，READY revision6aa94f4f40d791bedd6b4e75489e4ede6504d10d，历史86文件/4070375578字节回读；本次未重复下载或重新回读，不能把历史记录称为新的现场readback。v3.1原manifest标记NONE_NOT_PUBLISHED，本次未证明其后来已交付。

v3.1使用自身TRAIN拟合statistics时，完整149048 TRAIN/28247 DEV窗口的原CPU验收通过；后来R4把122条增量接到固定v3.0统计时，73322窗口出现114裁剪分量、最大XYZ误差0.000208620m/SO3误差0.000620702rad，形成[DATA_VIEW_BLOCKED](../../reports/PI0294/20261014T170930Z-incoming-pi0294-r4/PILOT_AND_CODEC_PROGRESS.md)。这两项不矛盾。原始成功片段可作为候选来源，不能沿用被阻断view直接训练、重新拟合live统计或抹去FAIL。

固定底座记录是真实任务证据，但从预置局部手部状态开始、监督EEF10，没有整机接近与BASE标签。优先提取接触/抓握/推送/释放；原实际状态与已提交目标分别保存。不能把未来实际EEF轨迹替代Teacher提交标签，也不能追加三个零后宣称BASE学习。

## 2. 官方小样本已锁定（十进制MB）

完整revision、Git blob/LFS SHA256、metadata文件SHA及源码链接在[DEMO_SOURCE_METADATA_LOCK_V1.json](../../reports/PI0294/design/DEMO_SOURCE_METADATA_LOCK_V1.json)。LFS SHA来自官方文件索引，尚不是本地payload完整校验。

| 来源 | 固定对象 | payload大小 | 当前可确认及限制 |
|---|---|---:|---|
| [ManiSkill官方](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations) | d674485bbffdd533914e52d272fdda34c0515608，demos/PickCube-v1/motionplanning/trajectory.h5 | 29.349195 MB | JSON列1000个success=true；pd_joint_pos，obs_mode=none；尚未核验HDF5内部状态/RGB/EEF字段 |
| 同上 | 同revision，demos/PushCube-v1/motionplanning/trajectory.h5 | 26.480047 MB | 同为1000个元数据成功项、joint动作，无BASE；不等于原T2评分PASS |
| [RoboTwin官方](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) | 3dc3b798668feb99ac61cc9086d84cbcc3d79186，dataset/place_a2b_left/arx-x5_clean_50.zip | 190.721619 MB | 来自官方项目链接；双臂固定ARX-X5，ZIP成员/每条实际成功/工具坐标仍未核验 |
| 同上，后备 | 同revision，dataset/pick_diverse_bottles/arx-x5_clean_50.zip | 186.886136 MB | 抓瓶技能，不等于原方块任务；500条版本分别2.57/3.59GB，不作为首包 |
| [MimicGen官方注册源](https://github.com/NVlabs/mimicgen/blob/72bd767c255545f462e7ccfb2731f2e5d4c1d9bb/mimicgen/__init__.py) | amandlek/mimicgen_datasets，33016f8a62c02334f929f2913af8fdd2a8a129e1，source/pick_place.hdf5 | 92.016062 MB | 官方卡称source通常10条；本文件条数/DatagenInfo是否已补全仍待HDF5检查；不下载9.6046GB core包 |

两份ManiSkill H5合计55.829242MB。若未来采用首包Pick/Push＋RoboTwin place_a2b_left clean50＋MimicGen source，四个payload合计338566923字节，另有少量JSON；不包含随机500、全部RoboTwin或TB级真实ARX数据。

ManiSkill metadata固定生成代码652ad9353c0223507a938f0e8d990dd6f1c771ad；不能拿最近源码默认配置代替该数据版本。JSON的max_episode_steps=50而elapsed_steps实际范围Pick49–103/Push61–81，须保留原步数/时基，不按50截断或假定20Hz。source success标记是源任务声明，不是本次物理复现。小sample.mp4存在不证明HDF5存有训练RGB。

许可元数据：ManiSkill数据Apache-2.0、RoboTwin数据MIT、MimicGen数据CC-BY-4.0；MimicGen代码另有NVIDIA Source Code License，不能把数据许可套给代码。保留各自引用/notice，具体再分发依原文件条款审查。本次未复制供应商资产。真实ARX5“1704条”与XRoboTwin的具体repo/revision本次未完成独立确认，暂不进入可执行来源锁；Arena保持workflow参考，不加入43D joint监督。

## 3. 源码带来的具体适配要求

RoboTwin官方README把下载数据指向上述仓库；当前源码ea8b21121ebb3cd201ff5b3fe361944ac94eda3f的place_a2b_left按物体相对左右位置、二维距离及夹爪打开评分，未实现我们的完整保持/撤离标准。新代码与旧ZIP并不自动同版本。官方还明确历史/新JPEG存在不同编码语义，未来实际ZIP需核对版本并使用配套decode_image_bit，不能先假定RGB通道正确。未找到所猜env_cfg/robot/_robot_config.yml路径，记HTTP错误而不宣称已核验robot配置；后续按固定tree解析真实资产入口。

MimicGen固定源码72bd767c255545f462e7ccfb2731f2e5d4c1d9bb核对DatagenInfo及base/robosuite接口：EEF/object/target pose是不同字段；target_pose_to_action/action_to_target_pose绑定当前EEF和源controller尺度，delta旋转复合方向也必须保留。不能把7D源动作直接当我们绝对EEF10。该类允许缺字段，类中有字段不证明HDF5中已有该字段。

拟统一只读片段schema：source repo/revision/fileSHA、episode/split/index、原始时间与控制频率、实际EEF与物体SE(3)、已提交target SE(3)、夹爪物理开度/符号/限幅、子任务边界及证据、源成功定义。缺失项写UNOBSERVED；不以视频猜测或从实际EEF伪造提交目标。

对象相对表达先分别构造inv(T_world_object)×T_world_tool_actual和inv(T_world_object)×T_world_tool_target；通过显式源工具→目标工具标定重定位到当前物体。TCP、夹垫中心/闭合轴、物体尺度、碰撞体和joint限位均保留来源，不能只拷四元数。自由空间接近与接触段分别保真；规划q只作FK/碰撞候选，最终仍只提交原软参考，由冻结WBC实际执行。

## 4. 下一步的最小科学问题（建议，非执行GO）

优先回答：**已有成功接触片段能否给出当前整机可达、可安全执行且完成原物体任务的参考？** 不再修一轮FSM误差阈值，不开展WBC优化，也不立即大规模finetune。

1. 先固定自有v3.0 TRAIN T1/T2各最早3条完整准入episode，保留全部实际状态/提交命令/接触边界；v3.1只作后备候选资产，不能偷偷把旧DEV并入TRAIN。
2. 外部首包只用上表四个payload；按源episode id各前3条做有限schema/片段审核，缺失/不合格保留分母、不改选容易样本。ZIP只检查目录和安全类型，不执行pickle；HDF5只读，不加载未知序列化。
3. 冻结来源选择、工具映射、retarget公式、时间参数化、候选顺序及内部IK/RRT预算，再采用独立有界目标执行系列。源接近/接触重定位改变MP-S2既定候选，不能在旧FAIL后偷偷热换。原同场景、安全、任务score不改，先争取一个技能的完整真实种子；“1–5条”愿望不能代替预注册episode cap。
4. 原任务成功＋原安全＋观测/提交标签/时间完整才入新fullbody成功bank；真实失败不补seed。BASE来源必须目标高层意图与实际执行记录清楚，W0零BASE只证明近场，不声明身体协同。扩增与FB13训练另按真实小包资格推进。

当前MP-S2明确external_demo_downloads=0、新扩增/训练=0；本次只完成来源审计与准备，不从旧余额推导新来源物理权限，也不把未回复的真人参与问题当已同意或已拒绝。详见[MP-S2真实进度](../../reports/PI0294/20261014T170930Z-incoming-pi0294-r4/TASK_DATA_FIRST/maniskill-search-v2/PROGRESS.md)。旧FAIL、原时钟和累计预算保留，selected/deployment=NONE。
