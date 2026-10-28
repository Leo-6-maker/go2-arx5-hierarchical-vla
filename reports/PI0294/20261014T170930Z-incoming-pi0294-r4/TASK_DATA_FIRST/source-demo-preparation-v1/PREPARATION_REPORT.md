# 自有成功片段与固定外部小包：实际准备交付

**已完成下载、完整SHA核验和只读来源检查；有界重定位/整机执行方案已准备，尚未运行新物理。** 用户本轮授权范围见[下载与准备采用收据](DOWNLOAD_PREPARATION_AUTHORITY.json)。旧MP-S2卡不改，旧有效FAIL/原时钟/累计账保留。

## 1. 四个固定文件已到本地项目数据目录

|来源|文件|实际字节|完整SHA256|
|---|---|---:|---|
|ManiSkill|PickCube motionplanning trajectory.h5|29349195|03ca60546541a7f18321d9d32721f0254bc75217828c0cadacd217d0c014576a|
|ManiSkill|PushCube motionplanning trajectory.h5|26480047|d74a4c0efdbf68cb5512dabfc9dfa2759fb53c7f221b15ded92b0d03533f3f2b|
|RoboTwin|place_a2b_left/arx-x5_clean_50.zip|190721619|29f7506a02ff1d7c45a7d3ed2c5dcf65ab321615cd9a7ee6ceaebaace76d49ad|
|MimicGen|source/pick_place.hdf5|92016062|18cb7d4e6947120e200b9ab27c041297f0056aaac20e43f1b79a49a3b3e4f723|

合计**338566923字节**，每文件实际大小和流式全文件hash与之前固定LFS SHA一致，不仅检查下载命令返回成功。完整revision与文件路径在[DOWNLOAD_RECEIPT.json](DOWNLOAD_RECEIPT.json)。二进制不进Git、不公开再分发；本地目录位于现有Projects/go2-arx5工作区的SOURCE_DEMO_SMALLPACK。

隔离PC环境huggingface_hub2.2.0/h5py3.16.0，实际版本在[PREPARATION_ENV_LOCK.txt](PREPARATION_ENV_LOCK.txt)，没有改CoreLab原生环境。初次CLI同时指定local-dir/cache-dir被拒绝，保留日志/原manifest，只去掉互斥cache-dir参数后继续同四个固定文件，不扩大下载清单。

## 2. 不再只依赖网页声明：已读取文件结构

[SMALLPACK_STRUCTURE_AUDIT.json](SMALLPACK_STRUCTURE_AUDIT.json)记录每源数字排序最前3条的实际schema、所选ZIP成员SHA及外部链接检查；只读、未解码图像、未加载pickle、未导入源simulator、未运行物理。

- ManiSkill：两个HDF5确各1000组；所选前三条有8维joint actions、空obs、Panda articulation及cube/goal actor状态。元数据success仍仅是源任务声明；没有直接EEF标签/RGB可移植的证据。后续若转EEF须有生成版本FK/状态布局依据，本版不为此装另一整套仿真器。
- RoboTwin：ZIP实际202成员、未压缩258138327字节；50 HDF5/50 pkl/50视频/51 JSON/1 TXT。前3个HDF5包含两臂endpose/gripper、joint action、三相机编码像素/标定；所查HDF5无物体位姿/时间戳。未执行任何pkl或外部pickle缓存。存在endpose不证明是已提交target，也未证明源task评分等于我们的T1。
- MimicGen：实际10条；env_args为**Sawyer**、robosuite1.4.1、20Hz OSC_POSE、7D动作、84×84双RGB与实际EEF/object观测。所选前三条无现成datagen_info/target_pose字段，不能把MimicGen类定义直接冒充数据标注。没有在目标环境复现这些源成功。

## 3. 优先的自有TRAIN已在原服务器确认

原v3.0 manifest完整匹配固定`b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df`。固定选择非legacy/new、TRAIN、按id排序前三条/task：T1/T2各F0-000/001/002，共6条，不跨DEV、不挑后来更容易的样本。

[身份与schema](OWN_TRAIN_IDENTITY_AND_SCHEMA.json)记录逐numeric SHA与实际有限值；[源阶段/成员hash](OWN_TRAIN_PHASE_AND_SOURCE_HASHES.json)记录原raw samples/references/events/card/score/audit等的真实索引。6条raw均存在，原success/admitted均true，每条641观察/80参考；导出601锚点不等于完整raw时间轴。本次只读索引、成员hash与原事件，没有生成新重定位路径或训练标签，没有重新采示范、下载整套旧数据/模型或重拟合stats。

例000-T1在12.8秒进入hold，完整抓放活动片段已存在；000-T2在8.25秒进入close、11.55秒开始真实push、24.4秒进入hold。新方案将前置整机自由空间连接与这些已成功的操作片段分开，避免重新发明接触FSM。

## 4. 已准备的唯一有界执行方案

见[完整协议](../../../../../docs/protocols/PI0294_SUCCESS_DEMO_SEGMENTS_TO_FULLBODY_SEED_DRAFT_V1.md)和[机器卡](../../../../../reports/PI0294/design/SOURCE_DEMO_RETARGET_TASK_CARD_V1.json)：只用6条自有TRAIN做可执行候选；外部小包作为已下载参考库，不因外部缺字段阻塞主线，也不在失败后热加外部来源。

冻结工具功能frame、两锚任务重定位、源actual/committed分离、原source阶段边界与确定性减速；各task首个静态合法候选在物理前锁定。采用后直接两项完整32秒T2→T1，前12秒只是同一episode侧报，不再插入新短里程碑质量前置。原WBC/安全/物体评分不改，无RMS前置或FSM阈值循环。

新拟上限4启动（2完整＋2严格任务前工程替换）/42012积分/1282双RGB/14400CPU秒/168bundle；内部至多2016目标IK/32256初值/1344RRT/174174dense，旧成本不扣回。按准备时父账用满预览42启动/395520积分/13530双RGB，仍小于原43/404023/13771 slice。原10月16日01:02:39新工作截止、02:10收口不重置，采用时须刷新真实余额/消费者。

**新方案的16项CPU契约、retarget IK/RRT、物理、接触、成功seed均未执行，不能写成PASS。** 下载/只读schema审计不是目标执行资格。当前只完成用户授权的具体下载与可审核方案；执行部分需要明确采用本次固定release，随后同合同工程自主继续，不再请求例行GO。

source/data/WBC/raw原件留原授权私有位置；无新GPU/PBS/模型/训练/扩采/邮件/monitor/merge，selected/deployment=NONE。
