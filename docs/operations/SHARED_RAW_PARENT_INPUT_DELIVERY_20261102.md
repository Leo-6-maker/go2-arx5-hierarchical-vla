# 四任务共享基线：固定raw父模型输入交付

本入口只交付`nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`的原7文件，文件身份与既有`minimal_model.FILES`和`RAW_PARENT_FILES_ID`保持一致。总计6,914,922,850字节。它不加载模型、不训练、不运行物理，不将E1/book128/b248权重替代为raw。

`scripts/pi0294/multitask/raw_parent_delivery.py`可由现有owner通过既有Journal部署为独立用户级有限输入任务。父进程退出不影响后台传输。所有新输入、代码、日志和结果置于已核实的Core节点本地独立目录；不使用Home/HF默认缓存、不读取或修改凭据、不修改现役capsule或系统服务配置。

输入manifest固定公开仓库、revision、7个SHA与精确字节数，默认最多8MiB/s、180 CPU秒和3600秒墙钟。每次读取前检查旧任务剩余保护量＋尚未交付输入＋30GiB真实可用空间；运行使用空CUDA可见设备、单核低优先级和独立内存上界。HTTP只沿固定公开HF路径，无自动重试。每文件先留intent，再流式全SHA校验；缺失、超长、损坏和中断文件留在独立PARTIAL中，不能发布为有效输入，也不覆盖已存在文件。

全部7文件完成后，`RAW_PARENT_AUDIT.json`提供完整流式字节核验及节点dev/inode/size/mtime/ctime，格式可由现有`minimal_model.unchanged_raw_parent`复用。后续在身份不变时无需反复重读6.9GB；身份变化则重新核验。原预训练statistics和将用于适配的共同TRAIN统计分别保留。

**输入交付不等于共享训练准入。** 共同59条目的原官方Processor/真实RGB资格、实际新数据冻结、任务/评价入口、raw零更新基线、M32前后向和完整恢复/显存资格仍需实际完成。既有D2或book128的资格不能替代这项新raw共享实验。当前D2和PB队列不因本文件修改。
