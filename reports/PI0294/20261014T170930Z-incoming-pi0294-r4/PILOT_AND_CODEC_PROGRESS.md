# R4 实际 pilot 与 codec 进度

四条 pilot 全部完成原32秒、16000积分、641双RGB记录；原任务成功且原高频安全通过。
两条最初缺少 ATTEMPT_PLAN 索引的只读审计已修复，四条独立审计通过。
旧报告曾将这两项工程审计缺失标为质量FAIL，该报告和hash永久保留；R1收据仅更正此分类。
修复新增physics/render/model查询均为0，未修改实际轨迹、任务、seed或安全门。

正式48条在原封存场景列表上使用四worker进行首次采集；无失败seed替换。
Linux真实入口的3项CPU边界检查已通过。Windows因缺少Linux resource模块的尝试保留，未计作PASS。

已有122条v3.1增量使用固定v3.0统计时出现114个action裁剪分量；73322窗口完整检查的最大XYZ解码误差为0.000208620m、SO3为0.000620702rad，超过原门。
这是有效DATA_VIEW_BLOCKED，不重拟合统计、不放宽门、不按成绩删样本；条件C1当前不能执行。
C0独立，但须先获得真实M2048 TRAIN4/DEV8面板。R3评价作业666401仍健康排队，模型行为尚未得到评价。

执行进度和能力PASS分开；selected/deployment保持NONE。所有原始数组、视频和原生环境私有。
