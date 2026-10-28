# PI0281 已保存 WBC 证据交接

PI0282只读交接。升级版 UNRECEIVED；没有执行新模型、actor、物理或渲染。

|实际尝试|修复接入|末2s位置RMS mm|真实SO3 RMS rad|资格|
|---|---|---:|---:|---|
|native_hold|NATIVE|9.063|1.105218|False|
|native_monotonic|NATIVE|12.038|0.629325|False|
|native_return|NATIVE|UNOBSERVED|UNOBSERVED|False|
|repair_hold|ENGINEERING_REPAIR_NOT_INSTALLED|9.063|1.105218|False|
|repair_hold-engineering-001|TRUE_REPAIR_CONSUMED|9.063|1.105218|False|
|repair_monotonic|ENGINEERING_REPAIR_NOT_INSTALLED|12.038|0.629325|False|
|repair_monotonic-engineering-001|TRUE_REPAIR_CONSUMED|12.038|0.629325|False|
|repair_return|ENGINEERING_REPAIR_NOT_INSTALLED|UNOBSERVED|UNOBSERVED|False|
|repair_return-engineering-001|TRUE_REPAIR_CONSUMED|9.008|0.543153|False|

HANDOFF.json 给出每个原件及真实视频路径/SHA、actor/资产/软件身份与修复实际调用回执。poses.npz 包含完整已保存时间、requested/raw/installed/actual姿态、关节、真SO3、原代理、progress/time-law和硬软限；representative-actor.npz 仅原first/middle/last实际history/preview，完整actor日志按原件索引引用。大模型与完整actor日志未复制。

原代理范数为无量纲 sin(theta/2)，不等于弧度角误差。full10s不是资格PASS。真正return修复完成10s仍有progress残差及姿态FAIL；源任务学习成功不能解除这个目标执行器阻断。

重现算术：打开poses.npz，SO3=Rotation.from_quat(installed)*Rotation.from_quat(actual).inv()的magnitude；final2s用actual_task_time_s∈[8,10]；原actor pack/history/FK验算脚本及既有审计SHA见索引。不得用此包启动WBC或新版本。

metadata-original-bytes.zip 保留所有小metadata原始字节（包括CRLF），跨Git文本换行处理后需从此包核对原SHA；普通metadata文本是方便阅读的副本。
