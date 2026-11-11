# NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1 · 结果报告模板

> 模板没有实验结果。实际报告必须填写真实产物与来源；未执行／未取得字段用NOT_RUN或PENDING，不以0成功代替未知分母。

## QUESTION

固定官方父模型、新body+wrist目标观测、目标TRAIN100与目标500更新，增加source500预适配是否提高完整站立整机的真实抓持、≥20mm抬升、≥1s保持和原站立／安全评分？

说明：单训练seed，条件扩样pilot；B比A多source计算，非等总算力对照。只测试本任务／新观测版本，不推导行走操作或WM结论。

## CHANGED

| 变更 | 固定身份／hash | 相比什么 | 实际验证 |
|---|---|---|---|
| 代码／官方父模型 | PENDING | 官方共同起点，旧历史保留 | PENDING |
| body相机／Processor | PENDING | 实际状态重渲染 | PENDING |
| target／source数据 | PENDING | 标签语义与统计分开 | PENDING |
| 训练／decoder | PENDING | 同白名单、同target预算、同endpoint basis | PENDING |

记录相对本协议的所有偏差及发生时间；是否在看新DEV输出前冻结。不要把偏差隐藏在run名称中。

## ACTUAL

### 1. 数据与相机

| 分母／指标 | 预期／上限 | 实际 | 证据 |
|---|---:|---:|---|
| target episode | 100 | PENDING | |
| target监督窗口 | 1450 | PENDING | |
| target全部原capture | 原manifest（预期1500） | PENDING | |
| body8px边距窗口 | ≥1378/1450 | PENDING | |
| 每阶段raw可见≥64px | ≥ceil(.90*n_phase) | PENDING | |
| 每阶段Processor后可见≥64px | ≥ceil(.90*n_phase) | PENDING | |
| source TRAIN独立组／episode／window | window≤3200 | PENDING | |
| source保留独立组／episode／window | window≤400 | PENDING | |
| 数据排除／未获取／缺文件 | 全部列原因 | PENDING | |

附target非视觉数据／腕图一致性、新相机标定、source actual-feedback语义、坐标／单位／RPY、时钟age/skew、split与统计证据。

### 2. 训练与端点

| 阶段 | 计划optimizer calls | 实际calls | 非零LR calls | 实际visits／unique windows | 250／500端点hash |
|---|---:|---:|---:|---|---|
| A_TARGET | 500 | PENDING | PENDING | PENDING | PENDING |
| B_SOURCE | 500 | PENDING | PENDING | PENDING | PENDING |
| B_TARGET | 500 | PENDING | PENDING | PENDING | PENDING |

另列资格更新／恢复探针与candidate更新，不能混计。保存完整optimizer/scheduler/RNG/cursor身份；B仅承接source白名单权重。列出普通推理重建和新Core同卡资格。

### 3. 新DEV注册与完整请求分母

列出old-population input-only审计、scene原点／初态／assets／reset／timeout／score身份。不包含旧TEST结果。

| 阶段／方法 | planned槽 | requested | attempted | reset_ready | technical_valid | success | 未启动／未激活 |
|---|---:|---:|---:|---:|---:|---:|---|
| Stage1 A | 8 | PENDING | PENDING | PENDING | PENDING | PENDING | |
| Stage1 B | 8 | PENDING | PENDING | PENDING | PENDING | PENDING | |
| Stage2 A（条件激活） | 32 | PENDING | PENDING | PENDING | PENDING | PENDING | |
| Stage2 B（条件激活） | 32 | PENDING | PENDING | PENDING | PENDING | PENDING | |

若不扩样，Stage2 requested=0并标CONDITIONAL_NOT_REQUESTED，不算32次失败。Stage1已请求但未启动项单列。success/requested是主要完整分母；technical_valid条件率只是附表。

| 配对结果 | Stage1（8对） | Stage2（32对，若激活） | 合并（描述） |
|---|---:|---:|---:|
| 共同成功 | PENDING | PENDING | PENDING |
| A-only n10 | PENDING | PENDING | PENDING |
| B-only n01 | PENDING | PENDING | PENDING |
| 共同未成功 | PENDING | PENDING | PENDING |
| 技术无效／不完整配对 | PENDING | PENDING | PENDING |

每布局两次restart逐项附录；实际初始base/EEF/object/qpos/qvel/grip差异和噪声key单列。不能把80请求当80个独立场景。首筛门值、触发与否、B新增技术／硬安全问题及决定时间写明。

### 4. 闭环机制与离线诊断

- 真实抓持／实际抬升／保持／原站立与安全评分；
- 首失败阶段，模型参考拒绝、技术异常、安全停机分别计数；
- raw H40与重建H40的包络／差异；实际提交与消费40区间及未消费后缀；
- 夹爪nominal、机械状态、接触证据的时间关系；
- source保留集FM与物理误差，按source Processor；仅作遗忘诊断；
- 四个SFT2000 TRAIN诊断，独立于A/B；
- wall推理延迟／暂停时间／sim时间，不声称已满足连续实时控制。

### 5. 资源成本

| 环节 | 实际wall时间 | GPU时间／峰值 | CPU解码时间 | 传输／存储字节 | 失败／恢复成本 |
|---|---|---|---|---|---|
| 原始获取与验证 | PENDING | | | | |
| 重渲染与覆盖 | PENDING | | | | |
| source转换与编码 | PENDING | | | | |
| target编码与cache | PENDING | | | | |
| 资格与恢复 | PENDING | | | | |
| 三阶段训练 | PENDING | | | | |
| 四TRAIN诊断／DEV | PENDING | | | | |

## NEGATIVE

保留旧BF16 REJECT、CORR2048/SFT2000/C500、旧DEV/TEST、用户Teacher24最终19/24及终态receipt状态。列出本轮所有失败、拒绝、技术终态、缺失／未运行数据及收益未出现的证据，不删不以成功重跑覆盖。

## DECISION

填写是否按预注册门扩样，首筛与新增布局是否支持相同方向；明确只是pilot。若未过门，写“本预算内未观察到足够的增量收益”，不要写成已经证明公开迁移无效。

若只完成A或source未准入，说明对照尚未完整，不发布虚假的B成绩。

## NEXT

只提出由失败谱系支持的一项或少数独立后继：数据语义／时钟修复、定向接触／恢复数据、新视野覆盖、更多训练seed／独立场景、总算力匹配等。没有证据时不自动追加同分布训练、WM或新大卡。

## 产物索引

填写相机报告、两域manifest、source split／时钟证据、缓存／白名单资格、六个完整250／500端点、评价全部请求、资源成本及实际实现SHA。公开GitHub只存允许公开的代码、manifest与去敏结果；大权重／原始私有数据使用既有允许artifact引用。
