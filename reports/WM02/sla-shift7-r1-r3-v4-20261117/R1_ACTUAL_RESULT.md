# V4 R1实际终态：SHIFT7单模型CN38 180/180

按PR397固定`a51f7a02d23414304f631b8720aec5fcb95d5705`和用户分别授予的R1／条件性R3共享GPU授权，复用PR396已完成R0/R2。R1实际执行源码`fa1477b1ad18c3f01f8ef26b4c523e42814b205b`；未热改或重投旧服务。

首波固定8EP×5阶段×2noise=80次真实生成，完整H40／原编译器80/80、CLOSE16/16≤100ms、无末端意图遗漏与夹爪参考接缝超限；同条件EEF位置误差p95=1.443217mm，对照HYBRID1.465944mm，单点最大2.495736mm vs2.609793mm。全部事前门通过，才实际追加其余10EP的100次；总180，没有追加180或模型恢复额外生成。

全银行18EP的180/180原完整H40通过，F1 140/140、F2 40/40、CLOSE36/36≤100ms。EEF位置误差p95=1.562773mm vs HYBRID1.624511mm；单点最大3.046194mm vs2.985168mm（增加0.061026mm，在事前1mm界内）。p95退化门0.5mm与单点最大退化门1mm均通过。首点夹爪参考速度≤40mm/s，原门与后缀全部保留；终末宽度另报，不新加事后验收阈值。

**全局冻结下一阶段架构为SHIFT7_SINGLE；R3只允许该架构最多32新请求／32生成。** 不再执行HYBRID性能候选，不按query择优。对照复用原HYBRID保存文件，0对照新生成。相近布局专家状态开发诊断不是盲TEST、学生on-policy、抓持或原生资格；未证明动态接触净空。

资源与分母：180实际attempt／180完整生成，0新训练更新／访问、0原生请求／积分／任务评价。一个父模型与一个adapter；GPU数组峰值7,315,062,528字节、pool峰值8,592,031,744字节。worker wall182.336s、CPU50.523s；模型热态p95约180ms，第一次JIT4.966s单列，详见JSON。墙钟不是GPU忙时。独立回读180原始输出的SHA／shape／有限性、180 Journal COMPLETE，并重算首80与全180门，结果完全匹配；面板与criteria SHA事前／事后相同，服务exit0、lease CLOSED、PID/cgroup空。

固定身份：spec `250bca78c4c7f7ed1b6407dd16d951d10e30016e16a1f2953795eaf606a8451e`；结果`0321115e51307b6e16d2444d5494c359b71cd8c7cd0cecae442fcd65cb08a3c0`；记录`a3dd741cbc4ebd28ec204d7e34b6535d1b95a3d448dd2ad6763777b43e82d903`。模型／parent／统计／银行和资源上限见[授权与固定SHA](AUTHORIZATION_AND_PINS.json)，完整逐episode／phase／noise及首超限／峰值见[R1机读实测](R1_ACTUAL_RESULT.json)。

下一项已获准工作：用独立新source／lease执行唯一SHIFT7架构R3连续非物理32请求；原allocation截止不延长。S0安全隔离、三碰桌、1/38误拦、RP8／JOIN10／旧DEV_TEST失败保持，独立READMISSION未签，原生0。
