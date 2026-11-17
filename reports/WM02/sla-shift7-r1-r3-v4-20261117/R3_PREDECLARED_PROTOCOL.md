# R3 V4事前固定执行细节 — 单一SHIFT7架构

入场凭据是R1封存实际结果`0321115e…`与用户独立R3授权。最大32联合请求／32完整单模型生成，F1/F2各16，0训练／0物理；不消费HYBRID64分支方案，也不自动增加第33请求。

连续片段在生成前确定：CN38−C20每family按ID排序第一条合格EP（F1 `72H-F1-T0-A003`、F2 `72H-F2-T0-A003`），从第一次CLOSE前6个query开始连续16个，全部noise0。phase仅用于离线选段／评分，不是模型输入。每个原记录query间隔250个2ms tick=500ms，上一条真实发出夹爪命令与source/query时钟均从原MOBILE_SAMPLES绑定，不虚构学生反馈。

先封存输入银行和criteria，再启动GPU。离线提取记录RGB为私有文件；测量中只在该请求的名义发布时刻读取当前输入，不提前把未来RGB送入模型。分别计录制文件交付时间与bytes-available→request→GPU→原完整H40→mock ACK；它们不包含未实测的现场相机成本。32请求都是WALL_CLOCK_STRICT，原400ms TTL、完整剩余50ms及原参考续期不变。

请求0是唯一冷启，import／load与首个JIT分别报告；若冷JIT使400ms过期，原拒绝保持，0安装／消费，保存完成的原始分支并只做明确标注的离线编译，不迟装或重生成。冷请求终态后，以+500ms为第一热态release，随后31条按固定500ms网格送入；冷启动间隙单列，绝不合入热态2Hz统计。热态p95≤350ms、所有有效ACK最大严格<400ms；全部原完整H40、无丢失一个完整500ms发布槽、无队列容量背压、0保护失效、归档全SHA且pending≤4。冷请求可被TTL正确拒绝，不能把31个有效ACK写成32/32安装。任何热态拒绝均使全链资格失败，不自动补做。

严格Receiver的P10点是零积分替身消费，native时钟是原记录时钟／mock推进，原子不会被当作实时物理执行。即使持续发布和请求TTL通过，也不证明WBC对整段预览、原生续期、相机或接触安全。R1和R3均不能解除隔离。

使用PR396 VerifiedObservation／只读H40／AsyncRequestArchive；派发前保留最多4条、4MiB预留；总tmpfs仍≤64MiB。新增capacity_wait_s仅测容量背压，不改排队或时钟逻辑。保留所有完成原始normalized/physical_H40，包括过期冷请求；归档失败／UNKNOWN禁止继续派发。现场只保有一个父模型和一个adapter。

工程预检：完整32循环CPU合成传输集成测试（不是GPU实验）发现名义release边界的负sleep竞态，最小修复为剩余等待下限0，不改任何TTL或采样；旧本地失败保留。原响应schema／绑定／H40合同保持。实际结果会独立封存，不以测试通过代替GPU测量。
