# 共同reset到T2源夹爪状态：显式连接过渡

原T2真实安全FAIL保留：4505包、81命令均只映射一次到2mm目标，逐步PD重放误差0N，无任何pad接触；右slide4个原始点低于0，最大约12.07微米。它不支持缺包/双映射bug。13个保存状态只读forward显示转腕时两slide轴投影变化、bias方向相反；这是伴随证据，不是唯一动力学原因证明，不授权重力补偿或调gain。

确切接续边界是：本次实际共同reset为open42mm，源首contact已经closed2mm；旧连接直接继承源首值，使整段自由转腕均贴近closed端。新连接显式采用现有working stroke中点22mm（两既有端点均值，无参数搜索），在自由连接期间保持；到源入口后使用原0.04m/s接口真实闭爪，额外hold仅按20mm行程/原速度＋一个原20Hz周期派生为0.55s，然后开始原源clock。所有源pose/gripper/time/source_indices/事件保持，源开始只整体平移0.55秒；原完整最小时域复核后仍不得在线延长。

这只改变reset→source连接的夹爪指令，不改变working endpoints、转换helper、速度、force、PD、damping、WBC、限位、安全或推送评分。连接值/原源值/实际达到值分别记录，不能宣称22mm是原专家源值或跳过真实闭爪。T1默认路径及无该卡字段的旧T2完全保持。

先CPU正负例、旧路径回归与同一保存入口的完整H40/接缝/假设夹爪几何检查；不重做IK/RRT、不换source或RNG。仅通过后执行原剩余第6个科学开发slot，全部历史工程/安全/成本仍累计；不增加科学预算，不原样重跑有效失败。若再次有效失败，不启动第7个科学slot。新GPU/PBS/GR00T/训练/真机仍0。

旧C2/C3 retime后的entry_duration元数据曾保留raw uniform时长，实际组合轨迹与独立重建始终使用retimed time；新运行同时记录raw_uniform_entry_duration与实际entry_duration。旧报告/数组不覆写。
