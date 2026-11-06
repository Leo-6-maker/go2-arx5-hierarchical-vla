# FUSION：原 S-CORR-2048 冻结条件与资产接续

2026-11-06。此文件登记一个独立冻结条件；尚未在本轮 Core 上加载、推理或执行。

原 E1 条件与失败全部保留。它在原源 SafetyGate 上因速度、角速度与路径长度拒绝，不能用修改阈值、平滑、clamp 或更换权重后回写为通过。

原 S-CORR-2048 作为后继候选的依据是历史 TRAIN4 的 3/4 与 DEV8 的 5/8，以及原动作/processor ABI；不是根据 SOURCE_CONFIRM 或正式 TEST 成绩择优。依据为 `docs/handover/2026-10-18/MP1_N1_CONFIRMATION_TO_NEXT_STAGE_EXECUTE.md`。SOURCE_CONFIRM 只可为该固定旧 proposal 提供机制审计，不参与 fit 或选择。

固定身份：

- 原 checkpoint manifest：`94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`，global step 2048。
- 公共 inference manifest：`ec46067a1179405d6c75081cfa333f0daa994f8e1c54bd3658eac0b5fb4eff61`。
- 既有获准交付：dataset `leoyuer/pi0294-public-artifacts`，revision `b37a3d959335b548abb45f87ad48a5186ec662d9`，prefix `PI0294/20261018T182956Z-localpc-pv1/DEC1_CORR2048_INFERENCE_DELTA_V1`。
- statistics：`fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`。

只复用原 `inference_assets.consume()` 与 manifest 验证函数，不调用旧 DEC1 scope 的 guard/run，不复用 CLOSED lease。新 CPU 作业通过现有 FUSION owner/registry/Journal 登记，CUDA 为空，不加载模型。所有新下载、缓存和输出留在已资格 HDD。已有 E1 的同 SHA 第一 shard 流式复制到新 HDD inode，再由原 consumer 在该盘内 hardlink；原文件 inode、mode 与字节保持不变。

13 个 delta 文件及 manifest 共 14 次逻辑文件调用；原 consumer 完整核验，随后重新核验九个 inference 文件与原第一 shard。逻辑文件调用与返回文件大小不是 HTTP 请求或实测网络流量。失败与部分文件保留，不自动重派。该资产只支持 inference，不包含 optimizer/RNG，不能称完整训练恢复。两处 Cosmos 名称元数据迁移按原 manifest 明示，权重、index 与统计不变。

资产验收后，只有实际 GPU 余量足够时才另行登记单次冻结 shadow；全部权重、processor、seed、数值精度与观察身份必须记录。目标相机/状态桥、完整参考支持域与原安全仍需分别验收。旧 CORR 的已执行 first8 与真实 400 ms 后果不能配给新 E1 动作，也不能扩展成 H40/H16 真值。目标 replay 不是在线闭环，更不是任务迁移成功。
