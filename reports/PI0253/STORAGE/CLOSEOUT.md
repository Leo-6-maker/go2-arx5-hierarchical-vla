# PI0253 storage retirement closeout

Status: PASS_BOUNDED_PURGE_WITH_UNKNOWN_SKIPS. Authorization eb2ac1b76338fc5b3486809f6bb28e80529ed4b1. Published preview: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5875018676. No reauthorization needed; explicit list applied after identity/reference checks.

|Host|Deleted files/links|Logical bytes unlinked|Unique allocated inode bytes removed|Measured scoped du decrease|Free bytes change|
|---|---:|---:|---:|---:|---|
|hopper-l-01|10|47646006520|47646023680|47645343744|47645196288 (/scratch/e1192548)|
|corelab-4080s|26|66704410966|66704433152|66704429056|0 (/home/liuyu-nfs/Projects/go2-arx5), -10809344 (/var/cache/users/liuyu-nfs)|
|corelab-amd-server|0|0|0|NOT_MEASURED_NO_DELETION|0 (/var/cache/users/liuyu-nfs)|

Total logical unlink: 114350417486 bytes. Shared NFS counted once on corelab-4080s. Corelab NFS free delta was zero at immediate statvfs observation despite verified file absence and directory du decline; no claim of a 62.1 GiB free-space increase. Local GPU cache free declined 10,809,344 bytes during the interval; no deletion targeted that disk. Hopper free increased 47,645,196,288 bytes. Quota headroom remains UNKNOWN on every host.

## Asset states

|Asset|Local state after this round|HF / protection|
|---|---|---|
|G2000|Hopper active model-only initializer retained with shared inode protection; unused Corelab inference payload retired to metadata-only|PUBLIC full Trainer revision efecea485ae77000599d19171b02491f8d514f42, tree cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226; prior full readback reused + fresh anonymous reachability|
|FULL2500|Corelab model-only candidate workset retained; loaded Hopper old resume source was retired before this round, not redownloaded|PUBLIC full Trainer revision289b3a8c98a8f8aa8cc62a94c26c3a827c154a4b; formal candidate retained despite intermediate FAIL|
|FULL5000|Current complete durable Trainer protected on Hopper; local GPU model-only inference workset retained|PUBLIC full Trainer revision a8ab7d38970294b52320beba636a10cb6a2d5afe, tree8cfe43038921d53e27f5408f2ac2307fadd05f568fe9171acc3d3e18db4ff570; complete readback verified|
|FULL7500/FULL10000|Future formal candidate save/workset directories protected; not yet complete at closeout|Same sealed saving/evaluation plan; no candidate retirement|
|FULL1000|Completed intermediate model-only observation; local weights retired, metadata retained|PUBLIC revision6cdfd62f0a8ee15a6f6fdd4ab4f2fae95416f888; model-only archive, never resume/candidate|
|A2000|Corelab weights/aliases retired, evidence retained|PUBLIC full Trainer revision2a7945d49d17f3ec1801f2b2af6dba5e5bd6c100; prior full readback + fresh anonymous reachability|
|U2000|No retirement in this round|Prior verified PUBLIC revision0296f5dca5cb46daf05b51f90c53e1df905a7203; prior full Trainer tree dace0dd423c4c1c6297b26bd6da09aee1e99edf4238638c5a317356e417a920f, preserved index; no new full readback claimed|
|Original adapted1000 baseline|Required model payload present, protected under PI0227; no resume claim from config|Retained for original selector; no retirement in this round|
|K/X/C/E, PI0243 terminal FAIL, one-step smoke declared copies|Weights permanently intentionally retired on exact listed paths; configs/cards/logs/receipts/arrays retained|No new archival required; existing HF history untouched; tombstones deny local reload/resume|
|Legacy UNKNOWN assets|Retained and individually skipped|No assumed retirement; exact path/reason in manifests|

Actual science snapshot: job637051 remains R, optimizer6803/10000, restored=true; cumulative rollout and backward recompute head counts27212 each. Same sealed source/optimizer/RNG/loss/data/budget. No restart/hold/qsub. Storage actions made0 model queries/0 optimizer updates/0 episodes. Separate original-budget FULL5000 screen128 queries/512 heads completed, independent CPU replay PASS; intermediate quality FAIL is not candidate retirement. selected_checkpoint=NONE.

Original TRAIN/expert/processor/stats, current cards/seals/keys/schedule/execution, WBC/gate, unique tensors/NPZ, small failed evidence and all HF/Git history preserved. No monitor/mail enable, no DEV/F/DAgger/hardware/merge. Exact manifests, before/after identities, skip reasons and fsync unlink journals are adjacent.
