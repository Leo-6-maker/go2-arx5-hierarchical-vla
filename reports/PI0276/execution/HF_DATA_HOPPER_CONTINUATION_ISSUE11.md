## PI0276 incremental execution — fixed task data fully archived/read back; Hopper same-state staging resumed

Authority unchanged: Issue #11 canonical 6000169298, release `57fd8a9c53ce241c05a8d5b351034bed7ba56b14`; continuation branch `pi0267/executor-decoupled-recovery`. This is progress, not a terminal or quality PASS.

### Actual scientific/source/data evidence preserved
- DEV joint/EEF 8/8 each; TRAIN 93/96 each; VAL 15/16 each. All original failures and the horizon deviation remain preserved; no source episode was rerun for this transport/archive continuation.
- 248 actual source/fixture attempts independently CPU-recomputed. Four always-open negatives have no task success/persistent success/sustained grasp.
- Actual data/processor admission: 64,908 genuine H40 windows (55,893 TRAIN / 9,015 VAL); TRAIN-only statistics; all saved codec/input/mask/pad/gate checks passed. Existing source/data canonical 6006768862 remains valid.

### Newly closed durable data artifact
- Public repo `leoyuer/pi0246-gate-pass-campaign`, prefix `evidence/PI0276/maniskill-stackcube-20261006/training_data`.
- Fixed revision `0f64f053ad69b6b4803e712000a0b136cc4623d3`; tree `d30e98ac1979e1f6a8ad98c3131d50f98eb97492fb087a89895921a2f25c9fd1`.
- Actual receipt `PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS`: 249 members, source bytes 28,736,239,352; complete tar 28,736,430,080 bytes / SHA256 `fc85ce844d732f6dd4289edbedc2eb22e3f17f4f8e05c5a800859fe450952fa5`.
- Exact receipt copied into execution commit `7553d4c9`: `reports/PI0276/execution/HF_TRAINING_DATA_FULL_READBACK.json`, SHA256 `bc2ce44a717d5c84132a1021ea82deab3d4f978b77d240d1f4641e436ed7c350`.

### Actual runtime and engineering repairs
The original PC SSH connection reset was preserved. The finite owner waited read-only through the VPN outage and resumed automatically after connectivity returned; it did not repeat the completed archive. An attempted detached Hopper provisioning wrapper then disappeared without a completion receipt. A bounded stdlib check found both plain and nohup descendants started but did not survive login-session closure. That failed wrapper, logs, and deviation are preserved under the PI0276 runtime repair directory.

Root transport repair is host-specific: Corelab keeps exact-argv detached adoption; Hopper staging/CPU phases retain foreground SSH with keepalive; actual model compute remains sealed PBS only. Narrow CPU checks cover actual RPC generation, exact process identity/adoption, read-only reconnection, and wrapper receipt execution with a fake subprocess—zero learned queries.

Same-state Hopper provision process PID 1929037 was actually verified running under the kept-open SSH session, downloading the fixed artifact (3,573,547,008 bytes staged at the latest recorded sample). The earlier failed provision never reached model computation. No training job has yet been submitted, no actual canary PASS exists, and TASK8192 has not started. This progress must not be relabeled TRAINING_PASS.

Next automatic path remains pinned data materialization → Hopper CPU/module binding/seal → real 8 vs 4-save-restore-4 canary → fresh TASK8192 → final-only fixed offline/physical evaluation → independent CPU/video/HF/terminal. No new recipe, sample/seed replacement, teacher call, WBC, real hardware, mail/monitor or merge; selected/deployment remain NONE.
