## PI0288 actual execution closeout — ENV_READY_INTERFACE_BLOCKED

Authority remains #6087400691 / `ede6d93b0ced2b0d409f992e7168b066da1f2c32`; this is an execution receipt, not a new authorization or a rewritten old verdict.

Implementation and raw evidence: **`2e8bbecfabb2f9c57e4b397f83cdcc310e58c12e`**, on `pi0267/executor-decoupled-recovery`.

[Full report](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2e8bbecfabb2f9c57e4b397f83cdcc310e58c12e/reports/PI0288/execution/REPORT.md) · [Adjudication](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2e8bbecfabb2f9c57e4b397f83cdcc310e58c12e/reports/PI0288/execution/ADJUDICATION.json) · [Zero-step native model audit](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2e8bbecfabb2f9c57e4b397f83cdcc310e58c12e/reports/PI0288/execution/evidence/DEFAULT_MODEL_CPU_AUDIT.json) · [Original first integration](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/2e8bbecfabb2f9c57e4b397f83cdcc310e58c12e/reports/PI0288/execution/evidence/55f23c661cdc4c8aa50a8731928dc101/physics.jsonl)

### Actual result

- User confirmation fixes this OCS2+MuJoCo package as the future deployment route. Parent native build/hold PASS remains unchanged. The nine current selected core binary/model/RL/config hashes match. No controller/gain/model/cost/solver change.
- One new setup episode, **one real integration (0.002s)**. Delivered native startup calls `mj_makeData`; it does not apply its supplied `home` keyframe. Four calf joints start at q=0 versus raw upper=-0.83776rad. A separate zero-integration CPU load reproduces this 0.83776rad default violation.
- First integrated calf values ≈-0.008233rad, violations ≈0.829527rad. All exceed registered immediate hinge threshold0.001rad. Read-only observer raised native SIGINT. Original owner result `Feedback unavailable` is preserved; independent raw adjudication identifies prior safety stop, not network/compile failure. The arm-joint numerical warning is also retained.
- Six G1 movements **NOT_RUN**, not a fabricated 0/6 tracking score. G0 clock/FK/expiry and G2 equipped stack are not qualified. No reference was sent, no teacher/pilot/formal data or RGB was collected. The stop precedes a task video; no substitute video is claimed.

PI0288§3 explicitly rejects illegal initial conditions and forbids temporary pose substitution. `home` exists but was not tried or declared safe. Changing initialization or bypassing the recorded violation requires an explicit initial-state/reset binding before resuming; no new pose/seed search performed.

### Reusable work actually completed

Original reference generator/scorer, named state maps, provisional scene/gripper/camera overlay, contact-based teacher, committed desired-command recorder, H40/data codec/admission modules and finite native owner. Local CPU tests **13 PASS / 1 private-source skip**; that actual private callback test independently passed on Corelab (original accept/send exactly once, invalid JSON leaves reference unchanged). Zero-step observer-copy check leaves original integration state byte-identical. Task runtime collection modes are not connected/claimed complete; official GR00T CPU batches are NOT_RUN because real admitted data does not exist.

Separate G2 blocker: deployment gripper model/revision, verified working speed/force and contact-pad geometry/material/friction still missing. Delivered 2025 model declarations alone are not hardware operating specifications. One clarification already requested; provisional assets remain unfrozen.

### Accounting / preservation

One setup slot; no reference/replacement/pilot/data slots. Physics observed1; RL/MPC/direct-dispatch counts **UNOBSERVED**, retain prelaunch upper reservations6000/12000/36000 rather than fabricated zeros. GPU service interval≤8.011s; total CPU core time UNOBSERVED, no all-cap PASS claim. Measured isolated owned outputs543,852 logical bytes /692,224 allocated bytes at inventory time. No native owned processes remain. Nine small original files (16,893B) have byte-exact server readback and Git blob hash checks.

GR00T forwards/optimizer/hardware=0. No H200/PBS/mail/monitor/cleanup/merge. Private source/assets stay private. Old PI0287 and all prior results unchanged. **Overall PARTIAL; task data NOT_READY; selected/deployment=NONE.**
