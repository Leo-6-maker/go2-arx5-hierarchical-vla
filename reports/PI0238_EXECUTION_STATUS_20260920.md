# PI0238 execution status

Status: `BLOCKED_BEFORE_G0_COMPLETION`

Protocol: `044e4d5ef8b4c1effb8236190c0eb083fa49979b`
Audited execution baseline: `998eba5dc571a600a6c25a350b4a0b5dd2dad40a`
Execution branch: `pi0238/audit-to-sim-execution`

## Attempts

| Job | Result | Evidence |
|---|---|---|
| `624206.hopper-m-02` | cancelled before script output after the site injected the standard 1-GPU allocation; no scientific read | scheduler history only |
| `624209.hopper-m-02` | G0 stopped on a false tree-drift check; zero model activity | `/scratch/e1192548/pi0238_audit_to_sim_20260920T031209Z/evidence/g0_failure.json` |
| `624229.hopper-m-02` | G0 stopped because the PBS wrapper named a non-existent modality path; zero model activity | `/scratch/e1192548/pi0238_audit_to_sim_20260920T034100Z/evidence/g0_failure.json` |

Both executed attempts used the platform auto route (`auto -> small`) and the site allocation `1 GPU / 12 CPU / 225 GB / 2 h`. The diagnostic script exported `CUDA_VISIBLE_DEVICES=''`; no model weights, policy forward, or `get_action` was reached.

## Repairs sealed

- Tree hashing now matches the existing PI0228 implementation: file entries are sorted by relative path before digesting. Commit `67bda86f38c8d1c2bd1fb1797f087f5b8b856944`.
- The canonical modality binding is `/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py`, SHA256 `244f37f28e62b7b392a1598b78e78f00e67693f6c08a606b3c5358830ada8b25`. Commit `b31874ccac95d3144b11cda9fdd90e5829a9dba1`.
- Remote exact-environment unit test passed: `7/7`.

## Gate state

G0: `BLOCKED` before final evidence closure; G1–G5: `NOT_RUN`.

G4 attempted queries: `0`; model weight loads: `0`; model forwards: `0`; `get_action`: `0`; training/backward/optimizer: `0`; WBC/simulation/DAgger: `0`; DEV-E/F reads: `0`; `selected_checkpoint=NONE`.

The filesystem quota result remains `FILESYSTEM_QUOTA_UNOBSERVED`; no G4 admission is allowed. The one permitted CPU replacement submission has been consumed, so execution stops here under the fixed protocol budget.
