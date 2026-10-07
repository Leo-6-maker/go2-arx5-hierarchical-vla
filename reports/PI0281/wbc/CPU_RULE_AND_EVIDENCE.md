# PI0281 WBC CPU proof and one repair

Authority: protocol release `360b412f55be8aec1568054d7d039ce7de2bc597`, canonical comment `6040516315`. This is a prepared implementation awaiting owner publication. No PI0281 physics or actor/model forward has run in this lane.

## Proven defect and limits

The CPU reproduction runs in the actual bound WBC Python 3.8.20 / PyTorch 2.3.1 / CUDA 12.1 environment, with CUDA hidden and every tensor on CPU. The bound private `modules/trajectory.py` (`8386841b29b58f324cd2e9afa086439add4183992be2466f688382be8ed9d76a`) projects globally onto 16 forward candidates within 0.15 m. The prescribed return path is only 0.10 m long and contains two positions for most heights. On ideal raw reference poses, the exact delivered float32 function makes a single 0.1000000015 m jump. This is branch selection, independent of physics and policy quality. It can falsely place progress at the returned endpoint while the time law is near the start.

On PI0280 saved actual TCP positions and FK orientations, native progress reaches 0.10 m by the historical 0.52 s sample. The repair reaches 0.0349037 m. The maximum saved native/FK position disagreement is recorded in `CPU_PROJECTION.json`. Historical native link orientation and actor input are absent: this replay uses an explicit FK orientation proxy and cannot establish exact historical observation equality.

The CPU counterevidence matters: the repair's lateral residual on the final saved pose is about 0.0369 m, versus native's 0.00316 m. Avoiding a branch jump is not evidence of tracking success. New fixtures may still fail the unchanged native stop or tracking criteria.

## Frozen rule

Use the same prior valid progress, native 0.15 m window, `M=16`, native `L`, candidate poses and SE(3) scores. Select the **first ordered local minimum**: score less than or equal to both neighbors; an absent endpoint neighbor is positive infinity. Ties and flat plateaus select the smallest index. Comparisons use the existing float32 values; there is no added tolerance, fitted term, search bound, or tuning knob. Progress remains monotonic. The existing reference replacement resets reference-local progress to zero; this repair changes no lifecycle state.

The rule modifies one in-process imported projection function. Original source files, weights, gains, time law, measured progress derivative, `s_ref`, `timing_err`, native stops, scene and TCP stay bound. The ideal hold result is unchanged; monotonic translation reaches 0.05 m with the same approximately 3.3 mm largest step. The ideal return reaches 0.10 m with a largest repaired step of approximately 6.7 mm. Existing-progress, dwell, no-regression and replacement CPU cases are retained.

## Orientation chain

Raw world quaternion is xyzw. Canonicalization stores world xyzw and the external anchor translates positions only. The native preview applies inverse heading rotation; the target body pose applies inverse full base rotation. The actual actor helper reorders xyzw to wxyz, then the delivered PyTorch3D shim packs the first two **matrix columns interleaved**. This packing is preserved. Nonidentity CPU checks cover raw → canonical → anchored curve → preview and heading/body packing; numerical errors are in the JSON evidence. No orientation defect has been proved.

The native error vector remains the sign-corrected quaternion vector part, the original `sin(theta/2)` proxy. It is not renamed or replaced by a true rotation-vector error. Native measured orientation is link6 orientation; its TCP adds `[0.1424, 0, 0.0001057]` with identity rotation. FK uses that same offset and orientation. New fixture evidence records the actual rigid-body link quaternion, native TCP quaternion, FK, target/body/preview quaternions, and the exact actor history input before each arm forward. Actual chain equality and tracking accuracy remain pending.

## Bounded execution prepared

Exactly native and repair versions, each with seed 4242 hold, monotonic 5 cm and original return fixtures, each 10 s. Per-fixture reservation is 1 physical start, 501 controls, 1,000 actor forwards, 2,004 integrations, 2,004 gripper dispatches and 100 RGB pairs. Six fixtures total 12,024 integrations, 3,006 controls and 6,000 actor forwards. The shared published-code guard and central attempt reservation are mandatory; per-event counts use a local journal. The finite wrapper saves completion status, real elapsed time and `RUSAGE_CHILDREN` after waiting for the process.

Final gates are unchanged: full 10 s with native hard stops; final 2 s position RMS/max ≤ 0.02/0.04 m, SO(3) RMS/max ≤ 0.15/0.30 rad, hard joint-limit tolerance 1e-6. Soft-limit negatives remain separate evidence. No quality retry. A technical replacement requires the same state and owner notification. Before each start, live GPU0 free memory must be at least 12 GiB. The source resident process has priority; low headroom is a resource blocker.

Only a three-fixture pass admits the four frozen target expert attempts; at least 3/4 persistent expert success admits the four recorded model EEF attempts. A failed fixture panel leaves these dependent results null. No fresh model query, training, statistics fitting or data generation is authorized here. Machine-readable rule, allocation and code hashes are in `REPAIR_CARD.json`.
