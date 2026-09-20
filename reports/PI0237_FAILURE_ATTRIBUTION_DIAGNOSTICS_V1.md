# PI0237 failure-attribution diagnostics v1

Status: `STOP_FOR_PI_ATTRIBUTION_REVIEW`
Selected checkpoint: `NONE`
Protocol: commit `6e83c5494772190f5c8082f730f06723d9c9bfda`, blob `9e055861c0c36789bab8098591855040a3ed66f5`

## Execution closure

The diagnostic ran in the independent branch `pi0237/failure-attribution-execution` from the immutable protocol commit. The remote output root is:

`/scratch/e1192548/pi0237_failure_attribution_20260920T014727Z`

P0 confirmed the sealed PI0236 family is `623593`, the TRAIN/CALIB source-rollout split is disjoint (`612/68` rollouts), and the fixed cohort is 64 TRAIN plus 64 CALIB. P0–P3 used no GR00T/VLM weight loads, `get_action`, or model forward. No PI0237 PBS job was submitted; no WBC or simulator execution occurred.

| Phase | Result | Evidence |
|---|---|---|
| P0 | `PASS` | `pi0237_p0_admission.json` |
| P1 | `PASS_READ_ONLY` over all 5,440 sealed records | `pi0237_p1_gate_diagnostics.json` |
| P2 | executed; `EXPERT_NONREJECT_PASS=false` | `pi0237_p2_expert_gate.json` |
| P3 | `PASS_READ_ONLY` with review-required normalized excess | `pi0237_p3_codec_roundtrip.json`, `pi0237_p3_lossy_transform.json` |
| P4 | `NOT_RUN` — blocked by P2/P3 admission conditions | no inference artifacts created |
| P5 | complete CPU synthesis | this report |

## P1: sealed model-record description

All 5,440 historical rows were read without new inference. Issue occurrences are not mutually exclusive:

| Issue | Occurrences |
|---|---:|
| `LINEAR_SPEED_REJECT` | 5,431 |
| `ANGULAR_SPEED_WARN` | 3,557 |
| `ANGULAR_SPEED_REJECT` | 228 |
| `INITIAL_ROTATION_DISCONTINUITY_WARN` | 5,258 |
| `INITIAL_ROTATION_DISCONTINUITY_REJECT` | 25 |
| `INITIAL_POSITION_DISCONTINUITY_WARN` | 1,074 |
| `INITIAL_POSITION_DISCONTINUITY_REJECT` | 12 |
| `LINEAR_SPEED_WARN` | 9 |
| `PATH_LENGTH_SANITY` | 40 |

All-row reference-distance summaries are descriptive only and are not task-success scores:

| As-evaluated profile | Median reference min-distance (m) | Selector non-REJECT rows |
|---|---:|---:|
| baseline-1000 | 0.258416 | 0 |
| fresh-2500 | 0.759239 | 0 |
| fresh-5000 | 0.664165 | 0 |
| fresh-7500 | 0.668037 | 7 |
| fresh-10000 | 0.713394 | 0 |

Per-tick spike index, expert XYZ/SO(3) error, normalized clipping/saturation, achieved TCP, WBC tracking, and native termination were not present in the sealed PI0236 records and remain `NOT_OBSERVED_IN_SEALED_ARTIFACT`.

## P2: independent expert-label gate

P2 evaluated all 1,088 CALIB windows plus the fixed 64 TRAIN windows through the actual provider/chunk and `TrajectorySafetyGateV1`, with the actual StateV1 row and the pinned rot6d-to-quaternion helpers.

| Split | ACCEPT | WARN | REJECT |
|---|---:|---:|---:|
| CALIB (1,088) | 37 | 1,039 | 12 |
| TRAIN (64) | 2 | 62 | 0 |
| Total (1,152) | 39 | 1,101 | 12 |

All 12 rejects were `INITIAL_ROTATION_DISCONTINUITY_REJECT`; no gate threshold was changed. The first P2 attempt incorrectly assumed every parquet window started at timestamp zero and stopped before producing evidence. Source inspection showed the Stage19 materializer preserves absolute source timestamps; the minimal diagnostic correction now validates monotonic 0.05-second spacing and records the source offset. The frozen selector's original `_build_chunk` remains unchanged and reconstructs its evaluator contract at `observation_timestamp=0.0`, so both timestamp facts are recorded side-by-side rather than silently re-timed.

This is direct negative evidence against the P4 expert-label admission condition. It does not establish that the model cannot learn, and it does not authorize relaxing the gate or changing labels.

## P3: real train/inference codec round-trip

P3 used the actual offline processor, TRAIN stats/relative stats, embodiment, dense-relative H40 action mapping, StateV1 anchor, slot ordering, and the real decode path. It covered the same full CALIB label set as P2 plus the fixed TRAIN cohort: 1,152 windows.

| Quantity | Maximum observed | Pinned tolerance |
|---|---:|---:|
| XYZ L2 (m) | `1.5147e-7` | `1e-5 m` |
| SO(3) geodesic (rad) | `1.4250e-5` | `1e-4 rad` |
| Gripper absolute | `0` | `1e-5` |
| Normalized round-trip | `1.6525e-5` | `1e-5` |

Two rows exceed the normalized float32 round-trip tolerance. No new clipping, smoothing, or retiming was added; the processor does not expose an independent clip-outlier counter. This is therefore recorded as `LOSSY_TRANSFORM_REQUIRES_PI_REVIEW`, not silently treated as numerical zero.

## Attribution matrix

1. `EXPERT_LABEL_GATE_MISMATCH_OBSERVED` is supported: 12/1,152 independent expert labels are rejected by the frozen gate, all on initial rotation discontinuity. The exact cause remains open among label/state anchor, frame/rotation semantics, and the frozen evaluator time contract; no one is selected as the root cause here.
2. `LOSSY_TRANSFORM_REQUIRES_PI_REVIEW` is supported: two real codec rows exceed the pinned normalized tolerance. This is independent of the P2 gate result and needs a scientific decision before any model inference is interpreted.
3. `INCONCLUSIVE` for model fit versus generalization: P4 was correctly not admitted, so no new baseline/fresh TRAIN-vs-CALIB policy queries were made. P1 alone cannot distinguish underfit, codec/gate incompatibility, or generalization.

## Counters and boundaries

- Historical PI0236 records read: 5,440; historical PI0236 queries remain historical and are not counted as PI0237 queries.
- PI0237 reused tensors: 0.
- Expert gate evaluations: 1,152.
- CPU codec-transform windows: 1,152.
- PI0237 attempted/completed `get_action`: `0/0`.
- Model weight loads / model forwards: `0/0`.
- Training, backward, optimizer, scheduler, DAgger: `0`.
- WBC steps / simulator steps / hardware: `0/0/0`.
- DEV-E / Final-F reads: `0/0`.
- New PBS jobs / retries: `0/0`.
- P4 H200 diagnostic job: not submitted.
- `selected_checkpoint`: `NONE`.

## Evidence

The complete machine-readable inventory is `pi0237_evidence_inventory.sha256` under the remote output root. The small branch manifest and protocol binding identify every phase artifact and hash. Large P1/P2/P3 tables remain in the declared remote diagnostic root and were not copied into Git.

Final decision token: `STOP_FOR_PI_ATTRIBUTION_REVIEW`.
