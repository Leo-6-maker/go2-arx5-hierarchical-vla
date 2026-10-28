# PI0242 train/infer bridge attribution closure

Date: 2026-09-21 (Asia/Singapore)  
Terminal: `PI0242_ATTRIBUTION_COMPLETE — STOP_FOR_PI_SINGLE_RECIPE_DECISION`

## Decision boundary

This is a bounded read-only attribution result. It does not promote a model,
select a checkpoint, authorize training, or authorize WBC/simulation.

`selected_checkpoint=NONE`  
`training=NOT_AUTHORIZED`

The fixed protocol was `07d96cf750b7374ab43fd53a044d48e8ab7fab53` and the
audited parent was `c1b2bd4634cc8cd5b756c9e3d9423d70be88b397`. Issue #11
comment `5754799090` is the canonical authorization; PR #77 comment
`5754800159` is only the execution pointer.

## Evidence and execution identity

P0 was completed read-only under the sealed evidence root:

`/scratch/e1192548/pi0242_train_infer_bridge_20260921/p0-r4`

| artifact | SHA256 |
|---|---|
| `pi0242_p0_admission.json` | `39e35b336bf88eae00c3e57dde9b62f5b3781892cb3a55fe5843ddebc11827e2` |
| `pi0242_evidence_index.json` | `8e60abb4b72d6d539f4ad84760c6ad33250616c709a3cecdd2ebbd3376eb538e` |

P0 status was `PASS_READ_ONLY`. It recomputed the original 64 screen cells,
classified recovery evidence as observed/reconstructed/unavailable, compared
the actual training-pipeline condition `C_T` with the screen condition `C_I`
on the original 16 windows, and verified the action/mask/132D physical
roundtrip. P0 found the actual conditioning difference only in
`inputs/pixel_values`; action and mask were exact, with active slots `0:10`.
The P0 runner reported zero model loads/forward/get_action/backward/optimizer,
training, WBC, simulation, CALIB, DEV-E and DEV-F operations.

The recovery audit does not treat `actual_sample_consumption=1000` as proof of
historical sample consumption. The valid `500 -> 625395 -> 1000` lineage and
the missing independent recovery-consumption rows remain separately labeled in
the P0 evidence. No missing history was reconstructed as an observed event.

P1 used the existing single job, without a duplicate submission:

| field | value |
|---|---|
| PBS job | `625549.hopper-m-02` |
| actual host/device | `hopper-04`, one H100, `cuda:0` |
| initial request | one GPU, 12 CPU, 225 GB, `gpu_model=H200` |
| scheduler recovery | same job altered to `gpu_model=H100` after H200 queue shortage |
| exit | `0` |
| code | `818f7a330431a984a4eeb489c2376a724849378c` |
| GR00T source | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| Python / torch | `3.12.3` / `2.9.0+cu128` |
| CUDA | `12.8` |
| seed | `22301` |

All paired cells were executed in this one H100 process; no Corelab/Hopper
outputs were combined into a paired comparison. The output directory retains
the legacy name `p1-h200`, but the scheduler/device identity above is the
authoritative runtime identity.

P1 artifacts are under:
`/scratch/e1192548/pi0242_train_infer_bridge_20260921/p1-h200`

| artifact | SHA256 |
|---|---|
| `pi0242_run_card.json` | `d20e8b9efcd9f537239357ab65b4e33f78c3f054bfccfd20d254ab8b6dd6fe37` |
| `pi0242_p1_closure.json` | `c66e6e402476cdebde5753e51a99368d0244650a467ab6b9da1d0917546c43a8` |
| `pi0242_p1_summary.json` | `2d7fda85aa970a244791360fce66f67b38b7c18c691df9fd8e18e80a64420897` |
| `pi0242_bridge_records.jsonl` | `c2afa4fe777e98480e43c1f295f5adaaf34369614e6c9059f40c7e7b07b2f9e7` |
| `pi0242_p1_stock_traces.npz` | `15320c8c0bb5181451490c6466bfeaec6c0d1df476e2c2b8b9ded96face1681d` |
| `pi0242_p1_direct_arrays.npz` | `817161d3148a4f56567b01b775c876ae9f7909bb4378809f6f156e414f09a9b0` |

The P1 closure is `PASS_READ_ONLY`, with `selected_checkpoint=NONE`.

## Fixed cohort and budget

The preregistered hash rule selected one original PI0241 window per target
family, independently of performance:

| target | source window | episode | phase |
|---|---|---:|---:|
| A | `pi0227_persistent_train_a_j03_yp015_s03_yp050_r00_r00_227072:obs-0033` | 1013 | 1 |
| B | `pi0227_persistent_train_b_j00_nominal_s00_c0_r02_r02_227182:obs-0033` | 2485 | 1 |
| C | `pi0227_persistent_train_c_j01_xp015_s04_ym050_r03_r03_227399:obs-0013` | 5442 | 0 |
| D | `pi0227_persistent_train_d_j03_yp015_s00_c0_r02_r02_227602:obs-0046` | 8247 | 1 |

The fixed noise hash was
`d7512d2879ff2b0d0ffa98555c2692d079206c37d3b7e59ba93c97ef57d75639` for
each source row, and was paired exactly across initializer/final1000 and
`C_T`/`C_I`. The four time probes were `t=0`, `0.06776241796562847`,
`0.36966943557751086`, and `0.7837719744658148`, discretized as
`0, 67, 369, 783`.

Observed P1 counts:

| counter | count |
|---|---:|
| stock sampler queries | 16 |
| model `get_action` calls | 16 |
| stock action-head sampler calls | 16 |
| stock denoising forwards | 64 |
| backbone forwards | 16 |
| direct-flow cells | 64 |
| direct action-head forwards | 64 |

Forbidden counters were all zero: training, backward, optimizer steps, WBC,
simulation, CALIB, DEV-E, Final-F and DAgger.

## P1 result matrix

### Stock free-running sampler

All 16 stock trajectories were `SafetyGate=REJECT`. The aggregate issue
counts were: `LINEAR_SPEED_REJECT=16`,
`ANGULAR_SPEED_WARN=14`, `INITIAL_POSITION_DISCONTINUITY_WARN=8`, and
`INITIAL_ROTATION_DISCONTINUITY_WARN=12`.

| model / condition | XYZ RMSE median (m) | SO(3) RMSE median (rad) | target min distance median (m) |
|---|---:|---:|---:|
| initializer / `C_I` | 0.484396 | 0.522472 | 0.189387 |
| initializer / `C_T` | 0.488266 | 0.515371 | 0.195394 |
| final1000 / `C_I` | 0.213020 | 0.098862 | 0.676870 |
| final1000 / `C_T` | 0.209237 | 0.099455 | 0.672922 |

The final1000 stock trajectory therefore improves orientation and endpoint
error relative to the initializer on these four windows, but remains a
physical task miss and gate reject. Its target minimum is about 0.67 m, not
the 0.035 m task radius.

### Teacher-forced local flow

The table reports median active-masked velocity MSE, physical XYZ RMSE from
the implied teacher-forced endpoint, and SO(3) RMSE. The endpoint is an oracle
diagnostic containing expert information; it is not a policy or WBC success.

| model | condition | q0 flow / XYZ / SO(3) | q10 flow / XYZ / SO(3) | q50 flow / XYZ / SO(3) | q90 flow / XYZ / SO(3) |
|---|---|---|---|---|---|
| initializer | `C_I` | 0.1879 / 0.4669 / 0.5938 | 0.1790 / 0.4856 / 0.5946 | 0.2201 / 0.5758 / 0.6063 | 0.2962 / 0.8161 / 0.6027 |
| initializer | `C_T` | 0.1717 / 0.4634 / 0.5667 | 0.1640 / 0.4905 / 0.5596 | 0.2043 / 0.5816 / 0.5814 | 0.2802 / 0.8245 / 0.5878 |
| final1000 | `C_I` | 0.0144 / 0.2456 / 0.1880 | 0.0178 / 0.3297 / 0.1817 | 0.0192 / 0.3616 / 0.1893 | 0.0382 / 0.4398 / 0.2174 |
| final1000 | `C_T` | 0.0171 / 0.2469 / 0.1905 | 0.0213 / 0.3338 / 0.2066 | 0.0224 / 0.3703 / 0.1993 | 0.0386 / 0.4478 / 0.2450 |

Final1000 reduces active local-flow MSE by roughly an order of magnitude at
the early probe (for example `0.1879 -> 0.0144` for `C_I/q0`), but the
teacher-forced physical reconstruction remains approximately 0.25–0.45 m in
XYZ and 0.18–0.25 rad in SO(3). The local objective is therefore not fitted
to the frozen expert scale on this diagnostic cohort.

## Attribution

### Supported

1. **`LOCAL_OBJECTIVE_FIT_DEFICIT_SUPPORTED`** — the final1000 model is much
   better than the initializer on the measured local active-flow loss, but its
   teacher-forced physical reconstruction remains far from the expert and its
   free-running stock trajectories remain gate-rejected and target-inaccurate.
   This is direct evidence for a remaining local action-head/flow fit deficit,
   not merely a sampler-only explanation.
2. **`CONDITIONING_PATH_EFFECT_MEASURED`** — `C_T` and `C_I` differ in the
   observed pixel preprocessing path, while action/mask/active slots and
   physical decode contract agree. Paired output differences are small relative
   to the initializer-to-final model effect. This is a measured conditioning
   effect, not a confirmed contract bug and not a complete explanation of the
   failure.

### Not supported by this campaign

- `CONTRACT_MISMATCH_CONFIRMED`: no target/stats/embodiment/slot/frame/ABI
  semantic mismatch was found in the measured rows.
- `LOCAL_TO_FREE_RUNNING_GAP_ON_PROBED_SUPPORT`: the local teacher-forced
  physical reconstruction is already far from expert, so the data do not
  isolate a downstream sampler gap.
- Any generalization claim: four windows and four time probes are a bounded
  mechanism diagnostic, not a distribution estimate.

The P0 recovery gaps remain evidence gaps, not inferred failures. The P1
matrix also does not prove whether the next best recipe should change the loss,
learning rate, capacity, preprocessing, or sampler internals.

## One next decision, not executed here

The single recommended next intervention is a PI-reviewed, one-arm
action-head/projector adaptation targeted at the measured local-flow deficit,
with the actual training preprocessing path `C_T` explicitly frozen and
declared. It must be a new PI decision; no sweep, extra seed, loss/LR/capacity
change, sampler change, or training is authorized by PI0242.

No WBC, simulation, CALIB/DEV-E/F, DAgger, checkpoint promotion or selector
operation was performed. The execution branch contains only the diagnostic
runner and this evidence closure; PR #77 remains unmerged.
