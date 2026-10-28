# PI0242-R1 native-flow parity and metric-correction closure

Date: 2026-09-21 (Asia/Singapore)  
Terminal: `PI0242_R1_NATIVE_PARITY_REVIEW_COMPLETE — STOP_FOR_PI_SINGLE_RECIPE_DECISION`

`selected_checkpoint=NONE`  
`training=NOT_AUTHORIZED`

This is a read-only correction and affected-matrix review. The original
PI0242 stock result and the PI0241 negative result are retained; this report
does not authorize training, WBC, simulation, CALIB/DEV-E/F, DAgger, hardware,
selection, or PR merge.

## Identity and preserved attempts

- Fixed R1 protocol: `030bfa291c0f12dd83141974db846a103f04f2b2`
- Audited parent: `b3a6c92012ef6949bb08dc4a4ed41dd9760a907e`
- Original PI0242 closure: Issue #11 comment `5755139669`
- Execution branch final commit: `99cf13d06b360097da5ea8186f8d8e1de78ff7de`
- GR00T source: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`

Engineering attempts were sequential and preserved:

| job | result | scientific status |
|---|---|---|
| `625603.hopper-m-02` | PBS direct-output/usecp failure before runner | zero model calls |
| `625604.hopper-m-02` | fail-closed because logs were placed inside the required-fresh output root | zero model calls |
| `625606.hopper-m-02` | fixed `sample_time(device,dtype)` keyword bug after 1 stock + 4 direct partial records | partial, not sealed; no parity |
| `625607.hopper-m-02` | all matrix records/NPZs written; post-run summary layer had a metric-key bug | valid matrix preserved; finalized CPU-only |

The valid matrix ran on `hopper-04` H100 with Python `3.12.3`, torch
`2.9.0+cu128`, CUDA `12.8`, and science-runner commit
`29ec76af09437c2a4f7bd36e88223bbebedcf8dd`. Finalization used no model load or
forward and was recorded separately in the run card.

## R1 output inventory

Output root:
`/scratch/e1192548/pi0242_r1_native_parity_20260921_v3`

| artifact | SHA256 |
|---|---|
| `pi0242_r1_run_card.json` | `1e3df7dbe72ca003e7d8709c4d46537cc8188c1584753f7209aad67941224a62` |
| `pi0242_r1_summary.json` | `3ba32571eafa3d0acf5079f14c13ed7c7f41f211becbc2a0b9ccdc23f0f3269e` |
| `pi0242_r1_closure.json` | `d2038b6b6df2d48f095e802bdbc05a6f18d5b9f703eced23f4315c2d4e01bff1` |
| `pi0242_r1_metric_correction.json` | `af3b1342c514fb461aa657d09261ff663fc70614d5ed95736a8447dd8bea451e` |
| `pi0242_r1_stock_records.jsonl` | `dcc932ea23a519a4352aad5013c9ea6d099a07acca4d7c82f6ab67cb4963faba` |
| `pi0242_r1_direct_records.jsonl` | `dce079966f7f91b67c85bbcd73bbe57bb59fb2fc1c97916699f569b84e7cdabd` |
| `pi0242_r1_native_parity.jsonl` | `ce6ca46222a1ea2dc49bc492d3f5b6dc1e3b9af57f7c4db361ddf08d0480a9f7` |
| `pi0242_r1_stock_traces.npz` | `fbaaab0c9d7db1225f3a2eb829ec96860a634841902318b40b12083766fe273c` |
| `pi0242_r1_direct_arrays.npz` | `80080df92f1b112863b875f7c7de6813622ca96e0bb7bf44ec6d2ecbca7933d2` |
| `pi0242_r1_native_parity_arrays.npz` | `d0270d6fd770fbc740e5fe191548252976fbc28e9d5b2f753e380bd2038950a5` |

The closure is `PASS_READ_ONLY`. Counts are exactly 16 stock captures, 64
corrected direct-flow cells, and 32 native/helper controls. Forbidden counters
are all zero: training, backward, optimizer, WBC, simulation, CALIB, DEV-E,
Final-F and DAgger.

## Feature single-pass provenance

The corrected runner snapshots the mutable backbone output before the official
native action head mutates it, then constructs independent processed-once and
processed-twice snapshots. The checkpoint actually uses `use_vlln=true`,
`LayerNorm`, a 4-layer `SelfAttentionTransformer`, `AlternateVLDiT`, and
position embeddings.

Across the 16 stock captures:

- raw → processed-once median feature max-absolute difference: `15247.0`;
- processed-once → processed-twice median max-absolute difference: `48.0625`;
- processed-once → processed-twice exact cases: `0/16`.

Thus the old PI0242 direct branch did not receive the native `F(h)` snapshot;
it received `F(F(h))`. The numerical effect is measured, not assumed. The
stock trajectory itself still uses the official native path.

## Native/helper parity

For every model/window/condition at `t0` and `q50`, the native action-head
forward was run with fixed noise and the actual captured `x_t`, discrete time,
processed features, state/embodiment, masks and attention arguments. Stage26's
helper was then run on those same captured inputs.

| control | result |
|---|---:|
| native/helper pairs | `32/32` |
| exact parity | `32/32` |
| maximum action-velocity difference | `0.0` |
| maximum action-encoder x difference | `0.0` |
| maximum diffusion hidden-state difference | `0.0` |
| maximum encoder-feature difference | `0.0` |

The helper implementation is therefore closed against the native action-head
forward on the measured support. This does not make the stock sampler and
training graphs identical; it proves the corrected helper is not introducing
another implementation mismatch for the captured native inputs.

## Angular-speed correction

The old sidecar used prediction-versus-expert orientation error divided by
`dt`. The R1 sidecar uses adjacent predicted rotations:

`geodesic(R_pred[t], R_pred[t+1]) / dt`.

All 16 rows were recomputed from the sealed old stock NPZ. The original gate
was not rerun or changed; all original stock decisions remain `REJECT 16/16`,
and the corrected captures agree with the original gate decision for `16/16`
cells. The complete old/new/delta rows are in
`pi0242_r1_metric_correction.json`.

Representative median sidecar changes by model/condition:

| model / condition | old max angular speed (rad/s) | corrected (rad/s) |
|---|---:|---:|
| initializer / `C_I` | 11.85 | 2.40 |
| initializer / `C_T` | 11.84 | 2.45 |
| final1000 / `C_I` | 2.88 | 2.11 |
| final1000 / `C_T` | 2.91 | 2.09 |

This correction changes a diagnostic sidecar only; it does not overturn the
original SafetyGate result.

## Original versus corrected direct flow

Each corrected JSONL row contains the original record locator, original flow
and physical metrics, and the cell-level numeric deltas. The table below gives
the final1000 medians; values are `active-flow MSE / XYZ RMSE m / SO(3) RMSE
rad`.

| condition | q0 original → corrected | q10 original → corrected | q50 original → corrected | q90 original → corrected |
|---|---|---|---|---|
| `C_I` | `0.0144/.2456/.1880 → .0129/.2333/.1691` | `.0178/.3297/.1817 → .0176/.3276/.1748` | `.0192/.3616/.1893 → .0180/.3568/.1729` | `.0382/.4398/.2174 → .0300/.3962/.1635` |
| `C_T` | `.0171/.2469/.1905 → .0132/.2309/.1640` | `.0213/.3338/.2066 → .0171/.3319/.1702` | `.0224/.3703/.1993 → .0182/.3591/.1685` | `.0386/.4478/.2450 → .0313/.4073/.1705` |

The correction materially changes the old direct numbers, especially at later
time probes, but the corrected final1000 physical reconstruction remains about
`0.23–0.41 m` XYZ and `0.16–0.17 rad` SO(3) on this four-window support. It
remains far from the expert scale. The original direct metrics must therefore
not be used unchanged, but the corrected evidence still supports a remaining
local action-head/objective-fit deficit.

## Updated attribution

1. `NATIVE_HELPER_PARITY_CONFIRMED`: the corrected helper and native
   action-head forward are exact on all 32 controls.
2. `DOUBLE_PROCESSING_NUMERICALLY_MEASURABLE`: the old direct path used
   `F(F(h))`; `F(h)` versus `F(F(h))` is not inert on the measured inputs.
   This explains why the original direct-flow numbers required correction, but
   it does not explain the full task failure by itself.
3. `CORRECTED_LOCAL_OBJECTIVE_FIT_DEFICIT_SUPPORTED`: after correction, local
   teacher-forced physical reconstruction remains far from expert while the
   native stock trajectories remain gate-rejected. This conclusion is limited
   to the fixed bridge4 support.
4. `CONDITIONING_PATH_EFFECT_MEASURED`: C_T/C_I remains a paired pixel-path
   difference with smaller effects than the model/fit deficit; no target,
   stats, embodiment, slot, frame or ABI mismatch was confirmed.
5. `LOCAL_TO_FREE_RUNNING_GAP_ON_PROBED_SUPPORT` remains unsupported because
   the corrected teacher-forced reconstruction is already poor.

No generalization claim is made from four windows and four time probes.

## One proposed next recipe, not authorized here

The single PI decision candidate is a controlled one-arm action-head/projector
adaptation targeted at the corrected local-flow deficit: initialize from the
PI0226 adapted baseline1000, keep the official dense-relative H40 model,
StateV1/ABI, original 16 TRAIN windows, `C_T` preprocessing explicitly bound,
the existing AdamW/loss family and seed, and change only the preregistered
update budget selected by PI for this diagnostic. No sweep, new seed, new
data, sampler change, gate change, WBC, or DEV/F access is implied.

This is a recommendation only. Training/backward/optimizer remains
`NOT_AUTHORIZED` under PI0242-R1.
