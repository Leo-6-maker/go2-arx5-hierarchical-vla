# PI0163 synchronized baseline

Status: **PARTIAL**  
Return token: `CORELAB_PI0163_SYNCHRONIZED_BASELINE_FOR_PI_REVIEW`

Authority: Issue #11 comment `5652818601`, PR #69 comment `5652821773`.  The
execution branch is `stage45f/corelab-local-initial-state-pilot` at
`f0cb83de0fa373152611a37814320f86eef76588`.  The old PI0160 overage and its
v1/v5 unknowns are retained as historical unknowns; this campaign never
refunds, rewrites, or re-audits that ledger.

## Scope and boundary

The native Python 3.8 import-only admission passed before the experiment
(Python 3.8.20, Torch 2.3.1, CUDA available, no simulator construction).  The
model-side recovery used the sealed Cosmos **snapshot** directory, not its
cache root.  The first model attempt is preserved as a preprocessor-path
failure; it reached no model construction/forward and produced no raw output.
The single allowed mechanical recovery then completed the C1 q0 diagnostic.

No training, backward pass, optimizer step, PBS/H200, ICL, Validation-E,
Holdout-F, hardware control, WBC consumption of GR00T output, or new retry was
performed.  The only native WBC calls are the two bootstrap ticks and the
already-authorized C1 expert ticks recorded below.

## Fixed identities and runtime

| item | identity |
| --- | --- |
| N2 checkpoint | `seed2223@2000`, tree `09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152`, 19 files, 20,889,823,145 bytes |
| N2 stats | file `0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb`; payload `508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572` |
| fixed noise | file `83590e27b22847260a5c7cc2af365c088b352ccafc056ad741418a6e98168922`; array `5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd` |
| BASE config / stats | `85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be` / `c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9` |
| modality | `e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031` |
| GR00T / Cosmos | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` / revision `9ce19a195e423419c349abfc86fd07178b230561`, tree `1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9` |
| import-only evidence | `/var/cache/users/liuyu-nfs/stage45f_pi0163_import_only_20260913.json`, SHA256 `0de49fe2e16ce517014d0646dbd500c4c84008a860de7792e6f1a0c916033fdd` |
| model host/device | `corelab-4080s`, RTX 4080 SUPER GPU0, UUID `GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad`; admission snapshot 10,033 MiB used / 22,197 MiB free |
| model runtime | `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/runtime/venv/bin/python`, Python 3.12.3, Torch 2.9.0+cu128, model dtype `torch.bfloat16` |
| Cosmos snapshot used | `/var/cache/users/liuyu-nfs/stage40_frozen_inputs_20260907_v1/materialized/cosmos_reason2_2b_hf_cache/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561` |

## Native synchronized expert results

The fixed conditions were `C0_NOMINAL` and `C1_X_PLUS_5CM`.  A shared loader
failure would have stopped the loop; the loader passed, so the second
condition was run without retrying C0.

| condition | result | synchronized evidence and expert counters |
| --- | --- | --- |
| C0 nominal | **BLOCKED** at q0 | target not observable; bootstrap WBC `1`, reference updates `0`, q0/snapshots/rows `0`; failure SHA `ff66d4ca8861c7208ce087e77473d9e4fb01d2399942ef041b5704e1aecaa710` |
| C1 +X 5 cm | q0 **qualified**, expert episode **PARTIAL** | q0 body/wrist/StateV1/EEF captured after the native tick; q0 NPZ SHA `e0376e138bb1610a557164d3afffa3da1d7d0446a3b28406374360bbe7c441ad`, q0 JSON SHA `a4d53a962af98b074b8a9f4a88a2b238e15254db362b2be06b2134d454c67e06`; 50 rows were present on disk, 2 snapshots, WBC `51`, reference updates `2`; the next segment ended `TRACKING_ABORT` |

For C1 the target is the rendered `stage21_red_target` root
`[0.3000000119, 0.1000000015, 0.5799999833]` m (evaluation target
`[0.3, 0.1, 0.58]` m).  The q0 body and wrist PPM hashes are
`20db457b4d60f4c893f2e8c4e2e49858fa6930f09778ae0b7cbf055f3e8420b3` and
`471873ad7ffac7149251feb42474264e2e6c3f3268f0a036d934e3f99095ea73`;
StateV1 SHA is `15fca8d9eb02f7c6267d8a5376f62bcade1b67e07550357fd9a08a3129e93a5c`.
Both q0 and q1 SafetyGate decisions were `ACCEPT`; the handoff records
`physics_reset=false`, `physics_teleport=false`, `policy_history_reset=false`,
and `gr00t_output_sent_to_wbc=false`.

### Reference versus achieved C1 trace

The q0 expert reference itself starts at the measured EEF and reaches the
target exactly.  The live achieved trace is only a partial execution trace;
its first recorded point is after the bootstrap and therefore the start jump
is reported explicitly rather than folded into tracking error.

| trace | initial error (m) | minimum error (m) | final error (m) | progress (m) | path (m) | roughness (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expert q0 reference H40 | 0.194902 | 0.000000 | 0.000000 | +0.194902 | 0.194902 | 0.000375 |
| achieved C1 rows (50, partial) | 0.194902 | 0.089077 | 0.089077 | +0.105824 | 0.295349 | 0.002025 |

Achieved-vs-desired RMS is `0.031201` m, maximum pointwise error is
`0.169003` m, and the first-recorded achieved point is `0.146566` m from the
measured q0 EEF (`start_jump`).  The terminal row has
`termination_code=TRACKING_ABORT`; it is not a completed expert rollout.

## Fixed N2 q0 diagnostic

Only the qualified C1 q0 entered the model process.  The raw arrays were
written before CPU decoding.  Exactly one E4 free-running query and three
matched local-field probes ran under `torch.inference_mode()`:

| mode | t bucket | final error (m) | progress (m) | best progress (m) | active field RMSE | true zero-field RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FR | — | 0.258581 | −0.063680 | +0.099590 | — | — |
| matched teacher | 67 | 0.249051 | −0.054149 | +0.096860 | 0.830409 | 1.069131 |
| matched teacher | 369 | 0.196281 | −0.001379 | +0.104262 | 0.834956 | 1.069131 |
| matched teacher | 783 | 0.116898 | +0.078004 | +0.110028 | 0.884614 | 1.069131 |

The FR trajectory has path `1.870410` m and roughness `0.042375` m.  The
teacher rows have paths `1.894364`, `1.418538`, and `0.526996` m respectively.
All four decoded rows use the N2 stats route above and have finite raw arrays.
The table preserves the historical script's reconstruction-residual values
separately; the `true zero-field` column is recomputed as
`-(x1-x0)`, so it is not confused with the `(1-t)`-scaled reconstruction
residual.

The T16 mask is `[40,132]` with 190 active scalars.  Reconstructed label
round-trip errors are zero after the label-only anchor representation repair;
the original position error was `1.49e-8` m and original rotation error `0`.
The active pre-clip audit finds 7 values outside `[-1,1]` (max `1.738614`),
and 7 post-clip boundary values.  This is the actual pre-clip count, not a
boundary-saturation count mislabeled as clipping.

## Evidence and accounting

Model output root:
`/var/cache/users/liuyu-nfs/stage45f_pi0163_model_20260913_v2`.

| artifact | SHA256 |
| --- | --- |
| model result (`CORELAB_PI0163_N2_Q0_MODEL_RESULT_V1`) | `2f3f7e0c3afd5672a2cca273a6919192c0da1c74de520017b62d61001aa8ba87` |
| model admission | `4411fdd6a4ba253ccbed15dc479bc121c242a10daf9c5a51e2baa5b784690351` |
| matched table | `d602d3cc36f71cce722356a8db8750ceeaacd3456f1701af0d78a5a7c8ddd402` |
| raw manifest | `3b1c7b81c4675da25055e026290c8f3fae310a8d32ec7850190ce7596398c735` |
| FR raw | `e4cd63edc76fc6bec46c7eec221d39e171e1846f719a9b62da977dd9e1ff40dc` |
| teacher raw t067 / t369 / t783 | `833635de33d891d53580b527a8acf445b2943701e5d7e123b5401b052787ca4b` / `20b58b62680d98144deb0c34a11b2fa309d2480559889b7ccee4ea2a6515bcf6` / `47e69ff2b6854c656f26142dfcb77da0266988b8e8725d54bdde9d05eb31b705` |
| native rows / snapshots / SafetyGate | `613a0c3baa5392edccc57b89b3407f57bcd17774811ed6118739df15b2277a95` / `96aea970dca7ba7d3aaa248e230fdbc632f29973a6255dac039bc0a56a9e8e50` / `1bd04f8354419a32ba35d741bee8e6683be6e13dcf5c239ddabb03f29534e5e2` |
| handoffs / state events | `67f41154c870574e26f543fbdf269609c7bb73a11c89615dc4cae0f17900d6b6` / `5d9671d6bdcc6ca84343259f1a6babc5591edb47b17b71147fd37e0ce970d7f5` |
| persistent PI0163 ledger | `4b5f9fa4de61d38ef23be9740be1119e8800f5c9b22f64c0148bfae6aa647616` |

The immutable PI0163 caps are simulator `4`, native ticks `984`, reference
updates `16`, GR00T constructions `2`, conditioning `6`, action heads `21`,
model-side calls `27`.  Final used counts are simulator `2`, native ticks
`52`, reference updates `2`, GR00T construction `1`, conditioning `2`, action
heads `7`, model-side `9`; active claim is null, with no cap rejection.  The
first failed model process was reclaimed after exit and did not enter a
forward or construction reservation; it is retained as a failed attempt, not
treated as a refund.

## Conclusion and one recommendation

The qualified C1 q0 is a valid synchronized input for a bounded N2 diagnostic,
but the overall two-condition baseline is **PARTIAL**: C0 was blocked before
q0 because the target was not observable, and the C1 independent expert later
terminated during the next segment.  On the one qualified condition, FR ends
with negative target progress while the latest teacher probe is positive; this
is a model-side diagnostic observation, not a learned-policy success claim.

Recommend exactly one next intervention: perform one fresh, gate-limited C0
synchronized capture after resolving the target-observability fixture, keeping
the same N2 checkpoint/stats/noise and recording the post-tick image/StateV1/EEF
bundle.  Do not train or alter the decoder from this result automatically.
