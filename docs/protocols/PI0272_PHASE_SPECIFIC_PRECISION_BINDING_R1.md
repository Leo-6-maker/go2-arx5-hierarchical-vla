# PI0272 R1 — Explicit phase-specific native precision binding

**Amendment authority: `PI0272_PHASE_SPECIFIC_NATIVE_PRECISION_RESOLUTION_GO_R1`.**
Date: 2026-10-04 SGT.
Review base: `9ce2a5a983558ae63a75f12c9ac8bbf7d751e35f`, branch `pi0267/executor-decoupled-recovery`.
Base authority: `PI0272_MATCHED_DIRECT_TRAJECTORY_READOUT_FITTING_GO_V1`.
Base release: `e62e6a469dce1f4d3df5d49b38bad4a3c627e969`; original Issue #11 canonical: `5979236384`.
Base protocol: `docs/protocols/PI0272_MATCHED_DIRECT_TRAJECTORY_READOUT_FITTING_V1.md`, blob `111260a1c3371ef817c84313ce87415290a7a91e`.
The fixed amendment release and its source-bound Issue #11 canonical are supplied on publication. This is an amendment to the SAME PI0272 study, not a new experiment or a budget reset.

## 1. Decision — resolve the specified blocker now

**The documented `BLOCKED_SOURCE_PRECISION_CONTRACT` is resolved by explicit scientific authority. Preserve the qualified phase-specific implementation: generated query FP32 during training; generated query BF16 during quality evaluation. DIRECT's zero query follows the actual native processed-feature dtype in the corresponding phase.**

The prior blanket BF16 generated-query wording was a protocol binding error. It was not evidence of a training failure and must not be satisfied by casting the qualified training path to BF16. The execution owner's decision not to change scientific precision without clarification was correct.

This R1 overrides ALL conflicting generated-query/feature/Euler precision statements in base sections 3–5, the original canonical, design JSON and requested-mode cards. In particular, statements that S4 training begins with BF16 query, or that DIRECT's query must always be BF16, are superseded. All non-conflicting base provisions remain mandatory. Do not rewrite the immutable original release or delete the blocker receipts.

Once this amendment and canonical are read back, the owner may materialize the resolution record, make the narrowly authorized binding/assertion/sidecar changes, finish actual runtime admission/sealing and proceed to the ALREADY authorized real canaries and matched training. No additional PI/helper/user confirmation is required for this exact resolution. Do not keep returning the same resolved blocker as a new decision request.

Scientific continuation is authorized; actual runtime admission and canary success are NOT claimed by this document.

## 2. Evidence actually reviewed

At the review base, the following source-bound records and implementation were read:

- `reports/PI0272/admission/PROGRESS_AND_PRECISION_CONFLICT.md`, blob `e752bb13b88d04b5bd387ff2f6def17690681f61`.
- `reports/PI0272/admission/A2_PRECISION_CONTRACT_CONFLICT.json`, blob `78673cf3a520c771a5993e27774c9d32801b5b92`, reported SHA256 `4e928d8337e2d5979023c3b2dab9a7b92fc939bd644704f9b047a7df45bb059c`; checked UTC `2026-10-04T12:49:50.661308+00:00`.
- `reports/PI0272/admission/A2_PHASE_PRECISION_EVIDENCE.json`, summary and initial inventory entries, reported SHA256 `e7484efc9aec9a300166dfb3ba6d57eab9e6fddc3b9b8e72332e39874886aaa3`; checked UTC `2026-10-04T12:55:48.574637+00:00`. Its remaining 1168-receipt inventory was not individually replayed by this reviewer.
- `scripts/pi0272_execution/readout.py`, blob `554dccdc8aa23ad6f8a094a7f96167789cf9e8c0`: query construction already uses `vl.dtype`, but an unconditional BF16 assertion prevents native FP32 training.
- `scripts/pi0272_execution/train.py`, lines 1–145, blob `bd2e0c4e38c64ea486b978c49ae7b526d3a6bbaf`: actual continuation requires `evidence/PRECISION_CONTRACT_RESOLUTION.json`, runtime binding, CPU admission and the existing canaries.

The completed AGG training audit reports the same signature in all8192 microbatches: native epsilon BF16; processed VL features FP32; S4 initial query FP32 after conversion to feature dtype; all four Euler states FP32. Source S4 audit SHA256 is `0d2bd6253c433425d6e7dda3963d81a06da19abb8321fac9f667508e02f9b917`. Closed stock parity has raw and XYZ maximum differences 0.0.

The phase record reports all1168 closed AGG A1/A2 quality receipts using BF16 query/native S4/four heads, matched to the closed output index. Its fixed quality worker SHA256 is `7e70551b613b218986efae1dfe6f2fa9be2d347779352166274e8d31686ceeb4`; evaluation loading is `.eval().to('cuda:0', dtype=torch.bfloat16)`.

At those receipts, new PI0272 updates, training heads, quality queries and PBS submissions are zero. The296-condition data, both2048-micro CPU streams and fixed evaluation membership are preserved. These are dated repository records, not reviewer SSH/PBS observations. No new neural call, optimization or physics was performed by this review.

## 3. Binding precision by execution phase

| Quantity | Training, including both canaries | Final quality evaluation, including DIRECT0 |
|---|---|---|
| Model parameter storage/loading | Preserve initializer's exact mixed dtype/name/shape map:471 trainable FP32 tensors;559 frozen tensors with original mixed BF16/FP32 storage | Preserve existing model-only `.eval().to(cuda:0,dtype=torch.bfloat16)` policy loading; no conversion of quality evaluation to FP32 |
| Native flow epsilon | Preserve native BF16 sampling and exact packet/RNG lifecycle in both arms | No native training loss or epsilon sampling; S4 uses the fixed registered evaluation noise |
| S4 source/query | Native epsilon converted to actual processed VL feature dtype FP32; four native FP32 Euler states | Registered source-noise tensor remains FP32; actual initial S4 query is its BF16 conversion; keep qualified S4 execution |
| DIRECT source/query | All-zero `[B,40,132]` tensor with `dtype=vl.dtype`, expected FP32 | All-zero `[B,40,132]` tensor with `dtype=vl.dtype`, expected BF16; consumes no generation RNG |
| Autocast and tensor operations | Preserve existing native BF16 autocast/mixed-precision operation boundaries | Preserve the qualified quality worker's inference context |
| Physical-loss / decode boundary | Existing physical losses in FP32 with original unclamped ROT/SE(3) contract | Existing raw-to-FP32 decode and saved-array/metric conventions unchanged |

FP32 training query does NOT mean all training operations or parameters become FP32. BF16 evaluation query does NOT mean saved FP32 arrays were originally computed in FP32. Native epsilon, feature dtype, generated-query dtype, parameter storage and autocast are separate fields.

Do not force the DIRECT output to match query dtype: record its actual native readout dtype and retain the existing FP32 physical-loss/decode conversion. The amendment changes no timestep, zero-query value, readout interpretation, modules, weights or loss.

Implement phase-aware assertions, not an unrestricted accept-either-dtype test. Use an explicit execution-phase binding. `head.training == False` is NOT sufficient to identify quality evaluation, because the generated branch intentionally uses an eval-mode context WITH gradients during training. In each phase, assert the expected native feature/query dtype and construct DIRECT zeros from those features without a compensating cast of the features or model.

## 4. Versioned continuation record and minimal repair

The accompanying `reports/PI0272/design/PRECISION_CONTRACT_RESOLUTION_R1.json` is the scientific authorization record. After publication/readback, materialize a source-bound runtime `evidence/PRECISION_CONTRACT_RESOLUTION.json` with status `RESOLVED_BY_EXPLICIT_SCIENTIFIC_AUTHORITY` and `launch_allowed=true`, referencing this fixed amendment release/canonical, the original release and both precision evidence hashes. The launch flag resolves this scientific gate only; do not fabricate runtime-binding, resource or canary PASS.

Retain the original requested-mode cards and blocked records. Create versioned effective mode sidecars that record training FP32 query, quality BF16 query, native mixed parameter/autocast boundaries, actual feature/query/output dtype, and base plus amendment authority. Bind these sidecars into every new complete Trainer, model-only quality view, execution fileset and result receipt. Do not reuse a V1 mode hash while silently changing its contents.

Authorized narrow source changes: replace unconditional BF16 assertions in the isolated PI0272 readout and related admission/audit/fixtures with phase-bound checks; propagate the explicit phase through training/inference/parity wrappers; update the resolution guard and provenance/sidecar/seal plumbing. Preserve `query = zeros(..., dtype=vl.dtype, device=vl.device)`. The S4 training path itself must remain qualified, not be recast to comply with the old text. Equivalent path/import/serialization fixes remain autonomous under the base protocol.

Reuse the existing fitting card, inputs, labels, sample hashes, schedules, data CPU admission and evaluation membership where their dependencies are unchanged. Do not re-collect, relabel, refit statistics, repeat all historical readbacks or rerun both full CPU encoding streams solely because authority metadata changed. Check only changed binding logic/fixtures and rebuild the affected runtime fileset with explicit reuse links. A newly detected changed data dependency must be checked honestly rather than waved through.

## 5. Parity and real checks without a new precision experiment

The base numerical tolerances remain unchanged: raw1e-4, decoded1e-4 where already required, XYZ1e-5m and complete original-gate agreement. These compare matching paths on the SAME packet, parameter-dtype state, phase/autocast context and native query/feature precision: training generated path versus its label-free reference in training precision, or equivalent quality paths in quality precision.

Do NOT require FP32-query training with mixed parameter storage to be bitwise/numerically identical to separately BF16-loaded production evaluation. They are the preserved phase-specific policies, not an equality assertion. No tolerance is relaxed to conceal a same-phase mismatch. This is not a new cross-precision sweep or a claim that the phase difference is optimal or irrelevant.

Use the already-budgeted first passes, parity allowance and real continuous8 versus4/full-save/fresh-resume4 canaries to record actual parameter, feature, epsilon, query, generated-output and relevant Euler dtypes. Source fixtures may test FP32 and BF16 with constructed tensors, clearly labelled as fixtures. A training FP32 query is no longer a blocking surprise; an actual deviation from the explicitly bound phase or a genuine semantic/gradient/nonfinite-optimization defect remains a specific blocker.

Both real canaries remain mandatory and have not yet run according to the reviewed records. After actual admission and both canary PASS, execute the existing S4FIT1024 and DIRECT1024 serially, then the fixed evaluation, CPU replay, four formal Trainer readbacks, new result archives and source-bound terminal return. No helper signoff loop or repeat request for this precision permission is needed.

## 6. Everything else stays frozen

Keep the same AGG4096 initializer revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`,296 conditions, two2048-micro streams,1024 formal updates per arm, native flow regularizer, physical losses/coefficients/PREFIX weighting, ROT statistics, input boundary, score/gate/thresholds, scopes and evaluation cards. DIRECT remains bucket0, one readout, no Euler/no new parameters/no teacher control.

Budgets are unchanged and cumulative:2048 formal plus32 planned canary updates;2208 completed-update hard cap;32768 training heads INCLUDING the256 parity allowance;2148 planned quality calls,2560-attempt/7168-quality-head caps. New teacher/data collection/episodes/renders/WBC/physics/hardware remain zero. No new arm, seed, dataset or performance probe. Preserve actual prior counters rather than resetting them.

This publication removes the specified scientific-precision authorization blocker; it does not claim a job is now running, model correctness or quality PASS. If runtime has advanced since the dated receipts, adopt its valid work rather than restarting. Preserve all historical FAIL/positive findings/WBC PARTIAL. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.

Terminal remains `PI0272_MATCHED_DIRECT_READOUT_FITTING_COMPLETE — STOP_FOR_FULL_SUPPORT_FEEDBACK_OR_FITTING_DECISION` under base V1 plus this R1.
