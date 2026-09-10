# Stage45C PI0100 TRAIN24 Checkpoint Evaluation — Completion Report

**Status:** PASS  
**Success token:** `HOPPER_STAGE45C_TRAIN24_CHECKPOINT_EVALUATION_COMPLETE`  
**Next gate:** `GPT_PI_REVIEW_AFTER_STAGE45C_TRAIN24_EVAL`  
**Execution authority:** `PI_GATE_STATE_0100` — Issue #11 comment `5611355932`  
**Machine-readable return:** Issue #11 comment `5611850414`  
**Scientific execution HEAD:** `74e77d83d85ef9b290c8f3642f242fb25aaa87cb`  
**PBS job:** `609482.hopper-m-02` on NVIDIA H200

## 1. PI review verdict

Stage45C TRAIN24 checkpoint evaluation is **execution-complete and provenance-complete** for the frozen protocol. The authorized H200 job completed with PBS terminal state `F` and `Exit_status=0`, produced the complete planned checkpoint/window cohort, emitted all expected raw prediction records, and performed no forbidden training or downstream validation actions.

This report is a **post-execution reporting artifact**. Its commit is not an execution-code identity and must never replace the scientific execution HEAD above.

The run establishes that the Stage45C production evaluation harness can execute the frozen TRAIN24 protocol end-to-end under the admitted runtime stack. It does **not**, by itself, establish that FULL24 learned the TRAIN distribution or generalizes. Metric interpretation remains a PI-level scientific decision after review of the sealed aggregate outputs.

## 2. Authority and protocol identity

| Item | Sealed value |
|---|---|
| Execution authority | `PI_GATE_STATE_0100` |
| Mapping provenance gate | `PI_GATE_STATE_0091` |
| Execution authority gate | `PI_GATE_STATE_0100` |
| Frozen protocol SHA256 | `0f8c564f534ce3b29a5029cc927f93a03697181b4b1df0aa9e80566af4cd020f` |
| Checkpoint mapping SHA256 | `d2281b451970ef408566ded15851bd252021ccf4de1bf5e59df9b8edf5b4c1d7` |
| Execution release SHA256 | `3750f6d7d01021631f93660d5438e498860b6f7b19e3b30e8d696ee222a60532` |
| Runtime binding raw SHA256 | `28c3e76dadc0d0f3f94498bf21b05a2d782a0244f87aef8319d6d5bb253f846e` |
| Runtime binding canonical SHA256 | `ea93e76b664d865461c4e109aa99ec91150d993c3780111fe9bbb03d3ff8c882` |
| Execution HEAD | `74e77d83d85ef9b290c8f3642f242fb25aaa87cb` |
| Integration PR | `#45` — history-preserving merge |

The production return explicitly records the historical checkpoint-mapping provenance gate separately from the current execution-authority gate. This closes the `609377.hopper-m-02` identity contradiction without rewriting the historical PI0091 mapping provenance.

PR #45 changed only the execution-plumbing identity model and its tests: immutable mapping provenance remains `PI_GATE_STATE_0091`, while the runtime execution authority is late-bound from the exact external binding. The frozen scientific protocol, checkpoints, sampler, decoder, metrics, data cohort, and model inputs remained unchanged.

## 3. Execution completeness

| Counter / property | Expected | Observed | Verdict |
|---|---:|---:|---|
| Real qsub calls | 1 | 1 | PASS |
| BASE windows | 24 | 24 | PASS |
| STRUCTURED@1000 windows | 24 | 24 | PASS |
| FULL24@1000 windows | 24 | 24 | PASS |
| FULL24@2000 windows | 24 | 24 | PASS |
| Total checkpoint-window evaluations | 96 | 96 | PASS |
| Four-step sampler/action-head calls | 384 | 384 | PASS |
| Raw prediction records | 96 | 96 | PASS |
| Raw prediction shape | `[1,40,132]` | `[1,40,132]` | PASS |
| Raw prediction dtype | `float32` | `float32` | PASS |
| Raw persisted before decode | required | true | PASS |
| Same per-window noise across checkpoint passes | required | true | PASS |
| PBS exit status | 0 | 0 | PASS |

The count identities close exactly:

- `4 checkpoints × 24 windows = 96 checkpoint-window evaluations`;
- `96 evaluations × 4 sampler steps = 384 sampler/action-head calls`;
- one raw universal action record was persisted for every checkpoint-window evaluation.

No adaptive extension or extra checkpoint cohort was reported.

## 4. Forbidden-action audit

The machine-readable return records all prohibited actions as absent:

| Boundary | Observed |
|---|---|
| Training | `false` |
| Backward | `false` |
| Optimizer steps | `0` |
| Resume | `false` |
| Checkpoint write | `false` |
| Validation-E read | `false` |
| Holdout-F read | `false` |
| WBC steps | `0` |
| Simulator | `false` |
| Hardware | `false` |
| Network fetch | `false` |

Therefore the result remains strictly a **TRAIN24 inference/evaluation** result. There is no validation, holdout, WBC, simulator, hardware, or generalization exposure.

## 5. Runtime-repair closure evidenced by the successful production run

The successful 96-window production execution provides end-to-end operational evidence that the previously exposed Stage45C execution blockers are no longer preventing the frozen evaluation path:

1. PBS submission/environment envelope reaches a valid H200 allocation.
2. Golden native FFmpeg/TorchCodec preflight no longer blocks production execution.
3. The sealed gitless GR00T/Cosmos offline model-loading path reaches model construction and inference without network fetch.
4. The model-owned processor/collator offline path no longer blocks construction.
5. The inference/autograd boundary repair is exercised through all 384 sampler/action-head calls with no backward/training exposure.
6. PI0100 mapping-provenance versus execution-authority separation is exercised by the exact production binding and no longer rejects the job pre-model.

This closure is **operational**, not a change to the scientific protocol.

## 6. Evidence seals

Primary machine evidence remains on the authorized execution host; GitHub records identities and hashes rather than duplicating large prediction artifacts.

| Evidence | SHA256 / location |
|---|---|
| Terminal return | `979cf567db33ffe0f8c2f6f362d402cb3757bfe84d3b803a07322b763a5cb331` |
| Output manifest | `c3aea2bbd1e81aa79df474be364fd991121d3fdae4e0fa89f1838401311809a4` |
| Submission manifest | `7df7a4a9cdfec93ba0eae2861c08f50c4134bf56f6eb750cd6fa0fdafde68212` |
| qsub argv | `5fd9e1f41808d61b01c1815c7303a3a13852fad9e7fad91c997f363ef70b7aca` |
| Submission wrapper | `9ce5e7f36eb1467238edea7e712a91ad456274c127e63e54851e5d18a6576b5f` |
| Terminal qstat | `9e8cc1b44830b4dc3f39bc79451b2d9770a64c1df5204c2ab928d6aa38209adb` |
| Terminal-return directory | `/scratch/e1192548/stage45c_pi0100_terminal_return_609482_20260910_v2` |
| Production output root | `/scratch/e1192548/stage45c_pi0100_h200_train24_eval_74e77d83d85e_20260910` |

Canonical GitHub machine-readable return: Issue #11 comment `5611850414`.

## 7. Scientific claim boundary

Accepted claim at this gate:

> The frozen Stage45C TRAIN24 checkpoint-evaluation protocol completed end-to-end on the admitted H200 runtime for BASE, STRUCTURED@1000, FULL24@1000, and FULL24@2000, with complete planned counters and sealed raw predictions, without training or locked-data exposure.

Not yet accepted:

- that FULL24@1000 outperforms STRUCTURED@1000;
- that increasing coverage improves TRAIN24 target progress;
- that FULL24@2000 improves over FULL24@1000;
- that any checkpoint generalizes to unseen targets/states;
- any Validation-E, Holdout-F, WBC, simulator, hardware, or deployment claim.

Those conclusions require PI review of the sealed checkpoint-level aggregate metrics. In particular, the primary scientific comparison remains STRUCTURED@1000 versus FULL24@1000 at matched 1000-step exposure; FULL24@2000 is the secondary exposure comparison, and BASE remains the reference.

## 8. Gate disposition

**Stage45C execution disposition:** `PASS / COMPLETE`  
**Blocker:** none  
**Further H200 retry under PI0100:** not needed and not authorized  
**Next mandatory gate:** `GPT_PI_REVIEW_AFTER_STAGE45C_TRAIN24_EVAL`

At the next gate, inspect the sealed aggregate metrics before deciding whether TRAIN24 learnability is strong enough to justify any unseen-target Validation-E experiment. No generalization claim should be promoted from this completion report alone.
