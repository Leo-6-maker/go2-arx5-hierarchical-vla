# PI0168 execution-chain and failure-mode audit

Status: `READY_FOR_PI_REVIEW` for the zero-model repository preparation only.

This audit is bounded to the PR #71 reviewed snapshot `940df1ab6f428fd0260fe836e36d4cd98c6161e3` recorded when Issue #11 comment `5658063963` issued `PI_GATE_STATE_0168`. The controlling pointer was read first from PR #69 comment `5658065067`; the gate and all restrictions were then read from Issue #11. The later live PR #71 movement is classified in the final section and is not silently treated as audited source.

## Scope and evidence boundary

The audit used AST, source text, file hashes, Git metadata, and the published Issue/PR comments listed below. It did not read Validation-E or Holdout-F payloads, enter an experiment output directory, import project/runtime modules, access a GPU, submit or poll PBS, or execute a model/training/inference path.

Published control evidence:

- Issue #11 `5658063963`: `PI_GATE_STATE_0168` authority and allowed-path contract.
- PR #69 `5658065067`: launch pointer, exact Vanda base snapshot, and Draft-PR requirements.
- PR #71 `5655421312`: allocation-8 six-preserved/30-missing accounting.
- PR #71 `5655491444`: allocation-9 pre-admission import-pollution failure and zero new-call boundary.
- Issue #11 `5657908043`: published PI0167 diagnosis and Vanda ownership of the repair.

## Static execution chain

| Stage | Source and static anchor | Process/import boundary | Reads, writes, state, and fail-closed behavior | PI0168 conclusion |
|---|---|---|---|---|
| PBS/bootstrap and checkout | `hpc/pbs/stage45f_pi0163_validation_e_gen1_h200.pbs:1-58` | Shell validates exact Python, config, branch, HEAD, clean tree, and fresh output paths before Python execution. | Reads environment/config/path metadata; no output is created by the pre-submit checks. `set -euo pipefail`, required variables, exact branch/HEAD, and clean-tree checks fail closed. | The single-job entrypoint is an operational boundary, not a reusable scientific primitive. |
| Runtime capsule and import-only probe | `..._pi0163_validation_e_gen1_h200.pbs:70-107`; `scripts/stage45f_pi0163_ffmpeg_capsule_bootstrap.py`; `scripts/stage45f_pi0163_runtime_import_probe.py:74-151` | Capsule bootstrap is stdlib-only. The probe is a separate process and dynamically imports Torch/TorchCodec/Hugging Face for CPU media/cache checks; it does not construct a model or read Validation-E. | Capsule record is durable; probe record is durable. Offline cache, FFmpeg library, and local resolver identities are checked. | Keep this process separate from the parent admission process; do not use it as proof that the model process is import-clean. |
| Recovery pre-submit and source inventory | `scripts/stage45f_pi0163_validation_e_recovery.py:375-435`, `:216-297`, `:643-681` | `raw_inventory()` has a function-local `import numpy as np` at line 220. In the reviewed recovery path it is called at line 870 before evaluator admission at line 1028. | Real recovery reads the raw manifest and `.npy` cells, hashes and validates them, and prepares a fresh output root. The inventory is intentionally payload-facing; this audit did not call it. | **Known PI0167 blocker:** NumPy enters the parent `sys.modules` before the frozen guard. The existing recovery static guard does not see this function-local ordering. Vanda owns the isolated-child-process repair. |
| PI0159 continuation/cache binding | `scripts/stage45f_pi0159_validation_e_continuation.py:488-548` | Top-level imports are stdlib plus the stdlib-only admission helper. The evaluator import is local at line 546 after prior-admission and offline-cache setup. | Writes continuation admission/effective-config/pre-model records in a fresh root; validates prior evidence and binds offline paths. | Static surface is model-side-free at module import. Runtime cache binding still requires the separate execution environment and must not be run by Corelab. |
| Frozen admission | `scripts/stage45f_vanda_validation_e_admission.py:114-208`; `scripts/stage45f_vanda_validation_e_gen1.py:462-480` | Admission module is stdlib-only. `validate_artifacts()` calls `admission.model_imports_present()` first at line 467, then checks freeze identity, code lineage, six arms, six windows, statistics/noise/E4, and locked boundaries. | Reads committed freeze/config/metadata and Git identity; rejects preloaded model-side modules before payload admission. | The import guard is correctly fail-closed and must not be weakened. The gap is upstream parent contamination, not an admission-policy defect. |
| Runner pre-model checks | `scripts/stage45f_vanda_validation_e_gen1.py:680-702` | `static_guard()`, checkout observation, GPU observation, and `validate_artifacts()` precede model/runtime imports. | Requires fresh output files and writes static/admission/runtime records only after checks pass. | This is the intended frozen boundary for a real Vanda run; no Corelab execution was authorized. |
| Post-admission processor construction | `..._vanda_validation_e_gen1.py:703-721`; recovery `:1033-1044` | NumPy, Torch, runtime adapter, Stage26 helper, and `checkpoint_eval` are imported after admission in the reviewed runner/recovery source. | Builds processor, embodiment, six windows, and CPU float32 E4 noise bank; writes noise metadata. | These imports are allowed only after the boundary. The same helpers are unsafe if reused by a pre-admission utility. |
| Model load, conditioning, and E4 action-head loop | `..._vanda_validation_e_gen1.py:738-790`; recovery equivalent after `:1044` | Model load is delegated to the runtime adapter after admission. Each arm has six windows and four timestep/action-head calls per window. | Reads accepted checkpoint/data/cache paths; writes raw `.npy` and raw manifest **before** decode/derived metrics. Counters are explicit and fail closed on shape/finite/count drift. | Model calls, GPU, and training were not executed by this audit. The six-arm/six-window/144-call assumptions remain frozen source facts, not new experiment authorization. |
| CPU decode, metric, and six-window aggregate | `..._vanda_validation_e_gen1.py:794-860`; `src/go2arx5_vla/evaluation/checkpoint_eval.py:221-364` | `checkpoint_eval` has a top-level NumPy import at line 22 and must remain post-admission. | Decodes from sealed raw arrays, derives metrics, aggregates exactly six windows, preserves raw/decoded hashes, and writes paired/cpu summaries. | CPU post-hoc mechanics are reusable only when their import boundary and raw-first contract are explicit. |
| Final inventory and seal | `..._vanda_validation_e_gen1.py:861-948`; recovery `:1248-1435` | No new scientific stage is introduced. | Writes telemetry, inventories output files, then writes the final seal/result with counters, hashes, no-training/no-Holdout-F flags, and no automatic retry. | Seal is evidence of a particular run, not permission to select a scientific winner or change the recipe. |

## Import/process purity result

The new stdlib-only auditor records AST line/scope evidence in the committed manifest. The result is `PASS_STATIC_AUDIT_WITH_KNOWN_PI0167_BLOCKER`; this is not a request to modify the frozen evaluator or PR #71.

| File | Top-level runtime imports | Function-local runtime/project imports | Classification |
|---|---|---|---|
| `scripts/stage45f_pi0163_validation_e_recovery.py` | None; the continuation helper is imported at module scope and is stdlib-only | NumPy at `:220`; later NumPy/Torch/runtime/Stage26/checkpoint evaluator at `:1035-1039` | `KNOWN_PI0167_BLOCKER` for the raw-inventory call before admission; later imports are post-boundary runtime dependencies. |
| `scripts/stage45f_pi0159_validation_e_continuation.py` | None | Evaluator at `:546` | Static-only continuation wrapper; execution order still requires the runner's own guard. |
| `scripts/stage45f_vanda_validation_e_admission.py` | Stdlib only | Local AST import at `:115` | Pure admission helper; do not add runtime imports. |
| `scripts/stage45f_vanda_validation_e_gen1.py` | None; admission helper is safe | NumPy/CV2/snapshot helpers at `:137`, `:323-399`; window/runtime imports at `:546-549` and `:706-710` | `STATIC_ONLY_SUSPICION` for helper reachability; the explicit runner order puts model-side imports after `validate_artifacts()`. |
| `scripts/stage45c_h200_eval_runtime_adapter.py` | Stdlib/AST only | Torch/GR00T/NumPy/project imports in runtime functions, including `:1084`, `:1105-1108`, and `:1355-1429` | `SCIENTIFIC_RUNTIME_DEPENDENCY`; separate Train24/H200 execution machinery, not an admission module. |
| `scripts/stage26_h1_c1_fixed_noise_local_flow.py` | Stdlib only | Torch/GR00T/NumPy/project imports in processor, sample, flow, metric, and checkpoint helpers | `SCIENTIFIC_RUNTIME_DEPENDENCY`; high risk if a pre-admission static helper imports it transitively. |
| `src/go2arx5_vla/evaluation/checkpoint_eval.py` | NumPy at `:22`, plus relative project metric/decode imports at `:24-40` | N/A | `IMPORT_PURITY_DEFECT` if imported before admission; valid post-admission CPU metric dependency otherwise. |

Mechanical fix classes are deliberately narrow:

1. Run NumPy-backed raw-array verification in an isolated child process and return only sealed JSON to the clean parent.
2. Keep model/runtime imports after the frozen admission call; do not relax `model_imports_present()`.
3. Keep reusable admission modules stdlib-only and put gate IDs in metadata rather than module-level scientific mechanics.

The first class is the already-published PI0167 repair owned by Vanda, not a new PI0168 P0. No PR #71 source was changed.

## Historical failure matrix through allocation 9

The rows below are an evidence matrix, not a replacement for payload inspection. Counts and phases come from the published cumulative accounting and the published allocation-8/allocation-9 seals in the Issue/PR comments above.

| Attempt/job | Failure class and first failing phase | Model calls | Preserved raw | Root cause / CPU pre-submit detection | Guard and owner |
|---|---|---:|---:|---|---|
| 1 / `614203.hopper-m-02` | Pre-Python/session/scheduler failure | 0 | 0 | Failure before the model boundary; no Python evidence surface | PBS/session boundary; scheduler/operator |
| 2 / `614226.hopper-m-02` | Partial model-load failure | 1 attempted, 0 completed | 0 | Capacity/runtime failed during model loading | Model-load counter and failure seal; Vanda/runtime |
| 3 / `614574.hopper-m-02` | `FAILED_BEFORE_MODEL_IMPORT` | 0 | 0 | Pre-model runtime/entry failure | PBS and pre-model seal; Vanda |
| 4 / `614586.hopper-m-02` | `FAILED_BEFORE_MODEL_IMPORT` | 0 | 0 | Mechanical retry reached the same pre-model boundary | Same fail-closed preflight; Vanda |
| 5 / `614704.hopper-m-02` | Partial after one valid raw cell | 1 load, 1 conditioning, 4 action-head calls | 1 | Post-raw decode/derived path failed after raw-first preservation | Raw-before-derived seal and fresh-root rule; Vanda |
| 6 / `614858.hopper-m-02` | `BLOCKED_STALE_EXPECTED_HEAD` | 0 | 0 | Checkout/expected-head identity mismatch | PBS Git identity check; Vanda |
| 7 / `615030.hopper-m-02` | `BLOCKED_PBS_PRE_SUBMIT_CHECKOUT_DIRTY` | 0 | 0 | Dirty checkout before admission | PBS clean-tree check; Vanda |
| 8 / `615034.hopper-m-02` | Partial evaluation-contract failure | 1 load, 6 conditioning, 24 action-head calls | 6 raw + 6 derived | Aggregate expected 24 windows but observed six for the affected per-arm path; raw-first records remained | Six-window behavioral assertion and aggregate check; Vanda |
| 9 / `615124.hopper-m-02` | `BLOCKED_PRE_ADMISSION_IMPORT_POLLUTION` | 0 new calls | 6 preserved; 30 missing | Recovery `raw_inventory()` imported NumPy before frozen admission; the guard correctly rejected the contaminated parent | Isolated child verifier; Vanda, PI0167 ownership |
| Cross-cutting / PI0166 recovery source | Snapshot dataclass-versus-mapping ABI mismatch | 0 attributable new calls | N/A | Metric adapter expected mapping keys while snapshot was dataclass-shaped; compatibility bridge was added in the recovery source | Keep adapter ABI explicit and test both representations; Corelab static record only |
| Cross-cutting / Train24 versus Validation-E | Window/cardinality assumption split | 0 in this audit | N/A | Stage45C helper is hard-coded to 24 Train24 episodes (`runtime_adapter.py:1110-1112`), while Validation-E freezes six windows per arm | Separate RunSpec contracts; future consolidation owner |
| Cross-cutting / post-raw CPU/seal | Raw-first versus derived/seal ordering | 0 in this audit | N/A | A derived/aggregate failure must leave raw evidence auditable and must not be relabeled as a clean scientific result | Raw-first and posthoc-seal schemas; Vanda execution owner |

## Zero-execution accounting

| Counter | Value |
|---|---:|
| Validation-E payload reads | 0 |
| Holdout-F payload reads | 0 |
| model construction/load/forward calls by Corelab | 0 |
| training/backward/optimizer/checkpoint writes | 0 |
| GPU/CUDA/PBS submissions or polling | 0 |
| simulator/WBC/probe additions | 0 |
| dependency installations/environment mutations | 0 |

## Live PR #71 movement check

At the final check, the live PR #71 head must be read again. If it differs from the issuance snapshot, the exact classification is appended here and copied into the manifest. The audit never edits or rebases PR #71.

```text
VANDA_HEAD_MOVED: pending_final_live_read
NEW_HEAD: pending_final_live_read
DELTA_FILES: pending_final_live_read
DOES_DELTA_INVALIDATE_AUDIT: pending_final_live_read
UNAUDITED_LIVE_DELTA: pending_final_live_read
```

The bounded conclusion remains: this packet is repository-consolidation preparation for PI review, not a claim that a moved Vanda execution head has been fully re-audited.
