# Cold owner integration entry: source preview ABI remains unsupported

Base: 60b787edf2268f124f0aaed32ec32433be9ddaf0. These are CPU contracts, not source/target native qualification. DEC1 remains original CORR@2048/R0. A/B endpoints, WBC, safety rules, scoring and interface parameters are unchanged.

The generic `BoundProgressExecutor` callback boundary is available for a future qualified native preview executor. The registered source path currently cannot be claimed ready:

- `scripts/pi0291_arx5_data/engine.py:Engine.step(target, opening)` consumes one 4x4 target pose and scalar opening. `scripts/pi0294/episode.py` repeats that single pose for 25 native ticks.
- Original `s1_preflight.predict` checks eight such poses by invoking that same single-pose step 200 times. It does not consume a nine-point WBCPreview. Equal first anchor or matching proposal digest cannot make this a check of all lookahead points actually dispatched.
- `SourcePrefixShadow` therefore declares supported=False and NOT_SUPPORTED_NINE_POINT_PREVIEW_ABI. It never calls the old predictor and never returns PASS. It returns an identity-bound NOT_SUPPORTED/zero-tick receipt; `bind_before_dispatch` always raises. `BoundProgressExecutor` records that known zero-effect result, latches stop and never dispatches. Same-anchor/different-lookahead CPU negatives verify this.
- `full_h40_validators` still provides useful raw/mapped H40 numerical legality through unchanged `gate_action`, accepting only a non-None dict with original ACCEPT/WARN decisions. This is not physical raw S1 or a mapped preview shadow qualification. Raw physical S1-before-mapping remains unbound/unqualified.

Do not cold-enable native DEC1 source execution using this unsupported adapter. To complete native wiring, the owner needs an actual executor/clone that consumes the exact preview+gripper ABI, original native safety on that identical consumed reference, measured tick/cost receipts, live/shadow parity, actual feedback/ACK/reset/stop and original finite commissioning admission. This patch adds no replacement safety predictor, no implicit first-point conversion and no new physical executor.

The generic boundary retains these invariants:

1. Raw full-H40 callback runs before canonical/frame mapping, then mapped full-reference and mapped prefix callbacks before any dispatch. Owner callbacks must implement the real frozen native chain; a numerical gate alone does not certify physical S1.
2. A paired preview/gripper dispatch is called once with a durable original-owner intent beforehand. Actual command/episode/executor identities, actual ticks/end time and post-execution feedback predecessor are checked. Errors latch and cannot retry. The original outer loop remains responsible for eight reference intervals per model chunk.
3. A known native S1 reject can return NativeShadowReceipt(decision='REJECT', ticks=actual_partial_ticks), or raise NativeShadowRejected carrying that receipt. Identity-bound known ticks are recorded and charged, unknown_shadows is cleared, known_shadow_rejections increments, and no live dispatch occurs. A valid early rejection is distinct from an unreceipted exception: the latter retains unknown_shadows and the original native accounting liability. The wrapper does not write/refund any central ledger.
4. NOT_SUPPORTED must have zero ticks; PASS must match all original expected prefix ticks. Neither unsupported ABI nor missing data becomes a successful safety check.
5. Stop must invoke the actual executor stop and return NativeStopReceipt for the same episode/executor only after acknowledged stop. Missing stop ACK stays a visible safety error. Python latch state alone is not native stop proof.

Target remains TARGET_NOT_DELIVERED. No source repeats, ideal tracker, welded base or tar archive can fill the target manifest/ABI. Source/target commissioning has not run. This gap does not block already-qualified A/B training. Actual mapped shadow cost cannot be inferred from CPU mocks or the single-pose forecast.

# Saved result aggregation

Cold CLI after publication: `python -m scripts.pi0294.dec1.paired_results ROWS.tsv SEALED_IDENTITIES.json`.

Reuse the existing 64-row template and supply all original row keys. Status vocabulary is SUCCESS, VALID_TASK_FAILURE, REFERENCE_REJECT, S1_REJECT, NATIVE_SAFETY_ERROR, ENGINEERING_MISSING or NOT_RUN. The sealed identity input supplies exactly eight distinct geometry IDs and one predeclared noise-bank identity. Every row must match it. Completed rows additionally require the actual checkpoint_manifest_sha256 and statistics_sha256 above and native_qualification=QUALIFIED. A matching model display name alone does not prove weights/statistics identity.

NOT_RUN and ENGINEERING_MISSING retain UNKNOWN task_success. All 64 keys remain explicit; missing rows, duplicate rows, mixed model/noise/geometry, false success flags and unqualified completed rows reject. Incomplete/native-fault panels report status counts and per-task strata but no paired effect estimates or fabricated zero rates. Complete valid panels calculate source/target progress-minus-time deltas and interaction with geometry as the independent block: all eight task/executor/interface cells in a geometry stay together during cluster resampling. The default draws/seed are disclosed. This is a bounded pilot interval, not 64 independent scenes, model selection, general WBC causality or release acceptance.
