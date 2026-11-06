# WM02 reference qualification V1

This is a new, finite `NON_VLA_CALIBRATION` program for a future sealed source
capsule. Implementation and offline tests do not qualify a native action. No
registration, model query, renderer, native process or integration is started
by this document or by the default check command. Existing FUSION calibration
episodes, source1ca inference and source3d asset jobs keep their original bytes
and contracts.

The only V1 native candidate is `Z_PLUS_0P5MM`: +0.0005 m in the existing world
frame, constant anchor orientation, original measured initial BASE pose hold,
and fixed working-open mechanical gripper fraction `teacher_physical_fraction(1)`
= `.042/.044`. It is neither a VLA command nor a repair of the historical CORR
raw mechanical gripper near 1. That raw command must not be clipped or rescaled.
The Z-minus, Z±1 mm and X/Y±1.25 mm catalogue entries are offline definitions;
this version rejects their native execution. A new admitted version and actual
native evidence are required before expanding the action set. This does not
qualify rotations, contact, full tasks, or any WM correction.

## Reference and desired outcome

Choose and pin exactly one anchor mode before execution:

| Field | Reference command anchor | Calibration desired outcome anchor |
|---|---|---|
| `INIT_TCP_2503_REFERENCE_V1` | actual initial TCP at native step2503 | actual decision TCP at step3703 |
| `DECISION_TCP_3703_REANCHOR_V1` | actual decision TCP at step3703 | actual decision TCP at step3703 |

The candidate offset is added to each corresponding anchor. The second mode
is an explicitly new reanchoring experiment, not a silent fix to the first.
Subsequent renewals continue the planned reference; they never reanchor to
achieved motion. Evidence retains both measured poses, both goals and their
difference. Calibration intent is not an observed source outcome or an expert
label.

## Exact episode JSON

The new worker has its own immutable `EPISODE_MANIFEST.private.json`. Replace
only the run, existing TRAIN scene and parent placeholders with actual pinned
values. This example is a schema, not a site grant:

```json
{
  "schema": "WM02_FUSION_REFERENCE_QUALIFICATION_EPISODE_V1",
  "phase": "collect-target-fusion-reference-qualification-v1",
  "purpose": "REFERENCE_QUALIFICATION",
  "experiment_role": "NON_VLA_CALIBRATION",
  "split": "TRAIN",
  "run": "<fresh-registered-worker-root>",
  "scene_id": "<existing-statically-passed-TRAIN-scene>",
  "parent": "<that-scene-parent>",
  "candidate_id": "Z_PLUS_0P5MM",
  "reference_anchor_mode": "INIT_TCP_2503_REFERENCE_V1",
  "reference_rule": "FIXED_HOLD40_BURN8_DECIDE_RAMP8_REFERENCE_QUALIFICATION_V1",
  "task_atoms": 64,
  "prehold_atoms": 40,
  "screen_atom": 48,
  "screen": {
    "window_s": 0.4,
    "relative_TCP_diameter_m": 0.002,
    "max_relative_TCP_speed_m_s": 0.025
  },
  "desired_outcome_provenance": "CALIBRATION_INTENT",
  "rotation_program": "HOLD_ANCHOR_ORIENTATION",
  "gripper_program": "HOLD_ORIGINAL_WORKING_OPEN",
  "base_program": "MEASURED_INIT_BASE_POSE_HOLD",
  "physics_dt_s": 0.002
}
```

Arbitrary offsets, rotation deltas, gripper values, clipping, rescaling,
certificate skips, safety overrides and execution-grant fields are rejected.

## Fresh binding through the existing owner

This phase uses the original AUTO1/PILOT1 path. It must not use
`fusion_binding`, `night1_binding`, `next1_binding` or `next2_binding`. Do not
reopen old CLOSED leases, replace the owner or introduce another scheduler.
The actual owner must prepare a fresh registered worker, a fresh worker lease
and an eligible active PILOT1 window under the actual active AUTO1 campaign.
If an eligible window does not exist, use the original registration API only
when the external site grant and parent lease actually exist.

The original PILOT1 window binding schema is unchanged:

```json
{
  "schema": "WM02_PILOT1_SCOPE_V1",
  "plan_sha": "fec841e74d406b3b556c84b733d2de1615581bf7",
  "scope": "WM02_PILOT1_SIM_ONLY",
  "owner": "EXISTING_LOCAL_PC",
  "host": "corelab-4080s",
  "uid": 2009,
  "engineering_budget_caps": null,
  "max_native_slots": 2,
  "C08_native_binary_sha256": "29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2",
  "campaign_root": "<actual-active-AUTO1-root>",
  "activation_epoch": "<actual-number>",
  "deadline_epoch": "<actual-finite-number>",
  "parent_campaign_deadline_epoch": "<actual-parent-deadline-number>",
  "source_sha": "<new-sealed-qualification-source-SHA>",
  "registry": "<existing-actual-registry>",
  "environments": "<existing-site-profiles-file>",
  "runtime_python": "<actual-existing-native-Python>",
  "physical_owner_lock": "<original-physical-owner-lock>",
  "evidence_files": {"<pinned-local-evidence-file>": "<actual-SHA256>"}
}
```

The `C08_native_binary_sha256` field is the historical window schema field;
it does not substitute C08 for the actual frozen C10 native binary. The worker
spec, existing C10 successor/build proofs and actual loaded executor identities
must continue to pin the unchanged qualified C10 executor. This change does
not rebuild C10 or rerun G1.

`scripts.wm02.register_pilot1.main(['--binding', binding_path])` delegates to
`pilot1_scope.register`: it checks actual host/UID, parent lease, original site
profile, evidence hashes and terminal prior research windows; it acquires the
original campaign and physical locks and uses
`scripts.pi0294.night10.native_relay.snapshot/once` for registration. It creates
the active window lease; it does not register this worker or grant native PASS.

The existing owner's worker registration must append one fresh actual registry
entry using the same `snapshot/once(expected_before_sha256=...)` transaction:
`role`, `path=registered_root`, `source_sha` must match the worker spec;
`pilot1_binding` must equal the exact window binding path. Include the actual
`run_uuid`, `executor_node`, `owner`, `scope=WM02_PILOT1_SIM_ONLY` and
`local_lease` for reconciliation. Freeze the resulting `registry_snapshot`.
The worker lease is the existing shape:

```json
{
  "status": "ACTIVE",
  "run_uuid": "<fresh-worker-root-basename>",
  "spec_sha256": "<SHA256-of-exact-new-EXECUTION_SPEC.private.json>"
}
```

The owner supplies real lease metadata and finite endpoints according to the
existing registration route. No file in this implementation manufactures that
lease or its authorization. An existing `SUPERVISOR.private.json`, missing or
non-ACTIVE lease, spec-hash mismatch, registry mismatch, expired owner or STOP
must be reconciled through that original owner; do not retry blindly.

## Worker spec and required pins

Start from the same actual qualified C10/P1 single-native worker spec, retain
its exact real site/runtime/storage/frozen-asset fields, and set the following
new fields. Do not copy an active worker or guess missing paths.

| Spec fields | Required value / binding |
|---|---|
| `phase`, `profile`, `domain` | new phase above; `wm02-auto1-target`; `WM02_AUTO1_TARGET_COMMISSION_V1` |
| `physics`, `physics_dt_s`, `WBC_changes`, `max_ticks` | `true`, `.002`, `false`, `4103` |
| `registered_root`, `lease`, `role`, `source_sha`, `source_root` | fresh worker; new sealed capsule; exact actual registry and lease |
| `pilot1_binding`, `pilot1_binding_sha256` | exact active original P1 binding and its file SHA256 |
| `campaign_root`, `campaign_binding`, `registry`, `registry_snapshot` | actual active original AUTO1 owner and actual registry/snapshot |
| `episode_manifest`, `episode_manifest_sha256` | exact new episode bytes above, with `run==registered_root` |
| `scene_allowlist`, `scene_allowlist_sha256`, `model_path` | original catalogue file; one TRAIN scene with `PASSED_NO_INTEGRATION`; exact parent/model SHA |
| `candidate_index`, `candidate_source_sha`, `native_binary`, `observer_library` | same physical candidate/executor lineage as actual qualified predecessor |
| `previous_stage`, `previous_stage_sha256` | actual same-campaign/candidate/binary/observer/scene native PASS RESULT; nonempty all-zero child exits, clean shutdown, no stops, PASS trace and reference qualification |
| `reference_binding_version` | `MEASURED_EEF_AND_BASE_POSE_HOLD_V2` |
| `repair_stop_result`, `repair_stop_result_sha256` | preserved original standing-reference negative result; unchanged original ancestry check |
| `cold_history_build_receipt/spec` and their SHA fields | existing exact C10 successor proofs, including clock/command-coherence members if applicable |
| `frozen_files` | original controller/WBC/leg policy/gains/costs/solver/safety/dynamics/dt/native/observer/trace/outer/helper/tool pins |
| `native_environment`, `xvfb_run`, `historical_target_lease` and SHA | exact existing native loader/CPU entry and unchanged CLOSED historical lease |
| `host`, `uid`, `site_profile`, `owner`, `cpus`, `wall_limit_s`, `deadline_epoch` | actual original site; finite worker within active owner deadlines |

`source_files` must hash the new contract, new native module, new check CLI,
`auto1_native.py`, `target_sequential.py`, `pilot1_native_worker.py` and
`commission_target.py`; retain the original additional frozen source members.
The read-only check verifies these changed members. Original runtime admission
also verifies every supplied source member and that the imported interpreter
is inside this exact capsule.

The first new action is one world only. `parallel_batch_spec`,
`standby_shutdown_diagnostic`, `reference_readiness_probe` and
`diagnostic_debugger` are rejected. Existing finite CPU resources, owner locks,
site permission and clean cold deployment remain required. A missing actual
external grant is an admission failure, never a qualification result.

## Commands and native lifecycle

After freezing a future capsule and exact spec, the first command is read-only:

```sh
PYTHONPATH="$CAPSULE/src:$CAPSULE" "$NATIVE_PYTHON" -B \
  -m scripts.wm02.fusion_reference_qualification_check --spec "$SPEC" --check-only
```

This checks recipe bytes and actual predecessor file hashes, not current site
admission. Only the actual original owner launches the first native command
under the pinned native loader environment and resource settings:

```sh
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$PINNED_NATIVE_LD_LIBRARY_PATH" \
PYTHONPATH="$CAPSULE/src:$CAPSULE" "$NATIVE_PYTHON" -B \
  -m scripts.wm02.fusion_reference_qualification_check --spec "$SPEC" --execute
```

These variable names denote actual fields resolved by the owner, not new
endpoints. `--execute` delegates to the original `pilot1_native_worker`, which
checks the fresh worker lease and duplicate supervisor, calls
`commission_target` under original `admitted_target`, then writes the worker
exit and closes the worker lease. `admitted_target` retains actual AUTO1/P1
registration, original campaign/physical/run locks and original `admitted_run`.

The adapter uses original initialization2500, original WBC mode ACK and actual
initial state2503. It retains 40 hold atoms, explicit same-hold refresh3503,
the original nine-sample .4 s prefix screen at3703, then the first predeclared
candidate for16 atoms. H40 has an eight-point ramp followed by fixed goal;
renewals at3878/4053 continue that planned suffix. No target integration occurs
during screening, decision evidence or reference certification.

Every install first checks the entire exact new H40, fixed original BASE and
raw working gripper. It then runs the unchanged original
`require_working_target` and `certify_chunk` with actual measured state and the
last two executed reference-history points under `forbid_integration`. A
certificate rejection is retained and aborts; there is no clipping, slowing
or automatic alternate candidate. Original BASE/gripper wire publication,
accept/installed ACK, measured source clock, .4 s TTL and .35 s renewals remain.
Each25-tick atom must bind an installed full-body target hash to an actual
solver policy after the receiver send. The controller does not export a
selected policy ID, so the report preserves that limit. Original native safety,
watchdogs, operator STOP, cancel/HOLD ACK, trace sealing and process closeout
remain in charge.

Private outputs are `REFERENCE_QUALIFICATION_PREFIX.private.json`,
`REFERENCE_QUALIFICATION_DECISION.private.json`,
`REFERENCE_PROGRAM_QUALIFICATION.private.json` and original command, native,
trace, loaded-identity and worker evidence. The program report alone cannot
claim native PASS: outer `native-01/RESULT.json` must show exact4103 trace rows,
PASS trace audit, zero stops/warnings/outer rejections, actual reference
qualification and clean all-zero process exits. Preserve every failed report.
STOP qualification remains `NOT_TESTED` unless actually recorded separately.
No result from this entry establishes the historical CORR/E1 command domain,
learned WM accuracy, task success, target embodiment transfer or F4 benefit.
