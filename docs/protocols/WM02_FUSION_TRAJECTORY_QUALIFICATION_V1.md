# Full trajectory geometry qualification V1

This entry implements a future, single-world `NON_VLA_CALIBRATION` / `TRAIN`
qualification. It does not establish online VLA control, source task transfer,
an expanded WM domain, or a native PASS before execution. Sealed historical
capsules and receipts retain their contracts.

The phase is `collect-target-fusion-trajectory-qualification-v1`. Its geometry
comes from the one predeclared, successfully closed target shadow query
(seed366102, capture3703, CORR2048 manifest94ad). The exact query spec, result,
already decoded `physical_H40`, observation and frame-map hashes are constants
in `fusion_trajectory_qualification_contract.PINS`. The full original saved
capture is verified by the unchanged `recorded_payload()` reader, including
the original sidecar and all three producer snapshots. No Processor decode,
new model load, new query, or source-outcome substitution occurs.

Each registered worker has one scale from `.03, .1, .25, .5, 1.`. The first
requires the byte-pinned actual Z+0.5mm PASS4103 predecessor. Each later scale
requires the immediately preceding scale's complete native PASS, same
candidate, scene, native binary, observer, query and adaptation family. The
checker reads the predecessor spec, CLOSED0 lease, worker exit, outer native
result, all four process exits, trace audit, exact endpoint and per-atom
pose/gripper evidence. A boolean qualification flag is insufficient.

The episode JSON has exactly these fields:

```json
{
  "schema": "WM02_FUSION_TRAJECTORY_QUALIFICATION_EPISODE_V1",
  "phase": "collect-target-fusion-trajectory-qualification-v1",
  "purpose": "TRAJECTORY_GEOMETRY_QUALIFICATION",
  "experiment_role": "NON_VLA_CALIBRATION",
  "split": "TRAIN",
  "reference_rule": "FULL_H40_RIGID_RELOCATE_SCALE_WORKING_GRIP_INTEGER_RETIME_V1",
  "reference_anchor_mode": "INIT_TCP_2503_REFERENCE_V1",
  "prehold_atoms": 40,
  "screen_atom": 48,
  "screen": {"window_s": 0.4, "relative_TCP_diameter_m": 0.002, "max_relative_TCP_speed_m_s": 0.025},
  "tail_hold_atoms": 8,
  "desired_outcome_provenance": "CALIBRATION_INTENT",
  "physics_dt_s": 0.002,
  "base_program": "MEASURED_INIT_BASE_POSE_HOLD",
  "query_pins": "replace with the exact PINS dictionary excluding first_predecessor",
  "run": "fresh registered root",
  "scene_id": "existing TRAIN scene",
  "parent": "its original physical parent",
  "geometry_scale": 0.03,
  "planned_end_atom": "integer computed from the pinned decoded query"
}
```

The strings explaining `query_pins` and `planned_end_atom` above are schematic,
and fail validation. Real input requires the complete dictionary and integer.
Plan the integer using the pinned query anchor and frame map. At the actual
decision3703, the adapter binds measured decision TCP, INIT2503 TCP and BASE7;
its actual length must exactly equal that predeclared length. The finite
endpoint is `3703 + 25 * (planned_end_atom + 8)` and must not exceed17000.
`wall_limit_s` must be positive and at most600. Physics remains2ms and atoms50ms.

The new spec retains the existing `wm02-auto1-target` profile,
`WM02_AUTO1_TARGET_COMMISSION_V1` domain, actual P1 binding, owner, registry,
Journal, fresh lease, locks, frozen physical files and finite closeout. It
adds `trajectory_query_root`, the episode path/hash, and
`trajectory_trace_capacity_build` / `_sha256`. All members in `SOURCE_MEMBERS`
must be byte-pinned in the new capsule. No old FUSION XY binding is accepted.

The trace build receipt must contain exact status
`TRACE_ONLY_131072_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED`,
`trace_capacity:131072`, `new_target_integrations:0`, new trace-library SHA,
unchanged native-binary/observer SHAs and `source_files` hashes for the original
`src/go2arx5_vla/target_mujoco/init_trace.cpp` and `.hpp`. The collector also
checks the actual owned simulator mapping and actual trace header at2503.
This phase uses the existing trace-first final simulator preload order
(`trace:outer:observer`), with constructor-bearing libraries absent from the
display wrapper environment. Other phases retain their original ordering.
The old Z predecessor legitimately retains its original65536-capacity trace.

Run the new sealed capsule's check module with `--spec PATH --check-only`.
The result supplies the exact finite endpoint and explicitly grants no
execution. Only `--execute` delegates to the original
`scripts.wm02.pilot1_native_worker`; that worker independently requires actual
current AUTO1/P1 admission and performs the original lifecycle and closeout.
This check command creates no scope, registration, lease or worker.

The collector preserves INIT2503 hold, history boundary3503, actual decision3703
and the original stable-prefix screen. The nominal reference uses the fixed
INIT transport; desired motion is anchored at decision TCP. Subsequent achieved
motion never reanchors the program. Full40 suffix previews, BASE hold, the
original `certify_chunk`, accepted/installed ACKs, source-relative400ms TTL,
350ms renewal, native safety, cancel/HOLD, trace sealing and clean process exit
remain required. No WBC, gain, solver, dynamics or safety threshold changes.

For this phase only, pose installation defers the old install-time gripper
write. The collector writes exactly one original packet before each actual
25-tick atom, sequence0 then strict+1, with fresh measured EE source. After
integration it verifies the original owned wrapper's actual sequence, opening,
expiry and step globals; actual finger positions remain separate evidence.
The original actuator speed limit and mechanical working-range guard remain.

Private records contain real decision9x106 history, eight actually consumed
EEF10+BASE7 commands, every actual installed full preview, per-atom source age
and remaining TTL, raw-query identity, desired/reference paths and separate
submitted/consumed/achieved gripper data. Final qualification is set only after
all40 retained knots, the whole program and tail are consumed. The outer
native result must still establish exact trace, safety and shutdown PASS.
These records are prospective response-model inputs, not a fit or a complete
MPC hidden-state restore claim.

The implemented trace-only build entry is
`python -B -m scripts.wm02.build_fusion_trace_capacity --spec PATH`.
It requires a fresh original-registry `core-wm02-init-build` registration,
`TARGET_INIT_BUILD_ONLY_V1`, `physics=false` and
`purpose=FUSION_TRACE_CAPACITY_ONLY`. It compiles the exact C10 receipt-bound
trace CPP/header at the already supported `WM02_INIT_TRACE_CAPACITY=131072`;
the executable, observer and actuation DSO remain byte-identical. It preserves
the original dependency SONAME for a trace-first preload and actually checks
old/new symbol owners, unchanged outer/observer next resolution, both capacity
headers and 70000 complete synthetic serialized-key events. The loader process
never calls the native executable, physics or a leg policy. This build result
is not dynamic qualification; the new native collector still checks its own
actual mapped DSO, header and complete trace. Build failures and their CLOSED
leases are retained, never overwritten or reopened.
