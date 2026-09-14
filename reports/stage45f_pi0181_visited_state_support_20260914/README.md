# PI0181 visited-state support pack

This directory is a CPU-only, `PREPARED_NOT_EXECUTED` support artifact. It is
derived from the accepted Stage43 five-condition seal and contains 20 real
policy-facing observations in fixed `C0..C4 × q0..q3` order, plus one
deterministic Stage22 same-scene expert relabel for each row.

The model-input boundary is fixed to `body_rgb + wrist_rgb + StateV1 +
language`. Target XYZ, historical policy actions, expert actions, distances,
progress, and WBC telemetry are metadata/supervision only.

`MULTI_VISITED20_V1` is a package, not an executed training result. Its base
checkpoint remains late-bound to the PI-selected Legacy/L1 checkpoint. The
existing `SINGLE_Q0_V1` package is not modified.

The historical replay sidecars include the real Stage43 body/wrist frames,
historical policy H40, expert H40, target/EEF telemetry, SafetyGate and WBC
metrics. The accepted Stage43 bundle has no native third-person camera
payload, so `third_person.mp4` is explicitly not claimed. The renderer also
found no pre-existing `ffmpeg` executable on `corelab-4080s`; therefore this
checkout records `PARTIAL_FFMPEG_UNAVAILABLE` with the portable frame,
trajectory-overlay and frame-metrics sidecars retained.

Authoritative summary and hashes are in [pi0181_result.json](pi0181_result.json),
[visited_pack_manifest.json](visited_pack_manifest.json),
[expert_label_manifest.json](expert_label_manifest.json),
[coverage_report.json](coverage_report.json), and
[s20_schedule.json](s20_schedule.json).
