# Stage41 L0 — sealed Stage40 M2 baseline profile

Status: `CORELAB_STAGE41_M2_BASELINE_PROFILE_PASS`

This is a CPU/read-only characterization of the accepted Stage40 M2 evidence.
It adds no pass threshold and does not rerun the model, simulator, or WBC.

## Identity

- M2 seal: `ad178ac49da1ced987f60224987ccab8dc1d3343674bb511b7c3fe13a3b8dcc5`
- M2 result: `5b0260d07a0fd43cae8ee30de18574547be7cd1a3599d4aacba74840aa9fb6db`
- evidence manifest: `a5aab8bcfc11593ab65ae4d626294fac3c7cad0bb5a6516a554f6e5e0cda0a99`
- M2 bundle: `9900f8e9f88b0af7316e0c3bf46909b77ae739dafaf8468eb78d84208f70963f`
- repository HEAD: `76a92a421b302ff119d18980fd5dffd233f6994c`
- machine-readable profile: `/var/cache/users/liuyu-nfs/stage41_corelab_long_horizon_20260908/l0_baseline/stage41_m2_baseline_profile.json`
- profile SHA-256: `81592ec1163c98659c03055270b14cebdea2a7fa79dbb50b8e134054fea66a03`

## Closed-loop baseline

- GR00T forwards: `4`; reference updates: `3`.
- Native WBC segments: `[49,49,49,98]`; total: `245` ticks.
- SafetyGate: `ACCEPT, ACCEPT, ACCEPT, ACCEPT`.
- Endpoint: `endpoint_reached=true`; native reference time `1.9599987268447876 s`.
- Native handoff schedule: `49` ticks at `native_dt=0.019999999552965164 s`.
- Executor action prefix: `20` samples at `action_dt=0.05 s`.

## Tracking characterization (no new threshold)

- EEF position error RMS/max: `0.019460859 / 0.048956220 m`.
- EEF orientation error RMS/max: `0.084014203 / 0.389261492 rad`.
- Maximum SafetyGate metrics: linear speed `0.787752953 m/s`, angular speed `0 rad/s`, initial position discontinuity `0.142400042 m`, SE(3) arc length `0.478875589 m`.
- StateV1 fingerprints are distinct for q0–q3. Adjacent state deltas (L2): `0.087451106`, `0.112782437`, `0.095168773`.
- All `245` rows had `termination_code=NONE`, `done=false`, `reset_buf=false`, and `invalid_state=false`.
- `joint_limit_violation` was recorded on all rows, with minimum margin `-0.002198273`; this is preserved as characterization evidence only and is not converted into a new pass/fail threshold.

## Runtime characterization

- GR00T latency mean/p95/max: `0.225615 / 0.652553 / 0.652553 s`.
- Target RTX GPU nvidia-smi used-memory baseline/peak/after: `20434 / 30054 / 21335 MiB`.
- GR00T worker peak allocated/reserved: `6089.454 / 6180.0 MiB`.
- Host RAM and per-replan GPU snapshots were not recorded by the Stage40 M2 evidence; they remain unavailable rather than inferred.

## Boundary

This profile characterizes one sealed TINY1 development fixture at seed 4242.
It is not a robustness, generalization, real-time, or hardware claim.
