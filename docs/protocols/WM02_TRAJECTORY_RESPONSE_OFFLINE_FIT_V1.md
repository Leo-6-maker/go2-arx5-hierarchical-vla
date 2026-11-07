# WM02 trajectory-response offline fit V1

This entry is READY code, not an executed real72 fit. It reuses the frozen
trajectory response contract/replay/NumPy study. Collection source and fitting
source are recorded separately: a later fitting capsule does not require
recollecting data from its own Git SHA. Existing native workers/capsules and
historical failures are unchanged.

```text
python -B -m scripts.wm02.fit_fusion_trajectory_response \
  --spec /qualified/fresh-worker/EXECUTION_SPEC.json \
  --output /qualified/fresh-worker/fit-01
```

Execution still requires a fresh original-owner admitted CPU worker. The new
profile is `wm02-fusion-trajectory-response-offline-cpu`, site
`CORE_FUSION_TRAJECTORY_RESPONSE_CPU`, kind
`FUSION_TRAJECTORY_RESPONSE_OFFLINE`, recipe
`WM02_TRAJECTORY_RESPONSE_TRAIN_FREEZE_THEN_DEV_CPU_V1`. The shared worker map
and dedicated runtime branch are a separate integration step; no old kind is
borrowed. The CLI explicitly calls existing `fusion_scope.validate` and
`pilot1_storage.local_path` so an unrecognized generic runtime site cannot
silently omit the original binding/disk checks. Original `admitted_run`
still validates the source capsule, registry, active worker lease, host/UID,
CPU affinity, output and owner lock. No new scheduler, registration or grant
is created here.

The finite spec requires CPU / `NUMPY_FLOAT64`, ridge1, empty CUDA visibility,
physics/WBC changes false, GR00T/Actor updates zero, predeclared-program
conditioning, positive finite `wall_limit_s`, deadline and original scope
fields. Stop/deadline/wall checks occur at file and NumPy-solve boundaries.
`noise_fraction` and positive `noise_floor_mm` are frozen in the spec; the
threshold value is computed exclusively from new TRAIN same-cell restart
scatter. No old XY threshold or DEV-derived choice is substituted.

The spec supplies hashed `panel`, `source_manifest` and original `raw_action`
NPZ paths. Its `source_files` includes the CLI and every frozen dependency.
The source manifest schema is `WM02_TRAJECTORY_RESPONSE_SOURCE_MANIFEST_V1`:

- `panel_sha256`: canonical scientific panel hash, distinct from its file hash.
- `collector_source_sha`: the panel/actual collection source, not fitting SHA.
- `frozen_model_source_files`: exact file-hash map for response contract,
  response replay, response model, trajectory adapter and SO(3) implementation.
- `episodes`: all72 prospective source descriptors, with embedded exact
  episode identities and these `{path, sha256}` items: `execution_spec`,
  `episode_manifest`, `native_result`, `collection_report`, `decision`, `lease`,
  `worker_exit`. Each actual artifact is inside its declared physical run root.

Preflight validates unique physical run identities, six whole parents
(four TRAIN/two DEV), all context×theta×restart cells and a common clock.
This only validates declared metadata. It does not authenticate DEV source
files or labels before the freeze.

The CLI streams48 TRAIN runs through the frozen replay loader after checking
actual file bytes and cross-joining collection source/spec with all five
frozen dependency pins, episode, CLOSED
zero-exit lease/worker, native report, decision and raw-H40 identity. Only one
predeclared H16 window per run enters fit; all actual installed H40 evidence
is retained in the real TRAIN replay. A12-run single-parent pilot cannot fit.

The frozen model API requires a72-role layout. Internally, unread DEV roles
therefore use NaN history/past-reference/label/clock sentinels and finite
design-only nominal/pulse placeholders. This scaffolding is never written as
real replay. Fit/normalization/LOPO/threshold read only the authenticated48
TRAIN rows. The freeze explicitly says
`AUTHENTICATED_TRAIN48_DEV_DECLARED24_UNREAD`; model-report panel counts do not
mean DEV evidence was authenticated.

The order is exclusive TRAIN replay/model save, distinct file reload, exact
arrays and all three-candidate TRAIN forecasts for every predictor, exclusive
TRAIN freeze and directory fsync. Only then may the24 DEV runs' artifacts be
opened/hashed. The real72 replay retains manifest order, with run IDs beside
prediction row IDs; it must reproduce frozen TRAIN forecasts exactly. No
refit follows DEV reads.

Saved metrics include TRAIN whole-parent LOPO, absolute and temporal-step
H1/H4/H8/H16 XYZ/XY errors, per-parent/context/theta strata, strong static,
same-history additive lag/velocity, conditional, correction-blind and
deterministic lag/velocity/HOLD forecasts. Blind candidate/path invariance is
checked. Results remain one-direction, one-phase prediction diagnostics,
not native feedback, exact counterfactuals, full-state models or task transfer.

The scoped solver observer forwards the unchanged NumPy call and restores it
even on failure. It reports attempted/completed solves (normally20), zero
optimizer updates, one portable NumPy study reload, CPU/wall/RSS and logical
pinned scientific input bytes. These bytes exclude code validation/output
reload reads and are not measured physical disk traffic. Failures preserve
partial artifacts and exact completed counts without retry. Bad DEV leaves
the TRAIN model/freeze intact and produces no successful ACTUAL72 result.
Immutable artifacts use exclusive writes; progress uses the existing durable
atomic-replacement helper. No remote operation or real fit is started by
this document or its synthetic tests.
