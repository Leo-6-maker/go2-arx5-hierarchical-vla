# Stage20T Next Simulation Gate

## Current gate state

Stage20T did not qualify the unchanged actual-model visual-reach rollout for
general closed-loop promotion. It did qualify the attribution path: the
provider, binding, native preview, native WBC execution, and lifecycle trace
are all usable for the next controlled simulation stage.

The next stage must not treat an artificially re-timed success as unchanged
model qualification. It should first make the timing contract explicit for
the data/inference/execution path and keep the native reset condition
observable.

## Proposed Stage21 gate

```text
SAME_SCENE_RECEDING_HORIZON_VLA_WBC_GATE
```

Required controls:

1. Render RGB from the same native WBC simulator scene as the robot state.
2. Re-infer a fresh single chunk after executing only a prefix of the prior
   chunk; include `chunk_id`, observation/receive/execute timestamps, age,
   freshness, stale, and out-of-order decisions.
3. Keep StateV1 and desired-action timestamps explicit. Do not infer a
   production action period from camera FPS.
4. Run a fixed-reset/anchor control alongside the existing native reset
   perturbation control, so initial discontinuity is not confused with model
   profile error.
5. Preserve the four error terms:
   `VLA_PREDICTION_ERROR`, `REFERENCE_RECONSTRUCTION_ERROR`,
   `CONTROLLER_TRACKING_ERROR`, and `END_TO_END_TASK_SPACE_ERROR`.
6. Keep actual model execution simulation-only and do not add ZMQ, ROS, or
   hardware transport at this gate.

Promotion requires completion on the declared easy task at the unchanged
model timing contract, repeatability across a clean reset cohort, and no
unexplained native termination. Progressive relocation remains a later
stress task, not a prerequisite for the first same-scene visual reach gate.

## Explicit non-claims

Stage20T does not establish real-robot readiness, production action frequency,
general manipulation, or arbitrary high-curvature 40-point equivalence. It
also does not authorize more training; the next attribution-preserving step is
same-scene receding-horizon orchestration with the current frozen controller
baseline.
