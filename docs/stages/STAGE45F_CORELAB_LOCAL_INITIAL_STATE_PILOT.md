# Stage45F Corelab local initial-state pilot

This support branch implements PI0150's bounded diagnostic. It consumes the
sealed Stage43 q0 observations (C0--C4) and the first sealed TRAIN24 window,
then evaluates one fixed N2 seed2223@2000 checkpoint with one common seed-4242
noise tensor and the accepted four-step T16 Euler action-head graph.

The runner has three modes:

- `audit`: CPU-only provenance and sealed-input checks;
- `preflight`: the same checks plus a live GPU0 UUID/free-memory admission;
- `execute`: one model construction/attempt, six conditioning passes and 24
  action-head calls, with raw arrays written before any CPU decode.

Stage43 target coordinates and achieved-world metrics are evaluation-only. The
processor receives only body RGB, wrist RGB, StateV1, and the frozen language
instruction. No target XYZ, WBC state, simulator, training, Validation-E, or
Holdout-F input is passed to the model.

The experiment is deliberately a local diagnostic, not a closed-loop success
claim or a Stage45E route selection. GPU failure, missing sealed input, a
nonfinite output, or any runtime exception produces a failure seal and stops;
the runner never retries or switches devices.
