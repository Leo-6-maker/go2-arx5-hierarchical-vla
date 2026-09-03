# Stage23 Hopper Overfit Protocol

Status: `PHASE_A_EXECUTED_NEGATIVE_DIAGNOSTIC; PHASE_C_PREPARED_ONLY`

This document records the Phase A O1 evidence and freezes the boundary for
future work. Phase A executed fresh `BASE -> N` runs at 10/25/50/100/200
steps on Hopper and stopped at each requested maximum. Phase B performed no
new training or inference. Phase C is preparation-only: it has not submitted
PBS, created a new checkpoint, loaded a model, or selected a model.

## Candidate runs

```text
O1 = TINY1_H2STATS deliberate overfit (Phase A executed; profile failed)
O1-secondary = TINY1_TINYSTATS retained one-snapshot diagnostic (not executed)
O6 = TINY6_TINYSTATS conditional diagnostic (not executed)
```

All candidates must use the official materialized-data path already qualified
in Stage23. They may use only the train split from `red_target_train_a`; they
must not read validation target `e` or locked final target `f`.

`TINY1_H2STATS` is the only primary O1 input. Its one selected sample uses
the exact Stage22 frozen `stats.json` and `relative_stats.json` bytes. The
tiny-stat variants are retained precisely because they use regenerated
statistics and therefore answer a different diagnostic question.

## Candidate recipe

```text
action_horizon       = 40
tune_projector       = true
tune_diffusion_model = true
tune_visual          = false
tune_llm             = false
tune_vlln            = false
```

The base checkpoint, GR00T revision, Cosmos revision, processor configuration,
StateV1 ABI, action representation, and timestamp semantics must remain the
ones audited by Stage23. No new loss, action representation, state ABI,
retiming, smoothing, clipping, or trajectory correction is admitted by this
protocol.

## Required attribution outputs

Any future authorized run must separately report:

```text
training target / decoded physical action
reference reconstruction error
controller tracking error (only if a separately approved WBC run exists)
end-to-end task-space error (only if a separately approved WBC run exists)
path-length ratio
mean-speed ratio
max-speed ratio
roughness
initial discontinuity
SafetyGate decision
```

Training loss alone cannot select an overfit checkpoint. A future threshold
proposal remains uncommitted until the PI freezes it before the run.

## Stop conditions

Stop before Hopper if a future audit changes any of the following:

```text
10D physical action is not the only active loss partition
122 padding dimensions are active in the mask
relative reference is not the last observed eef_9d sample
rot6d ordering or inverse transform is inconsistent
desired and achieved labels are mixed
timestamp ordering is not preserved
official loader/processor no longer reproduces the Stage23 shapes
```

The Stage23 result cleared the preconditions for the Phase A O1 diagnostic,
which is now complete. This document does not authorize an O1 rerun,
O1-secondary, O6, or any Phase C model execution.
