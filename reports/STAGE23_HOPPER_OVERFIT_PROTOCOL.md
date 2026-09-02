# Stage23 Hopper Overfit Protocol

Status: `PREPARED_ONLY_NOT_AUTHORIZED`

This document freezes the diagnostic boundary for a possible future Hopper
run. No Stage23 training, PBS submission, checkpoint creation, or model
selection was performed in Stage23.

## Candidate runs

```text
O1 = TINY1 deliberate overfit
O6 = TINY6 conditional diagnostic
```

Both candidates must use the official materialized-data path already qualified
in Stage23. They may use only the train split from `red_target_train_a`; they
must not read validation target `e` or locked final target `f`.

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

The Stage23 result clears these preconditions for consideration, but does not
itself authorize O1 or O6.
