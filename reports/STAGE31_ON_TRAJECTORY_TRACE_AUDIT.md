# Stage31 stock-sampler trace audit

Gate: `PI_GATE_STATE_0011` (Issue #11 comment `5548763347`)

Accepted base: `8ba410e7185e6508a59d7022fbba28da96941a40`

Pinned GR00T source: `/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t`

Pinned GR00T HEAD: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`

Pinned sampler source SHA256: `6bd66ef768f1484ac5a7889c40358a599f1862b1d554d47d357e113f2f96501c`

## Canonical path

The only selected scientific path is the same public path used by Stage29:

`Gr00tN1d7.get_action` → `Gr00tN1d7ActionHead.get_action` → `get_action_with_features`.

The pinned action head creates one `torch.randn([B,40,132])` state, fixes
`dt = 1 / num_inference_timesteps`, executes exactly four iterations, computes
timestep IDs `0,250,500,750`, passes the current `actions` tensor into
`action_encoder`, slices the last H40 rows of `action_decoder` output as
`pred_velocity`, and applies `actions = actions + dt * pred_velocity *
vel_strength`.  With no RTC input, `vel_strength` is all ones.

## Instrumentation boundary

Stage31 does not copy or reimplement that loop.  Around the public model call it
temporarily registers:

- an `action_encoder` forward-pre-hook which appends references to the positional
  `actions` and timestep tensors;
- an `action_decoder` forward-hook which appends a reference to the returned
  decoder tensor.

Both hooks return `None`.  They do not clone or move tensors while the sampler is
running, replace inputs or outputs, consume RNG, alter dtype/device, or call a
second model path.  The final `action_pred` is taken from the unchanged public
return.  Only after the call returns are the last H40 decoder rows identified as
the four used velocities and copied to float32 CPU evidence.

The harness requires exactly four state and four decoder observations, timestep
IDs `[0,250,500,750]`, exact tensor equality for each captured Euler update, and
the Stage29 raw initial-noise and float32 final-output hashes for every call.
The first mismatch stops the campaign and forbids autonomous retry.

## Scientific and recovery boundaries

Exactly six calls are permitted: seeds 2223, 2224, 2225, each with BASE then
H2@2000.  No training, backward, optimizer/scheduler step, resume, checkpoint
write, alternate sampler, changed step count, smoothing, clipping change,
validation/holdout read, WBC, or hardware path exists in the selected wrapper.

Before the first scientific call, Z0 revalidates both historical artifact seals
and all six Stage29 NPZ files, reconstructs the normalized universal target with
the frozen processor/data path without constructing a VLA model, and reproduces
the Stage29 conditioning hash.  Source and mechanical repairs must be committed
before execution.  After the first call begins there is no autonomous rerun or
replacement campaign.
