# Corelab Stage40 CT3 runtime preparation

Classification: `NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE`

This preparation package adds a fail-closed, direct RTX launcher and two
barriered hold processes for CT3. It does not submit a job, construct a model,
create a simulator, or query CUDA. The current Issue #11 gate is still
`PI_GATE_STATE_0065`, and the Vanda execution-ownership release has not been
observed.

## Protocol frozen for the next authorized boundary

The launcher requires a one-line Vanda ownership-release file containing either
`VANDA_STAGE40_A40_QUEUE_RELEASED_CORELAB_PRIMARY` or
`VANDA_STAGE40_A40_EXECUTION_ENDED_CORELAB_PRIMARY` before its first GPU-facing
operation. After that check it runs `nvidia-smi`, binds the explicit
RTX UUID, and requires at least the preferred 32-GiB card. It starts only the
native constructor and exact GR00T constructor on that UUID; both wait at a
release barrier.

The native hold captures body RGB `[96,96,3]`, wrist RGB `[96,96,3]`, and finite
StateV1 `[10]` using the current native WBC source. Its construction-only
trajectory is neutral and is not a scientific reference. The model hold binds
the sealed local Cosmos snapshot and constructs the exact STRUCTURED@1000
model without submitting an observation or calling `get_action`.

CT3 budgets remain exactly zero: GR00T forward, native WBC step, reference
update, and scientific rows. The launcher combines the two runtime counter
records, writes the evidence manifest, and seals only a PASS result. Any
constructor error, missing input, wrong UUID, non-32-GiB card, invalid token,
or nonzero counter stops the run without downstream admission.

## Current disposition

`PREPARED_NOT_EXECUTED` — no GPU/model/WBC execution was performed while Vanda
remains queued. Freeze the `RTX-runtime` bundle from the final clean pushed
HEAD immediately before the explicit ownership cutover; do not reuse the old
Vanda bundle or create a second GPU lane.
