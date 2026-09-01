# ADR-011: Actual GR00T Output Is Not Task Evidence

## Status

Accepted for Stage17.

## Context

The base `nvidia/GR00T-N1.7-3B` checkpoint can produce finite tensors through a
custom embodiment processor without having been adapted to Go2+ARX5 data.
Finite output, correct tensor shape, and runtime latency do not establish
meaningful mobile-manipulation commands.

## Decision

Run at most the approved offline custom inference in Stage17. Store only a
node-local output artifact and a reproducible manifest. Route the output
through the provider and non-corrective safety gate, and stop before WBC if it
is rejected. Actual unadapted model-to-WBC execution is not a Stage17 success
criterion.

## Consequences

Runtime closure, interface validity, safety-gate outcome, controller
orchestration, and task performance are reported as separate claims. A gate
rejection is useful evidence about the unadapted output and is not called an
integration failure.
