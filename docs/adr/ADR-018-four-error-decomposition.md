# ADR-018: Separate model, reference, controller, and end-to-end error

## Status

Accepted — Stage20S

## Decision

Every model-driven WBC simulation reports four separate quantities:

1. `VLA_PREDICTION_ERROR`: decoded model action versus desired label;
2. `REFERENCE_RECONSTRUCTION_ERROR`: model reference versus native consumed reference;
3. `CONTROLLER_TRACKING_ERROR`: native consumed reference versus achieved EEF;
4. `END_TO_END_TASK_SPACE_ERROR`: desired label versus achieved EEF.

Post-reference idle samples are excluded. Native lifecycle termination is
reported independently from these metrics.

## Rationale

A final target error cannot identify whether the model, interface conversion,
or WBC caused the deviation. The four-way split preserves that diagnosis.

## Consequence

An accepted model trajectory and a completed task remain separate claims.
Partial rollouts retain their termination code and valid-prefix metrics.
