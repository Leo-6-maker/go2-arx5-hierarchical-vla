# ADR-016: Desired EEF is the action label; achieved EEF is audit state

## Status

Accepted — Stage18

## Decision

The primary action target is future desired absolute EEF trajectory plus
gripper. Achieved EEF remains a separate raw audit stream and is never
silently relabelled as the target.

## Rationale

The final VLA-to-WBC interface predicts a desired EEF reference. Achieved EEF
measures controller/robot response and is needed for quality, lag, and failure
analysis. Mixing the streams would hide tracking error and change the learning
contract.
