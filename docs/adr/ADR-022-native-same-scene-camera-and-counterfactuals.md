# ADR-022: Native Same-Scene Camera and Counterfactual Evidence

- Status: Accepted for Stage21
- Date: 2026-09-01

## Decision

Stage21 visual evidence must come from the native IsaacGym WBC scene that
contains the robot, target actor, camera sensors, and native state. A
procedural or cross-simulator image may be used only as a separately labelled
fixture, never as same-scene evidence.

Visual influence is audited with matched-state/language counterfactuals and
image occlusion diagnostics. Output sensitivity is reported separately from
visual grounding and task success.

## Rationale

An RGB tensor can be accepted by a model without affecting its decision. A
fixed target or an image-only input check is therefore insufficient to claim
visual grounding. The Stage21 camera smoke closes the native-camera blocker,
while its counterfactual results remain sensitivity evidence only.
