# ADR-017: Qualify visual reach before contact-rich manipulation

## Status

Accepted — Stage20S

## Decision

The first simulation-only robot-specific adaptation task is
`SIM_VISUAL_REACH_V0`: RGB target + language + compact EEF state produces a
future desired EEF trajectory and gripper channel. Pick-and-place and contact
physics are deferred until the visual-intent-to-EEF-to-WBC chain is qualified.

## Rationale

Contact, grasp dynamics, object physics, and grasp-success definitions would
mix several unqualified failure sources into the first closed-loop test. A
visual reach isolates task-space intent and native whole-body realization.

## Consequence

Success on this fixture is only a controlled simulation-family claim. It does
not imply general manipulation or real-robot readiness.
