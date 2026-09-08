# ADR-023: Reference-Only Receding-Horizon Hot-Swap

- Status: Accepted for Stage21
- Date: 2026-09-01

## Decision

The first receding-horizon implementation replaces only the external EEF
reference. It must not reset physics, teleport the robot, or reset arm/dog
policy history. GT desired chunks qualify this infrastructure before actual
model claims are made.

The primary test mode is explicitly named
`SYNCHRONOUS_PAUSED_SIM_RECEDING_HORIZON`. Async transport and real-time
readiness are separate gates.

## Rationale

Resetting or retiming at every chunk boundary would hide whether a failure
comes from hot-swap lifecycle, controller tracking, or model profile. Keeping
the episode continuous makes those causes observable. Stage21 therefore
reports both predicted-profile handoff and achieved-state handoff rather than
collapsing them into one continuity score.
