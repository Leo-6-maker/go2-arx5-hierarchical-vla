# Stage21 GT-First Receding-Horizon Gate

## Result

```text
GT_RECEDING_HORIZON_INFRASTRUCTURE_PASS
```

Fixed primary reset seed: `13`.

| Field | Result |
|---|---:|
| Prefix K | 4 (declared choices 4/6/8/10) |
| Native control dt | 0.01999999955 s |
| Native prefix duration | 0.0799999982 s |
| Chunk test period | 0.05 s (`TEST_ONLY`) |
| Steps / valid steps | 24 / 24 |
| Initial chunk + updates | 1 + 5 |
| Reset events | 0 |
| Safety gate | 2 ACCEPT, 4 WARN, 0 REJECT |
| Lifecycle | 5 FRESH updates |

The runner paused neither by resetting the robot nor by teleporting physics.
Every update retained root state, DOF state, episode length, arm policy
history, and dog policy history. The run ended at the declared harness
schedule boundary (`SCHEDULE_END`), not at a native task termination.

## Interpretation

This is a reference-update/infrastructure qualification. The fixed Stage16
GT replay is not a fresh policy-conditioned label at each observation, so its
large achieved-to-new-first discontinuities are not used as a tracking claim.
The relevant invariant is that each update was applied without resetting
physics or policy history and with predicted handoff measured separately.
