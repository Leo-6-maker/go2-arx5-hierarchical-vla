# Stage15 TCP contract

## Controlled pose

The native policy's controlled semantic is:

```text
link:          x5_link6
position:      gripper_center
orientation:   x5_link6 orientation
link -> TCP:   [0.1424, 0.0, 0.0001057] m
```

The source configuration names `x5_link6` as the end-effector body and
provides `ee_local_offset`; runtime state construction applies that local
position offset while retaining the link orientation. Evidence:

- `config/wbc.py:64-67`;
- `wbc_env.py:798+` and `wbc_env.py:1023-1031`;
- native URDF fixed `gripper_center` child of `x5_link6`.

## Interface rule

`x5_link6_gripper_center` is the TCP semantic ID used by the Stage15 contracts.
The external trajectory describes this pose. It must not be substituted by the
flange/link origin, the P1M provisional TCP, or a real-robot calibration
assumption.

The native WBC does not own the GR00T gripper scalar. The 10D GR00T physical
projection is split as EEF 9D to this contract and gripper 1D to a later
parallel channel.

## Kinematic check

The source-backed URDF model and independent repository FK were compared at
100 deterministic native simulator configurations. Maximum discrepancies were:

| pose | translation | rotation |
|---|---:|---:|
| `x5_link6` | `5.31e-07 m` | `1.65e-06 rad` |
| `x5_link6_gripper_center` | `6.50e-07 m` | `1.65e-06 rad` |

These are FK agreement checks, not tracking results and not real calibration.
