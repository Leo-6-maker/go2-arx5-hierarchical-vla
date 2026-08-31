# Stage15 reset anchor audit

## Exact native operation

For each reset, the native runtime computes:

```text
shoulder_world = base_position_world
                 + R(base_quaternion_world) [0.05, 0, 0.10]

direction_body  = normalize([0.36, 0, 0.10])
direction_world = R(base_quaternion_world) direction_body

R_max           = reach_table.query(direction in shoulder frame)
anchor_world    = shoulder_world + direction_world * (0.60 R_max)
```

Source: `wbc_env.py:325-332` and `wbc_env.py:1302-1338`; configuration values
are in `config/wbc.py:419-430` and the native reach table is
`resources/reach_tables/go2_x5_2d.pt`.

## Temporal semantics

The anchor is **reset-dependent** and **frozen at trajectory creation/reset**.
The native loader adds the anchor to Gamma positions once; it does not update
the anchor as the base moves. The policy preview is different: at every
observation the current base orientation is reduced to a yaw/gravity heading
and applied to the preview. Thus:

- anchor placement is frozen;
- preview frame orientation is dynamic;
- dynamic heading re-expression must not be mistaken for a moving anchor.

## Trace evidence

The non-invasive trace tool ran 12 deterministic resets on the native runtime.
Example reset 0:

| quantity | value |
|---|---:|
| base position (m) | `[0.000571, 0.008825, 0.366512]` |
| shoulder position (m) | `[0.044166, 0.019566, 0.468904]` |
| world direction | `[0.928579, 0.194401, 0.316147]` |
| reach radius (m) | `0.7687425` |
| anchor (m) | `[0.472469, 0.109232, 0.614725]` |

The independent `NativeAnchorInputs` implementation reproduces all 12 anchor
positions with maximum error `7.2759e-08 m`.

## Binding consequence

For a generic rigid binding the repository exposes the explicit conceptual
equation:

```text
T_world_eef(t) = T_world_anchor T_anchor_task T_task_eef(t)
```

For the delivered native loader, the position operation is intentionally
specialized to:

```text
p_world(t) = p_anchor + p_anchor_task + p_shape(t)
```

because native Gamma positions are world-aligned and are translated without
being rotated. Orientation is still composed explicitly. This distinction is
encoded as `NATIVE_TRANSLATION_ONLY_WORLD_ALIGNED`, not hidden behind a generic
SE(3) assumption.
