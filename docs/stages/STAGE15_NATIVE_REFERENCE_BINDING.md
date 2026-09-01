# Stage15 — Native reference binding and aligned qualification

Stage15 closes the semantic gap between an embodiment-neutral benchmark shape
and the reference consumed by the delivered native WBC.

```text
TrajectoryShapeV1
  + PlacementProfileV1
  + native reset state / shoulder / reach anchor / TCP
        |
        v
EpisodeReferenceBindingV1
        |
        v
native world-aligned trajectory
        |
        v
9-point SE(3) arc-length preview
        |
        v
native dynamic heading-frame arm observation
```

The delivered native anchor is frozen at reset. The native loader adds its
world-aligned translation anchor to the Gamma positions; it does not rotate
those positions. Orientation is carried explicitly as the native x5_link6
orientation plus the bound shape orientation. The preview uses the exact
offsets `0.010, 0.025, 0.050, 0.090, 0.150, 0.225, 0.325, 0.425, 0.500 m` in
the composite metric `sqrt(dp^2 + (0.15*dtheta)^2)`.

Stage15 uses `MODE_NATIVE_CONTINUOUS` only. The GR00T 40-point ablation remains
closed until the aligned native baseline is expanded and qualified.

The current evidence supports the EEF-only external interface, but the seven
trajectory diagnostic set is not yet a broad native generalization result:
five nominal/centered cases completed, one native static-relocation case
terminated under the native timing rule, and the reused E2 continuous case was
reclassified as native geometrically infeasible.
