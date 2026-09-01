# Stage20S Runtime Orchestration

## Interface seal

The actual-model path was sealed as:

```text
decoded GR00T [40,10]
 -> GR00TTrajectoryProvider
 -> TrajectoryChunkV1
 -> TrajectorySafetyGateV1
 -> EpisodeReferenceBindingV1
 -> GR00TToWBCAdapter
 -> WBCPreviewV1
```

The seeded easy candidate had finite values, valid SO(3), explicit timestamps,
0.608 mm initial position discontinuity, 0 rad initial rotation
discontinuity, and no corrective clipping or retiming. Its gate decision was
`ACCEPT`; the provider emitted EEF `[40,9]` plus a parallel gripper `[40]`
channel. The WBC preview was finite with shape `[9,9]`.

The progressive offline candidate also passed the interface seal. Its initial
position discontinuity was `4.391 mm`, path length `0.36765 m`, and maximum
declared-timing speed `0.5375 m/s`. It was not executed through native WBC
because the easy actual-model closed-loop gate had not qualified.

## Gated native execution

`stage20s_native_closed_loop.py` requires a matching interface seal with
`TrajectorySafetyGateV1=ACCEPT` and `actual_wbc_execution=false`. It refuses
WARN/REJECT trajectories and writes a separate orchestration manifest. With
the correct native IsaacGym environment, it executed the unchanged accepted
easy candidate and recorded `EXECUTED_TRACKING_ABORT`; this status separates
successful orchestration from successful reference completion.

No ZMQ, ROS2, shared-memory transport, or PolicyServer was introduced.
