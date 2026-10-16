# Q4 Fixed Upstream Portability Audit

All upstream/native files and Git blobSHA identities are in
`third_party/flowdagger/UPSTREAM_SOURCE_LOCK.json`. Microsoft MIT license is
byte-identical to upstream; NOTICE maps adapted components to original files.
No LIBERO assets/environment/expert or upstream full training loop was installed.

| Upstream Component | Our Mapping | Actual Status |
|---|---|---|
| groot_inversion | inversion.py/native.py | adapted tensor math + original head/masks; mockPASS |
| groot_expert_inversion | codec.py | newEEF10 adapter through existingSAP; no7D encoder |
| NoisePolicy | steering.py | full initial latent MLP and detached loss; synthetic gradient only |
| FlowDaggerBuffer | aggregation.py | pinned audited correction index, not uncheckedringbuffer |
| InterventionHandler | contracts.py | explicit external decision and original safetyproof binding; scheduleUNSET |
| LiberoExpert | oracle.py | not copied; existingsource Teacher.plan wrapper and explicitINVALID |
| GR00T rollout | transaction.py | CPU handshake only; no model/env/training loop |

Official example:16x7 OSC action, gripper+1close/-1open, prefix8 on student
branch, livefeedback16 on expert branch, model40x132. Default intervention1.0
selects expert every block; generic beta/disagreement Handler is not called by
the GR00T trainer. Its default is not evidence of student-state DAgger.

Our actual metadata: M2048,40x132padded, activeEEF10,4nativeEulersteps,
1000timebuckets, alternateVLDiT; relativeEEF once and absolutegripper through
original v3.0 processor. Processor kwargs are nested under processor_kwargs,
not top-level. Source officialSHA51d4c89 differs from upstream example9c7e746;
code inspection alone is not compatibility qualification.

Feature bridge calls observation-only existingprocessor, backbone once, head
_encode_features once. It does not call get_action just to capture features,
consume/discard a stochastic draw, or concatenate Teacher/objecttruth. Frozen
eval parameters are checked before each velocity evaluation. Native optional
positionembedding and alternateVL masks have CPU test doubles, notrealweights.

Counted inverse uses the same k/Ntimegrid as forward. Foursteps*11inverse
evaluations+fourreconstruction =48velocitycalls, plusfeatureencoding; ordinary
draw has4. This is call accounting, not a latency or throughput measurement.
The affine mock has an approximate inverse; a divergent finite mock is not
qualified. PhysicalXYZ/SO3/gripper timing and original safety must still be
tested on realM2048; paddedMSE alone cannot prove expressibility.

Future intervention proposal: predeclare beta or dimension-aware XYZ/SO3/grip
disagreement on TRAIN, preserve immutable safety overrides, request same-state
expert labels during genuine student visitation; independent eval expertOFF.
No new probability/threshold is selected by this CPU package.
