# FD2 Actual Dynamic Expert Clearance Pilot

Status: **CLOSED_COMPLETE_ALL_EIGHT_ACTUAL_AUDITED_CELLS**.
Runtime318445f17ed2093595e9df252cc56bb24280f805, new independent
run20261018T171916Z-localpc-fd2; no FD1/N1 reopening, training or WBC modification.

## Actual Results

Original4mm: **4/4** complete task successes.
Fixed12mm: **4/4**. Both arms' actual native faults
total **0**; original S1 vetoes
total **0**. These are four paired TRAIN
layouts, not eight independent layouts or DEV/generalization evidence.

| Cell | Nominal mm | Task Success | Min Actual Vertical Gap mm | S1 | Hold |
|---|---:|---|---:|---:|---:|
| PI0294-FD2-G0-C04 | 4 | True | 0.729830 | 0 | 41/41 |
| PI0294-FD2-G0-C12 | 12 | True | 8.673590 | 0 | 41/41 |
| PI0294-FD2-G1-C12 | 12 | True | 8.246315 | 0 | 41/41 |
| PI0294-FD2-G1-C04 | 4 | True | 0.322677 | 0 | 41/41 |
| PI0294-FD2-G2-C04 | 4 | True | 0.603885 | 0 | 41/41 |
| PI0294-FD2-G2-C12 | 12 | True | 8.543109 | 0 | 41/41 |
| PI0294-FD2-G3-C12 | 12 | True | 8.273940 | 0 | 41/41 |
| PI0294-FD2-G3-C04 | 4 | True | 0.347871 | 0 | 41/41 |

The matrix contains actual object contact/displacement,500Hz no-lift,
separation/withdrawal, every original terminal41 subpredicate, all stage
events, original S1 decisions/parity, saved-byte audits and video indexes.
Success means the complete unchanged object-task score, not contact alone.
The original41-point hold scoring window is28.0-30.0s, not an invented replacement window. All eight first attempts retained; no seed search or replacement. Initial
scene XML, qpos/qvel/TCP and both-camera pixels match exactly within all
four pairs.499 historical layout records were excluded geometrically,
ignoring visual changes; these are not claimed499 independent scenes.

## Dynamic Clearance

Across cell minima, exact vertical pad/table gaps span
**0.322677 to 0.729830mm**
for the nominal4mm arm and
**8.246315 to 8.673590mm**
for nominal12mm. Nominal reference, actual vertical clearance, native-query
output and original safety verdict are distinct quantities.

All saved500Hz pad/table geometry is independently reconstructed without
integration; four additional audit constructors are charged. The saved
geometry JSON gives every reconstruction error and finite-table footprint
qualification. No extra rollout or unrecorded hidden-state parity is claimed.

The3.2.7 native mj_geomDistance(distmax=2m) outputs are retained and reproduced,
but not silently doubled/rescaled or promoted to exact Euclidean separation.
That API uses the collision function with distmax as margin in this version:
[official3.2.7 implementation](https://github.com/google-deepmind/mujoco/blob/3.2.7/src/engine/engine_support.c#L1295).
Primary physical clearance uses actual box vertices and the actual table
top, qualified when pad footprints lie inside the finite table. Native
S1 safety continues to use its original unchanged rules, not this diagnostic.

Recorded stage partitions are the active requested-prefix Teacher stage,
not a privileged-label substitute for object-task phase or success.500Hz
actual robot-contact subsets are joined separately. Observer callbacks
add no forward/snapshot/integration and check native/Python/RNG fingerprint
nonmutation. Original live/shadow parity remains1e-10. The independent
saved-byte audit includes every raw member/image hash, reference label,
original score and separately implemented complete32s push predicates.

## Decision And Limits

Both experts completed the four fresh layouts. Original4mm dynamic task feasibility is demonstrated here; expert infeasibility or a unique neural training cause is not established. Qualified contact-phase gap gains >=4mm: 4/4 pairs. The separately authorized PV1 P0 rule selects CLEARANCE_PAIR, not an assumed12mm winner.

Separate PV1 P0 decision **CLEARANCE_PAIR** is authorized by9eff29a44a193dd4d3620dc0bfe5764a402eff42, not by unused FD2 resources. This pilot does not identify
a unique GR00T training cause, prove a training benefit or repair T1 release,
T2 Oracle0/2, Go2/WBC transfer or real-robot performance. The expert was
controlling these trajectories; no GR00T query occurred.

## Evidence And Closure

[All8 actual results](FD2_ALL_EIGHT_ACTUAL_RESULTS.json),
[exact four initial pairs](FD2_FOUR_PAIRED_INITIAL_IDENTITIES.json),
[saved geometry qualification](FD2_SAVED_GEOMETRY_QUALIFICATION.json),
[all8 public videos](FD2_ACTUAL_VIDEO_ARCHIVE.json),
[resource final](FD2_RESOURCE_FINAL.json),
[reproducible checks](FD2_REPRODUCIBLE_CHECKS.ipynb).

Core accountedCPU8152.063277s, including
explicit900+3600s control uppers; local3600s conservative upper. Actual
256000ticks=128000live+128000S1;
684starts including clones and four zero-integration
audit constructors. GPU/PBS/NN/updates0. All registered workers terminal,
new lease CLOSED, future reserved0, unknown effects0; old lease bytes,
FD1/N1 closure,26 assets and15 numerical files preserved.

Public HF READY was uploaded last, exact file sets/sizes/hashes read back
at both payload and final revisions. No old model/data was downloaded in
bulk, deleted or made public. No healthy job cancelled, daemon, tunnel or
second owner created. WBC/legs/plant/gripper/S1/score frozen, TF-R3 STOP
preserved, selected/deployment NONE.

Original N1 autonomous results remain T1 10/20 and T2 4/20. These new Teacher outcomes do not replace that denominator.
