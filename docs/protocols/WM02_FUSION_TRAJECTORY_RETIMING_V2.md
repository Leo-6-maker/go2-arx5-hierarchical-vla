# Full-H40 time-stretch qualification V2

This is a new finite qualification after the sealed V1 half-scale rejection.
It does not reopen that lease or reclassify the failed attempt. Completed V1
3%, 10% and 25% checks are retained and are not repeated.

V1 half-scale stopped at10178 when the unchanged `certify_chunk` rejected a
future preview. A read-only replay of its exact final native packet, proposed
H40 and two actually consumed command-history points reproduced
`ORIGINAL_JOINT_ACCELERATION_REJECT`: cloned-IK reference acceleration
1.0860116774361717rad/s² at preview38/joint1, versus the original1.0 bound.
The join maximum was0.37984478569882685rad/s². These are proposed-reference
kinematic quantities, not a measured native joint-limit violation. The failed
run had no native STOP, joint warning or outer guard rejection and closed all
four children cleanly. Its result SHA is
`77bea49654a7d33007312d20d416f3796896a508d3e93dbce134a7d2ca5e1a8d`.

The new phase is `collect-target-fusion-trajectory-slow-qualification-v2`.
Its episode schema is `WM02_FUSION_TRAJECTORY_QUALIFICATION_EPISODE_V2`,
reference rule
`FULL_H40_RIGID_RELOCATE_SCALE_WORKING_GRIP_INTEGER_RETIME_2X_V2`, and explicit
integer `time_stretch_factor:2`. All other geometry/gripper/frame/anchor
conventions remain those of V1. Only geometry scales0.5 and1.0 are admitted.
The adapter multiplies the original bridge and uniform segment interval counts
by2, then samples the same quintic time law from the same exact40 knots. It
does not linearly interpolate the old samples, clip geometry, delete knots or
change the original between-grid native interpolation. Default factor1 retains
the original V1 evidence and path values.

For the pinned actual query, half/full planned end atoms are716/1034;
including the unchanged8-atom tail, native endpoints are21803/29753. Every job
must independently declare its exact computed endpoint, at most33000, and
positive wall limit at most1200s inside the unchanged parent deadline/reserve.
Only these new long phases use the original already-supported task quota
bucket64; the exact registered `max_ticks` still bounds every advance. No quota
header, physical timestep, control law, WBC or safety threshold is changed.
Other dynamic phases retain bucket32 and600s admission; non-dynamic phases keep
their existing quota selection.

A fresh trace-only build must first verify262144 capacity and140000 complete
synthetic serialized-key events, no drop and no physics. The trace CPP adds
only262144 to its compile-time capacity assertion and message; its default65536,
record layout, emission, writer, queues and callbacks stay unchanged. The
builder restores that exact assertion text and checks the resulting source SHA
against the original C10 build, while requiring the original header bytes. The
new trace DSO/source SHAs are new identities. Frozen executable, observer and
actuation DSOs stay byte-identical. The reader only adds the explicit262144
header capacity to its accepted layouts. Actual native mapping/header and full
trace still require separate execution evidence.

The first V2 half-scale requires the actual V1 quarter-scale CLOSED0 result
`2732f75836d8d89f5789d53bb0b9de4f28716728bf44819a5fc46513b76be9c0`
as successful predecessor, plus the exact terminal failed-half result above.
V2 full-scale requires the immediately preceding complete V2 half-scale PASS,
same query, scene, frozen executor, time law and full pose/gripper evidence.
The old failed half is never used as a successful predecessor.

Use the existing `scripts.wm02.fusion_trajectory_qualification_check` entry.
Check-only remains read-only and creates no grant. Execute still delegates to
the original native worker, owner, registry, Journal, fresh lease, locks,
full-preview certificates, ACK/TTL, per-atom gripper, trace and shutdown.
Code/tests or a successful trace build do not establish a new native PASS,
response-model domain, online VLA, full task success or correction gain.
