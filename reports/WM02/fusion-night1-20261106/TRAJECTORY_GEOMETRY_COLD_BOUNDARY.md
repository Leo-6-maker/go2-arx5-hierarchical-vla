# Full-H40 qualification: three complete scales and one retained rejection

QUESTION: can the complete, already decoded CORR target proposal be compiled
and consumed through the frozen native executor, without clipping it into the
old1mm XY domain?

CHANGE: source `d699145e847d20bc1c9261112e74f1c3c2605786` preserves all40
knots, explicitly maps the mechanical gripper stroke to the original working
stroke, separates INIT reference from measured decision intent, and applies a
declared quintic integer retime. PR367 priorities were adopted at the closed
boundary; its plan SHA was not deployed as a source downgrade. These are
`NON_VLA_CALIBRATION` runs using the predeclared archived target-query geometry,
with no new Processor decode, model load or query.

ACTUAL_RESULT:

| Explicit geometry scale | Native ticks | Task50ms transitions | Exit | End XYZ error norm | Whole post-decision XYZ trajectory RMSE |
|---|---:|---:|---:|---:|---:|
|0.03|6878|175|0|1.598783mm|0.883831mm|
|0.10|7878|215|0|2.034941mm|1.185717mm|
|0.25|10853|334|0|2.899182mm|1.936211mm|
|0.50|10178actual /12853planned|307|1|Not scored at the unfinished endpoint|Not a completed-trajectory score|
|1.00|0|0|NOT_RUN|Not measured|Not measured|

Each of the first three runs consumed all40 knots and every recorded pose and
gripper atom; all traces had zero drops. Half-scale rejected a future preview
before installation: `ORIGINAL_JOINT_ACCELERATION_REJECT`. All four runs are
CLOSED with MainPID0 and clean four-child shutdown. No native STOP, joint
warning or outer guard rejection was recorded; STOP qualification was not
exercised. Failed-half raw suffixes and rejected candidates are not executed
labels. There are four episodes and one physical parent, not1031 independent
physical conditions.

The exact rejected native packet, H40 and last two consumed command points
reproduced the original guard in a read-only cloned-MjData calculation:
planned IK acceleration1.0860116774361717rad/s² at preview38/joint1, original
bound1.0; join maximum0.37984478569882685. Thus the measured cause is an interior
future-reference acceleration rejection. This is not an observed native joint
violation, and the evidence does not implicate a duplicate/missing history
knot. Clone replay exit0,1.318244wall/1.081583CPU seconds, zero integrations.

The four workers used487.243093wall seconds,219.445563own CPU and727.734835child
CPU seconds. Systemd reports952.802910CPU seconds covering overlapping work;
do not add that to worker CPU. New effects:35787native2ms steps,1031task50ms
transitions (839post-decision), zero GPU work, model updates, new model fits,
VLA queries, Actor updates or full tasks. FUSION totals are now267attempts /
266nativePASS /1nativeFAIL /1114876ticks. With inherited PILOT2:
921attempts /915PASS /6retained failures /4102169ticks. Existing192DEV and
its negative WM-versus-static result are unchanged and were not rerun.

Full SHA identities, each worker cost/exit, commands with private path aliases,
data-report hashes, error components and the cloned-replay result are in
[TRAJECTORY_GEOMETRY_ACTUAL_V1.json](TRAJECTORY_GEOMETRY_ACTUAL_V1.json).
The frozen checkpoint manifest remains
`94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`;
the archived target action file is
`e447e92a9b63371a560efe01d7c5a6e6a1337c5adf22657012f591c7211fc090`.
This query's maximum anchor displacement is17.179319mm and raw waypoint path
281.623587mm. It is a different frozen condition from the retained E1
150.645696mm proposal; that older proposal was not silently reduced/executed.

NEXT_DECISION: test the new explicit2×time version, retaining geometry/all40
knots and the original joint bounds. It requires an independently built and
actually loader-verified262144trace capacity, a fresh native scope and the
actual quarter-scale PASS plus failed-half evidence. First run repaired half,
then full only after complete PASS. Original executor control law, gripper safety,
WBC/observer/actuation bytes and2ms physics remain frozen as specified in
[the V2 protocol](../../../docs/protocols/WM02_FUSION_TRAJECTORY_RETIMING_V2.md).
Only the trace capacity assertion/reader layout permit and new finite quota
selection change; this does not grant general new physics support.

The new response code and pause-assisted current-observation helper are
prospective. They have no actual response cohort, fit, captured live input or
inference yet. A successful geometry qualification will still not establish
WM gain, source-task transfer or online VLA. The main goal remains ACTIVE;
weak-teacher distillation stays deferred, and no new task Actor/Critic or
H200/GPU-physics/distributed expansion is started.
