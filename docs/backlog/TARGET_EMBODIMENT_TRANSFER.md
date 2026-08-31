# Backlog: target-embodiment transfer to P1M

This is deliberately not a Stage14 baseline.

Stage13C ran the delivered WBC through the unqualified P1M bridge for 100
steps. The run had arm target clipping on 93/100 steps, position error mean /
p95 of `0.526898 / 0.863888 m`, and rotation error mean / p95 of
`1.761295 / 3.018101 rad`. These numbers are labeled
`UNQUALIFIED_BRIDGE_TRACKING`.

Material differences were found in ARX5 joint axes, joint origins, joint
limits, Go2-to-ARX5 mount, TCP definition, and drive/actuator semantics. The
P1M canonical asset remains immutable. A future transfer stage must bind an
explicit embodiment adaptation and re-evaluate observation/action semantics;
it must not alter the checkpoint or enlarge joint limits to hide clipping.
