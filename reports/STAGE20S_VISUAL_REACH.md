# Stage20S Visual Reach Qualification

## Task definition

`SIM_VISUAL_REACH_V0` uses a visible red target, the instruction
`move the end effector to the red target`, and the current 10D EEF/gripper
state. The first task deliberately excludes contact, grasp dynamics, object
physics, and a pick-success definition.

## Result

The data-to-model path passed its controlled fixture checks, and one actual
overfit N1.7 output was accepted, sealed, and executed by the native WBC
simulation. The native rollout nevertheless stopped after 25 samples with a
timing `TRACKING_ABORT`. Therefore the result is:

```text
SIMULATION_ONLY_CHAIN_EXECUTED_WITH_BOUNDARY_DEVIATION
SIM_END_TO_END_VLA_WBC_CLOSED_LOOP_QUALIFIED: NOT YET
```

The known-good Stage16 benchmark control completed normally, which isolates
the runtime orchestration from the actual model candidate. The four-error
report shows that provider/reference reconstruction stayed near 1 mm while
controller tracking was about 3.6 cm mean and 0.255 rad mean on the partial
model rollout.

## Camera and claim boundary

The visual input is a procedural cross-simulator fixture. It is not a native
same-scene camera rendering from the IsaacGym WBC process. The result supports
the software contract and a controlled simulation fixture only; it does not
support claims of real-robot readiness, general manipulation, or robot-wide
visual grounding.
