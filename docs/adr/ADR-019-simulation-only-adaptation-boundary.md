# ADR-019: Keep Stage20S robot-specific adaptation simulation-only

## Status

Accepted — Stage20S

## Decision

Stage20S may materialize simulation data, run the official GR00T adaptation
entry point, and execute accepted model output through the delivered native
WBC only in simulation. It may not connect to Go2/ARX5 hardware, send motor
commands, start hardware ROS, or collect physical data.

## Rationale

The data, calibration, clock, and physical safety contracts are not yet
validated on the robot. A simulation-only gate gives useful architecture
evidence without turning a finite model output into an unreviewed command
path.

## Consequence

Stage20S can establish a controlled simulated task-family result at most. Any
real pilot requires a separate authorization and physical calibration gate.
