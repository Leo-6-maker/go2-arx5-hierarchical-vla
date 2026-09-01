# System Architecture V2

## Runtime thin waist

The external interface is EEF-only, with a parallel gripper channel:

```text
body RGB + wrist RGB + language + compact proprioception
                         |
                         v
                  GR00T N1.7 runtime
                         |
             decoded physical [H,10]
              [EEF 9D] [gripper 1D]
                    /          \\
                   v            v
        GR00TTrajectoryProvider  GripperCommandV1
                   |
          TrajectoryChunkV1
                   |
        TrajectorySafetyGateV1
                   |
       EpisodeReferenceBindingV1
                   |
             GR00TToWBCAdapter
                   |
              WBCPreviewV1
                   |
              delivered native WBC
        arm action + internal structured 9D body/gait plan
                   |
             native Go2+ARX5
```

The thin waist deliberately hides raw GR00T universal tensors from WBC code.
It carries explicit timestamps, frame, TCP, orientation representation,
provenance, and lifecycle state. The 9D body/gait plan remains an internal
controller output, not a VLA latent or an external GR00T output requirement.

## Safety and stop boundaries

`TrajectorySafetyGateV1` validates without correcting. `REJECT` stops the
actual model path before binding and WBC. `WARN` remains visible to the
caller; it is not silently accepted as a clean trajectory. Threshold labels
distinguish controller contract, generator support, benchmark-derived,
test-only, and real-robot-TBD values.

The Stage17 actual-model path stopped at the gate because its decoded motion
exceeded the test-only linear-speed threshold. The known-good benchmark path
was separately validated through native WBC. These are two different claims.

## Transport boundary

The provider and lifecycle contracts are transport-independent. Stage17 uses
an abstract no-network lifecycle and introduces no ZMQ, ROS2, shared-memory,
or PolicyServer path. A deployment stage may select a transport after the
actual system architecture and clock requirements are fixed.

## Evidence status

- Stage16 native general-WBC baseline: qualified with a known boundary/timing
  limitation.
- Stage17 actual custom N1.7 inference: runtime pass, no task evaluation.
- Actual model interface: safely rejected by gate; no WBC execution.
- Known-good benchmark orchestration through native WBC: pass.
- Robot-specific GR00T adaptation: not performed.
