# System Architecture V2

## Data-readiness path

Future robot recordings enter through an offline, versioned path before any
model-facing dataset is produced. Stage19 qualifies the same path with a
node-local D3 fixture:

```text
native-rate robot/camera streams
             |
             v
       RawEpisodeV1
             |
       CalibrationBundleV1
             |
        SynchronizerV1
             |
     EpisodeQualityGateV1
             |
   GR00TTrainingExampleV1
             |
      LeRobot / GR00T v2 materializer
             |
      official GR00T loader + processor
```

The raw episode remains the audit authority. The compact model state is
`StateV1` (EEF XYZ + GR00T first-two-rows rot6d + gripper = 10D), and the
primary label is future desired EEF plus gripper. Achieved EEF and full
whole-body state are retained for audit, not silently used as labels. Source
timestamps are preserved; production action timing is
`TBD_BY_ADAPTATION_DATASET`.

The Stage19 D3 fixture has 50 Hz native state/command timestamps and 25 Hz
camera timestamps, despite stale 20 Hz camera metadata. The discrepancy is
classified as `FRAME_SUBSAMPLING_METADATA_DEFECT`; source timestamps remain
authoritative and no retiming is performed. The official GR00T loader path
qualifies the materialized v2 layout; the optional standalone `lerobot`
package is not required by that path.

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
- Stage19 LeRobot/GR00T materialization, official loader, synchronization
  stress, and recorder replay: pass with the classified D3 metadata deviation.
- Robot-specific GR00T adaptation: not performed.
