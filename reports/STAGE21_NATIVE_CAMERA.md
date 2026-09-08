# Stage21 Native Same-Scene Camera Qualification

## Result

```text
NATIVE_CAMERA_GATE: PASS
```

The native WBC scene created a red target actor, attached body and wrist
cameras, and captured 96x96 RGB images while the native Go2+ARX5 WBC scene
remained active. The camera smoke artifact is:

```text
/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage21/continuity/camera_smoke/camera_smoke.json
```

The smoke run recorded simulator instance `native-isaacgym-pid-2670481`,
native WBC active, target actor `stage21_red_target`, attached body/wrist
camera handles, unchanged robot root/DOF state while the marker moved, and
changed body/wrist image hashes between target placements.

Raw red-pixel counts were `body_a=0`, `body_b=7`, `wrist_a=160`, and
`wrist_b=0`. These counts show that visibility is camera/placement
dependent; they are not evidence that both cameras visibly resolve both
target placements.

## Actual model execution scene

The actual-model receding runner independently captured its own camera/state
observations in one native simulator instance:

```text
native-isaacgym-pid-2691550
```

All six requests (`model-0000` through `model-0005`) carried this same
simulator instance ID, `simulation_only=true`, `hardware_control=false`,
source frame `NATIVE_WBC_RELATIVE_REFERENCE`, and TCP semantic ID
`x5_link6_gripper_center`. The actual model path therefore used native
same-scene observations; it did not relabel procedural cross-simulator RGB.

The camera smoke counterfactual and actual receding run are separate native
simulator instances, as expected for separate experiments. Their IDs are not
conflated.

## Boundary

The worker/mailbox exchange was an offline shared-filesystem harness. It is
not a real-time camera transport and does not qualify deployment latency.
