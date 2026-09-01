# Stage21 Visual Counterfactual and Occlusion Audit

## Construction

The native camera smoke generated observations with:

```text
same StateV1
same language: move the end effector to the red target
same native simulator instance
different target location/image
```

The model used the same RNG seed (`2121`) for all three inputs. The actual
inference manifest is:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage21/same_scene_model/same_scene_groot_inference_manifest.json
```

All inputs produced finite `[1,40,132]` universal and `[1,40,10]` decoded
outputs. The provider split `[40,9]` EEF and `[40,1]` gripper channels; the
gripper channel remained parallel and was not sent to WBC.

## Counterfactual results

| Pair | Same state/language | Image change | Decoded EEF mean position delta | Endpoint delta | Max decoded delta |
|---|---|---|---:|---:|---:|
| target A vs target B | yes | body+wrist | 0.9475 mm | 2.3263 mm | 0.002334 |
| target A vs body-occluded A | yes | body only | 0.6873 mm | 2.6451 mm | 0.002024 |
| target B vs body-occluded A | yes | body+wrist | 1.6122 mm | 4.5063 mm | 0.003968 |

Rotation deltas and gripper deltas were zero in this fixture. All three
counterfactual changes exceeded the `1e-5` sensitivity check. The three
decoded trajectories were finite and each interface seal returned
`TrajectorySafetyGateV1=ACCEPT` without correction.

## Interpretation

This establishes output sensitivity to changed RGB under a matched state and
language. It does **not** establish visual grounding, target localization,
or task success. The occluded input was an image diagnostic derived from the
native capture, not a second physical camera scene. A future visual-grounding
gate needs task-linked held-out targets and an explicit success criterion.
