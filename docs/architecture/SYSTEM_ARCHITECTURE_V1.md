# System architecture V1

The first scientific baseline is a hierarchical controller, not an
end-to-end joint-space VLA:

```text
vision + language + compact proprioception
                 |
                 v
          GR00T N1.7 (future)
                 |
       40 timestamped EEF points
       XYZ + rot6d + gripper
          /              \
         v                v
  EEF XYZ + SO(3)     GripperCommandV1
         |
         v
  WBCPreviewV1: nine spatial SE(3) preview points
         |
         v
  delivered WBC
   arm policy -> arm action + structured internal 9D body/gait plan
                         -> dog policy -> Go2 legs
         |
         v
  native Go2+ARX5 embodiment
```

Before the WBC preview, Stage15 makes the reference boundary explicit:

```text
TrajectoryShapeV1 (relative SE(3) shape and timing)
  + PlacementProfileV1 (native placement intent)
  + reset state / shoulder / reach anchor / TCP
        -> EpisodeReferenceBindingV1
        -> native world-aligned trajectory
        -> WBCPreviewV1
        -> native dynamic heading-frame arm observation
```

The first qualification isolates the WBC from GR00T and P1M. Stage15's
12-reset self-consistency check reproduces the native anchor and policy
preview. The aligned diagnostic set then uses native-bound continuous
references. The 40-point interface ablation and GR00T runtime integration
remain closed until the aligned native baseline is expanded and qualified.

## Claims kept separate

- A geometrically feasible EEF trajectory is not a unique whole-body motion.
- Internal WBC redundancy resolution can still make an EEF-only external
  interface viable.
- Whole-body requirement is a property of trajectory shape plus embodiment and
  placement; original E2 labels are not native labels.
- The P1M mismatch is a transfer/embodiment issue, not native-WBC evidence.
- A generator's expressivity does not establish policy generalization.
