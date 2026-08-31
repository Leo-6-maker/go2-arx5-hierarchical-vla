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

The first qualification isolates the WBC from GR00T and P1M. It injects the
E1/E2 continuous benchmark references into the delivered native IsaacGym
runtime. Only after that baseline is coherent will the 40-point interface
ablation and GR00T runtime integration be considered.

## Claims kept separate

- A geometrically feasible EEF trajectory is not a unique whole-body motion.
- Internal WBC redundancy resolution can still make an EEF-only external
  interface viable.
- The P1M mismatch is a transfer/embodiment issue, not native-WBC evidence.
- A generator's expressivity does not establish policy generalization.
