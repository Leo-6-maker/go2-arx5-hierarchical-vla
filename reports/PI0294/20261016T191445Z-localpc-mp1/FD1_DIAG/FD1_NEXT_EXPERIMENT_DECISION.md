# FD1 Single Next Experiment Decision

Status: **NOT_AUTHORIZED**. No remaining FD1/N1 allowance is a future GO.

Decision: **TRAIN_ONLY_DYNAMIC_EXPERT_CLEARANCE_PILOT**, before any new training.

Hypothesis: the ordinary T2 expert's static4mm pad-table design margin may be dynamically fragile and make imitation sensitive to downward reference errors. Current evidence shows student pad/table vetoes, not expert-caused learning failure. Test that gap directly rather than assuming it.

Proposed controlled pilot, requiring a separate fixed protocol/card and actual site binding:

- Four fresh TRAIN-only scene groups, two pushing +Y and two pushing -Y with fixed appearance, excluded from all opened CONFIRM40 states/geometries/appearance siblings.
- Original expert reference4mm versus exactly one preregistered larger nominal clearance (proposed12mm),8 total Teacher episodes. Do not alter robot/plant/gripper control, gains, friction, S1, score, observation or object task.
- Fix all geometry/appearance/initialization within each pair; reference clearance is the only changed variable. All attempts stay in the denominator; no automatic replacements or seed search.
- Measure dynamic signed geom distance, pad vertical clearance, actual contact/no-lift, force/pose/reference errors, original task success and terminal41 predicates. Larger clearance may miss the object and fail; it is not presumed better.
- Suggested NEW ceilings for review only: Core8core-hours/local2core-hours,2cores/site,6h wall,8 new Teacher episodes,300000 total physics ticks including S1 shadows,700 starts including clones,8GiB scratch/2GiB transfer. GPU/PBS/neural generation/training updates0.
- If original references are dynamically robust, the expert-fragility hypothesis is weakened; do not automatically train a modified dataset. If larger margin is safe but loses push contact or task success, it is not a valid demonstration generator.

This pilot would qualify a reference/data question, not prove training benefit. Any later ordinary-BC controlled comparison needs its own new GO and fresh evaluation population. T2 Recovery Oracle remains unqualified0/2; no correction labels or privileged Teacher phase/object pose is added to policy inputs.

Current action: stop after FD1 publication/readback. Selected/deploymentNONE, TF-R3 STOP preserved.
