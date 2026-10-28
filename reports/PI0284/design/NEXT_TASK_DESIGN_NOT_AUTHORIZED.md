# Read-only successor design

Status: NOT_AUTHORIZED; launch_allowed=false. This is not a new experiment or a launch request. PI0284's fixed StackCube, sampling, and caps are unchanged. selected/deployment remain NONE.

## Two object task families

1. Object relocation: grasp a specified object, transport it to a specified support region, release, and maintain stable placement. StackCube is the current source case; a future non-stacking placement case tests whether the learned grasp/transport/release representation transfers.
2. Articulated-object manipulation: grasp a handle, change the object's configuration along its physical constraint, release, and preserve the requested configuration. This tests sustained contact and completion beyond free-space EEF tracking.

These are proposed families, not collected data or qualified simulator tasks. A later protocol must freeze scene/reset partitions, object identities, actual observations, instruction variations, native safety termination, task-specific success predicates, and finite collection/train/evaluation budgets before execution. No old TEST/DEV-F assets enter them.

## Common high-level interface

Preserve observed two-view RGB, measured task-frame EEF pose and gripper, language, and a future EEF/gripper chunk with declared TCP/frame/time. Keep source-specific EEF opening conversion explicit. Joint action is not the learning target. Future target integration remains reference-only into the frozen WBC; successful source physics does not establish WBC transfer.

Use task-balanced, episode-balanced, event-aware sampling with denominators for all attempted episodes. Retain failed demonstrations and safe partial windows under explicit label/observation validity rules; do not filter task training data by target-WBC success. Partition geometry and episodes before cutting windows; statistics use TRAIN only. Event labels describe measured approach/contact/grasp/transport/release/hold, not model-visible phase tokens or simulator truth.

## Matched comparison and retention

A future authorization should specify one matched intervention, a fixed pretrained donor, paired samples/noise, full recovery checkpoints, and final-only model choice. Fix task-specific persistent object success before scores; EEF error, first success, grasp, installation, tracking and task completion remain separate metrics. Source and target failures must not be combined into a single unexplained success score.

Retain the original DROID processor/history/noise probes and saved pristine outputs as a descriptive regression panel. Also retain fixed source task panels. Neither panel proves broad multitask retention. Any future optimizer or physics call needs a separate published authority; PI0284 cannot supply unused budget to it.
