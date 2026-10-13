# Real RGB observation-only input qualification

Actual Hopper CPU run at fixed source cefcb3a23fe899aa75dacfea1fcc7c322e523b88
passed on six recorded TRAIN/DEV examples: first, middle and final legal window
of each split. The new official VLAStepData input has `actions=None`, not dummy
future supervision. It accepts only the two original camera RGB images, original
language and actual EEF/gripper state plus frame/TCP/snapshot identity.

Every inference tensor dtype/shape/bytes hash matches the qualified native
dataset/collator view with action/action_mask removed. Frozen stats are unchanged.
The real CPU child used 9.02705 seconds, exited, and its 900-second reservation
was reconciled to actual CPU in the original Hopper ledger. No model was
constructed, CUDA initialized, physics advanced, training started or PBS submitted.

These are recorded-bank observations for a **CPU input-equivalence test**, not new
live physical observations. This result does not satisfy teacher relay2/2, model
canaries, compute-node route or paired confirmation. Real observation/action
endpoints and their runtime qualification remain required. The official c187d957
training/generation recipe remains untouched.

The launch manifest was validated before dispatch at the registered run. Its
inherited output-role text incorrectly said PBS_AND_TRAINING; actual outputs were
registered CPU evidence/operations paths. That imprecision is explicitly retained
in the receipt, not retroactively rewritten. Subsequent launcher preparation
corrects the label. The first deployed CPU check finished within its 300-second
cap; observation verified the exact child identity and terminal absence. A later
source hardening adds in-child alarm/CPU limits; it is not falsely claimed to be
part of that completed run and does not hot-edit its sealed capsule.

Local synthetic protocol tests cover original pixels/state/language and reject
future labels, task truth, camera reorder, CRC/hash mismatch and nonfinite state.
The successful real RGB result is separate from those fixtures.
