# PI0294 R1 Actual Safety Return

Status: `BLOCKED_REAL_NATIVE_FINGER_JOINT_LIMIT_NO_P1_GATE`.
The full scientific objective is incomplete, not a working-policy result.

The user confirmed continuation under the existing GO and reuse of mature
training scripts with queue `auto`. The same official pinned GR00T Trainer
chain that actually produced M512 is preserved; all18 numerical training,
physics, teacher, decoder and scorer paths are byte-identical to the parent.
Source021d813f changes only inference seed compatibility and its CPU tests.
129PI0294 tests/93subtests and758repository tests/53subtests passed;14 explicit
skips. GitHub CPU CI38511924296 is SUCCESS. Three changed Git blobs were
read back exactly, then both idle nodes adopted new immutable capsules.

First PBS665153 ended with the fixed63-bitseed failing the legacy NumPy32
API before model forward. Its321allocatedGPU seconds and1query reservation
remain charged. The repaired explicit pre-forward recovery used a new output
namespace and the original fixed draw plan, retaining the exact Torch seed.
PBS665199 then completed48real H40 diagnostic arrays; expert decoding and
both expert/model SafetyGate checks passed. This is not closed-loop success.

The same allocation then ran a resident on Hopper and a serial physical
worker on CoreLab. In the first original TRAIN F0/000/T1 case, six real model
actions were generated/sent once, five complete prefix ACKs returned, then
the sixth prefix stopped after81integrations:1081total,44actual RGB pairs.
The native executor raised `JOINT_LIMIT`; remaining three cases were NOT_RUN.

Saved high-rate CPU audit verified the right_slide at0.045004271727453214m,
above the frozen effective upper0.045m, at tick1081/time2.162s. All six arm
joints and left_slide remained within their frozen effective limits. The
last20Hz sample is2.15s, so its sampled score says safety_valid=true; the
500Hz terminal flag and native abort take precedence. No safety threshold,
physics parameter or WBC code was changed. Task success is0/1attempted,
with3/4planned cases NOT_RUN; T2 has no model behavior result.

An actual44-frame/20fps/2.2s saved-RGB video was encoded and fully decoded
using the mature saved-video encoder, with only the PI display label adapted.
It is not padded into a32s success video. Raw evidence and video stay private;
a small exact-hash video copy is available to the human on local-PC.

CoreLab stopped natively. The exact own local route was interrupted after
verification; resident received an owner STOP and the PBS parent a finish
directive. PBS665199 reached native F/exit0, which means orderly software
shutdown, NOT scientific PASS. No qdel or unrelated-job cancellation.
The controller's separate local-terminal handshake timeout is preserved;
its bounded5000CPU-second reservation was conservatively charged after
process-terminal proof. All model/physical/CPU reservations are now zero.

Final new totals:2/3PBS,890H200allocated seconds,511.035013Hopper CPU seconds;
CoreLab charged5229.491659548CPU seconds (includes5000conservative upper,
not a claim of5000measured work),1physical start/1081steps/44RGB pairs.
55query reservations=1failed pre-forward+48offline+6closed-loop generated.
No new formal/engineering updates; global512/parentfullstate retained.
No new HF upload, collection, deletion, service/tunnel, WBC/Go2/hardware.

The second PBS script requested128GiB but native allocation reported225GiB;
this distinction is retained rather than rewriting the intent as allocation.
Both submissions requestedauto and native routing chose small. The exact
memory normalization cause is NOT_VERIFIED. Personal quota had
175628976128bytes headroom at final native readback. Current new-owned
files were8,827,697bytes Hopper/12,879,756bytes CoreLab, not shared-df claims.

P1's required zero-real-safety-error/four-development-attempt gate has FAILED.
M2048/M4096 full-state continuation, DEV, selection, independent confirmation
and regression are NOT_RUN. No bypass, reset or third submission. Original
T0/cutoff/deadline and caps remain fixed. Selected/deployment NONE.

Next priority for scientific review: explain the saved six gripper command
profiles versus achieved slider dynamics and the high-rate stop, without
replaying physics or loosening thresholds. A future task-space/gripper
contract or executor change requires an explicit versioned decision; it must
not be hidden in a training-speed fix. The existing mature Trainer remains
the continuation baseline, but this GO's failed P1 gate cannot be passed by
calling a safe shutdown or offline diagnostic a policy success.
