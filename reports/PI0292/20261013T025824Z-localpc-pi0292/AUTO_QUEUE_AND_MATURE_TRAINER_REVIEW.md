# Auto Queue and Mature Trainer Review

This is an actual read-only launcher review and CPU admission repair preparation,
not another PBS submission, model smoke or adoption of a new scientific budget.

## Project and Queue Are Different

Hopper's native `hpc --help`, `hpc project`, `hpc hopper`, and
`qstat -Qf -F json auto` were read. No `projects` or `project` executable was found
in the verified login shell. The installed project-list command is `hpc project`.
The real `auto` queue is enabled and started, has queue_type Route, and routes to
small/medium/large/special. It is not a billing project and does not select a new
scientific project automatically.

The intended future script declarations are:

```bash
#PBS -P CFP04-CF-007
#PBS -q auto
#PBS -l select=1:ngpus=1:ncpus=12:mem=225gb:gpu_model=H200
```

This snippet is NOT_SUBMITTED; it omits the stage-specific authorized walltime,
manifest and exact frozen launcher, so it is not a runnable replacement job.
Both failed PI0292 jobs already recorded PBS_O_QUEUE=auto and routed to small.
Changing the queue cannot fix their pre-model amgr errors.

## Reuse the Trainer, Not an Old Experiment

The actual migrated official checkout is still pinned to
51d4c89f72fda44cbf77285c6a8114b52676b8a1. Its launch_finetune.py and trainer.py
exist; the native launcher separately assigns global_batch_size and gradient
accumulation from FinetuneConfig. PI0292 trainer_adapter.py already subclasses
this official Gr00tTrainer, calls its native loss and train methods, and retains
the qualified dataset/cursor/resume instrumentation. There is no need to replace
the model engine because the admission layer failed before reaching it.

Mature historical PBS and PI0286 training scripts were inspected, not executed.
They demonstrate project selection before submission and scheduler/allocation
checks on compute, without a mandatory same-path amgr executable on compute.
They cannot simply be relaunched: PI0286 has microbatch4/accumulation2/effective8,
seed28203, different statistics, training scopes/lineages and zero weight decay;
PI0292 fixes microbatch2/accumulation16/effective32, seed29200, new TRAIN-only bank
statistics, one pristine initialization and weight decay1e-5. Historical scripts
also reference retired September environment/model paths. Reuse their PBS
mechanism and the pinned official Trainer, not their old paths, authority or
scientific recipe.

## Correct the Admission Assumption

The previous custom guard incorrectly required login's amgr executable to exist
with identical bytes on compute. The native cluster submission example places
project selection on login and submits a named project to PBS. The demonstrated
expired-project hook rejection also shows that scheduler project admission is not
provided by our own script. This review does not bypass that hook.

A new explicit mode FRESH_LOGIN_RECEIPT_NATIVE_PBS_V1 is prepared in
allocation_admission.py. It reads an immutable hash-bound attestation generated
by native amgr ls/checkbalance on Hopper login, bound to run/request/canonical
submission nonce. Existing validation still requires a positive balance,
membership, active project/dates, observation age0..1800s, exact own running PBS
job/project/resources, current quota, original authority/clock and qualified
inputs. The unchanged sealed wrapper then independently checks allocated node,
H200/visibility, operation seal and the exact job binding before any model.
An overaged queue receipt is rejected, not relabeled fresh. No passwords, tokens,
login executable copies, authentication proxy or new network service are needed.

The explicit mode is opt-in for a newly sealed request after review. Old failed
capsules have not been modified, no old request was rerun, and unknown modes are
rejected. It does not authorize a fourth smoke, main training, or a new goal.

## Actual Evidence and Limits

108 CPU tests/150 subtests pass, including changed hashes, wrong run/request or
nonce, stale/future observations, and Q/F/X jobs. On Hopper login the reader
accepted a real fresh native CFP04-CF-007 attestation and rejected the actual
terminal job664308.hopper-m-02. This is real login CPU evidence, NOT actual
compute-node model qualification. Model loads/updates/generation remain0, PBS
calls3/4, allocated GPU17s; all allocations are terminal and the healthy peer is
untouched. Hopper CPU upper is2910s after accounting for these inspections.

The one remaining old submission is not spent here. The guard mode still needs
new request/manifest integration and actual compute acceptance under a reviewed
submission plan. The existing old main/eval composition has not been updated to
the revised four-call failure chain. Successful S3 is still required before main.
No deadline, recipe, data/statistics or robot safety gate was reset.

Concurrent upstream release67034abb9d7fb046b8d38e1cdcca65c8121c40f9 contains a
new PI0293 design and an adoption delta consistent with this login-vs-compute
repair. Those files are preserved during merge; this review does not adopt its
new goal, clock, collection or GPU budget. Such adoption must be explicit and
must not erase PI0292's failed attempts or unfinished scientific result.
