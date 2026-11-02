# Source camera and reset native qualification

PR317 requires independent reset qualification before a new BOOK_RESET_DEV10
panel. PR329 supplies the same source constructors and frozen task-specific
scorers for all 68 shared DEV definitions. Neither CPU fixtures nor a GitHub
merge satisfy the remaining camera/fault requirement.

The finite `source_reset_qualification` command reuses the existing Night10
supervisor for resource waiting. It introduces no scheduler, budget allocator,
training recipe or model load. The original standard entry/profile, canonical
path registry, node-local capsule, onceIO staging and existing per-device GPU
lock remain in use. D2 and its immutable successor queue are not modified.

## Native command

The existing owner supplies a sealed request with one predeclared original
condition per task, CLOSED source asset templates, current source hashes,
node-local runtime paths and a finite lease. The command:

1. Constructs each original T1/T2/BOOK/DOOR scene in two independent processes;
   checks home, controller/IK, actual state, original noise keys for all 80
   queries, and the original two cameras at time zero. It uses the same initial
   camera gate as the actual policy episode. Cube goal visibility and cabinet
   metadata retain their original implementations.
2. Uses a third disposable process per task to inject nonzero initial velocity.
   The original reset witness must reject it and leave the fault unchanged.
3. Starts one new finite qualification-owned user probe, deliberately presents
   CLOSED metadata while its actual PID is alive, and verifies rejection. It
   requests that probe's normal exit through its own file, verifies real unit
   and cgroup closure, and checks that CLOSED permits progress.
4. Checks an UNKNOWN close and an actual separate writable descriptor against
   the same predecessor gate. Both must block progress. After the owned writer
   closes normally, the positive control must pass again.

At most 12 source constructors and eight RGB pairs are permitted; physics
integrations, GR00T loads/queries, optimizer updates and formal conditions are
all zero. RGB and private XML stay on Core. Only this qualification's files and
probe are used for fault injection. No old STOP, dataset or active actor is
changed. Missing closure remains UNKNOWN, with no automatic replay.

## Resource and certificate boundaries

A single finite Night10 task may wait for authorized shared GPU0, using the
original source entry of 2048 MiB free, a 120-second stability observation,
one CPU, a 2 GiB cgroup hard limit and 8 GiB available host memory, protected storage plus 30 GiB and a ten-hour dispatch
window. GPU1's active D2 and its spare memory are not borrowed. Actual rendering
also checks the selected UUID and at least 1 GiB free. GPU0 sharing permission
does not guarantee zero bandwidth or throughput interaction.

Only all successful native processes, real fault rejections and positive close
controls produce `TASK_NATIVE_QUALIFICATION.json` and
`RESET_NATIVE_QUALIFICATION.json`. The producer's native result and CLOSED
lease must be independently read back before those files are bound to a later
run. They qualify the exact camera/reset implementation and asset templates;
they do not qualify task dynamics, the raw neural model, native training,
inference capacity, a full DEV panel or target WBC execution.

## Validation and deployment state

106 affected CPU checks passed, including original book/shared evaluation,
reset, sampling/data/training and Night10 regressions. These are CPU fixtures,
not native qualification. Source compilation and diff checks passed. Full CI
has not been claimed.

At 2026-11-03 01:19 SGT, the corrected immutable capsule was actually
registered, passed the original Core entry preflight and independent 37-file
readback, and was mounted in the existing Night10 user mechanism. PID4174758
and InvocationID 07e2faa06f144ec9bcb935d3d51fa8a8 stayed identical after separate
control connections, with advancing heartbeat. The task remains **PENDING**:
GPU0 has 1668 MiB free, below its original 2048 MiB entry. No qualification
constructor or RGB pair has run and no qualification certificate exists yet.
The ten-hour dispatch window ends at 2026-11-03 11:17 SGT; expired waits are
reported unrun rather than restarted automatically.

The first staged capsule was closed before any launch when review found the
probe needed an explicit node-local working directory. Its source, preflight
and costs are retained. No active job was patched. D2 continued to 976/4096
updates on GPU1, and the original M13 training source and queue are unchanged.

See the [actual finite queue receipt](../../reports/PI0294/design/controlled-finetuning-20261102/SOURCE_CAMERA_RESET_QUEUED_ACTUAL_20261103.json).
This is READY_FOR_RESOURCE_WAIT_ONLY; native camera/reset qualification and
all subsequent neural training/inference and formal DEV remain unproven.


## Resource-only cold correction at 01:31 SGT

Actual GPU0 free memory had risen to 7176 MiB, but the first wait plan still
required 42 GiB available host memory while 40.10 GiB was available. That was
an excessive future neural-load reserve for a zero-model task bounded by a
2 GiB cgroup. Under the existing GPU0 project lock, the owner proved the task
PENDING with no native intent/output/result, stopped only its idle waiting
controller, and verified actual MainPID 0 and an empty child cgroup. It then
started controller03 (PID 4188746) with 8 GiB available-memory admission.
Worker code/manifest/lease, neural training, all original GPU and safety
thresholds and scientific limits remain unchanged. This is a controller
handover before any native attempt, not an experiment retry.

The former plan and both controllers are retained in the existing result
paths. See the [actual cold correction](../../reports/PI0294/design/controlled-finetuning-20261102/SOURCE_CAMERA_RESET_RESOURCE_CORRECTION_ACTUAL_20261103.json).


## Actual automatic execution and retained engineering failure

After capacity admission the server started worker 4190775 automatically. All
12 children returned PASS: four tasks each had two distinct native processes
with identical fresh witnesses, all 80 fixed noise keys and identical original
RGB pairs, plus one dirty-initial-state rejection without repair. Independent
readback streamed and verified all 16 PNGs. There were zero physics
integrations, neural models, optimizer updates or valid model conditions.

The predecessor checks then failed on permission to read a same-UID systemd
user manager descriptor under `/proc`. The manager has a distinct init.scope
and protected descriptor links. It was not modified or silently skipped. The
qualification returned an explicit engineering failure and CLOSED lease; the
finite controller and probe both have actual MainPID 0. No native qualification
certificate was issued, and no attempt was automatically retried. This
requires a safe correction to the process-inspection scope before new DEV.

D2 independently reached 1039/4096, and checkpoint 1024 has actual model/42-state
AdamW, scheduler, RNG and sampler-cursor32768 restoration evidence. Its formal
4096 endpoint remains incomplete. The existing training successor queue is
still running with its original source and identity.
