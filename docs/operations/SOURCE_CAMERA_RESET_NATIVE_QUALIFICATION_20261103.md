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
one CPU, bounded memory, protected storage plus 30 GiB and a ten-hour dispatch
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

At this publication the new native command is implemented but **not yet
mounted or GPU-qualified**. Deployment and actual outcome must be reported
from the real Core unit and its separate connection readback.
