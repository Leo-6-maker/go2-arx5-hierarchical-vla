# WM01 implemented entry points

The immutable handoff is `de64c252246e4746315f2a324535b5d48912783c`, PR #342,
which remains unmerged. `plan.json` is the original design snapshot; its
publication-time `NOT_IMPLEMENTED` and zero counts are historical. Actual
execution is recorded in [the dated report](../../reports/WM01/wm01-20261103-w0-w2/FINAL_REPORT.md).

Implemented finite commands:

```bash
python -m scripts.wm01.collect --help
python -m scripts.wm01.fit --help
python -m scripts.wm01.diagnose --help
```

Imports and help do not construct MuJoCo, import Torch, download a model, or
submit a job. The commands require a bound run/output/profile. The existing
host-aware registry, Journal/onceIO, CPU owner locks, native S1, and PBS remain
the execution mechanisms. This adds no scheduler or budget platform.

The source adapter imports `scripts.pi0294.scene.make_engine`, the original
PI0291 finite-torque driver, `scripts.pi0294.teacher.make_teacher`, and
`scripts.pi0294.score.score`. Native 50ms intervals retain all actually consumed
2ms commands, installed references and achieved state separately. Eight
contiguous intervals form one macro. H40 preview tails are not executed labels.
Complete restore includes MuJoCo integration/IK state, warmstart, controls,
targets/cache, RNG, native warnings and monitoring/Teacher history. Extra
`mj_forward` calls alter warmstart; the wrapper preserves original snapshot
cadence rather than increasing the original S1 tolerance.

WM inputs are 110 privileged structured state components and 56 task-space
command parameters. Three independently initialized/bootstrap members each
have three 512-wide SiLU layers: 675,451 parameters per member, 2,026,353 total.
Rotations use SO(3) updates and geometric loss. Four-step loss is autoregressive.
The typed replay normalizer uses TRAIN only; no TEST replay is accepted.
Pre-execution safety rejections supply an observed termination label only,
weighted by observed prevalence. They supply neither expert commands nor
unknown future physical/contact targets. BC/PPO is not implemented or started.

Example arguments actually used by the private Journal helpers, with paths
expressed using their existing run bindings:

```bash
python -m scripts.wm01.collect --spec "$WM01_NATIVE/RESOLVED_SPEC.json" \
  --run-id canary-01 --profile local-PC-native \
  --output "$WM01_NATIVE/canary-01" --attempts 4 --restore-check
python -m scripts.wm01.collect --spec "$WM01_NATIVE/RESOLVED_SPEC.json" \
  --run-id train64-01 --profile local-PC-native \
  --output "$WM01_NATIVE/train64-01" --attempts 64
python -m scripts.wm01.collect --spec "$WM01_CALIB/RESOLVED_SPEC.json" \
  --run-id calib16-01 --profile local-PC-native \
  --output "$WM01_CALIB/calib16-01" --attempts 16 --split CALIB
```

W1 uses one real PBS GPU, effective batch256, FP32, AdamW3e-4, WD1e-5,
gradient clip1, a200-update smoke, and exact full-state resume to5000. Checkpoint
restore also reproduces the next actual update/sample indices; its additional
updates are explicitly counted. The native TRAIN replay, CALIB replay and
full restore originals stay private; only the minimal numeric TRAIN view and
model package use the existing private HF artifact route at pinned revisions.

W2 requires that actual5000 endpoint and exactly64 predeclared CALIB states:
16 scenes at macro indices2/6/14/24, fixed from TRAIN timing before optimization
and before CALIB capture. Five arms execute at most4 macro steps each. After
the first candidate, all use zero task-frame increments at the current actual
TCP and hold the last gripper command. Feasible gradient/negative/random
changes have the same actual normalized norm, with one random draw and no
rejection resampling. All branches and safety preflights count toward cost.
The interval bootstrap clusters the four states belonging to each scene.

W3/W4 policy training and final TEST evaluation are subsequent work, conditional
on the real W2 result. Their CLIs are not supplied as empty implementations.
This state experiment does not qualify a visual policy, Go2 WBC transfer,
Core GPU inference, or a real robot.
