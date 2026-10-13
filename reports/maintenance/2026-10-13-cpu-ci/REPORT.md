# CPU CI Repair

Status: LOCAL_VALIDATED_REMOTE_RUN_PENDING. No scientific execution or server deployment.

## Observed Failures

- Active branch run 38350580090 at 16d38237c660c7ae8b38c27fcfada274ca5670fd: seven collection errors. CPU Torch was absent, legacy sibling scripts were not on the test import path, and task-data tests eagerly imported MuJoCo.
- Main run 38350694628 at 84acbc517ff790bc0b24ca1370e8a2932a2ee628: stale README assertions, T1 knot-count regression, missing history in a depth-one checkout, a host-specific scratch fixture, incomplete fail-closed wrapper inputs, and unavailable historical statistics.
- Resolving collection exposed further active-branch fixture defects: missing Pillow, a download mock that ended before the call, a process-wide import guard contaminated by other tensor tests, incomplete recovery identity fields, and an incorrect expected difference of zero for 2000/1000 - 1000/1000.

## Repairs and Boundaries

- CI installs the CPU-only Torch wheel, Pillow and safetensors, fetches history for exact historical byte checks, and has a 15-minute cap. No CUDA, model weights, simulator, SSH or scheduler workload is installed/launched. Synthetic tensor/autograd unit fixtures are not GR00T training.
- CPU test imports include the legacy scripts directory. MuJoCo Engine imports are lazy and three engine checks require explicit `--run-mujoco`; GitHub Actions rejects that option before collection. All non-engine task-data checks still run.
- T1 constant-speed timing uses its established two endpoints. Extra knots still fail; T3/T5/T9 and Stage38 count/spacing checks remain intact. No running source capsule consumes this code change.
- Portable fixtures retain the production scratch-path restriction and exact protocol rejection. The HF mock remains active through the tested call. Pre-import admission is tested in a fresh process, not by disabling its guard.
- README tests check the current workspace and separate science/maintenance authorities. Historical audit assertions remain. Existing tracked binary evidence is limited to exact path/blob identities at the reviewed pre-fix SHA; new or altered binaries still fail. Nothing is deleted.
- The sealed Stage45C adapter and later published mapping/binding are incompatible. Neither the sealed source nor historical manifests are rewritten. Compatible temporary unit fixtures cover the original adapter contract; negative tests explicitly verify rejection of the published mismatches. CI success does not make that historical runtime executable.
- Missing historical private statistics, decode evidence and the uncommitted PI0147 predecessor archive are explicit skips, not passes. Missing predecessor admission also has a mandatory fail-closed regression. Existing private WBC-source and Hopper-only evidence tests remain explicitly skipped on this host.
- PI0292's 22 zero-model/zero-physics CPU contracts are now included as a separate CI step when present on the branch.

## Local Verification

Isolated Python 3.10 environment; pytest 9.1.1, NumPy 2.2.6, Torch 2.14.1+cpu. GitHub independently verifies Python 3.11.

- Full active suite: 748 passed, 14 skipped, 45 subtests passed; one non-failing synthetic-tensor scalar conversion warning.
- PI0292: 22 passed, 8 subtests passed.
- `git diff --check`: PASS.
- New GPU/PBS/GR00T/WBC/physics execution, data/checkpoint deletion and active server source changes: zero.

The 14 skips are five private WBC-source checks, three explicit MuJoCo engine checks, two private Stage40 data checks, two full admission checks requiring the absent PI0147 archive, and two existing Hopper-only historical evidence/configuration checks. These are not claimed as validated by CI.

Remote Actions receipts and final publication identities will be reported in Issue #47 after actual completion. Main receives only the shared CI/math/fixture repairs, not the active scientific branch history.
