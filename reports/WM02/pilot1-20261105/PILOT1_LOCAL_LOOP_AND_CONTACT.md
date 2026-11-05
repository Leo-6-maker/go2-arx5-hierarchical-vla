# PILOT1: actual learning loop, residual comparison and reachable contact

2026-11-05, Asia/Singapore. Execution PR #362, fixed plan
`fec841e74d406b3b556c84b733d2de1615581bf7`.
This supersedes the *current status* of the [first partial report](PILOT1_FIRST_LOOP_PARTIAL.md),
whose ENOSPC failure and earlier measurements remain historical facts.

The minimum local learning loop now completed: train/reload V0 → native evaluation
→ append four new TRAIN windows → continue/reload V1 → fixed new DEV evaluation.
V1 matched the two strong baselines on this tiny DEV panel. Neither a reliable
control advantage nor full manipulation success has been established.

## Fixed new DEV: all 20 evaluation attempts

P21/P22 are previously unused DEV physical parents, two goals each. Both received
separate native qualification. Each method then ran each goal once in an
independently initialized native process. These are not exact restored forks.
No DEV outcome entered training. The frozen tolerance remains 0.1858445664 mm.

| Method | Local XY successes / attempts | Mean XY error, mm |
|---|---:|---:|
| Geometry reference direction | 0/4 | 0.771928 |
| TRAIN mean response | 3/4 | 0.126591 |
| Frozen ARX | 3/4 | 0.157724 |
| Actor V0 | 2/4 | 0.422103 |
| Actor V1 | 3/4 | 0.123765 |

The geometry comparator operates on the installed reference and carries reference
tracking bias; its failure alone does not establish a learning benefit. The
TRAIN-mean comparator remains a strong baseline. V1's 2.8 µm mean advantage over
that baseline is below the observed independent-restart variation. All three
strong methods failed P22 goal128; errors were 0.234303 / 0.252722 / 0.235332 mm
for mean / ARX / V1. This failure is retained. V1's improvement over V0 combines
more updates, new TRAIN observations and weak outcome feedback; it does not
isolate the causal effect of the feedback loss.

[Per-goal actions, errors, denominators and costs](FIXED_DEV_SUMMARY.json).
The DEV panel's 22 native attempts all passed the original native checks and
closed cleanly: 90,266 ticks, 1,369.650175 own-plus-child CPU seconds and
741.592240 summed worker wall seconds. Its 20 evaluation windows contain 480
actual 50ms learning atoms, excluded from TRAIN feedback. TEST P23/P24 remains
unexposed; the rest of the predeclared TRAIN panel remains unfinished.

## Actual residual WM training; ARX remains the predictor

Source `04897e5d665a85db2ddaa6bd8556f5ede7f5d31f` implemented and executed a
zero-output initialized TCP XYZ residual over the copied frozen ARX. Both
ACTION_CONDITIONED and NO_FUTURE_CHOICE arms used the same 24 TRAIN / 8 DEV
windows, architecture, seed, exposure and 1/4/8/16-step curriculum, with
250/250/500/1000 updates. The ablation removes the new future choice while
preserving causal past references; it does not erase all action history.
Normalization is TRAIN only. Neither new TEST nor contact data entered this fit.

| H16 old DEV prediction | XYZ RMSE, mm |
|---|---:|
| Frozen ARX | 0.111689 |
| ACTION_CONDITIONED residual | 0.136949 |
| NO_FUTURE_CHOICE residual | 0.497143 |
| TRAIN mean displacement | 0.104675 |
| HOLD current state | 0.439295 |
| Constant velocity | 1.361462 |

Both arms performed 2,000 actual optimizer updates, plus one separately counted
restore probe each. Full continuation/reload errors were zero; zero initial
residual exactly reproduced ARX. The action-conditioned residual did not improve
ARX, so it is not deployed as a better predictor. DEV action-effect RMSE is
0.128282 versus 0.363829 mm for the choice-blind arm, but the six pairs are
independent restarts with maximum prefix TCP mismatch 1.759795 mm, not causal
counterfactual forks.

The residual predicts only TCP XYZ, not rotations or joint states. Its DEV TCP
versus unchanged ARX-qpos FK mismatch is 0.099062 mm (choice-blind: 0.265346 mm).
This compact head does not satisfy full-body kinematic consistency. Finite
16-step rollouts remained finite without hard clipping; neither asymptotic
stability nor a native physical gradient was established.
[Model, baseline and FK measurements](RESIDUAL_SUMMARY.json).

Training ran under a fresh original-registry LOCAL_PC_PILOT1_CPU model-only grant,
not a fabricated Core GPU profile: exit0, lease CLOSED, 4,000 main WM updates,
2 restore probes, 0 new physics, 0 Actor updates. Training CPU 75.312817 s,
wall 38.773389 s, peak RSS 477,112 KiB. Including worker startup/imports:
79.799242 child CPU s and 43.272336 wall s. An earlier source
`e513ff9e095e8edf44b307b121b700f1c6f36dc8` failed with a forward-keyword mismatch
before any optimizer update (exit1, lease CLOSED, 5.801202 child CPU s). That
failed capsule was retained; the interface was repaired in a new source/run.

## Reachable contact is now measured

Scene preparation source `26a35bf69f57d5ce21e7157cfab4acabb0df375e` compiles a
separate MJCF/MJB with the original robot, keyframes, dynamics, actuators, physics
options and safety properties compared exactly. Only pre-episode cube placement
and static support layout change. There is no mid-episode teleport, object weld,
floating-base fixation, changed friction or altered controller. Static sweeps
use saved states and zero integrations; geometric overlap is not a force label.

| Commissioning scene | Native ticks | Force-positive ticks (≥0.05N) | Positive / negative task atoms | Peak normal force |
|---|---:|---:|---:|---:|
| C01 | 4,103 | 0 | 0 / 64 | 0 N |
| C02 | 4,103 | 820 | 36 / 28 | 0.193420 N |

Both qualification workers used native runtime source
`b7c75b045b09aff60ec568b2ea4d34cb6b5e6c99`, forced HOLD, original safety and
fresh registered identities; both exited0 with CLOSED leases and no STOP,
warnings, outer rejection or dropped trace records. Their XY goal scores are
ignored for this contact commissioning task. C01's actual minimum sampled gap
was 0.639630 mm. C02 moved the initial cube 1.2 mm closer, and passed a new
165-state static sweep before native execution. It is a new curriculum, not a
retroactive success for the old task. C02 first force-positive step was3225,
last4103; early threshold flicker is retained rather than counted as controlled
withdrawal. Per-native-tick forces are aggregated over all25 ticks in each atom,
not just its endpoint. Contact timestamps retain the original t−2ms convention.

Same-process touch/withdraw source
`61e7df1c90730b1515dbc4b866909d32e599c76a` adds HOLD64 → qualified +Y1mm for32
atoms, with the original ramp, 350ms renewals, source-relative400ms TTL,
reference certificates, physical owner and native STOP. It records the actual
gripper globals at each pause, executed reference indices, all25 native force
labels and object/robot observations. A valid exit requires real prior contact,
at least100 terminal noncontact native ticks (0.2s), at least0.2mm actual away-Y
motion and final table support. It does not claim complete MPC/WBC hidden-state
restoration. The short worker exited0 with original native checks passed, lease CLOSED and
4,903 actual ticks. It moved TCP away by0.503441mm but retained contact at the
endpoint (0 terminal negative ticks), so **contact exit failed**. The force
declined from approximately0.180N to0.052N, suggesting an unfinished response.

A separately preregistered V2 source
`4d1c5c18d0181636ca1793658331f3ab04e00b7d` extended only the same +Y1mm hold
from32 to64 atoms, preserving the short negative result and the exit predicate.
It exited0 after5,703 actual ticks, lease CLOSED, original native checks passed,
no STOP or warnings and clean shutdown. It measured **0.985856mm away-Y motion,
668 terminal noncontact ticks /1.336s, and final table support**. Same-process
touch→withdraw is qualified on this new curriculum. It has1,639 force-positive
native ticks and74 positive /54 negative task atoms. The two runs use independent
initializations; this is evidence for the longer-response hypothesis, not an
exact counterfactual duration experiment or a full manipulation success.
[Native results and actual event counts](CONTACT_SUMMARY.json).

Source `b1a0216dd86d6b444abc976c69126d796de974e9` binds a fresh two-action
HOLD / +Y1mm coverage protocol to qualified scene identities and whole-parent
TRAIN/DEV/TEST partitions. The HOLD condition uses the same128-atom endpoint. Its native worker completed
exit0 / lease CLOSED /5,703 ticks / original native PASS. Despite0.513306mm
of natural TCP away-Y drift, it retained contact (only3 terminal negative ticks
/0.006s,2,276 positive ticks,96 positive /32 negative task atoms), and failed
the unchanged exit criterion. The +Y condition passed with0.985856mm motion
and1.336s terminal noncontact. This one independent restart per condition is
a useful real-action contrast, not a statistically reliable or exact-fork gain.
Neither2mm extrapolation nor an arbitrary action is admitted. Coverage collection
and an independent DEV parent precede contact-model evaluation.

The first withdrawal implementation source
`2e62ff767ca6b4fb3aee3243a8b0519f05ca2df3` failed after2,528 actual ticks on a
list/ndarray serialization mismatch; no native STOP occurred and shutdown was
clean (exit1, lease CLOSED, 20.871487 own-plus-child CPU s, 16.094352 wall s).
The repaired source adds an end-to-end synthetic adapter seam test using the
real list representation. A subsequent shortened source-transfer attempt
omitted the pinned root entry `WM02_PILOT1_START.md`: registration failed before
spec/registry/ACTIVE lease/dispatch/physics. Its original INTENT was independently
audited and sealed RECONCILED_ABORTED. A new complete transfer/run identity is
used; neither failed identity is replayed.

## Identities and actual command forms

The [first report](PILOT1_FIRST_LOOP_PARTIAL.md) retains the full V0/V1, task,
frozen binary and original replay hashes. Additional immutable identities:

| Artifact | SHA / SHA256 |
|---|---|
| M1 residual runtime | `04897e5d665a85db2ddaa6bd8556f5ede7f5d31f` |
| M1 input replay | `ea11b27e568443b5f9964f36da7563040a59e4dfb012391c0708a9b747d04734` |
| ACTION_CONDITIONED checkpoint2000 | `338c7f0e4028de738c124f46767ff14f0ecf606b098abc10e3002713414cff2c` |
| NO_FUTURE_CHOICE checkpoint2000 | `f4279f818fa1c757785b7dc12a6ad12bf30f24053d182010d908bac5239d4afd` |
| C01 scene | `ac46f16a8566de361f3c8d16a4a4f79e6898bf200d506420d510e05f073dabc9` |
| C01 actual state journal | `b936eda4c46233728389f209150a2a5e82808da37ad28e4d147dabad09bc6395` |
| C02 scene | `0dc4c7fe401a222593dff8b315ec8dec70c9cfb64ace6c261dec48b804164053` |
| C02 actual state journal | `08b7ee18d282b814b07ce3f0d0e969b5b899d300f86c4d5a32dc25385e30dbf0` |
| Short-withdraw actual state journal | `2e7a6bec6e85cfeb2d2277a4fa10f42fae78dc053001660c6ac7adefcb61fd6e` |
| Long-withdraw actual state journal | `c8c74347ab53296f950dc1458775f9cad94a4730fe2e5652686ccf6557d3779d` |
| Exact frozen robot contract, both new scenes | `5495b40fd499cdce97c78724cce49fa2eec9809b16e7aefc6559b06d6b5f6898` |

These are the actual executed program forms; private original argv/spec receipts
resolve the environment variables. They are implemented commands, not planned
interfaces. Source/data/checkpoints remain private; hashes are not public downloads.

```bash
PYTHONPATH=src:. "$MODEL_PYTHON" -B -m scripts.wm02.train_pilot1_residual --spec "$SPEC" --output "$FIT"
PYTHONPATH=src:. "$NATIVE_PYTHON" -B -m scripts.wm02.prepare_pilot1_contact_scene --spec "$SPEC" --output "$SCENE"
PYTHONPATH=src:. "$NATIVE_PYTHON" -B -m scripts.wm02.pilot1_native_worker --spec "$SPEC"
```

## User-authorized storage and GPU cleanup

The user explicitly authorized project cleanup and requested that new data stay
off the root filesystem. A separate existing local data disk passed actual
flock conflict, atomic rename/fsync, symlink and hash round-trip checks. No
mount, permission, driver, SSH, system service or other-user process changed.
New native bulk outputs, logs, caches and temporary files use this data disk;
the original canonical owner/registry/journal stays at its existing location.
No filesystem type is silently relabeled to bypass the separate GPU profile.

174 large files from24 eligible CLOSED historical runs were copied, verified
and replaced by original-path symlinks: 3,695,470,647 logical bytes and
3,694,878,720 measured root bytes recovered. Historical data bytes remain intact.

For the failed GR00T checkpoint128, dependency auditing found that other valid
models still reference the three parent weight shards. Those12,616,925,552 bytes
were hash-verified, moved to the existing private non-root artifact archive,
and retained through original-path symlinks (UID and0600 modes preserved).
Only the unused failed optimizer, scheduler, RNG and training-arguments files
were deleted: four files /81,480,772 bytes. Original failure/metadata evidence
remains. Independent complete SHA readback subsequently verified all three
symlinked shards. The cleanup worker exited0 and its fresh lease CLOSED.
Its measured root recovery was12,682,788,864 bytes. An earlier copy worker hit
its256MiB memory bound, exit137: no source bytes had been moved/deleted. The
partial copy was audited and removed, its lease closed, then a new worker used
bounded buffered copying with periodic fsync and512MiB. Failed-copy costs remain
separate (2.359694 CPU s); successful cleanup20.468313 CPU s /939.383971 wall s.

The audit inspected25 registered own GR00T training roots with checkpoint
directories. The other failed-status run was the user-stopped BOOK run; its
complete1024/2048 checkpoints remain valid and were preserved. No fresh complete
endpoint after2154 was invented. Other completed checkpoints and current WM
artifacts were retained. This is not blanket deletion of all old models.

Fresh GPU observations found no CUDA compute applications on either device;
the old727048/3892803/1425203 processes were absent. Reported free memory was
31,997 /15,868 MiB for GPU0/1. Display-context occupancy remains separate. No
unneeded GPU reset or other-user termination was performed. Root available
space after cleanup was approximately16.4GB; NFS and local-data-disk capacity
are distinguished. These are observations, not a perpetual GPU reservation.

## Actual totals at the sealed long-withdraw boundary

33 native attempts:32 native PASS,1 preserved recorder failure;137,824 actual
ticks and2,209 complete raw task atoms. Policy evaluation remains24 attempts
(4 TRAIN +20 DEV), yielding96 new TRAIN feedback atoms and480 DEV learning
atoms. Qualification, commissioning and future HOLD controls do not inflate
the policy-success denominator. Native own-plus-child CPU2,119.906534s and
summed worker wall1,145.725710s include the failed recorder. Two registration
failures had0 native steps and are counted separately. All33 native leases
are CLOSED. The original parent/PILOT1 scope remains ACTIVE under its unchanged
finite deadline. Finite successor collection is additional work, not included
in this sealed total. Development, transport, viewing, static audits, model
training and cleanup costs are separate; no monetary cost is invented.

## Validation and remaining work

At source2e62,171 world-model tests passed (12 subtests);11 affected tests passed
under Python-O. The subsequent list-serialization repair passed its5 contact
tests, including the complete96-atom adapter seam;18 affected native/clock/contact
tests passed after the duration variant. The final two-action scope passed its5
contact tests, including rejection of an unqualified2mm command. These checks do not replace
native qualification. The native WBC binary, observer, control law, solver,
gains, leg weights, physics step and original safety remain unchanged.

Actor main updates remain4,000 plus2 restore probes. Residual WM main updates
are4,000 plus2 restore probes. Full manipulation successes0; H200 / GPU physics /
distributed jobs0. The minimum free-space loop has completed, but robust control
gain, reserved TEST, contact-model training and
contact-policy native evaluation require their own actual results. The active
goal and original finite scope continue; a report is not a stopping criterion.

The next stage is a separately compiled/qualified DEV course C03, translated
1mm in object X with the same robot and support. Obtain both legal action
responses and seal a whole-parent-partitioned contact replay for finite model
training/reload. The private handoff records the exact original-route commands
and reconciles each finite worker before registering its successor. No contact model or contact policy is claimed yet.
