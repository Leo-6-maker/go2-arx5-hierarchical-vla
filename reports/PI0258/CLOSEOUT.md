# PI0258 executed result: KIN_DATA_BLOCKED

Authority: commit8944d569d9f83baa74f0cb28c1113ba9659e528a, canonical5889346948.

**A: CONTROLLER_EVIDENCE_PARTIAL. B: KIN_DATA_BLOCKED.**
The registered data prerequisites failed. No task reaches its10/12 development
admission threshold, so TRAIN/KIN-VAL collection, C_KIN_MT fitting, Base/G canaries,
6000-update arms and model evaluation are not released and were not executed.
This is a data-admission negative, not a multitask model quality failure.
The protocol's requested trained-model end state was not reached.

## Fixed development results

| Task | Necessary start visibility | Subsequent geometry | Admission consequence |
|---|---:|---:|---|
|M1 pose target|6/12|Not run after task visibility ceiling<10|FAIL; opposite target/axes absent|
|M2 specified-side route|0/12|Not run after task visibility ceiling<10|FAIL; goal absent|
|M3-v2 open270-degree contour|12/12|8/12 meet>=95% active windows and all quartiles|FAIL; four fixed solver trajectories lack coverage|

All36 complete numerical references and six actual common-start camera/state
packets remain. All12 M3 sequences retain361 timestamps, prescribed joints,
desired poses, independent FK, IK residuals, velocity flags and collision data.
No easier family, pose, solver parameter, reference, seed or task was substituted.

M3 failures: family000 command0/1 each245/361 IK-valid samples,61.875% qualified
H40 geometry windows, missing the first motion quartile. Family002 command0/1
each151/361 IK-valid samples,32.5% qualified windows, missing first and last
quartiles. Each has360/361 velocity-valid samples. All361 samples in all12
sequences are collision-free under the registered checker. These are failures
of the fixed deterministic solver/reference program, not proof that no possible
IK solution exists.

The other eight M3 sequences pass all361 IK/velocity/collision samples and all
160 active H40 geometry windows. They have full actual body/wrist renders:
181 image pairs each at0.10s,1448 pairs total. All1280 raw H40 gate decisions
are non-REJECT. These facts do not overcome the fixed12-sequence task denominator.
No TRAIN/VAL codec or normalization PASS is implied.

## Geometry, cameras and integrity

The isolated renderer uses original private mesh assets as prescribed link
actors with complete URDF FK. Its transform and camera readback requires no
physics integration. Direct articulated state placement initially failed the
transform check; those engineering logs remain. The final independent-link
adapter changes only synthetic rendering, never the physical WBC environment.
Placed/changed/restored fixtures prove geometry moves in the pixels and a
restored pose reproduces exact RGB. At the recorded fixture, maximum link
position readback error is1.25e-7m and rotation-matrix error6.81e-8.

Original96x96 camera resolution, attachments and135-degree local rotation are
retained. Actual images and independent projection support the visibility
failure. For family000, the second target projects outside both cameras; the
orange goal is outside the body image and behind the wrist camera. Pixel counts
are actual producer observations; segmentation masks were not separately saved,
so no independent segmentation-reconstruction claim is made.

The collision checker uses42 original URDF collision components and758
nonadjacent pairs, with fixed-body/direct-adjacency exemptions indexed from the
source. It checks robot self/world and the original grey box, including the
mesh component. Pinned task-local python-fcl0.7.0.8/trimesh4.8.3 do not modify
the native WBC runtime. Nominal-source pose hard bounds and collision fixtures
pass. Kinematics does not establish balance, force feasibility or WBC tracking.

Independent CPU replay reconstructs all4332 M3 timestamps, residuals, velocity,
collision flags and1920 active geometry windows from saved joints. It also
reconstructs framed StateV1 and all1280 rendered raw H40 gate decisions.
See independent_cpu_replay.json for numerical differences and scope.

## Preserved deviations and interpretation limits

1. Preflight common_start.npz named its raw world-FK pose field state_v1. Those
   original bytes remain; they were never model input. Full sequence exports
   use the original world-aligned convention with immutable nominal-TCP
   translation anchor and explicit T_world_native. Geometry/pixels/references
   were not rerun or altered to fix this export naming/frame issue.
2. Full-path endpoint-marker visibility fractions in the render summaries are
   descriptive diagnostics. The pre-generation card's necessary visibility
   condition concerned the shared start. Do not promote those later fractions
   into a new acceptance threshold. M3 already fails the frozen geometry
   denominator8/12 independently of this diagnostic.
3. Some scene families differ only in target colors and share M3 geometry.
   Neither family counts nor overlapping windows are claimed as independent
   geometry-generalization evidence. No held-out model evaluation occurred.
4. The first nominal fixture was viewed before the generation card; this exposure
   is disclosed there. Registry, camera, geometry and solver settings were not
   adjusted to improve it. The finite program did not replace valid failures.

## A: controller findings and physical boundary

CONTROL_A0.md and per_joint_margins.csv decompose all1500 old Bfix ticks.
Actual target rot6d inputs match the frozen compatibility shim's two-column
flattening, different from upstream's two-row convention. Position conditioning,
the bounded quaternion rotation-error proxy, actor history and clipped joint
targets reconstruct from original bytes. This narrows a conditioning issue;
original checkpoint training-import identity/support remain UNKNOWN. No unique
cause or corrected physical PASS is claimed.

Gripper soft violations occur throughout, with additional active-arm violations
in three conditions. Small hard-bound penetrations are preserved. Hold SO3 and
angular motion remain failed even in conditions with no active-arm violations.
Original C0 proved a finite position hit, not arbitrary SE3 holding.

Optional A1 was not executed with that conditioning-integrity question open.
No WBC/gate/gain/shim was modified, and no physical attempt was repeated.
Future physical-transfer gaps include training-time rotation convention,
orientation authority/support, dynamic starting/holding ability, native
soft/hard/fixed-drive semantics, timing, camera noninterference and closed-loop
learned-policy execution. None is bypassed by a kinematic video.

## Actual deliverables and accounting

- New physics steps, WBC actor calls, GR00T loads/queries/heads, optimizer updates:0.
- Fixed development attempts:36; full M3 geometry sequences:12; full rendered
  sequences:8. No TRAIN/KIN-VAL sequences or TEST/DEV-F execution.
- Registry:336 planned executable sequences and240 TEST metadata rows;
  untouched conditional allocations are not transferred.
- Full raw references, joints, images, transforms, failure indexes, actual source
  identities, CPU receipts and labelled18-second video are delivered.
- Video: M3_fixed_pair_kinematic_reference.mp4,181 frames at10fps, full decode
  and first/middle/last visual review. Watermark: KINEMATIC REFERENCE / NO WBC
  EXECUTION. It shows the first complete fixed geometry pair, not model behavior.
- No C_KIN_MT, real recovery canary, Base/G6000 model, learned prediction video
  or offline semantic/O64 model table exists. Their required data gate failed.

Stopping condition is PI0258 section4/9's data block; this is not
KIN_MT2/3_OFFLINE_TRAINING_COMPLETE or STOP_FOR_PHYSICAL_TRANSFER_ADMISSION.
The physical-transfer end state remains unreached. No routine GO is needed for
engineering corrections, but changing cameras/scenes/solver/reference semantics
to seek admission would need a new scientific decision.

PI0256 valid failures, PI0257 work and old U/R are preserved. No healthy job was
restarted; no Hopper allocation, broad cleanup, email/monitor or PR merge.
selected_checkpoint=NONE; deployment_selected_checkpoint=NONE.
## Durable public delivery

HF repo: `leoyuer/pi0246-gate-pass-campaign`.
Prefix: `evidence/PI0258/control-audit-kinematic-development-v1-20260929`.
Immutable revision: `f36ea6251b9ca99deb92e3f0b150d28d75e4d297`.
Receipt: `HF_PUBLIC_RECEIPT.json`, status
`PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK`.

All four published files passed anonymous full-content size/SHA256 readback,
including the archive, its manifest, the complete video and its index. All199
archive members (17876587 uncompressed bytes) passed individual content hashes.
Source tree SHA256:
`eccfe72a484cb7e09fd6e29f834d2c49fa48244d34d14057cca8b0146431d426`.
Archive SHA256:
`1c9c732f19d977897e75815653e12614d5d6de58e51072d158a08f33336c7f90`.
Video SHA256:
`ed31af47f41525a84adc141bf69b687c3dd397de097a19f098e1a5952885df7d`.
Private WBC assets/source, model/tokenizer files and credentials are excluded.
The archived report predates this receipt addendum; its original bytes remain
unchanged. The receipt and this addendum are retained in Git.
