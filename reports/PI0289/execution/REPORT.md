# PI0289 actual startup and first EEF qualification closeout

Authority: release `1b9990c780e652a81cc585ef345af250de9bdfea`, Issue #11 canonical `6092294476`.

**Overall: VALID_QUALITY_FAIL. Safe native WBC startup passed; G1 is not qualified.**
Actual Corelab MuJoCo execution and independent Windows CPU recomputation are complete.
Original PI0288 42-step FAIL, original reports and accounting remain unchanged.
`selected=NONE`, `deployment=NONE`. Teacher and GR00T execution are0.

## Four actual finite runs

| Version / episode | Source-backed change | First result |
|---|---|---|
| V1 / `0fabf5e270194718900f198a1ede01f6` | Delivered MPC initial arm `[0,.9,.9,0,0,0]`; pause until native GetUp + HOLD ready | Valid safety FAIL: base-floor contact at .942 s /471 steps |
| V2 / `dd13faf1951544d1ba9abefd1dc6d0dd` | Original native RLLocomotion entry instead of GetUp crouch; same pose/laws | Safe standing; keyboard WBC route immediately returns HOLD, stopped at original15 s warmup bound |
| V3 / `a7ff3a9a26c34aab9843fa5c4abecef3` | Original arm mode request topic for HOLD/WBC, keyboard only ENTER | S1 PASS: 10.000178504 s WBC hold; reference quality FAIL |
| V4 / `7bdc332eae2843c0aa3edeffe4a52fde` | Same V3 startup plus versioned reference session/expiry adapter | Three actual negative cases reach native HOLD; receiver clock audit BLOCKED |

V3 is reused for first-valid G1 hold, without another quality run. Other five G1
fixtures are NOT_RUN: clock admission is not closed and existing hold already fails
the unchanged qualification. No claim of six completed fixtures or G1 PASS is made.
G2, teacher PILOT, TRAIN64/DEV16 and formal processor qualification are not admitted.

## Initialization and source evidence

Initial arm comes from delivered `task_floating.info` initialState.arm. Root z=.34 m
and original home legs are retained. CPU audit proves18/18 bounded joints, finite root
quaternion, zero initial velocity and exact shared reset. Joint2 lower/upper margins
are .9/2.765 rad; joint3 .9/2.34 rad. Initial front/rear foot-floor distances are
16.497/26.079 mm. No pose, height or seed search was used.

Shared initialization applies to actual native mjData at creation/reset paths.
First-step pointer and qpos match the actual audited initialization. Original safety
checks run from first integration. Native warnings and forbidden contacts stop execution.

Passive forces arm DAMPING with kp/kd/tau zero; joint2/3 model damping/limit margin
are zero. Old42 force channels were not saved: UNOBSERVED. Retained CPU mj_forward
reconstruction has zero integration and no original warmstart, and is not unique causal proof.

V1 exposes another startup problem: original GetUp's one-second crouch from standing
home drives the base down. At .940 s base-floor contact distance was +.000922794 m,
normal constraint force89.8345 N, before .942 s stop. This is contact within model
margin, not a claim of negative penetration. Arm limits remained legal.

V2/V3 use original FSM entry, leg RL, arm HOLD and WBC laws. V2's rapid WBC-to-HOLD
transition is preserved; source ClearInput retains last_keyboard. V3 uses the existing
mode topic. No controller/gain/dynamics change or same-version quality retry was made.

All nine original core pins still match: MPC/node binaries, policy/config, URDF/MJCF,
scene and original simulator. `evidence/SOURCE_POSTCHECK.json` verifies these plus
startup pins and absence of remaining owned control processes.

- Isolated V2 simulator, reused by V3/V4: `ef68498f14a18849448ae19ba982ef412045e647aaf452b04fdf301e1e9a2550`.
- Original parity-bound MJB: `168b1232df2a50a163709c1c1e57d13d39aa3efb92a4c98005e797924142376d`.
- Source build/reset identities: startup config, CPU/build receipts and `LOAD_INIT_CONTRACT*.json`.
- Later observer/force-audit products are independently runtime-pinned; the earlier
  build receipt is not retroactively claimed to contain those later build commands.

Private full source diffs, models, binaries and printed WBC config logs stay private.

## S1 safety and reference quality are separate

V3 records2,022 policies,2,021 arm requests and2,020 base commands; no joint excursion,
native warning or forbidden contact. FSM remains RLLocomotion with arm WBC through
the delivered external MPC ROS route.

| Original hold, last2 s | Requested/actual | Installed/actual | Gate |
|---|---:|---:|---|
| XYZ RMS | .011672954 m | .011672954 m | <=.015: PASS |
| XYZ max | .011712614 m | .011712614 m | <=.030: PASS |
| SO3 RMS | .065199717 rad | .065199717 rad | <=.050: FAIL |
| SO3 max | .065295823 rad | .065295823 rad | <=.100: PASS |

Actual native measured FK carries its MPC time and covers10.000178504 s:2,001 samples.
The unchanged scorer returns REFERENCE_QUALITY_FAIL. Independent normalized-quaternion
dot-product CPU arithmetic reproduces both RMS values within1e-10. Outputs are not corrected.

Separate sampled-physics score ends at derived t=9.998581992 s and retains
INCOMPLETE_10S_TRACE; no endpoint is invented. Its .065199674 rad tail agrees with
native feedback. Feedback clock is mapped from actual MPC time using measured affine
pairs, not described as an exact integration timestamp. Native bind/install receipts,
ROS command receipt times and actual native SetCommand snapshots remain separate.

At late hold, native EE versus actual MuJoCo x5_ee agrees within8.24 micrometres and
2.51e-5 rad, excluding a static TCP/frame offset of the observed magnitude. Saved
endpoint joint3 q=.589266, qd=-.000471, actuator/bias5.44713/5.44435 Nm; received target
q=.625330, qd=.194336. Native feedforward torque remains zero. These support a
physical holding/command tracking hypothesis, not unique causal proof; endpoint
timestamps differ explicitly in `ARM_HOLD_DIAGNOSTIC.json`.

**WBC-owner question:** verify delivered arm preview/velocity feedforward against
actual MuJoCo PD/gravity holding. Is an existing supported holding path missing, or
is a separately approved core change required? No added gravity torque, gain tuning,
preview sweep or MPC modification was applied.

## Clock / expiry

Adapter changes only episode/sequence/monotonic metadata and rejection/expiry to original
HOLD. Original parser/interpolation/geometry are preserved. Actual private callback CPU
check:6 panel windows byte-equal,4 metadata negatives pass, expiry still enforced after
native bound_start clears. Actual WRONG_EPISODE, STALE_PACKET and .1 s EXPIRY all observe HOLD.

Original doctor rejects V4 receiver pairs: .047803157 s max residual exceeds unchanged
.025 s. V3 hold map residual is .018100027 s. Source-clock diagnostic using an estimated
publisher origin reduces V4 residual to .002967734 s and exposes .048503120 s receiver
delay. Actual publisher origin was not saved: the estimate is not substituted as PASS.
Raw receiver pairs and BLOCKED_CLOCK_OR_RESET remain; no clock gate was loosened.

## Gripper choice / local package

User delegated choice. Selected SIM_ONLY baseline: delivered ARX5 parallel fingers
(`gripper_joint`, `joint8` mimic), each0..44 mm stroke, existing TCP `[.1424,0,.0001057]`,
delivered masses/inertias. Leg RL controls12 legs and no gripper. New independent
gripper DOFs cannot change the original18-joint MPC/RL mapping.

URDF effort100 N/speed5 m/s are model declarations, not certified operating limits.
Finger travel is not pad aperture. Hardware revision and pad material/friction remain
unverified in `GRIPPER_SELECTION.json`. Existing CPU scene/camera/schema helpers remain;
no equipment physics or official target collection was run.

At user's request, the215,194,129-byte original private archive is downloaded to PC
`Projects/go2-arx5/PI0289/private-assets/wbc_mujoco_minimal.tar.gz`, outside public worktree.
Source/local full-file SHA256: `6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`.
No public upload of that private package occurred.

## Accounting / preservation / runnable check

Actual episodes4/180; service union53.800073 s. Reservations: GPU240 s, CPU4,800 core-s;
these are not utilization. Observed integration lower bound18,186; final periodic
sampling can miss terminal steps, so exact total is UNOBSERVED. Actual leg RL forwards,
MPC solve attempts and CPU core-hours are UNOBSERVED, not inferred from messages.
Teacher, task RGB, GR00T queries/updates, WBC optimizer and hardware are0.
Before archive: logical bytes142,980,212; unique-inode blocks143,523,840.
Original PI0288 raw accounting is separate. No source/evidence/checkpoint deletion occurred.

Public archive whitelists owned numerical ROS/physics/control/planner/initialization,
references and receipts. Private WBC logs/source/models are excluded. HF PASS requires
fixed-revision whole-file and every-member readback: see `evidence/HF_ARCHIVE_RECEIPT.json`
once closed. Local `READBACK.json` records byte-hash verification of downloaded evidence.

Runnable independent CPU check from this worktree:

```powershell
py -3.12 scripts/pi0289_mujoco/verify_results.py
```

`start.py` remains the bounded single native entry; sealed first-valid FAIL plus clock
BLOCK now prevent further qualification under this unchanged contract. This is not
TASK_EXECUTOR/TEACHER/DATA_READY and not a deployment qualification.
