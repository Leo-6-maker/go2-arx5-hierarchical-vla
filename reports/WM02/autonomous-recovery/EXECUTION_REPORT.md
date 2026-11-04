# WM02 autonomous recovery · actual execution

2026-11-04 SGT. This branch continues the unmerged PR353 charter `a5619af01a8e508f170ecaec4aa994936098a985`, based on PR352 `76bb8115c3b588b70fd55bdde13f10926c60bce7`. Main and active sources were not switched. Original data, failed projections, checkpoints, STOP and CLOSED leases are preserved.

**Result:** complete PC native-mesh recorded replay and recorded-file subscriber; A0 dependency/startup evidence with bounded owner request; two fresh mechanism models400 main updates +2 continuation probes; **6,752 new actual S transitions**,116 executed method episodes out of136 intended,56 source restore/tail controls passed. **Target integrations0, Actor updates0, complete new tasks0.** M4-B passes the prespecified DEV gates, but no incremental decision benefit over the original rule is established. Target initialization/5s standing remains **OWNER_BLOCKED**.

## 1. A0 environment and causal evidence

`ENV_NATIVE_MATRIX.json`: all original frozen23 and candidate-build20 SHA checks match. Both outer preload chains were actually checked again with dlsym, exit0. Selected chain is `task_actuation_r1 → observer → frozen MuJoCo3.2.7`, plus the unique trace library. The unchanged native candidate is source `7ae02fd40bff0ca6e52255e4a23736f340f07bc5`, binary `7828f9f29e174ac7bf9e82edd897de577a2d365ab22e6dec438f0565b09ff488`.

Current native Python metadata is3.12.3/NumPy2.5.4/MuJoCo3.2.7, with ROS2 Jazzy package manifests and resolved native dependencies. This differs from the neural environment3.12.3/Torch2.9.0+cu128/NumPy1.26.4; environments were not silently unified. No Core target executor/controller was constructed or started. Real OCS2/WBC/receiver restoration and dynamic qualification are still missing.

`T1_T2_0_27_TIMELINE.json` and `FIRST_CAUSAL_DIVERGENCE.md`: original T1/T2 first saved qpos/qvel/ctrl divergence is **step4,8ms**. T2 ctrl repeats over4–23, then original limit/STOP at24 and **three later integrations**. The old sparse trace cannot establish actual PD callback rate, input/history versions, independent position/velocity consumption or inference/PD overlap; those fields remain UNKNOWN.

The conditional PD projection now uses actual XML sensor/actuator joint order and matches the first applied torque within1.02e-9. The wrong earlier body-order projection is preserved but excluded. This is a corrected offline diagnostic, not native controller repair. No unique leg root cause is established.

`TARGET_SCOPE_REQUEST.json` is **REQUEST_ONLY_NOT_GRANTED**: a new original-owner-bound scope, single environment, two50-tick cold starts, then1000 and2500 ticks only after preceding passes, total≤3600; original safety, quarantine and old CLOSED lease retained. It specifies hashes, resources, telemetry, risks, stop and rollback. Private absolute bindings are saved separately. Real cold starts, Policy/PD trace, trace dropped count, STOP qualification and2s/5s holds are all NOT_RUN/NOT_TESTED. A deployment command cannot be bound before actual owner admission; none is fabricated.

## 2. C0/C1 real PC viewer

Actual same-episode MJB SHA `70663bf41a235f8f2baf179447720bd1ca845ce1abf847898ba884a46559fc55`; T1 journal `3f0065eec1e499f5eb5998b965efee9a8d67ba7ed9445bec3fa8a4bd69ce77fa`; T2 journal `5f7eb9b7ffd010d8832413699bfb4aba07a42cad17c294cc89d76c2d3bd97e9c`. Isolated PC MuJoCo3.2.7 loads ABI34/32/20,25 meshes,28 bodies,79 geoms. Actual full Go2/ARX5/table/cube screenshots stay private.

- T1 offscreen120 frames960×640: **102.495FPS**. Native passive UI, capped30:29.78–29.85FPS. Pause, seek and rate change actually executed; mouse orbit/zoom uses the native viewer.
- Recorded-file producer/subscriber:141 received updates, final sequence2700,49 stale display frames after EOF; holds last saved pose. Original monotonic timestamps were unavailable and remain null; emission monotonic time is separate.
- Native integration guard receipts and Python assertions:0 step calls. Original bytes/time/poses retained, no interpolation. An initial transport-check receipt variable was mistyped; that5s run is preserved and a separately identified5s guard check completed, both costs included.
- The convenient standalone playback CLI also actually ran for3s at source `e48c674e7bbc07e46a106880f122163df27c4426`, exit0, native step calls0. See `PC_VIEWER_README.md` for the private-variable command.

Labels are `TARGET_FAILED_REPLAY`, never a new successful trajectory. The bridge is **RECORDED_REPLAY_ONLY**. Live target subscription awaits an independently admitted owner process; no ROS/action/network control publisher was added.

## 3. B0 → M4-A → M4-B

Same data SHA `c12293bc877758254a14d19ec112f147fe19c93ffacf4c8399ae8ebfcb42c808`, FIT24/DEV8,720 training windows,5,736 unique normalizer states, seed20261104,18,905,387 parameters and initial parameter hash `1f933cf8f0349eb198a9ed313adb462c7b0727c159025f65e2ac7bad8305be78`. Each fresh model has51,200 base sequence exposures,256 effective batch/32 microbatch, unchanged AdamW and200+1 endpoint. Extra absolute sequence exposures are logged; no pair supervision is falsely counted.

B0 source `983804b99c4e458134e4ce963a7a8476faa359c2`: FIT24/DEV8 all three phases and H1–16 contact/gripper/terminal diagnostics; local gradient probes are **20 matched FIT examples per phase**, not whole-FIT causal attribution. M3 preclose effect-vs-absolute gradient cosine−.364; base8 event gradients are small relative to absolute/effect objectives. Current grasp labels are nearly linearly predictable from measured geometry/gripper on these narrow anchors, not proof of future-contact Markov completeness. Positive abort labels are absent, so safety/terminal heads remain unqualified. No physics or optimizer updates in B0.

M4-A replaces difference-only extra loss with causal absolute TCP/object, full-state and event grounding. M4-B changes **only the object-pose interaction structure relative to M4-A**, using a causal geometric attachment gate and predicted-TCP SE3 carry plus learned slip residual; uncertain/opening modes keep independent learned dynamics. Same parameters/init/loss/data/exposure; no current/future native event or Teacher phase enters the model. Both hypotheses/specs were committed before execution.

| Existing DEV8 metric | M1 | M3 | M4-A | M4-B |
| --- | ---: | ---: | ---: | ---: |
| Transport H1 object mean-parent RMSE,mm | 1.151 | 1.947 | .241 | **.094** |
| Transport H16 object RMSE,mm | 1.958 | 1.641 | 2.369 | **1.253** |
| Transport H16 TCP RMSE,mm | 6.264 | 13.676 | .502 | .588 |
| Preclose H16 grasp Brier | .07340 | .69215 | .06052 | .06847 |
| Supported-release H16 grasp Brier | .05478 | .78160 | .000111 | .000137 |
| Prespecified combined gates | FAIL | FAIL | **FAIL** | **PASS** |

M4-B predicted toward-away object differences H1/H8/H16 are.228/7.221/7.196mm; real.200/7.952/7.906mm. Maximum per-parent errors.041/.802/.765mm meet.5/1/1.5mm limits; canonical candidate permutation error0 across32 parents. `M4_A/B_STAGE_DIAG.json` report full time curves, relative TCP/object pose and event metrics. FREE and privileged one-step/true-TCP oracle diagnostics are distinct. No old holdout was promoted to a new test.

| Actual run | Source SHA | Checkpoint SHA256 | Updates | Exit |
| --- | --- | --- | --- | --- |
| M4-A train | `72e5c4c5d014fdf162e909f0ae6f2cc0dd28a3cc` | `61a020c2b70ddf9286f1d65715cc9d07c29ccb40ea6a3c7c1371de484110fc5c` | 200+1 | 0 |
| M4-A evaluation | `e04702f4afdbdbf0077d7feeefe4a618220cb511` | same frozen endpoint | 0 | 0 |
| M4-B train/evaluation | `2666b32a8a8221f40f364c9fa408b76d76f617aa` | `9676312c5b810a757c465174ddc4302fc67b1f6896f8ba66dbed5242dee77f7b` | 200+1 /0 | 0 /0 |

First/endpoint reload and199→200 continuation errors are all0. Entire CUDA run reserved peak≤991,952,896 bytes; this WM fits16GB comfortably within the4GiB allocator cap. This does not size GR00T or certify target dynamics. No800 extension or third blind neural variant was started.

## 4. Prospective physical decisions, including failures

Both new populations were frozen before sampling, disjoint by parent/layout and2mm/2° near-duplicate checks, explicit DEV never-FIT split. No TEST outcomes or old once-holdout data were used. OLS uses only original FIT0–3×5 samples; it has no contact head and explicitly falls back to a rule outside transport. All proposals and recommendations were fsynced **before** any online/oracle effect; identical recommendations were actually executed separately, not aliased to one outcome.

**First new8**, source `e48c674e7bbc07e46a106880f122163df27c4426`, prereg SHA `0ca4c000aef66df8ba9cf43609f82d64f409c7e91e477fddbd83dbbcfd65fc31`:5,696 real transitions;96/96 method episodes,120 candidate branches,48 repeat/tail controls. Every original rule score already equals the sampled oracle at all24 anchors. Pure WM matches the rule in preclose/transport and is slightly worse at **3/8 releases**, mean diagnostic Q difference−.016996. Hybrid/rule/OLS fallback do not establish neural increment. These0.8s diagnostic Qs are not formal whole-task rewards.

**Disjoint coast-close8**, source `8c28eea015e626bfe5fb137a02efe4cd03da4a11`, prereg SHA `889ab113036a06de88c7f0034303b2a1e3f00042d803511406786d893b6241cd`:1,056 real transitions. Four positive-direction open-gripper prefixes were refused by original copied-state S1, with no coast live integration; all intended cases remain in the denominator. Static gate recomputation from saved states is ACCEPT in all8, so the short-circuit identifies S1 rejection; its historical specific reason was **not saved**, and is not invented. The415 additional copied-state ticks are charged. Future code now saves the admission details.

The other4 complete all five methods and five candidates: pure WM and original immediate-close rule grasp4/4, oracle headroom over that rule still0; the introduced delayed-close rule and hybrid fallback grasp0/4. The observed end gripper fraction is≈.505 for delay6 versus≈.437 for immediate closure; no claim that delayed closure is inherently better survives. Full native contact/gripper curves are in `COAST_READONLY_DIAG.json`. This OOD mechanism exam refutes that simple delay hypothesis; it does not justify Actor training.

The coast worker exits **1** after all eight scene results and aggregate RESULT were durably written: final stdout serialization encountered `numpy.bool_`. Original failure/lease remain, no physics rerun. Source `e1ad0c1717b85e25ddcb043f7c4a031352da4594` fixes the output and adds regression tests; a separate read-only audit verifies all6,752 real records, actual command64 hashes,2ms consumption, parent split and116 pre-effect method streams. All live unsafe/terminal rows0;56/64 intended source restore/tail controls actually pass,8 unrun with the refused prefixes.

The PC online inference environment is Python3.10.12/Torch2.14.1+cpu/NumPy1.24.4/MuJoCo3.2.7; Core development evaluation used Torch2.9.0. Frozen inference derivative SHA `cba5a6b162c9e162614855dfe9983dbaad10a2f630f5e3c4be6eb07e14a5d674`, copied tensors from the full M4-B checkpoint with no optimizer/parameter updates. Cross-version bitwise output equality is not asserted. Actual private stream manifests are pinned in `S_PROSPECTIVE_ACTUAL.json`.

**Decision:** prediction engineering improved; the present S exam still does not establish useful incremental action selection. Retain the original rule/OLS as strong comparators, keep negative hybrid evidence, stop blind retraining. Actor/PPO0, no target-WM training. A new mechanism hypothesis or target-owner scope is needed for a substantive next dynamic phase.

## 5. Real commands, validation and cost

Private variables below are the operator's resolved original paths. The actual registered systemd workers used `Restart=no`, fixed4 CPUs, finite2400s deadlines, GPU allocator4GiB or PC RAM5GiB, original Journal/owner locks and fresh immutable capsules. These rows describe **completed effects, not commands to rerun CLOSED identities**:

```bash
# Actual GPU unit entry (each worker calls its admitted model CLI with bound spec).
"$CORE_NEURAL_PYTHON" -B "$M4A_RUN/worker.py"  # source72e5..., OS0,200+1
"$CORE_NEURAL_PYTHON" -B "$M4B_RUN/worker.py"  # source2666..., OS0,200+1
# Actual PC source unit entries (4 spawned workers, existing native adapter).
"$PC_MODEL_PYTHON" -B "$NEW8_RUN/worker.py"    # sourcee48c..., OS0,5696 transitions
"$PC_MODEL_PYTHON" -B "$COAST_RUN/worker.py"   # source8c28..., OS1 AFTER saving1056 transitions
```

Original full argv, resolved paths, systemd invocation/terminal receipts and immutable source manifests remain in the private runs. The delivered report gives their full source/data/model identities; the convenient `-m scripts.wm02...` entries also exist, but no unimplemented target CLI is claimed.

Actual read-only audits at source `e1ad0c1717b85e25ddcb043f7c4a031352da4594`, both exit0:

```bash
PYTHONPATH=src:. "$PC_MODEL_PYTHON" -m scripts.wm02.audit_prospective_s \
  --root "$NEW8_RUN/validation-01" --output "$WM02_PRIVATE/audit-new8-01"
PYTHONPATH=src:. "$PC_MODEL_PYTHON" -m scripts.wm02.audit_prospective_s \
  --root "$COAST_RUN/validation-01" --output "$WM02_PRIVATE/audit-coast-01"
```

Validation: neural46 normal +46 `-O`, world-model50 normal +50 `-O`, all exit0, plus the two real GPU runs/reloads, real native S transitions/restore controls and PC mesh GUI checks. Synthetic/mock tests are not target native qualification. Repository CI is separately reported; no CI pass is fabricated.

`RESOURCE_AND_PROVENANCE.json`: GPU worker wall **282.783s (.07855h)** including probes/reload; measured compute/readonly CPU subtotal≈**1,633s (.454 core-hours)**; counted new private/checkout/transport storage≈**3.09GiB** at export. Staging/install/transport/test CPU overhead was not fully individually metered; the loose whole-execution available-CPU envelope is recorded, not counted as measured use. Successful and failed runs are charged. Real S live168,800 and shadow169,215 ticks, conservatively13,520.6 combined50ms equivalents, within the charter20,000 limit. No billing or energy figure is fabricated.

All own worker MainPIDs0, Restart0 and leases CLOSED. Old known727048/3892803/1425203 absent. Final GPU0/GPU1 have **other UID** compute work; none was signalled. Original target lease remains CLOSED and no new bound init-only scope was found. No old queue or target quarantine was reopened.

**Next immediately executable command (read-only evidence verification):**

```bash
python3 -m scripts.wm02.review_autonomous_round --report-dir reports/WM02/autonomous-recovery
```

For geometry review use the private finite playback command in `PC_VIEWER_README.md`. For A1 submit/review `TARGET_SCOPE_REQUEST.json` through the existing original owner/profile; actual admission must bind a new run before any cold start. No repeated exit78, new scheduler or fabricated authorization is used.
