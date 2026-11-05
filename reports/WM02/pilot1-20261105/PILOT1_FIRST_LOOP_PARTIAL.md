# PILOT1 first learning loop — partial, real execution

2026-11-05, Asia/Singapore. Plan PR #360: `fec841e74d406b3b556c84b733d2de1615581bf7`. Execution PR #362. This is a local four-action XY reach pilot, not a manipulation result.

## Actual source and artifact identities

| Artifact | SHA / SHA256 |
|---|---|
| Initial Actor learner / scope source | `3d37c570f2bb6cc8de772a62cb5b2f791809f52b` |
| Native evaluation source | `b7c75b045b09aff60ec568b2ea4d34cb6b5e6c99` |
| Feedback builder / successor learner source | `b20021094bbf68e02436cf62d8e3a0aa597fde02` |
| Frozen native binary | `29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2` |
| Frozen ARX | `c69ae25164b023cb005e17aaa0c6a45b9d75f756963a1887c782597e7acf55d1` |
| Original replay | `0b889a835a155b5e2697e3a139b49a4e35ae83e096e05a97d4e3b2c50d74a0de` |
| Four new TRAIN windows | `03da405d1e4cc7466486fc26d31804cf38a1cb01372d0e7e271dcf5c24cdf08a` |
| Appended replay | `ea11b27e568443b5f9964f36da7563040a59e4dfb012391c0708a9b747d04734` |
| Frozen task definition | `5e0653617bc735dd39ec75cac1598d4b76488d1c6d72ecc50e30ae6d6d6c7b24` |
| V0 checkpoint, 2,000 main updates | `46b0ef3dbe21c66f87f034e37cfe7ae6b9c9c1f237fad001ae8ee915dfcce8f3` |
| V0 native NumPy Actor | `8a4c4154bfc0dca0a0cb0ece7dc6705022e656dcc1f8e7d9a441f9b06539b458` |
| V1 checkpoint, cumulative 4,000 main updates | `1237ab0bd784c2ddbba5783d7f8792089d9f6075893f567a35be881ede6961c3` |
| V1 native NumPy Actor | `1477e50600407a2f741aa8514ed94555d7d7578db9be9524eb002da15e1f1a82` |
| P17 native scene | `9a8d20f0c76d23a2e0f946c69b0379eaf06c050587c89105e4b20d5791d9bdcf` |

Private scene assets, trajectories, checkpoints, exact host paths, UUIDs and specifications remain at their authorized locations. These public hashes are identities, not public download links.

## Executed programs and exit codes

These are the actual program/argument forms. `$SPEC`, `$FIT`, `$MANIFEST`, `$BASE_REPLAY`, and `$BUILD` resolve to the registered private paths retained in the full argv receipts; they are not newly invented CLIs.

```bash
PYTHONPATH=src:. "$NEURAL_PYTHON" -B -m scripts.wm02.train_pilot1_actor --spec "$SPEC" --output "$FIT"
PYTHONPATH=src:. "$NATIVE_PYTHON" -B -m scripts.wm02.pilot1_native_worker --spec "$SPEC"
PYTHONPATH=src:. "$NATIVE_PYTHON" -B -m scripts.wm02.build_pilot1_feedback --manifest "$MANIFEST" --base "$BASE_REPLAY" --output "$BUILD"
```

- Initial Actor: exit 0; 2,000 main optimizer updates plus one separate restore-probe update. Full model/optimizer/normalizer/RNG/cursor recovery error 0; endpoint reload error 0. Worker lease CLOSED.
- Native qualification: exit 0, 4,103 actual ticks, original safety passed and shutdown clean. Excluded from policy-evaluation and feedback denominators.
- Four native method evaluations: all exit 0 and native safety passed; 16,412 actual ticks, four actual 24-atom learning windows. Local task success is separate from native safety.
- Feedback build: exit 0, four new TRAIN windows / 96 actual transitions. Old eight DEV episodes and arrays preserved. No additional integrations.
- Successor Actor: exit 0, another 2,000 main updates plus one restore-probe update. Resumed complete V0 optimizer state; full continuation and endpoint reload errors both 0. Worker lease CLOSED.
- Fixed P21/P22 DEV driver: exit 1 during the first registration snapshot write, `OSError: [Errno 28] No space left on device`. No DEV native unit, native process, ACTIVE worker lease or integration was created. The original registry entry and INTENT were audited; the admission was sealed as `ABORTED_BEFORE_DISPATCH`, with a CLOSED non-started admission receipt and `RECONCILED_ABORTED` original intent. Do not redispatch this identity.

## Actual control results, all four evaluation attempts

Same predeclared P17 TRAIN parent and goal index 0; independent native restarts, not exact state forks. Frozen success tolerance: 0.1858445664 mm. Non-HOLD action cost: 0.020 mm.

| Method | Executed action | Actual XY error (mm) | Local XY success |
|---|---|---:|---|
| Geometry reference direction | −X 1 mm | 0.623051 | false |
| TRAIN mean response | HOLD | 0.091440 | true |
| Frozen ARX selector | HOLD | 0.050288 | true |
| Trained Actor V0 | HOLD | 0.190087 | false |

The three HOLD realizations differ despite selecting the same action. A single independent restart cannot establish an ARX-versus-Actor control advantage. The Actor miss remains a failure; the threshold was not relaxed. The empirical response table is retained as a strong baseline. No full manipulation success occurred.

V1 uses the four actual TRAIN outcomes: successful chosen actions receive cloning loss; failed chosen actions receive unlikelihood loss, weight 0.25, alongside the original ARX soft distillation objective. This is weak behavior feedback, not a counterfactual value oracle. Two local successes and two failures were retained. No DEV/TEST outcome entered fitting. The task generator and scoring file are unchanged.

Old DEV teacher agreement changed from 82.1289% to 84.5215%; model regret changed from 65.200 µm to 51.112 µm. These are offline teacher-agreement metrics. **V1 native benefit and fixed new DEV performance remain unmeasured.**

## Equivalence and reachable-contact status

Copied ARX coefficients, not a refit: FP64 H1/H8/H16 maximum errors across mixed state groups were 8.88e−16 / 3.64e−14 / 1.03e−13. Model autograd versus numerical goal-X derivative error was 4.14e−9. No physical Jacobian is claimed. FP32 TCP-position errors were 6.59 / 7.24 / 5.56 µm; larger mixed-state maxima belong to control units and must not be reported as TCP meters.

Eight new physical parents P17–P24 were statically prepared, with TRAIN/DEV/TEST split 4/2/2. Only P17 received native qualification. P21/P22 DEV and P23/P24 TEST are not physically evaluated. TEST remains reserved.

Readonly local MuJoCo 3.2.7 reconstruction of 165 actual P17 states used zero integrations. Stable late pad bounds were approximately [0.2410, −0.0675, 0.6868] to [0.2885, 0.0311, 0.7308] m; the initialization path reached pad height about 0.6484 m. Consequently, moving a support surface to the final TCP height without checking the whole initialization path can introduce a collision. This audit is workspace evidence, not contact qualification. No new reachable contact scene is dynamically qualified, no contact-positive data exists, and no contact model/policy is trained.

## Measured work and resource status

- 5 actual native attempts, 20,515 ticks, 320 raw task atoms. 4 policy-evaluation attempts; 96 new feedback learning atoms. Qualification is excluded from feedback.
- Actor: 4,000 main updates + 2 separately counted restore probes; WM/residual updates 0. Full manipulation successes 0.
- Native own-plus-child CPU: 307.721728 s; summed native worker wall: 162.221165 s.
- Two Actor workers: 12.835340 child CPU s, 14.915566 wall s; summed CUDA event time 7.706434 s. GPU peak allocated V0/V1: 136,822,784 / 149,453,824 bytes; reserved: 299,892,736 / 167,772,160 bytes. Context memory is separate.
- Feedback builder wall: 1.969020 s. Static scene preparation: 2.421167 CPU s, 2.509180 wall s. Readonly local workspace audit: 0.362293 CPU s. Development, transport and viewing overhead are not included in these compute totals; no invented monetary total is reported.
- Initial learner, successor learner, qualification and all four evaluation worker leases are CLOSED; finite first-four driver is terminal. DEV driver is failed and the aborted admission is sealed. Parent/PILOT1 scope remains ACTIVE under its original finite deadline.
- The local compute filesystem reported 0 free bytes at failure. A later readonly query saw only 109,465,600 bytes available, an unstable amount; this is not capacity acceptance. NFS free space is not a substitute for qualified node-local storage. No data/checkpoint cleanup, other-user intervention or infrastructure change was performed.

## Next executable action

The private handoff contains the exact existing-route command that runs `audit_dev_registration01.py` read-only. After sufficient actual local storage is available, register a **fresh** DEV identity with the same frozen panel and V0/V1 artifacts. Do not rerun `dev-panel-01` or its reconciled aborted registration. Finish fixed DEV evaluation before claiming the minimum learning loop complete; M/residual and reachable-contact tracks remain open.

The separate browser viewer from PR #361 is running locally at `http://127.0.0.1:8766/`, bound to the last actual P17 Actor evaluation. It correctly says SEALED and never advances physics. It is an explicit finite three-hour viewer session, not a deployed public service or an automatic follow-all-runs dashboard.

`LOCAL_PILOT_TRAINED=true`; `LOCAL_POLICY_EVALUATED=true` (V0 only); `LOCAL_CONTROL_GAIN=false`; `FIXED_DEV_REEVALUATED=false`; `CONTACT_DATA_AVAILABLE=false`; `CONTACT_POLICY_EVALUATED=false`; `H200_JOBS=0`.
