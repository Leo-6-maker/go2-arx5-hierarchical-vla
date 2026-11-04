# Physical-clock diagnosis, two private videos, and policy readiness

This is an intermediate NIGHT1 delivery. The original owner/deadline remain active. C06 prediction gains remain recorded in [PROGRESS_02](PROGRESS_02.md); they are not upgraded to action-selection success.

## What C07 actually passed and failed

C07 completed all11 original G1/G2 cold/hold/reference stages, the RootSafe1.2s original single/dual checks, and the late-decision P01/P02 single checks plus the late-decision dual check. Every reported PASS includes unchanged native safety and a clean trace/owned shutdown. The next single-scene attempt, `night1-c07-decision-p03-q01`, stopped at native step288, before any task action. Actual `RL_calf_joint` was **-0.8267404441rad** against the unchanged upper limit **-0.83776rad** (**0.011019556rad** penetration); velocity was8.651710rad/s. This is a joint-angle violation. q/dq mixed versions were0, so the pair repair did not solve all initialization defects. No C07 production replay or matched model fit was started.

The failing initialization had160 actual policy-query begins over0.576s physical time. A passing C07 full6.206s recording had416, and another had304. Native steps between PD records had maxima6,23 and9 respectively. Inspection of the actual frozen base configuration established **PD dt0.005s and decimation4**, while physical integration remains0.002s. The policy-history/gait update should therefore correspond to0.020s physical intervals. The native GUI catch-up loop can hold the simulation mutex across multiple integrations; wall-loop policy history can also advance repeatedly without equivalent physical progress. These facts motivate a clock-binding repair; they do not prove this is the sole remaining physical failure mechanism.

Source **`93362d53497fa1aaee58ff03e091153e6c893a6c`** implements C08 as a separately selected build option. Physics waits for the frozen5ms PD bucket and20ms policy bucket to have completed, releasing the original sim mutex through the existing outer loop rather than waiting inside inference. Policy history advances at most once per physical policy bucket. Existing LoopFunc periods, formulas, weights, gains, solver, dt and safety remain unchanged. Tests explicitly cover the5ms/20ms/2ms rates, repeated same-time callbacks, in-flight policy with continuing PD marks, and reset. The full component suite passes129 tests; clock-specific tests pass under Python-O. Native acceptance is still required.

The actual inherited base.yaml was absent from the earlier narrow runtime input list. Its current bytes were inspected and added to the new C08 frozen-input set, SHA-256 **`b42c80faa2bb9e910cba0de65e0de9952d9c27643a9184a2475972bc1b642381`**. The file was not edited. We do not retroactively pretend that the old receipts pinned it.

## Concrete current execution blocker

The C08 dispatch's outer SSH connection timed out (exit255) before reaching the relay. A readonly probe through the original configured SSH alias also timed out. There is **no C08 build/native receipt**. The pending PC qualification frontend was stopped, and no replacement target job was blindly dispatched. No host network/SSH/service settings were changed. The next remote action is readonly reconciliation of the exact C08 root, unit and lease after the existing route recovers, followed by at most one fresh build submission if the identity is absent. C08 is tested source, not an executed native fix.

## G1/G7: same-source audit and genuinely rendered private recordings

Two private videos were produced using the original compiled MuJoCo meshes and their own original qpos/qvel/ctrl/timestamps. The same native-result, scene, journal and command SHA pins are checked. The renderer forbids mj_step/mj_step1/mj_step2 and verifies saved files remain unchanged. It displays actual TCP/BASE/fingers, the installed command goal and original contact/failure information. There is no interpolation of a synthetic success, live controller or hidden-state restoration.

| Recording | Actual source / result | Video SHA-256 | Frames / render wall |
|---|---|---|---|
| `night1-c07-original-q24-single-01` | qualified finite FREE_SPACE, not full task | `6a35e637e3d67f1b0b6d57e9efedaae989ce703d2a9919ca4d132038c9552fa9` | 188 at30FPS /5.454042s |
| `night1-receiver3-panel-a-r4-pair4-slot-1` | FAILED, original joint-safety stop at4,255 | `b0cbe75caee702701534b77621cc618c3152a7948ba8d693af390ae2b7dd5fe9` | 257 at30FPS /7.247104s |

Render source is **`bcc23aa0818e27446d23dd8a2efd266d0849a273`**; MuJoCo3.2.7, private imageio-ffmpeg0.6.0 encoder,960×640. Process CPU was3.221879/4.518358s respectively; encoder-child CPU/GPU/power charges were not measured. The EGL log warned that its selected device does not honor forced software rendering, so CPU-only rendering is not claimed. New physical integrations, controller starts, WM updates and Actor updates are all0. No qualified contact video exists; none was fabricated. The private paths/complete local reproduction configs remain in the execution receipt, not public GitHub.

The subsequent actual recording audit, source **`c10244eddde56a539f3434a099d5c2f1af26c81d`**, found all24 qualified task atoms had real solver-target confirmation and minimum TTL expiry margin0.044s. Yet this zero-offset hold episode had **33.596817mm maximum TCP movement** and17.817252mm BASE movement; object movement/contact/grasp were0. This one transient is much larger than the1mm candidate action envelope and is a concrete limitation for the proposed decision exam. The failed recording's partial task prefix reached99.707062mm TCP movement and had a real0.007559134rad left-rear calf violation; it remains excluded from learning. This audits two bound original recordings, not the entire production replay population.

Implemented zero-integration reproduction commands, with already bound private paths:

```bash
PYTHONPATH="$WM02_REPLAY_EXTRA:$WM02_SOURCE/src" MUJOCO_GL=egl \
  "$WM02_REPLAY_PYTHON" -B -m scripts.wm02.render_recorded_video \
  --config "$WM02_PRIVATE/CONFIG.private.json" --output "$WM02_PRIVATE/video-01" --fps 30 --max-wall 90
PYTHONPATH="$WM02_SOURCE/src" "$WM02_REPLAY_PYTHON" -B -m scripts.wm02.audit_recorded_target \
  --config "$WM02_PRIVATE/CONFIG.private.json" --output "$WM02_PRIVATE/AUDIT.private.json"
```

Paths are expanded privately in the saved command. Both successful render invocations and both recording audits exited0. An earlier failed-video transfer timed out and its dependent render failed before reading a config; both failed attempts are retained. A later stream transfer verified all four original files and produced the failed video under a separate private output.

## G8: NOT_READY_FOR_FORMAL_POLICY

| Required seam | Actual evidence / remaining work |
|---|---|
| Action, clock, BASE hold | Real14D measured EEF+BASE hold and25-tick atoms exist. q/dq coherence repair qualified in C07; robust frozen-rate physical-clock binding is pending C08 native execution. |
| Legal Actor observations |106D WM engineering state is not a completed deployable RGB/proprioception observation binding; privileged object/scene metadata must not silently become student observations. |
| Object task score/reward | Existing `WM02_STANDING_T1_V1` offline wrapper already preserves the GO2_PLACE_V1 object/contact/release predicates while separating the historical BASE-translation task. Its native sensor binding explicitly remains `PENDING_E3`. Do not implement a duplicate score or substitute TCP distance for object completion. |
| Grasp/support/release/withdrawal | No qualified native approach prefix or contact/grasp/release demonstration; the two audited recordings have no object contact. No complete32s standing task is available. |
| Model response and action gradient | Real prediction/reload is established, including production DEV. Matched pre-response action/no-action fits and actual native action-response/gradient validation are not completed. |
| Counterfactual restore | Offline state container/tests exist; complete MPC/WBC/ROS state export/rebinding is not native-qualified. No exact same-state fork or oracle regret claim. |
| Unseen parents and prospective policy | P09–P16 remain reserved for a frozen-model prospective panel. Actor/PPO/GR00T updates0; no formal policy run authorized by this readiness assessment. |

R0 has a concrete primary PARTIAL with nonzero real data and model recovery. G1 is PARTIAL (two original-recording audits and strict replay construction; full population coverage pending). G2 contact remains blocked by its missing qualified approach. G3 additional reserved parents, G4 matched mechanisms and G5 prospective selection remain pending native readiness. Optional G6 strict throughput benchmarking was not run. G7's two truthful private recordings are complete; G8's readiness assessment is complete with the negative decision above. FINAL closeout is not yet complete and the original execution window is not extended by this report.
