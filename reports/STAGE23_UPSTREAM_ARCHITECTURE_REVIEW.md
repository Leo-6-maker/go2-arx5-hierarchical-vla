# Stage23 Upstream Architecture Review

Date: 2026-09-03 (Asia/Singapore)

## Scope and execution boundary

This is a source/documentation-only review of the two immutable upstream
revisions. Source text was read from pinned Git objects, not imported or
vendored. No model was constructed and no dataset, validation `e`, holdout
`f`, checkpoint, PBS/Hopper job, WBC runtime, ROS/ZMQ path, or hardware path
was executed.

Pinned revisions:

- Isaac-GR00T: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`
- GR00T-WholeBodyControl: `a0732b642c0333077e127a2f56ab0014c196bca4`

Current project evidence anchor: `de454440b2458f0f28789f45b0d34299553241a9` on
`stage23/action-pipeline-learnability`. Pre-existing working tree at generation: `clean`; the three expected audit outputs may be pending.

## Executive findings

1. The N1.7 path is a dense flow-matching action head: processor
   normalization/padding → category-conditioned state/action projectors →
   Gaussian noisy H×132 trajectory → DiT velocity → masked MSE flow loss →
   four-step Euler sampling → processor unapply/decode. The action head does
   not receive an explicit timestamp tensor; temporal semantics enter through
   modality `delta_indices` and the dataset/configuration contract.
2. `NEW_EMBODIMENT` maps to category/projector ID 10, a finetune-only custom
   slot. The audited source initializes category-specific matrices with an
   independent small random initialization and contains no special
   new-embodiment copy/transfer initializer. Shared backbone/DiT weights can
   be loaded from a base checkpoint, but the custom category-specific path
   still requires embodiment data and learning. This is a source-level
   conclusion; exact checkpoint tensor values were not inspected.
3. Stage23 O1@200 remains a real bounded negative for its exact recipe and
   the free-running temporal-profile failure remains strongly observed. The
   earlier phrase “single-sample memorization failure” is too strong as a
   general claim: O1 exposed one example for 200 optimizer steps, while the
   official custom example uses batch 32 and 2000 steps, a 320× larger nominal
   example-presentation count, a shorter H16 target, and different defaults.
4. The observed oscillatory profile is therefore not yet attributable to the
   four-step sampler alone. State dropout, augmentation, VLLN/projector
   settings, horizon/padding, batch/LR schedule, custom-slot learning, data
   scale, and free-running flow behavior must remain in the attribution set.
5. Modern SONIC is a materially different ABI: GR00T predicts a 64D learned
   motion token plus two 7D hand streams; a SONIC decoder/controller consumes
   tokens at 50 Hz. Its VLA runtime is asynchronous and latency-indexed, and
   its controller—not the VLA—owns low-level smooth whole-body realization.
6. The old decoupled path likewise separates sparse/time-stamped upper-body
   and navigation intent from continuous output: `target_time` waypoints are
   scheduled through an explicit interpolation policy with `max_change_rate`,
   while a lower-body Gear WBC/RL policy generates body actions.

## A. Exact N1.7 action-head trace

The pinned source supports this contract:

```text
physical action streams
  -> StateActionProcessor: relative transform (when enabled) and normalization
  -> pad to max_action_horizon=40 and max_action_dim=132
  -> action_mask: zero invalid temporal slots and inactive width
  -> category-specific state/action encoders selected by embodiment_id
  -> x_t=(1-t) noise + t action, v_target=action-noise
  -> action-conditioned DiT / AlternateVLDiT
  -> category-specific velocity decoder
  -> masked velocity MSE
  -> inference Gaussian H40x132 + 4 Euler velocity steps
  -> unnormalize / relative-to-absolute / slice configured modalities
  -> physical action dictionaries
```

The padding is not absent from the forward graph: padded H×D values are
constructed and flow through the action head, while the training mask removes
their direct loss contribution. Therefore 122 inactive width dimensions and
any unused temporal slots are masked targets, not proof that those channels
are computationally absent. This is a plausible nuisance-channel question,
not a demonstrated bug.

For relative EEF, the processor uses the last observed state as one fixed
reference for the chunk and composes `T_reference^-1 @ T_target`; it is not a
frame-to-frame delta chain. Gripper/action streams can remain absolute per
modality config. The upstream model/action head has no explicit `0.05 s`
input; Stage22's 0.05 s was a project-owned TEST_ONLY timing contract.

### Pinned source evidence

| pinned source | blob SHA-1 | content SHA-256 | lines / markers |
|---|---|---|---|
| `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | `346b597a4b9a115a9a5b1053621f47f07833da09` | `6bd66ef768f1484ac5a7889c40358a599f1862b1d554d47d357e113f2f96501c` | 627 / action_head_class=38; action_head_components=66,72,77,130,131,132,159,160,161,217,239,272,321,405,430; state_dropout=99,220,223; flow_noise=230; flow_time=107,170,231; noisy_trajectory=234; diT_call=254,264,417,425; masked_flow_loss=194,276,277,278,283; inference_noise=349; rtc_options=361,362,363,366,367,368,373,376,379,381,384,392 |
| `gr00t/configs/model/gr00t_n1d7.py` | `6026b1f1499d952dbb6d7874bee1a8df8fa4ab25` | `379ff708496a73b25b12f4656850284a16fde0e205828e2a0c0be0af61463744` | 180 / cosmos_backbone=40; max_dimensions=74,75,76; four_step_sampler=106; model_tuning_defaults=113,114,115; raw_state_dropout_default=118; relative_default=71 |
| `gr00t/model/modules/embodiment_conditioned_mlp.py` | `504785d57cc33a87613dd775cc415cc88574c2ee` | `d8c60775642016262e3df742e7ed10da38007d7a6bc6aa94fcb427f2ff9e1752` | 238 / category_specific_init=66,67; category_selection=77,78; category_mlp=143; multi_embodiment_action_encoder=177 |
| `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` | `ec0d59f771c206e56ebbaf8697a30c8913b2d75c` | `77fd0e1f6972cdcf0334f18f2e84724629f6746612721b6b27135ccb23f00b3f` | 889 / projector_id_table=69,74,75,76,112,119; inference_action_mask=539; inference_valid_horizon=533; decode_slice=282,531,539,541; training_action_pad=605,616; training_mask=635,636; processor_state_dropout=270,648,649,790; save_stats_config=243,270,648,649,758,790,796,805,828,835,844,873 |
| `gr00t/data/state_action/state_action_processor.py` | `3bebaca6db900f6b5d0637451009ab7251869b90` | `0151dd0bb6cb727fd401bf78121efc532e7184f2575e6ef25294085f629ac145` | 695 / relative_reference=386; apply_relative=389,639; unapply_absolute=513,666 |
| `gr00t/data/state_action/pose.py` | `a0d679fa24566eca6d98a5b8b8ed2077f26d7a77` | `4a09d0f1da9fe80bd8c3b0788e04c06979666f5dd636a91799a21e7d92c5f808` | 726 / relative_pose_composition=662,675 |
| `gr00t/data/state_action/action_chunking.py` | `a777555128fc5b39d50f12073d1f62adcaa7ac20` | `0b8379b903a87ec35e1671afa15f8bb4af8d948ee239b5e0eb1c90c8b4dd6b78` | 680 / absolute_pose_composition=651,652; slerp_interpolation=502 |
| `gr00t/configs/finetune_config.py` | `15918e206aa2b04b300886e674f827811dc8aa84` | `2084b094c717cefc2f7833df97a7cdac45b0f2ab0ac87504028ceffd2a02015b` | 211 / official_flags=55,58,61; official_dropout_default=61; official_batch_default=124; official_lr=131; official_warmup=169; official_max_steps=163 |
| `gr00t/data/embodiment_tags.py` | `3424b4e83ed6046b1c5b5b41fc293d844c068da5` | `eadc1a395175209cc0d57afa21541cc64be7d848300602209de81fe42713ca47` | 224 / new_embodiment_tag=50,128,219; finetune_only_tags=178,217; sonic_posttrain_tag=44,107,209 |
| `gr00t/experiment/launch_finetune.py` | `00cb3ef1fb25a4958fc3403e0975076e3465115d` | `5f259a9453957fb08d21e7a0ec5febd29c986cdb7e031ee2a4573628bfc30852` | 130 / launcher_relative_action=102; launcher_dropout=82; launcher_tuning_flags=78,79,80,81 |
| `gr00t/model/gr00t_n1d7/setup.py` | `742dcf7c391428f26e2bed57f2ad08c7d35e32c6` | `5212755cf18c1147206408e7250576b27504a6bd3af6402d9cd7f011b2cfd31a` | 242 / model_from_checkpoint=82; processor_from_checkpoint=157 |
| `gr00t/model/modules/qwen3_backbone.py` | `b7665be7ac84dc597bed786e6766bb83774af1b9` | `e3902c8385d586fb9f079abf58d37795cbcefc55f2f506f761c4b78a5afbaf52` | 372 / backbone_tuning_flags=139,140,154,155,198,249,250,251,254,256,264,265,280,282 |
| `gr00t/__init__.py` | `e9d017c3b5570e473369a106e05bb2104bc9c6ca` | `1f5bc8699a273afc0df1601cf9b7ac40e2587bf65907961f6714d6f4f18df185` | 388 / local_pretrained_construction=53,111,179,198,206,228,236,240,245,277,279,282,286,293,295,313 |
| `gr00t/configs/data/embodiment_configs.py` | `8a1189a75442727ce30871b375dcff696c1d146a` | `4c7626426cfd184034df772bca9f1ab5c2014c139c388f833012ca98ac6f948a` | 361 / sonic_modality=67,88; sonic_horizon=38,86; official_so100_horizon=203 |
| `getting_started/real_world_deployment.md` | `d85da41ff327a9ed65b9d9edf184f073f40996a4` | `a11002f2955fc16d65118a3bd15fa6647afdf405e89abfc78f25f2e5fdfe2fb0` | 464 / dataset_recommendation=140,142; relative_action_recommendation=146,287; chunk_recommendation=148,430; jitter_taxonomy=263,264,265; async_receding_rtc=288,406,407,409,411,415,444; rtc_not_policy_wired=413; intra_chunk_metric=298 |
| `getting_started/finetune_new_embodiment.md` | `707e893c3dad98f58541f402d253c07806d3bfff` | `0866df58044a77b4581590c44f05bad3fb3e8947a1e6cd16e1b62e4030f97c26` | 192 / so100_horizon=43; so100_relative_joint=49; official_finetune_scale=107,109,172; official_augmentation=110; open_loop_execution=138; training_curve_guidance=168,169 |
| `getting_started/policy.md` | `cf1020b9e46d9ef9474cfc5ece07086a63fd2382` | `f5ec4c9aa37714837c1d0758597b1f07e0dd7a3def8715be26660368828fec04` | 579 / base_horizon=84 |
| `README.md` | `6d86ea8af1a7f303b8cb71e8e281606f71f7a0f3` | `62f6c55a356be039287416f97e03a3b5541052d038a3daec208ea9ff8045a067` | 635 / relative_eef_claim=90; expanded_dimensions=100,118,210; inference_chunk_distinction=121 |
| `gr00t/eval/open_loop_eval.py` | `4461b84431f33634b953c9e563addd91e20ab9ec` | `e15706fac4cd8caff3229a1538eb0288b487919b28c0e26ecf30c32d0f61245f` | 373 / execution_horizon_arg=56,69,108,143,166,171,184,230,253,351,370; denoising_steps_arg=266,308,310,311,314,316,319 |
| `gr00t/data/state_action/droid_frame.py` | `4d804b517a3670c70ab24d64a7a90e4cf87c53ed` | `ecbeabdffb6e1e7429319b86bd2f77368b423c72a7cb9989148483f7330bfe23` | 75 / no requested marker match |

### `NEW_EMBODIMENT` finding

The source tags `NEW_EMBODIMENT` as finetune-only and maps it to category ID
10; `UNITREE_G1_SONIC` maps to a different posttrain category. The category
specific linear layers allocate `W[num_categories, input_dim, hidden_dim]`
with `0.02 * randn` and zero bias, then select `W[cat_ids]`/`b[cat_ids]` at
runtime. No audited loading or initialization path copies an existing
embodiment slot into ID 10. The safe conclusion is therefore:

```text
shared pretrained backbone/DiT: loadable from base checkpoint
custom category-specific slot: no source-level copy initializer; must be learned
exact slot tensor state in a particular checkpoint: not inspected in this review
```

### Effective state dropout and O1 scale

The Stage20S/22/23 launcher explicitly passes `state_dropout_prob=0.0` to the
model and processor. The official finetune CLI default is `0.2`; the raw N1.7
model-config default is `0.8`. Both the processor and action head contain
state-dropout logic, so this is a real recipe difference, not merely a log
label. Zero dropout is sensible for a deliberate one-example diagnostic but
means O1 is not a direct reproduction of the official regularized recipe.

O1 uses batch 1 and 200 optimizer steps: 200 optimizer-example presentations
and 200 stochastic flow noise/time draws when gradient accumulation is one.
The official SO100 command uses global batch 32 and 2000 steps: 64,000
nominal example presentations/noise-time draws, or 320× O1. That count is a
presentation budget, not 64,000 unique windows; the official dataset has
multiple episodes/windows. O1@200 is consequently a bounded exact-recipe
negative, not evidence of architectural unlearnability.

### Horizon/mask and RTC

The official SO100 modality asks for 16 action indices while N1.7's maximum
head horizon is 40. Training pads to H40 and width 132, then zeros the mask
after the valid horizon and after the active action width. In inference, the
head produces its configured H40 universal tensor, while the processor slices
the configured `delta_indices` horizon and un-applies only the configured
physical streams. The open-loop evaluator executes only a separate
`execution_horizon` (16 in the official example).

RTC is an inpainting-style inter-chunk primitive. It feeds the previous action
tail into the new action overlap, freezes an initial overlap prefix, and ramps
the remaining overlap velocity. It is intended for latency and inter-chunk
continuity. It can reduce a boundary jump, but it does not explain or repair
high-frequency oscillation already present inside a standalone predicted
chunk. The upstream docs also say this primitive is not wired into
`Gr00tPolicy`/server-client and has no ready-made example at this revision.

## B. Official recommendations and symptom mapping

The pinned deployment guide recommends state-relative actions for easier
convergence/inter-chunk consistency, treats both joint and task space as
valid (with low-DoF arms often using joint space), recommends H16 by default
and H≥32 with RTC, and separates inference/replanning frequency from action
execution/camera frequency. It recommends asynchronous inference plus
receding-horizon execution and RTC when latency causes stop-and-go.

Its jitter taxonomy is directly useful for Stage20–23:

| upstream case | source meaning | Stage20–23 mapping |
|---|---|---|
| intra-chunk jitter | model undertrained or data quality issue | Stage20T/23's long, oscillatory H40 model profile is consistent with this phenotype; causality remains open |
| inter-chunk jitter | adjacent chunk predictions inconsistent | Stage21's old-prediction/new-chunk inconsistency is consistent with this case; achieved-to-new-first remained small, so it is prediction inconsistency rather than an adapter jump |
| smooth chunks, robot shakes | hardware/low-level control | not tested and not a current claim |
| stop-and-go | end-to-end latency exceeds execution period | Stage21 synchronous paused loop deliberately did not qualify real-time behavior |

The custom-embodiment guide uses a five-episode SO100 fixture, H16 joint
relative arm plus absolute gripper, batch 32 and 2000 steps, color jitter,
and train-trajectory open-loop curves as a primary sanity check. The guide
also warns that poor held-out performance with only five episodes is data
scarcity, not necessarily a code bug. These are upstream examples and
recommendations, not robot-specific guarantees.

## C. Modern SONIC-backed whole-body VLA path

The pinned path is:

```text
VR/teleop/controller motion and proprioception
  -> universal-token encoder / FSQ representation
  -> controller token_state
  -> LeRobot action.motion_token (64D) plus hand streams
  -> GR00T UNITREE_G1_SONIC action chunk, H40
  -> asynchronous Python inference worker
  -> latency-compensated action_chunk_index
  -> one 64D token + 7D left/right hand action per 50-Hz publish step
  -> SONIC C++ decoder/controller
  -> whole-body joint command
```

The SONIC action ABI is 64D motion token + 7D + 7D hands = 78D per step.
Relative to N1.7's universal width, 64/132 = 48.48% for the motion token and
78/132 = 59.09% if the hand streams are counted together. The project
10D physical EEF+gripper row is 7.58% of 132, but the ratios are only a width
comparison—not a claim that the spaces are semantically interchangeable.

The source declares VLA inference at 2.5 Hz (0.4 s interval), action publish
at 50 Hz (20 ms), and H40. If all H40 outputs are consumed at 50 Hz, one VLA
chunk spans 0.8 s between its first and last sample window; the 0.4 s update
interval creates nominal temporal overlap. This is distinct from SONIC's
controller reference lookahead: the released controller uses 4 or 10 frames
at 20 ms, approximately 80 or 200 ms. Those lookahead values are not total
end-to-end latency.

The runtime marks prefix steps stale by rounding inference delay × 50 Hz and
starting at that index, while a background worker obtains fresh observations
and chunks. The exporter records `action.motion_token` from controller
`token_state` (or zeros when unavailable); the upstream docs describe VR
whole-body teleoperation demonstrations as the label source. Sensor-side
stale checks use 100 ms for SONIC pose and 200 ms for planner messages.
SONIC's pretrained decoder/controller owns the conversion from token to
smooth, dynamically feasible full-body joint motion. This is materially
different from the current dense EEF H40 path, where a task-space adapter and
native WBC own realization.

## D. Older decoupled-WBC path

The old design passes high-level fields including `target_upper_body_pose`,
`target_time`, `navigate_cmd`, and `base_height_command`. The teleop loop
attaches a monotonic timestamp and target completion time; the upper-body
policy is explicitly described as open-loop interpolation. `InterpolationPolicy`
stores time-stamped waypoints, drops waypoints before
`interpolation_garbage_collection_time`, and schedules each waypoint subject
to `max_change_rate`. `get_action(time)` evaluates the interpolator and emits
the current continuous pose. The composed policy then supplies these
interpolated upper-body values to a lower-body Gear WBC policy, which selects
standing/walking RL policies based on navigation command magnitude and emits
body joint action.

Therefore this interface is sparse/time-stamped intent followed by an
explicit declared interpolation/controller layer; it is not a high-level
model emitting every low-level smooth step. This is an architectural analogy
for option C below, not a recommendation to hide post-hoc correction in the
current benchmark.

## E. Architecture comparison

| system | high-level representation | active dims / 132 | H / sampling | duration / update | execution and latency | smoothness / label owner |
|---|---|---:|---|---|---|---|
| Current Go2+ARX5 ABI | Vision+Language+StateV1 -> NEW_EMBODIMENT GR00T -> dense EEF XYZ+rot6d+gripper | 10/132 = 7.58% | H40; 0.05 s TEST_ONLY | ~2.0 s nominal window / 1.95 s first-to-last; production TBD | task-space adapter -> native WBC; Stage21 paused/receding diagnostic, no production async | native WBC; expert desired EEF + gripper labels, achieved retained separately |
| Isaac-GR00T N1.7 custom example | NEW_EMBODIMENT SO100 joint-relative arm + absolute gripper | configured physical prefix; remaining width padded/masked | H16 configured in modality; model max H40 | source does not impose a universal physical period; execution_horizon=16 | open-loop evaluator executes prefix then replans; RTC primitive manual at pinned revision | model action head plus robot controller; LeRobot demo trajectories |
| GEAR-SONIC VLA | 64D motion token + 7D left hand + 7D right hand | 64/132 = 48.48%; full 78/132 = 59.09% width comparison | H40; 20 ms / 50 Hz publish | ~0.8 s if H40 is consumed at 50 Hz; VLA update 2.5 Hz / 0.4 s | async worker; latency-compensated stale-prefix index; SONIC C++ decoder/controller | SONIC owns whole-body smoothness/dynamic realization; token_state from teleop/controller exporter |
| Older decoupled-WBC | target_upper_body_pose + target_time + navigate_cmd + base_height_command | not a 132D GR00T tensor; sparse structured intent | time-stamped waypoints; interpolation evaluated at controller time | target-time dependent; no fixed VLA H | max_change_rate interpolation -> upper-body command; lower-body Gear WBC/RL | interpolator/controller owns continuous motion; teleop/planner supplies waypoints |

For the current project row, the exact frozen description is:

```text
Vision+Language+StateV1
 -> NEW_EMBODIMENT GR00T
 -> dense H40 x 10D EEF+gripper
 -> 0.05 s TEST_ONLY sample period (~2.0 s nominal chunk; 1.95 s endpoint span)
 -> task-space trajectory adapter
 -> native WBC
```

## F. Stage23 claim reassessment

The following prior conclusions are retained but bounded:

- `FREE_RUNNING_TEMPORAL_PROFILE_FAILURE_STRONGLY_SUPPORTED` remains an
  observed phenotype: O1@200's sealed profiles show approximately 25.57×
  expert median path length, median adjacent direction cosine -0.5493,
  back-and-forth rate 0.7632, and high-frequency first-difference energy
  fraction 0.8436; the three O1@200 outputs were finite but failed the
  declared test-only SafetyGate speed bound.
- `O1@200` remains a valid bounded negative for the exact `TINY1_H2STATS`,
  batch-1, 200-step, Stage23 recipe. It does not establish that a custom
  embodiment is unlearnable or that the representation is wrong.
- The phrase “single-sample memorization failure” should be demoted to
  “single-sample, bounded-recipe overfit diagnostic did not produce a healthy
  free-running profile.” The official example has 320× the nominal optimizer
  example/noise-time exposure and materially different H, augmentation,
  dropout, and default trainability settings.

### Attribution matrix

| hypothesis | evidence status | interpretation boundary |
|---|---|---|
| `input drift` | **NOT DEMONSTRATED** | No new model run was performed; source review cannot close the prior run's input provenance. |
| `GPU/resource` | **LOW PLAUSIBILITY FOR PHASE A** | Existing sealed Phase B evidence reports H200 resource/OOM ruled out; this review did not rerun it. |
| `optimizer/LR schedule` | **PLAUSIBLE UNRESOLVED** | O1 has only 200 steps and each max-step run has its own warmup/decay schedule. |
| `flow objective fit` | **NOT DEMONSTRATED** | Free-running profile failure does not prove teacher-forced objective failure. |
| `normalization` | **NO DEFECT FOUND IN AUDIT** | Tiny-stat degeneracy is retained as a diagnostic warning, not a demonstrated cause of H2. |
| `temporal oscillation` | **STRONGLY OBSERVED PHENOTYPE** | High path/tortuosity/frequency metrics support the phenotype, not one unique mechanism. |
| `132D inactive-noise burden` | **MECHANISTICALLY PLAUSIBLE** | Analytic inactive/active noise energy ratio 12.2x is not causal proof. |
| `4-step Euler sampler` | **UNRESOLVED** | Sampler is a candidate factor; no C1 or sampler comparison was authorized. |
| `data scarcity/generalization` | **PLAUSIBLE HIGH** | Official custom guidance uses substantially more presentations and multiple demonstration windows. |
| `state dropout / augmentation / trainability recipe` | **REAL RECIPE CONFOUNDS** | Stage23 differs from official defaults in dropout, color jitter, VLLN, batch, H, and custom-slot exposure; none is proven causal here. |

In particular, the four-step Euler sampler is only one unresolved factor.
The source review finds no action-mask defect, no relative-transform semantic
ambiguity in the audited path, and no evidence that the Stage23 evaluator's
identity test is wrong. It does find legitimate recipe differences: O1 uses
state dropout 0 rather than official CLI 0.2, no explicit color jitter while
the official example enables it, freezes VLLN explicitly while the raw model
default is trainable, uses B=1, uses H40 rather than the official H16 example,
and relies on the custom category slot. These differences are confounds to
control, not reasons to erase the oscillation observation.

## G. Candidate next-stage options (analysis only)

| option | scientific upside | engineering/data cost | falsifier / strengthening evidence |
|---|---|---|---|
| **A** | Directly tests whether the dense H40 EEF hypothesis becomes healthy under official-scale custom-embodiment training and matching runtime semantics. | Highest immediate GPU/data cost; requires a larger robot-specific or controlled simulation dataset and careful timing lock. | Healthy held-out profiles and overlap under the same dense ABI strengthen the current architecture; persistent oscillation after controlled scale weakens dense H40. |
| **B** | Separates prediction horizon/profile burden from execution latency by using shorter state-relative chunks, async scheduling, and RTC where valid. | Moderate implementation/runtime cost; requires explicit timing, chunk-age, and inter-chunk accounting, and can mask full-chunk quality if not audited. | Healthy raw chunks plus stable receding execution strengthen a runtime explanation; raw intra-chunk oscillation remains unchanged and falsifies a runtime-only explanation. |
| **C** | Tests whether the VLA is better suited to sparse task-space intent while an explicit, declared interpolation/rate-limit layer owns smoothness before WBC. | Requires a new waypoint/target-time ABI, interpolation contract, and matched evaluation; increases interface design burden and may reduce endpoint expressivity. | Stable waypoints plus transparent interpolation with improved task-space outcomes strengthen it; failure before interpolation or dependence on hidden correction weakens it. |
| **D** | Moves temporal smoothness into a learned compact task-space manifold, inspired by SONIC without requiring the Go2+ARX5 VLA to emit low-level whole-body channels. | Largest research cost: trajectory corpus, tokenizer/decoder training, representation validation, and a new checkpoint/ABI lifecycle. | A frozen token decoder reconstructing WBC-compatible smooth references and improving held-out profile/overlap metrics strengthens it; no reconstruction gain or embodiment leakage weakens it. |

No option was implemented or selected by this audit. The evidence most
directly supports a PI choice between a narrowly isolating fixed-noise /
teacher-forced flow diagnostic and a properly controlled official-scale
custom-embodiment replication. A future runtime/representation change should
wait for that decision.

## Required return

```text
ROLE: CODEX_CORELAB
STAGE: Stage23 upstream architecture review before Stage24
GIT_SHA: de454440b2458f0f28789f45b0d34299553241a9
ISAAC_GROOT_SHA: 51d4c89f72fda44cbf77285c6a8114b52676b8a1
GROOT_WBC_SHA: a0732b642c0333077e127a2f56ab0014c196bca4
C1_EXECUTED: false
NEW_MODEL_FORWARD: false
NEW_TRAINING: false
HOPPER_PBS: false
VALIDATION_E_READ: false
HOLDOUT_F_READ: false
UPSTREAM_ACTION_HEAD_TRACE: COMPLETE; dense H40x132 flow head with masked padding/loss and four-step Euler inference
NEW_EMBODIMENT_PROJECTOR_FINDING: category ID 10 finetune-only custom slot; no source-level copy initializer; embodiment-specific path requires learning
EFFECTIVE_STATE_DROPOUT_FINDING: Stage23 launcher/model/processor 0.0; official CLI 0.2; raw model default 0.8
O1_VS_OFFICIAL_FINETUNE_SCALE_FINDING: O1 200 batch-1 presentations versus official SO100 nominal 64,000 batch-32 presentations; 320x exposure difference; O1 is bounded negative only
SONIC_VLA_ABI_FINDING: 64D motion token + 7D/7D hands, H40, 50Hz publish, 2.5Hz inference, SONIC decoder owns whole-body realization
ASYNC_LATENCY_RTC_FINDING: SONIC async worker skips stale prefix by latency-compensated index; N1.7 RTC is low-level/manual and mainly inter-chunk
DECOUPLED_INTERPOLATION_FINDING: sparse/time-stamped upper-body/nav intent becomes continuous output through max-change-rate interpolation; lower body is separate Gear WBC/RL
STAGE23_CLAIM_REASSESSMENT: phenotype remains strongly observed; O1@200 is a bounded negative; prior general single-sample memorization wording was too strong; sampler-only attribution is unsupported
ARCHITECTURE_OPTIONS: A/B/C/D compared analytically; none implemented or selected
RECOMMENDED_NEXT_EXPERIMENT_FOR_PI_REVIEW: PI review should choose between the already-prepared fixed-noise teacher-forced C1 diagnostic and a controlled official-scale custom-embodiment replication; no experiment executed here
RESULT: UPSTREAM_ARCHITECTURE_REVIEW_COMPLETE
BLOCKER: NONE_FOR_SOURCE_REVIEW; C1 remains on hold pending PI architecture decision
NEXT_GATE_REQUEST: GPT_PI_ARCHITECTURE_REVIEW_AND_STAGE24_DECISION
```

## Reproducibility

```text
python scripts/stage23_upstream_architecture_audit.py
```

The companion JSON records every inspected path's pinned Git blob ID,
content SHA-256, line count, and requested marker line numbers.
