# PI0285 R3 WP5 saved-array coverage and alternatives

Status: COMPLETE bounded descriptive CPU audit. Alternative design: **NOT_AUTHORIZED_FOR_PRODUCTION**. Normalizer/processor/labels unchanged; model/teacher/physics/render/optimizer calls all zero.

Inputs bind SOURCE_BANK, original source results and arrays, frozen statistics/processor/mapping, DIAG192 and original CPU codec bias by SHA256 in WP3/TEMPORAL_CONTRACT.json. Scope is both instructions, first eight TRAIN and first eight DEV layouts: 32 episodes, 19232 H40-eligible anchors, 192 fixed DIAG windows. This does not repeat the full 76928 TRAIN roundtrip. Original aggregate TRAIN maxima remain UNOBSERVED.

## Temporal contract

Observation index i is real RGB and StateV1 at t=i*0.05 before command i. StateV1 reconstructs physics sample 5*i; achieved state for command i is sample 5*(i+1). H40 begins with command i; first8 uses commands i..i+7, installations i..i+7 and achieved samples 5*(i+1)..5*(i+8). Teacher event time is relative to the 20 neutral controls: event index is 20+round(teacher_time/0.05).

Teacher target/gripper is computed from phase_before; transition mutates stage; returned stored phase marks phase_after. Thus align->close and lower->release transition rows issue the preceding gripper value for one 50ms control. This is confirmed bookkeeping semantics, not authorized relabeling. Measured opening is (q_left+q_right-0.00004)/0.08; commanded opening 0=closed, 1=open; native actuator is 2*g-1. All fixed original teachers succeeded. Installation/tracking residuals are reported per first8 in FAILURE_ATTRIBUTION.csv; they are not task losses or unique causal explanations. First model attribution is pending owner-provided first-valid arrays.

## Coverage

Bounded source phase occupancy: {"0": 1282, "1": 864, "2": 1711, "3": 832, "4": 14543}. Event-window exposure: {"first8": 1216, "no_event": 15010, "tail_only": 3006}. State/raw-command/relative-target ranges and excursions outside frozen TRAIN bounds are reported separately for T1 TRAIN, T1 DEV, T2 TRAIN, T2 DEV in JSON coverage.by_task_split. These are observed support of these 32 episodes, not population bounds.

Additional observed state support gap: both T1/T2 DEV285200 StateV1 channel5 at controls20..23 (1.00..1.15s) is below frozen TRAIN minimum. At control20 raw value -0.11755000 versus TRAIN minimum -0.08169304, unclipped normalized value -1.45055024. Raw observation identity and pre-command timing are correct. This is a held-out support excursion, not a newly proven runtime bug; production statistics remain frozen. The rotation-action-only counterfactual below does not resolve state support. Absolute raw command channel5 has three excursions per task; production actions use relative normalization, whose observed failures remain the preserved two fixed windows.

Existing full plan audit receipts are reused, without rescanning images:

- PLACE4096: 32768 draws, 10781 unique UIDs, 10767 unique two-view RGB, 10778 unique states, 10778 unique RGB/state pairs; first8 event draws 19560, tail-only draws 10354; phase draws {"2": 8192, "3": 6554, "0": 8192, "1": 8192, "4": 1638}.

- SHARED4096: 32768 draws, 18008 unique UIDs, 13939 unique two-view RGB, 13939 unique states, 13939 unique RGB/state pairs; first8 event draws 18942, tail-only draws 10517; phase draws {"3": 6554, "0": 8192, "2": 8192, "1": 8192, "4": 1638}.

## Rotation counterfactual

Raw commands remain evaluation truth. The retained frozen codec comparison has exactly two failures: T1-DEV-285200-20 and T2-DEV-285200-20, shared initial geometry. Their H40 SO3 RMS is 0.03232898 rad and first8 RMS 0.01070248 rad; original maximum SO3 deviation 0.035746579 rad remains preserved in R2 evidence. The valid-relative-SO3 counterfactual changes only six rotation normalization bounds to [-1,1], preserving original XYZ/gripper scaling and Gram-Schmidt finite/nondegenerate SO3 checks. Across these 192 fixed windows alternative maximum SO3 error is 1.6669641e-16 rad. This is label algebra, not a learned model result or error-floor proof.

Empirical rotation widths: [0.2874008222690907, 1.3838041010580984, 0.08314503653304575, 1.3808601247351446, 0.2878277300958392, 0.16001706488713713]. Existing normalized derivative per physical coefficient: [6.958922330874262, 1.4452912796477042, 24.05435229083225, 1.448372622378106, 6.948600815265617, 12.498666947869813]; alternative derivative is 1. For the same small physical coefficient error, alternative/empirical squared normalized loss ratios are [0.020649808160237364, 0.47872844752630295, 0.0017282742750203781, 0.4766936710208899, 0.02071120055303081, 0.006401365263773563]. Expanded bounds strongly reduce weighting of narrow rotation components. No clipping does not establish better training geometry, and arbitrary predicted 6-vectors still require nondegeneracy checks.

## Stored-history near-observation audit

128 fixed approach/release/hold observations were compared under the rule: Same instruction, different original episode; XYZ<=.01m SO3<=.1rad gripper<=.1; nearest per row in 16x16 two-view pixels. There are 10 nearest candidate pairs; exact full RGB/state matches among those 0. Up to twelve nearest records retain current pixel distance, physical state distance, first8 action differences and genuine 50ms prior-history differences in JSON. These surviving pairs are approach observations; no release/hold pair met the tight physical-state rule in this fixed sample. Their previous 50ms distances equal current distances, so this sample provides no demonstrated history improvement. Downsampled similarity does not prove contradictory labels; exact duplicate nonconflict does not prove full observability. Cross-instruction rows are excluded because language legitimately changes the desired goal.

## One prospective HISTORY/ROTATION design

NOT_AUTHORIZED_FOR_PRODUCTION: a later separately reviewed experiment may replace six relative rotation bounds with mathematical [-1,1] and add one genuine previous 50ms RGB pair and StateV1 plus a validity mask at reset. Explicit changed fields are relative_action.eef_9d.min/max[3:9], observation image/state temporal indices [previous,current], and reset history-valid mask; XYZ/gripper math, instructions, frame, H40/first8 and task scorer stay explicit. Use separate experiment/processor identities; never modify this sprint's sealed inputs. Required checks: valid SO3 roundtrip including both preserved DEV cases, nonfinite/degenerate rejection, XYZ/gripper parity, prior timestamp exactly current-0.05, no crossing reset/takeover, masked missing history, single-current-input baseline with identical noise and physical scoring. Current evidence motivates a diagnostic design only; it does not demonstrate history benefit or authorize gradients.
