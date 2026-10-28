# PI0153 tier/seed selection packet

CPU/read-only material derived from the sealed PI0152 H200 campaign. No model, checkpoint payload, sampler, GPU, PBS, training, Validation-E, Holdout-F, WBC, simulator, or hardware execution occurred while producing this packet.

- Gate: `PI_GATE_STATE_0153` / Issue comment `5649870843`
- Science execution HEAD: `eab15ec185203d69c03ab82b7843bcaea9200281`
- Analysis HEAD: `212dd449fe9b89f1059e470ce8caf4f52b1a456d`
- Physical rows: `432` unique rows
- Pair comparisons: `27` (12 intervention-vs-Legacy, 9 exposure, 6 L1-vs-N2)

## Row and recovery accounting

`432 = 312 current-direct + 24 recovery + 96 historical`. The reported `336 current-phase` count includes the 24 recovery rows, so `336 + 24 + 96` is intentionally not an additive formula. The 1296-entry canonical index is a phase-reference index and contains zero physical metric rows.

The accepted unique call identity count is `8976`: `8856` new current-allocation calls plus `120` retained recovery-source calls. The `120` failed-attempt calls and `120` recovery overlap refer to those same identities and are not added again.

## 18 arm summaries

| arm | rows | progress m | final target distance m | minimum distance m | path length m | roughness | endpoint cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEGACY(seed2222)@1000 | 24 | 0.031336074 | 0.100875424 | 0.042051574 | 1.625934702 | 170.307243842 | 0.557034313 |
| LEGACY(seed2223)@1000 | 24 | 0.031504250 | 0.100707249 | 0.034449282 | 1.527599868 | 157.845057489 | 0.621310137 |
| LEGACY(seed2224)@1000 | 24 | 0.005321938 | 0.126889560 | 0.026098386 | 1.831612595 | 199.182403990 | 0.368790989 |
| LEGACY(seed2222)@2000 | 24 | 0.042437054 | 0.089774444 | 0.030283096 | 1.618187486 | 171.434126108 | 0.631910996 |
| LEGACY(seed2223)@2000 | 24 | 0.035869892 | 0.096341606 | 0.033537529 | 1.538938624 | 163.115124228 | 0.627285147 |
| LEGACY(seed2224)@2000 | 24 | 0.010307158 | 0.121904341 | 0.029922852 | 1.865120796 | 207.597733404 | 0.422267521 |
| N2(seed2222)@1000 | 24 | 0.017187453 | 0.115024045 | 0.044334055 | 2.038336772 | 209.871531353 | 0.562539571 |
| N2(seed2223)@1000 | 24 | 0.012956346 | 0.119255153 | 0.046478452 | 1.952814060 | 196.576913529 | 0.546806578 |
| N2(seed2224)@1000 | 24 | 0.017410515 | 0.114800983 | 0.037186577 | 1.936663814 | 199.819522993 | 0.576606841 |
| N2(seed2222)@2000 | 24 | 0.016042972 | 0.116168526 | 0.036550165 | 1.950328521 | 212.881799103 | 0.529003700 |
| N2(seed2223)@2000 | 24 | 0.018335669 | 0.113875830 | 0.044118871 | 1.934147138 | 198.630923975 | 0.598600758 |
| N2(seed2224)@2000 | 24 | 0.022193499 | 0.110018000 | 0.038635300 | 1.918836852 | 200.679183011 | 0.598962479 |
| L1(seed2222)@1000 | 24 | 0.043974280 | 0.088237219 | 0.040472885 | 1.489748948 | 158.318298016 | 0.635507516 |
| L1(seed2223)@1000 | 24 | 0.046998596 | 0.085212902 | 0.035792101 | 1.458310846 | 149.489135258 | 0.684748273 |
| L1(seed2224)@1000 | 24 | 0.042296902 | 0.089914597 | 0.032443112 | 1.462398752 | 155.569157542 | 0.662164439 |
| L1(seed2222)@2000 | 24 | 0.050292559 | 0.081918940 | 0.032656760 | 1.440640829 | 154.596653087 | 0.691061726 |
| L1(seed2223)@2000 | 24 | 0.050113001 | 0.082098498 | 0.037166756 | 1.452793904 | 153.017599360 | 0.660165045 |
| L1(seed2224)@2000 | 24 | 0.048484075 | 0.083727424 | 0.033552815 | 1.456485466 | 156.860913978 | 0.664780886 |

The table is descriptive and keeps every seed/tier arm separate; it is not a winner selection.

## D1/D2A read-only diagnostic closure

### D1

- `legacy`: 864 derived rows, 4 arms, raw-before-derived=True
  - `normalized_local_flow`: rmse mean `0.874707412`, cosine mean `0.551095075`
- `n2`: 1296 derived rows, 6 arms, raw-before-derived=True
  - `normalized_local_flow`: rmse mean `0.876443517`, cosine mean `0.542582378`
- `l1`: 1296 derived rows, 6 arms, raw-before-derived=True
  - `normalized_local_flow`: rmse mean `0.883362263`, cosine mean `0.538859593`

### D2A

- `legacy`: 384 derived rows, 4 arms, raw-before-derived=True
  - `state_path_divergence`: rmse mean `0.327533122`, cosine mean `0.929553402`
  - `field_sensitivity`: rmse mean `0.063121248`, cosine mean `0.988754012`
  - `teacher_bridge_local_error`: rmse mean `0.878522335`, cosine mean `0.559745090`
  - `off_path_diagnostic`: rmse mean `0.876085359`, cosine mean `0.563046916`
- `n2`: 576 derived rows, 6 arms, raw-before-derived=True
  - `state_path_divergence`: rmse mean `0.327402921`, cosine mean `0.929044032`
  - `field_sensitivity`: rmse mean `0.074840083`, cosine mean `0.984632234`
  - `teacher_bridge_local_error`: rmse mean `0.880395874`, cosine mean `0.550823405`
  - `off_path_diagnostic`: rmse mean `0.877110798`, cosine mean `0.555137366`
- `l1`: 576 derived rows, 6 arms, raw-before-derived=True
  - `state_path_divergence`: rmse mean `0.331678876`, cosine mean `0.923012526`
  - `field_sensitivity`: rmse mean `0.061887833`, cosine mean `0.989067120`
  - `teacher_bridge_local_error`: rmse mean `0.889118889`, cosine mean `0.546058385`
  - `off_path_diagnostic`: rmse mean `0.887401145`, cosine mean `0.548298689`

## Metric/frame definitions

- `final_target_distance_m`: Euclidean distance from the final decoded EEF position to the target, in world-frame metres.
- `absolute_progress_m`: initial target distance minus final target distance; positive means distance reduction, not a success threshold.
- `minimum_target_distance_m`: minimum decoded-path distance to the target, in metres.
- `prediction_path_length_m`: sum of successive decoded EEF XYZ segment lengths, in metres.
- `roughness_ratio`: mean norm of decoded XYZ second differences divided by the expert XYZ second-difference mean; dimensionless.
- `endpoint_target_vector_cosine`: cosine between decoded endpoint displacement and the target vector from the decoded start; dimensionless and distinct from decoded-vs-expert error.
- `endpoint_error_m`: decoded final XYZ minus expert final XYZ norm, in the existing physical/control frame; it is not final target distance.
- All paired deltas use `left - right` on the same TRAIN24 window; no weighted score or significance threshold is used.

## Recommendation and limitations

Recommendation: make the decision unit one complete route×tier block with all three seeds, review the 24-window paired evidence and D1/D2A summaries, then have PI pre-register exactly one route/tier for the first Validation. Do not select from a single seed, pool tiers, or infer a winner from route means.

Limitations: this is TRAIN24 evidence only; 24 windows are repeated structured probes rather than 24 IID samples, and the physical E4 sources include both current/recovery and historical read-only provenance. D1/D2A summaries are descriptive and do not establish a new scientific intervention.

Files: `selection_packet.json`, `per_window_physical_metrics.json`, `stratified_summaries.json`, `paired_comparisons.json`, `phase_diagnostic_summary.json`, `metric_frame_definitions.json`, `row_and_call_accounting.json`, `source_inventory.json`, and `PACKET_SHA256SUMS`.
