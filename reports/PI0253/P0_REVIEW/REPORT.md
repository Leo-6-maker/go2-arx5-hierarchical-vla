# PI0253 P0 postmortem: label admission and support regression

Scope: direct user pasted review, P0 existing arrays/actual consumption records only. No new goal/scientific GO, model load/query, training, gradient query, physical episode, new cohort or selection. PI0253 frozen NO_ADMISSIBLE_CHECKPOINT/NONE remains unchanged. P1 remains a proposed experiment requiring frozen protocol before any new query.

## 1. Expert and model admission alignment

|Checkpoint|Model REJECT|Same source expert REJECT|Model-only new violation|Expert REJECT now non-REJECT|
|---|---:|---:|---:|---:|
|10000|10|10|0|2|
|2500|11|10|1|2|
|5000|9|9|0|3|
|7500|11|11|0|1|

Full source episode IDs including observation suffix were joined. Historical expert set12 and model union involve13 source windows;9 windows reject under all four candidates. For those13, original parquet action/state/timestamp byte hashes match PI0237, frozen gate config and original selector hashes match. No replacement label or retiming. Narrow current C_T encode/decode CPU checks: maximum XYZ error 1.22e-07 m, SO3 7.3e-08 rad.

This establishes expert labels themselves exceed the same frozen admission criterion at the overlapping windows. It does not prove the model is perfectly imitating those labels: model-to-expert first-pose median errors across rejected windows are 0.237/0.238/0.243/0.215 rad at2500/5000/7500/10000. Label mismatch and model fit error coexist; overlap alone is not a causal decomposition.

The sole model-new source is FULL2500 B:j08/s04/r01 :obs-0000 (full ID in CSV). Expert-current1.4752439217rad (below1.5 by.0247561), model-current1.5581403616 (above by.0581404), model-expert.1795324754. It is not classified as an expert REJECT. No binding-undetermined row remains among the13-window joined scope. This does not locate the underlying desired/achieved/window initialization cause.

## 2. Learning and retention decomposition

704 existing retention rows =128 existing G B0/B1 +64 FULL1000 B0 +512 formal FULL B0/B1. The576 new PI0253 rows are reused; no new inference. Each actual tensor content hash matches its receipt; model labels/state exactly match G for each source/noise. Raw arrays are not downloaded with model weights.

Per window e(t)=predictedXYZ-expertXYZ. Offset=mean_t e(t); centered shape RMSE=RMS_t(e(t)-offset). Exact RMS decomposition asserted per window. Group medians are separate summaries and do not obey a median-level squared identity.

|All64/B0 model|XYZ RMSE median mm|Mean-offset magnitude median mm|Centered-shape RMSE median mm|First-point median mm|Last-point median mm|
|---|---:|---:|---:|---:|---:|
|G2000|14.81|8.77|11.05|13.63|15.02|
|FULL1000|67.65|58.30|31.26|24.35|85.78|
|FULL2500|54.54|44.87|30.89|20.80|72.46|
|FULL5000|72.61|61.33|31.46|24.26|84.24|
|FULL7500|54.86|47.37|28.41|19.94|63.91|
|FULL10000|54.38|45.16|28.63|18.80|66.85|

FULL10000 vs G: XYZ worse63/64 in each noise group; centered shape worse61/64 in each group. Median per-window offset/SSE fraction75.20% B0 /74.19% B1. Regression is broad, includes both offsets and trajectory shape, and cannot be reduced to one global constant correction. No alignment, smoothing or correction is applied to predictions.

|FULL10000/B0 stratum|XYZ median mm|Centered-shape median mm|
|---|---:|---:|
|old16|52.34|23.64|
|other48|55.63|29.39|
|task_a|45.45|25.99|
|task_b|47.75|28.58|
|task_c|53.59|27.92|
|task_d|68.78|29.55|
|phase_0|69.13|44.27|
|phase_1|75.12|34.33|
|phase_2|56.19|25.15|
|phase_3|32.49|13.15|

All704 per-cell rows and both-noise all64/old16/other48/A-D/four-phase summaries are in the CSVs. Phase is the preexisting rollout-window rank quartile, not a newly optimized cut;0early,1/2middle,3late. Error is larger early/middle but also increases late. A-D and both old16/other48 regress. These are descriptive strata, not independently sampled experiments or proof of optimization causality.

Actual10000 pre-forward source records match frozen schedule:9584 TRAIN windows occur once and208 twice. Old64:63 once and1 twice (65 exposures total). Prefix exposure counts per checkpoint are included per source; G FULL-exposures=0 denotes before the FULL stage and says nothing about its ancestor training exposure. Actual exposure differs greatly from prior local repeated training, but limited repetition is not by itself proof that more identical updates would solve it.

Normalized XYZ absolute maximum across704 rows 0.914062. Boundary/exceedance fractions are included; actual clipping/outlier events are not exposed and remain UNOBSERVED. Normalization range statistics are not asserted as clipping or saturation. No new clipping intervention.

## 3. Fixed live input

Existing four live arrays recomputed against the already corrected A_attempt2 actual-time expert sample. Their original XYZ and SO3 summaries match. New columns include offsets/centered shape/first/last; reference-min world distance retains the actual anchor. These four single-input side predictions remain development diagnostics; no new achieved TCP or generalized success rate is inferred.

|Checkpoint|Expert XYZ RMSE mm|Centered-shape mm|Reference min world-target mm|
|---|---:|---:|---:|
|2500|96.84|31.19|57.38|
|5000|159.01|59.83|64.73|
|7500|115.84|53.40|49.39|
|10000|106.99|44.36|56.40|

## 4. Narrow CALIB statistic audit

Target identity matches original sealed authority on all4352 candidate records: field target_position_world_m, authority SHA06fa238a27485ecde39c75797fe9fd82dd1cdc5f5c79f5d5a5b3d9dcab7597e8. Actual source frame is NATIVE_WBC_RELATIVE_REFERENCE. Original selector and actual qualified adapter both compute min_t norm(decoded native-relative XYZ - target world XYZ), with no anchor/world conversion in that statistic. The example calculation from saved arrays reproduces the record exactly.

Therefore its large values are an unchanged historical mixed-coordinate statistic, not a world TCP error and not expert RMSE. Live side reports instead add their actual anchor before world-target comparison. No invented CALIB anchor, no rewritten statistic, no selector reranking. Candidate rejection gate is independent of this distance ranking; all candidates already fail zeroREJECT, so correcting interpretation does not change NONE. Separate TRAIN expert fitting regression also remains.

Complete profile denominator1088 each. Selector distance summaries include non-REJECT records1077/1079/1077/1078, with median. Worst-cell value is maximum of source-rollout medians over that same non-REJECT subset; WARN included. No change to historical summary or hash.

## 5. Remaining inference gap and proposed P1

P0 confirms label-admission conflict and broad old-support degradation. It still cannot decide new-TRAIN improvement versus broad underfit: there is no G/FULL paired prediction set across representative TRAIN outside old64. A single fixed live improvement does not fill that gap. Neither data shortage, architecture inability, generalization failure, catastrophic forgetting cause, nor mode collapse is proven.

The proposed64 new TRAIN windows excluding old64 × G2000/FULL10000 × one presealed shared actual noise =128 queries remains NOT_STARTED. It needs a frozen new scientific protocol/model-input/source/noise identity and budget; no new query authorized by this retrospective analysis. A replay/data-mixture intervention remains a hypothesis until that contrast is available. No additional10k, loss search, smoothing, gate change, candidate substitution or sim.

Evidence: expert_model_gate.csv, retention_decomposition.csv, retention_groups.csv, live_decomposition.csv, summary.json, supplement.json, input_inventory.json. Exact original main source and later supplement source retained separately. Historical reports/hashes, retired weights, fixed HF family and selected NONE unchanged.
