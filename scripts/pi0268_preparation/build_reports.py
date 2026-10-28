"""Assemble the preparation return from completed CPU receipts, without remote calls."""
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/PI0268/preparation'


def data(name):
    p=OUT/name
    return p.read_bytes() if p.exists() else gzip.decompress(Path(str(p)+'.gz').read_bytes())
def read(name):return json.loads(data(name))
def rows(name):return list(csv.DictReader(io.StringIO(data(name).decode())))
def write(name,text):(OUT/name).write_text(text.strip()+'\n',encoding='utf-8')
def table(headers,values):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in values])


def main():
    audit=read('TRAIN_AUDIT_RECEIPT.json');numeric=read('NUMERIC_SUPPLEMENT.json');exp=read('EXPOSURE.json');supp=read('EXPOSURE_SUPPLEMENT.json')
    closed=read('CLOSED_SUMMARY.json');episodes=read('CLOSED_EPISODE_ROWS.json');trace=read('OFFLINE_TRACE_SUMMARY.json');checks=read('CPU_CHECKS.json');cast=read('GRAM_AND_CAST.json')
    exposure=read('CONSUMED_CONDITION_INDEX.json');opt=rows('OPTIMIZATION_PHASE_VARIANT.csv');snapshot=read('SOURCE_SNAPSHOT.json')
    native_hash=numeric['native_loss_source_sha256']
    snapshot['native_model_source_Hopper_Corelab_equal_sha256']=native_hash
    snapshot['CPU_checks']=len(checks['checks']);snapshot['peak_observed_RSS_bytes']=max(audit['max_rss_bytes'],closed['max_rss_bytes'],read('NUMERIC_AUDIT_RECEIPT.json')['max_rss_bytes'],checks['max_rss_bytes'])
    snapshot['engineering_repairs']=['NumPy 1.x trapz compatibility in isolated analysis; original failed log retained.','Processor image transforms are parameter-free torch modules; allow only torchvision.transforms with no parameters, continue to reject all learned modules.','Completed 880 legacy prefix columns by hash-matched existing donor labels, without rerunning codec/exposure or models.']
    snapshot['analysis_code_scope']='Only scripts/pi0268_preparation and reports/PI0268/preparation; original PI0267 scientific files unchanged.'
    (OUT/'SOURCE_SNAPSHOT.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    scale_table=table(['Boundary (10 active representation coordinates)','abs p95','abs p99','fraction > 1000','nonfinite'],[
        ['state',f"{audit['state_normalized']['abs_p95']:.3f}",f"{audit['state_normalized']['abs_p99']:.3f}",f"{audit['state_normalized']['outside_1000']:.4%}",0],
        ['relative action + gripper',f"{audit['target_normalized']['abs_p95']:.3f}",f"{audit['target_normalized']['abs_p99']:.3f}",f"{audit['target_normalized']['outside_1000']:.4%}",0]])
    cast_table=table(['Arithmetic operation','maximum XYZ error (m)','maximum SO3 error (rad)'],[
        ['action FP32 encode/decode',max(r['FP32_xyz_m'] for r in cast['roundtrip']),max(r['FP32_so3_rad'] for r in cast['roundtrip'])],
        ['action BF16 CPU cast then decode',max(r['BF16_CPU_ARITHMETIC_xyz_m'] for r in cast['roundtrip']),max(r['BF16_CPU_ARITHMETIC_so3_rad'] for r in cast['roundtrip'])],
        ['state BF16 CPU cast then affine inverse',numeric['state_BF16_cast_inverse']['max_xyz_m'],numeric['state_BF16_cast_inverse']['max_so3_rad']]])
    write('CODEC_CONDITIONING_REPORT.md',f'''# PI0268 — actual codec conditioning

## Finding

The unchanged OLD codec is reversible but poorly scaled on actual PI0267 recovery TRAIN. Its X/Y pose changes activate near-constant rotation components. This is measured on all 24 frozen TRAIN families / 10,752 raw conditions, not inferred from a toy or fitted from EVAL. The evidence warrants a normalization-only comparison; it does not establish a unique cause of R's failures.

{scale_table}

These population fractions include all seven variants, not the actual training sampling weights. Padded dimensions are excluded. The ten active coordinates are a 9-coordinate EEF representation plus gripper, not ten independent physical DoFs.

## Binding and transformations

Actual `build/set_statistics` input SHA256: `{audit['source_statistics_sha256']}`. It matches the frozen R1 design and actual Hopper dataset statistics. Both checkpoint archival statistics have different serialized bytes but equal parsed numeric structure. A filename or formatting mismatch is not the finding.

Raw StateV1 and absolute labels are FP32. Actual stock pose conversion constructs orthonormal rotation rows, forms relative SE(3), then returns FP64 relative values. The frozen affine map is `2*(x-q01)/max(q99-q01,1e-6)-1`, without clipping. Packed state/action are FP32. Relative action EEF uses the 40×9 relative-action table; absolute-action EEF statistics are unused on this production route. Gripper retains its absolute map. State/action are padded to 132; the 122 extra coordinates remain zero and the corresponding action mask is zero.

Four actual RGB/state/language/label processor calls reproduce their consumed condition identities and raw-to-packed arrays exactly. Collator outputs: state FP32 `[1,1,132]`, action FP32 `[1,40,132]`, mask FP64 `[1,40,132]`, pixels FP32, tokens/embodiment INT64. See [PROCESSOR_BOUNDARY.json](PROCESSOR_BOUNDARY.json). No learned forward was executed. Model `prepare_input` casts floating tensors to the model dtype; the sealed native path uses BF16, while saved training full-S4 rollout tensors are FP32. Source equality between Hopper and Corelab was directly verified: `{native_hash}`. The S4 log hashes document its real stored dtype; this report does not conflate CPU casts with a model precision experiment.

State EEF forward gains, coordinates 0–8: `21.1145, 10.9642, 36.3641, 101.6841, 5.1001, 484316.4841, 5.1001, 101.6841, 2000000`. Relative-action feature maxima across horizons: `612.9576, 645.2610, 1293.0199, 152514.6506, 252.0907, 484316.4841, 252.0907, 156322.6795, 2000000`. State rotation coordinate8 reaches the 1e-6 floor; relative coordinate8 does so at every horizon. Constant gripper also has a large gain, but its observed constancy does not support blaming amplified gripper motion.

For example, TRAIN M1/c0/first quartile has state abs-p95 about119929 for X tilt and29042 for Y tilt; nominal and Z tilt stay around unit scale. At horizon39, X relative rotation coordinate8 is about117502. These are actual project measurements. Full task×command×variant×quartile, feature×horizon rows remain in [CODEC_TRAIN_STRATA.csv.gz](CODEC_TRAIN_STRATA.csv.gz), including benign dimensions and all failed cases. [CODEC_FEATURE_BOUNDS.csv](CODEC_FEATURE_BOUNDS.csv) records q01/q99/min/max/effective width/floor and both gains. [ABSOLUTE_ACTION_UNUSED_EEF.csv](ABSOLUTE_ACTION_UNUSED_EEF.csv) explicitly labels the unused absolute EEF route.

## Nonlinear conversion, arithmetic and sensitivity

TRAIN Gram–Schmidt first-vector and orthogonalized second-vector norms never fall below0.9999999404 for state/action, or0.9999999999999997 for relative action. Maximum observed row orthogonality errors are below3.4e-8 (FP32 inputs) and5.3e-17 (relative FP64); components lie within[-1,1] up to2.3e-16 roundoff. There is no measured near-zero TRAIN denominator to explain the scale blow-up. See [NUMERIC_SUPPLEMENT.json](NUMERIC_SUPPLEMENT.json) and [GRAM_AND_CAST.json.gz](GRAM_AND_CAST.json.gz).

{cast_table}

These are arithmetic roundtrips of stored numeric tensors, **not a model precision ablation**. A small inverse error does not make a 1e5 learner input or target well-conditioned.

Fixed function probes use the first metadata-ordered TRAIN item in each task/command/variant cell, exactly1e-7 and1e-6 metres/radians on each axis (336 probes). State normalized maximum changes: translation3.70e-6 /3.64e-5; world-left rotation0.203125 /2.0. Original absolute future labels are held fixed to isolate anchor-conversion sensitivity; no probes become training data or rendered observations. All values, including relative-target changes, are in [FIXED_CODEC_SENSITIVITY.csv](FIXED_CODEC_SENSITIVITY.csv). The nearly constant toy fixture is separately labelled constructed.

## Limits

Per-module neural activations are UNOBSERVED. No live normalizer, target, decoder, clipping rule or epsilon changed. Retrospective EVAL ranges are isolated in the trace tables and were not used to fit bounds. The proposed theoretical rotation bounds depend on the verified representation and do not apply to XYZ or arbitrary deltas.
''')
    exp_table=table(['arm','microbatches','distinct input conditions','distinct raw sources','distinct RGB pairs','state / label / language identities','available but unconsumed','terminal / near-terminal microbatches'],[
        [r['arm'],8192,exp['summary'][r['arm']]['distinct_conditions'],exp['summary'][r['arm']]['distinct_raw_sources'],r['distinct_images'],f"{r['distinct_states']} / {r['distinct_labels']} / {r['distinct_language']}",r['eligible_unconsumed'],f"{exp['summary'][r['arm']]['terminal_microbatches']} / {exp['summary'][r['arm']]['near_terminal_microbatches']}"] for r in supp])
    loss_table=table(['late-phase arm/variant','microbatches','median native loss','median target squared magnitude','median accumulated gradient norm'],[
        [r['arm']+'/'+r['variant'],r['microbatches'],f"{float(r['native_loss_median']):.5g}",f"{float(r['normalized_target_mse_median']):.5g}",f"{float(r['gradient_accumulated_norm_median']):.5g}"] for r in opt if r['phase']=='late'])
    trace_table=table(['M1 point / variant','rows / REJECT','median first S4 XYZ velocity absmax','median final normalized XYZ absmax','median inverse-relative displacement (m)','median prefix / full H40 XYZ RMSE (mm)'],[
        [r['point']+'/'+str(r['variant']),f"{r['rows']} / {r['REJECT']}",f"{r['trace_v0_xyz_absmax_median']:.4g}",f"{r['normalized_output_xyz_max_abs_median']:.4g}",f"{r['inverse_relative_xyz_max_norm_median']:.4g}",f"{r['prefix_xyz_rmse_m_median']*1000:.2f} / {r['whole_xyz_rmse_m_median']*1000:.2f}"] for r in trace['groups'] if r['task']=='M1' and (r['point']=='R4096' or r['variant'] in (1,4))])
    write('TRAIN_EXPOSURE_AND_OPTIMIZATION.md',f'''# PI0268 — actual exposure and saved optimization

## Actual consumption

All counts join actual sample logs to the unchanged source rows, not schedule length. 16,384 microbatches, 8,192 per arm, align with native/full loss records; 4,096 gradient records per arm align at the two-microbatch update boundary. Input identity includes the actual language and processed images/state. Numeric label duplicates and language repeats are retained.

{exp_table}

Terminal means original anchor15; near-terminal means anchors12–15. Each arm has384 terminal microbatches. R divides these into192 nominal and192 recovery: each recovery variant has32 terminal microbatches across tasks/commands. Neither absent terminal exposure nor infinite terminal stabilization training is supported. The training target is still local H40; repeated hold/late drift remains a separate closed-loop question.

N repeats exact condition identities2/4/6 times (2048/256/512 distinct conditions). R repeats1/2/3 times (6144/256/512). All10752 proposals were eligible; N sees1536 sources, R5632. R recovery source coverage by task is342 of768 for variants1–4 and340 of768 for variants5–6. Full family/task/anchor/variant/template joint cells and unconsumed denominators are retained in [EXPOSURE.json](EXPOSURE.json), [EXPOSURE_SUPPLEMENT.json](EXPOSURE_SUPPLEMENT.json), and the exact [consumed-condition index](CONSUMED_CONDITION_INDEX.json.gz). These identities do not establish how well either model fits those examples.

## Saved loss and gradients

Early/middle/late bins are fixed updates1–1365,1366–2730,2731–4096. Full joined native, position, increment, rotation and speed-margin losses, target magnitudes and clipping indicators are in [OPTIMIZATION_JOIN.csv.gz](OPTIMIZATION_JOIN.csv.gz) and [OPTIMIZATION_PHASE_VARIANT.csv](OPTIMIZATION_PHASE_VARIANT.csv). Stored physical loss values are reported as loss values, not mislabelled metre RMSE. Gradient norms belong to accumulated updates shared by two microbatches, not independent per-example gradient attribution.

{loss_table}

The native target is normalized action minus Gaussian epsilon; mask-weighted squared error is divided by active-mask sum. Thus target normalization changes effective native supervision weighting even if the declared loss coefficients are unchanged. Actual native epsilon values were not saved, so target squared magnitude is a scale proxy, not reconstruction of the exact sampled velocity target.

N clips4034/4096 updates; R clips4096/4096. Maximum accumulated norms are72.8648 and136810800. Median absolute step-to-step norm changes are0.6757 and842.9844; R's p95 change is1.077e8. Per-module gradients/activations and full training Euler tensor values are **UNOBSERVED** (training logs retain Euler hashes/dtypes and scalar losses). No fresh gradients were obtained. See [OPTIMIZER_STEP_SUMMARY.json](OPTIMIZER_STEP_SUMMARY.json).

## Saved inference localization — retrospective EVAL only

All2088 old offline rows are retained, including every REJECT and the440 reused donor outputs. Each row includes raw noise, each of four stored S4 x/v stages, final normalized XYZ/rotation, inverse-relative representation and composed physical motion. First-point and segment errors are retained. Matching legacy reference hashes permit all2088 prefix-vs-H40 comparisons; [LEGACY_PREFIX_COMPLETION.json](LEGACY_PREFIX_COMPLETION.json) records the targeted completion of880 legacy columns without model calls.

{trace_table}

Full M1/M3/legacy and all variants are in [OFFLINE_TRACE_SUMMARY.json](OFFLINE_TRACE_SUMMARY.json); all rows and gates are in [OFFLINE_TRACE_ROWS.csv](OFFLINE_TRACE_ROWS.csv). Representatives are the first UID per point/task/variant, chosen by metadata rather than outcome. The initial noise stays around unit scale, while R X/Y XYZ velocity is already elevated at the first stored model velocity. Later S4 stages further grow normalized XYZ. Inverse-relative translation and post-anchor displacement norms agree closely: rigid anchor composition does not create the metre-scale magnitude. This localizes the observed growth upstream of final physical composition; it does not identify which learned module caused it.

**Counterevidence to a sufficient-cause claim:** donor and N receive the same very large X/Y state values without R's gate failures. R also improves virtual M1 first-hit attainment. No genuinely seen-TRAIN neural fit panel has run. A numerical intervention must still be evaluated against these alternatives.

The existing axis/gate and paired findings are reused from basis commit7023316d912146e8219b57aae7e938982f3ffee7, especially `OFFLINE_VARIANT_GATE_BREAKDOWN.json`, `CLOSED_RESULTS_PAIRED_ANALYSIS.json` and `OFFLINE_STAGE_RETURN.md`. N improves60/128 paired recovery rows; median paired delta is+0.404mm, despite its smaller aggregate recovery median. No aggregate median is substituted for the paired comparison.
''')
    closed_table=table(['point','task/backend','episodes','original first-hit successes','collision episodes','REJECT','requested / installed / achieved entry episodes','successful but outside at end'],[
        [r['point'],r['task']+'/'+r['backend'],r['episodes'],r['success'],r['collisions'],r['REJECT'],f"{r['requested_entry_episodes']} / {r['installed_entry']} / {r['achieved_entry']}" if r['task']=='M3' else 'N/A',r['successful_final_outside']] for r in closed['groups']])
    prefix_table=table(['point','task/backend','replans','median first-point error (mm)','median 0.4s prefix motion (mm)','median first SO3 / prefix SO3 (rad)','median nearest nominal / recovery support distance'],[
        [r['point'],r['task']+'/'+r['backend'],r['queries'],f"{r['median_first_error_m']*1000:.3f}",f"{r['median_prefix_motion_m']*1000:.3f}",f"{r['median_first_so3_rad']:.5f} / {r['median_prefix_so3_rad']:.5f}",f"{r['median_nominal_support_distance']:.3f} / {r['median_recovery_support_distance']:.3f}"] for r in closed['groups']])
    collision_table=table(['point/case','collision time (s)','precursor first-point / prefix (mm)'],[
        [e['point']+'/'+e['case'],f"{c['time_s']:.2f}",f"{c['preceding_query']['first_error_m']*1000:.3f} / {c['preceding_query']['prefix_motion_m']*1000:.3f}"] for e in episodes for c in e['collision_events']])
    write('CLOSED_LOOP_GAP_REPORT.md',f'''# PI0268 — completed virtual point gap decomposition

## Scope and unchanged outcome

Entry snapshot closed donor/N/R:64/64 each. This analysis reads all192 completed episodes and18,988 existing predictions (donor6400, N6210, R6378). Result/card/trajectory/query/observation hashes are bound in [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) and [CLOSED_SOURCE_HASHES.json.gz](CLOSED_SOURCE_HASHES.json.gz). No episode or V3 scorer was rerun or changed. Main-agent WBC work is independent and is not claimed complete here.

{closed_table}

R's virtual M1 first-hit attainment is30/32, higher than N26/32 and donor13/32; that does not erase R's offline recovery failures or its M3 collision. Every successful M1 episode in every point ends outside the original pose bounds. All96 M3 episodes fail entry in installed and achieved motion. Three donor M3 episodes contain an unexecuted H40 that independently requests a qualifying entry, yet their actual installed prefixes never acquire it. Requested future, installed prefix and achieved entry are separate events.

## Exact executed-prefix decomposition

For every saved query, `first=prediction[0]-actual_now`, `prefix=prediction[8]-prediction[0]`, `total=prediction[8]-actual_now`; index8 is0.4s on the original0.05s H40 grid. The maximum `total-first-prefix` absolute residual is{closed['max_prefix_identity_abs']:.3g}m. SO3 first-point and relative-prefix geodesics are reported separately, never added as global rotation vectors. Gate REJECT rows would remain in the denominator; there are none in these actual closed-loop inputs.

{prefix_table}

Small first-point errors do not imply appropriate progression or retention. For N M3, the median first mismatch is about0.13mm and prefix movement2.1–2.2mm, yet no requested or achieved entry occurs. E0 matches installed motion exactly; deficient primary motion therefore exists without WBC tracking error. E1 lag is a separate executor.

[CLOSED_PREFIX_ROWS.csv.gz](CLOSED_PREFIX_ROWS.csv.gz) records each target-distance change, requested entry, same-tick reference/actual error and support distances. [CLOSED_EPISODE_ROWS.json](CLOSED_EPISODE_ROWS.json) records first2s approach progress, original first hold, first subsequent exit, post-hold path length, integrated excess distance, maximum drift and original collision events. For M3, approach distance is to the original entry; final-goal distance in the plot is explicitly different. No minima at different ticks are subtracted.

Nearest support uses the exact same-task/command TRAIN minimum of `sqrt((XYZ/.010m)^2+(SO3/.060rad)^2)`, separately for nominal and recovery. The scale is the pre-existing perturbation magnitude, not a tuned OOD cutoff. This is task-space proximity, not RGB or language equivalence. N M3 can stay within roughly one such unit of nominal support and still fail, so support distance alone does not explain the failure. M1's large late distances partly reflect recorded post-hold drift; they cannot establish that OOD preceded or caused it.

Prefix-vs-full-H40 **label** RMSE is reported on matched saved offline rows in the trace report. Closed-loop distance to target is a different diagnostic and is not relabelled teacher trajectory RMSE. No teacher corrects any executed model output.

## Collisions and retention remain failures

{collision_table}

Every precursor row retains the actual requested reference, original ACCEPT decision, and collision tick linkage in the episode JSON. Near-zero first-point mismatch did not prevent these collisions. No smoothing, return-to-goal controller, early stopping, late-failure removal or new success threshold is introduced.

![Metadata-fixed family028 command0 E0 trajectories](FIXED_CLOSED_TRAJECTORIES.png)

Plot selection is fixed by family/command/backend, not scores. Vertical lines mark original first-hit times. These are saved virtual EEF arrays, **not WBC, physical episodes, a second robot, realtime execution or general transfer evidence**.
''')
    write('HYPOTHESIS_MATRIX.md','''# PI0268 — evidence, counterevidence and one next comparison

| Hypothesis | Supporting evidence | Counterevidence / uncertainty | Falsifiable next observation |
| --- | --- | --- | --- |
| Rotation normalization damages recovery learning (highest preparation priority) | Actual TRAIN inputs/targets reach1e5; X/Y native losses stay1e7–1e9; R accumulated gradients reach1e8; R X/Y output growth begins in saved S4 velocity. | Donor/N tolerate the same large input state values. State, target weighting and neural adaptation are coupled. No seen-TRAIN inference or intervention exists. | Same donor/raw R visitation with old versus representation-bounded rotation maps, per-contract step0 and final; retain all original gates/negative strata. |
| Insufficient independent recovery support | At each anchor the coupled ±translation/rotation perturbations span rank3, not6; only5632/10752 raw sources consumed by R. | Same combined-variant EVAL can already fail catastrophically; rank alone does not explain that. Large state/target magnitudes exist on actual TRAIN. | Proposed112 seen inputs distinguish fit from transfer; proposed304 consistent factors separate translation/rotation while updating dependent RGB. |
| Local objective misses entrance/retention | All96 M3 virtual episodes miss achieved entry; all69 successful M1 episodes later leave original pose bounds. First-point errors are sub-mm and E0 tracks installed motion exactly. | R increases M1 first-hit success; terminal examples exist. M3 can remain close to nominal TRAIN support. | Later separately authorized prefix/terminal experiment with current V3 and full late failures retained; not combined with the normalization comparison. |
| WBC explains all primary failures | Secondary physical executor has its own transfer limitations. | E0 has the same primary M3 failure with exact installed tracking; primary virtual diagnosis needs no WBC explanation. | Main lane's independent secondary WBC results, without blocking this preparation. |
| Cross-host label bytes caused R failure |1359 normalized-label byte differences are real historical evidence. | Existing physical reconciliation passes; numeric magnitudes and losses are much larger than disclosed inverse errors. | Retain byte/runtime receipts; do not manufacture causal attribution or redo training for records. |

Only **one** training draft is recommended: the combined state-rotation + relative-action-rotation normalization treatment. Each six-component block is treated coherently as the verified rotation6D representation; XYZ, gripper, unused absolute-action EEF statistics, labels, timing and loss formulas are unchanged. This includes stable components within the rotation blocks and cannot establish a single-coordinate, state-only or action-only mechanism. A coordinate-subset search is not proposed.

Theoretical[-1,1] bounds follow from orthonormal rows, verified on actual TRAIN. They are not percentiles fitted to EVAL. Changing target normalization changes effective native-loss weighting; changing state and decoder changes the initial effective policy. Both facts are part of the treatment. This is not an inference-time repair of R.

No second coverage or terminal training arm is prepared. Those alternatives remain hypotheses. The112 TRAIN and304 factor panels are NOT_AUTHORIZED and supply no observed fit evidence today. Old NEW_SIM_EVAL families are development evidence after being examined; they cannot become blind TEST again.
''')
    own=[r for r in exposure if r['arm']=='R' and r['unit']<=2048]
    draft=dict(schema='PI0268_NEXT_RUN_DRAFT_V1',status='NEXT_RUN_DRAFT',launch_allowed=False,approval_status='NOT_AUTHORIZED',
        missing_authorization=['Separate immutable scientific release and Issue11 GO for diagnostics/rendering and the selected comparison','Final main PI0267 outcome must be read at launch decision','Prospective run-card approval of exact proposed costs, gates and independent-generalization scope'],
        donor=dict(repo='leoyuer/pi0246-gate-pass-campaign',revision='e4290e6ca387abd2de4337b5f12b9c72a5f85c99',prefix='arms/PI0264/L/checkpoint-2048',tree_sha256='2f61d8ed8c8fe8444b7ef23164cb4df2bcacd8a73c93f19be77fcdf459ef6fae'),
        comparison='NORMALIZATION_ONLY_COMBINED_ROTATION_BLOCKS',arms=['CONTROL_OLD','TREAT_ROT6D_BOUNDS'],
        one_factor_diff=dict(control='Exact old processor/decoder numerical contract',treatment='Only min/q01=-1,max/q99=1 for state.eef_9d[3:9] and relative_action.eef_9d[:,3:9]',
            untouched=['XYZ affine','gripper','unused absolute-action EEF map','raw labels','S4/loss formulas and coefficients','camera/RGB','teacher/timing','trainable names and precision','gate/scorer']),
        policy_warning='Identical weights under different processors are different initial policies. Native target scaling changes effective native-loss weighting. No R inference-only swap.',
        data=dict(plan_sha256=snapshot['source_identities']['/home/liuyu-nfs/Projects/go2-arx5/PI0267/R1/design/PAIRING_PLAN.json']['sha256'],
            selection='Original R indices/language/order for units1..2048, identical both arms; no sample replacement',updates_per_arm=2048,microbatches_per_update=2,examples_per_arm=len(own),distinct_input_conditions=len({r['condition_sha256'] for r in own}),distinct_raw_sources=len({(r['family'],r['index']) for r in own}),source_families=24),
        optimizer=dict(fresh=True,LR=1e-5,scheduler='cosine2048',warmup_steps=10,weight_decay=1e-5,max_grad_norm=1.,trainable_scope='Exact PI0267 sealed parameter-name/freeze contract; verify against REUSED_EXECUTION_FILESET before any approved load'),
        randomness=dict(seed=26741,unit_seed='sha256(PI0267:TRAIN:26741:{unit}) first8 little modulo2**31',native_and_S4='Paired actual packets; preserve exact draw order and audit actual bytes',no_new_seeds=True),
        save_resume=dict(save_updates=[1024,2048],complete_trainers=4,components=['model shards/index/config','optimizer','scheduler','Trainer state/global step','Python/NumPy/CPU/CUDA RNG','processor/statistics','actual sample/noise/loss/update audit'],
            exact_resume='Resume same arm/contract and cursor at checkpoint; restore RNG after zero-worker DataLoader iterator creation before first microbatch, as sealed PI0267 lifecycle. Verify complete state, never configuration-only resume.',
            preservation='Persistent complete source until fixed HF PUBLIC revision full-file readback; no tmpfs-only recovery source; no broad cleanup',
            canary='Proposed actual8 continuous vs4+complete-save/restore+4 per arm, separately authorized; compare actual inputs/noise/gradients/final weights/optimizer/scheduler/RNG. NOT RUN.'),
        evaluation=dict(points='Per-contract donor step0 and final2048 only;1024 inventory only',
            panels='112 seen TRAIN at two fixed packets +304 consistent TRAIN factors at one fixed packet +unchanged440 prior-domain retention rows',
            tentative_rows_per_point=968,tentative_contract_points=4,tentative_queries=3872,tentative_attempt_cap=4064,tentative_S4_head_cap=16256,
            proposed_final_virtual_episodes=128,virtual_scope='Existing64 frozen conditions per final,0.4s feedback, originalV3/V2 and gates, all late failures retained; not WBC or new independent TEST',
            model_queries_allocated=0,optimizer_updates_allocated=0,render_pairs_allocated=0,
            separate_diagnostic_panels=dict(planned=1584,attempts=1664,heads=6656,frame_pairs=304,status='NOT_AUTHORIZED'),
            no_double_count='Budget must deduplicate shared exact point/contract/input/noise tuples at authorization; no budget is granted here'),
        decision_plan=['Do not compare normalized loss values as equal-scale quality across contracts; use physical metrics and original gate failures per task/command/variant/noise.',
            'Separate treatment step0 effect from final-minus-own-step0 learning; report both arms regardless of outcome.',
            'A preparation-local mechanistic contrast is positive only if treatment reduces each M1/M3 X/Y rejection count and each corresponding median physical XYZ error versus control without any increase in legacy rejection count or legacy median XYZ/SO3; report all strata, no aggregate rescue. This proposed decision rule needs future scientific approval and grants no deployment status.',
            'Nonfinite output, REJECT, collision and late drift remain valid failures. No quality retries, seed/candidate switching, tuning or1024 selection.',
            'Integrity failure pauses only affected lane for same-state repair; preserve raw budget. Valid scientific failure stops that claim, not the other preauthorized arm.',
            'No independent-generalization or physical-transfer claim without a later preregistered authorization.'],
        dependencies=dict(statistics_sha256=audit['source_statistics_sha256'],TRAIN_source_index='TRAIN_SOURCE_IDENTITIES.json',consumed_index='CONSUMED_CONDITION_INDEX.json.gz',runtime_seal='REUSED_EXECUTION_FILESET.json',donor='REUSED_DONOR_BINDING.json',noise='PROPOSED_EXISTING_NOISE_IDENTITIES.json',panels=['SEEN_TRAIN_112_MANIFEST.json','FACTOR_304_MANIFEST.json']),
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    (OUT/'NEXT_RUN_DRAFT.json').write_text(json.dumps(draft,indent=2)+'\n')
    write('NEXT_RUN_DRAFT.md','''# One proposed comparison — NOT_AUTHORIZED

`launch_allowed=false`. [NEXT_RUN_DRAFT.json](NEXT_RUN_DRAFT.json) is the complete proposed design and dependency manifest. No optimizer, canary, query, renderer, physics, WBC or PBS budget has been granted by PI0268.

Use the same fixed original L2048 donor, not R. CONTROL keeps the old numerical contract; TREAT changes only the verified state and relative-action rotation6D bounds to theoretical[-1,1]. This is a combined normalization-policy intervention. Effective native target weighting changes; identical donor weights are not identical effective step0 policies.

Tentative two arms×2048 updates, two microbatches/update, identical first2048 original R visitation units:4096 examples per arm,3840 distinct conditions and3584 raw sources. Fresh matched optimizer/cosine, unchanged LR1e-5, loss formulas, trainable scope, precision and actual RNG plan. Complete1024/2048 saves per arm; only final quality. Actual8-vs4+save/restore+4 canaries, all training and all evaluation remain unallocated.

Proposed per-contract step0/final panels include112 TRAIN×two packets,304 factor conditions and440 prior-domain rows:968 per point, four points3872 planned queries. The separate three-existing-model diagnostics cost1584 planned/1664 attempts/6656 heads and304 future consistent frame pairs. Overlapping exact tuples must be deduplicated in the later approved run card. Any proposed final virtual follow-up retains original64 conditions per arm; it is not physical WBC. No selection or independent TEST claim follows.

The pure `proposed_rotation_contract` function is a code sketch. It deep-copies input metadata and leaves XYZ/gripper/absolute-action statistics untouched; it is never installed in the active runtime. The executable entry only validates metadata with `--dry-run`. Every launch path raises NOT_AUTHORIZED before any model import; a flag or edited boolean cannot authorize this preparation release. A later approved scientific release must supply its own audited launch integration.

No coverage expansion, prefix loss, terminal loss, adapter, new teacher or alternative seed is bundled into this comparison.
''')
    write('PREPARATION_RETURN.md',f'''# PI0268 preparation complete

**PI0268_PREPARATION_COMPLETE_AWAITING_SEPARATE_SCIENTIFIC_LAUNCH**

Authority: release80554763aaeeecdc631da8c5ded96074884ff434 / Issue11 comment5955686136. This isolated preparation does not complete or take ownership of the main PI0267 lane. selected/deployment remain **NONE**.

## Findings

1. Actual old statistics map X/Y TRAIN recovery rotation into up to~120k inputs/targets. Roundtrip is accurate and Gram–Schmidt is well-conditioned; reversibility is not a conditioning guarantee.
2. Actual R late X/Y native losses remain~4e7–8e8; accumulated gradient norms reach1.37e8 and all4096 updates clip. Donor/N tolerate the same amplified input state, so the causal mechanism remains unproved.
3. All2088 saved offline traces and192 closed virtual episodes were analyzed. R M1 original first hits30/32, M3 0/32, one collision; donor/N M1 first hits13/32 and26/32. All69 successful M1 runs later finish outside original pose bounds. Original qualification and late failures are preserved.
4. 112 truly consumed TRAIN inputs and304 factor specifications are ready as manifests only. No altered state was paired with old RGB. Existing packet bytes/keys were verified. Planned1584 queries/304 frame pairs are **NOT_AUTHORIZED**.
5. Recommend one combined state/action rotation normalization comparison, retaining all raw data, loss formulas and other contracts. Per-contract step0 is essential. [NEXT_RUN_DRAFT.json](NEXT_RUN_DRAFT.json) remains `launch_allowed=false`, `NOT_AUTHORIZED`.

## Deliverables

- [Codec conditioning](CODEC_CONDITIONING_REPORT.md), feature/horizon tables and fixed sensitivity probes.
- [Actual exposure, optimization and saved S4 traces](TRAIN_EXPOSURE_AND_OPTIMIZATION.md), complete consumed index and hashes.
- [Closed-point prefix, entry, retention and safety](CLOSED_LOOP_GAP_REPORT.md), all18,988 query rows and fixed-array figure.
- [Hypotheses and counterevidence](HYPOTHESIS_MATRIX.md), [112 seen inputs](SEEN_TRAIN_112_MANIFEST.json), [304 factors](FACTOR_304_MANIFEST.json), [single training draft](NEXT_RUN_DRAFT.md).
- [CPU checks](CPU_CHECKS.json):{len(checks['checks'])} positive/negative checks, including four real processor/collator calls with exact existing input identities, wrong units/frame/labels, missing/duplicate/heldout inputs, preserved gate rejection, partial-point exclusion and denied model/launch paths. **Static/processor checks, not a model canary.**
- [Source/concurrency/resource snapshot](SOURCE_SNAPSHOT.json), [original log identities](TRAIN_LOG_SOURCE_FILES.json), [file inventory](DELIVERY_INVENTORY.json).

## Resources and omissions

No new model or gradient queries, optimizer updates, renderer/physics/WBC calls or PBS submissions. No full model copied. One finite CPU analysis at a time, CUDA hidden, four-thread library caps, nice19/ionice3. The first process's threads were pinned to four cores after startup (before/after receipt retained); later processes inherit four-core affinity at startup. Peak observed RSS{snapshot['peak_observed_RSS_bytes']/2**30:.3f}GiB; working data at collection~{snapshot['resource']['working_bytes']/2**20:.1f}MiB, derived numeric data~{snapshot['resource']['derived_bytes']/2**20:.1f}MiB. No GPU evaluator process touched. Small hash/source reads on Hopper only.

Completed24-family source arrays were streamed once into a small CPU numeric cache. Subsequent arithmetic uses that cache; four bounded processor fixtures use their matching original frames. The880 missing legacy prefix columns were finished from hash-matched existing labels/outputs. Original failed analysis logs remain; no scientific query was repeated for a report error.

UNOBSERVED: neural fit on the proposed112 TRAIN inputs, per-module activation/gradient attribution and unsaved training Euler/epsilon tensor values. Main secondary WBC status is not polled or taken over. No normalization was fitted to EVAL, and examined EVAL remains development evidence. No live statistic, checkpoint, scorer, gate, original output or selected status changed.

Large plain metadata tables are losslessly gzip-compressed; inventory includes both stored and uncompressed hashes. Use `gzip.open` to inspect. No weights or image corpus are in this return.
''')
    inventory=[]
    for p in sorted(OUT.iterdir()):
        if not p.is_file() or p.name=='DELIVERY_INVENTORY.json':continue
        body=p.read_bytes();raw=gzip.decompress(body) if p.suffix=='.gz' else body
        inventory.append(dict(file=p.name,stored_bytes=len(body),stored_sha256=hashlib.sha256(body).hexdigest(),uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
    (OUT/'DELIVERY_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print('REPORTS_COMPLETE',len(inventory))


if __name__=='__main__':main()
