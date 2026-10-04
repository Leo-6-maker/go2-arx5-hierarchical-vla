"""Render the fixed saved-array results; stdlib only, no model execution."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


def main(repo):
    out=repo/'reports/PI0272/final'
    def read(name):return json.loads((out/name).read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    rows=list(csv.DictReader((out/'ALL_POINTS3504.csv').open()))
    pairs=list(csv.DictReader((out/'DIRECT_VS_STOCHASTIC_PER_CONDITION.csv').open()))
    assert len(rows)==3504 and len({(r['point'],r['panel'],r['input_id'],r['j']) for r in rows})==3504
    cpu=read('CPU_REPLAY.json')
    for name,r in cpu['tables'].items():assert sha(out/name)==r['sha256']
    assert read('QUERY_OUTPUTS_COMPLETE.json')['actual_attempts']==2148
    def choose(**fields):
        return [r for r in rows if all(r[k]==v for k,v in fields.items()) and
                (r['panel']=='early_guard' or r['j'] in ('','0','1'))]
    def med(rs,k):return median(float(r[k]) for r in rs if r.get(k))
    def table(headers,values):
        return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers),
                         *['| '+' | '.join(map(str,v))+' |' for v in values]])
    def quality(rs):
        return [len(rs),f"{med(rs,'prefix_xyz_rmse_m')*1000:.3f}",f"{med(rs,'prefix_so3_rmse_rad'):.6f}",
                f"{med(rs,'xyz_rmse_m')*1000:.3f}",f"{med(rs,'so3_rmse_rad'):.6f}",
                f"{sum(float(r['direction_cosine'])<0 for r in rs if r['direction_cosine'])}/{len(rs)}",
                sum(r['gate']=='REJECT' for r in rs)]
    points=('AGG4096','DIRECT0','S4FIT1024','DIRECT1024')
    primary=[(p,choose(point=p,panel='support',bucket='M3_TERMINAL')) for p in points]
    primary_table=table(['Policy','Outputs','Prefix XYZ mm','Prefix SO3 rad','Full XYZ mm','Full SO3 rad','Wrong direction','REJECT'],
                        [[p,*quality(rs)] for p,rs in primary])
    breakdown=[]
    for family in sorted({r['family'] for r in primary[0][1]}):
        for command in ('0','1'):
            for p in points:
                rs=choose(point=p,panel='support',bucket='M3_TERMINAL',family=family,command=command)
                breakdown.append([family,command,p,*quality(rs)])
    primary_groups=table(['Family','Command','Policy','Outputs','Prefix XYZ mm','Prefix SO3 rad','Full XYZ mm','Full SO3 rad','Wrong direction','REJECT'],breakdown)
    comparison=[]
    for before in ('AGG4096','S4FIT1024'):
        for task,bucket in sorted({(r['task'],r['bucket']) for r in pairs}):
            rs=[r for r in pairs if r['before']==before and r['task']==task and r['bucket']==bucket and r['K']=='2']
            comparison.append([before,task,bucket,len(rs),f"{med(rs,'delta_prefix_MSE_m2'):.9g}",sum(float(r['delta_prefix_MSE_m2'])<0 for r in rs)])
    pair_table=table(['S4 comparator','Task','Bucket','Conditions','Median per-condition DIRECT minus mean S4 MSE (m2)','DIRECT lower MSE conditions'],comparison)
    guard=[];guard_pairs=[]
    for task in ('M1','M3'):
        for command in ('0','1'):
            for bucket in ('ORIGINAL_Q0','ORIGINAL_Q1'):
                for p in ('AGG4096','S4FIT1024','DIRECT1024'):
                    rs=choose(point=p,panel='early_guard',task=task,command=command,bucket=bucket)
                    guard.append([task,command,bucket,p,*quality(rs)])
        index={(r['point'],r['input_id']):r for r in choose(panel='early_guard',task=task)}
        ids={r['input_id'] for r in choose(point='DIRECT1024',panel='early_guard',task=task)}
        for before,after in (('AGG4096','S4FIT1024'),('AGG4096','DIRECT1024'),('S4FIT1024','DIRECT1024')):
            changes={k:[float(index[after,i][k])-float(index[before,i][k]) for i in sorted(ids)] for k in ('prefix_xyz_rmse_m','prefix_so3_rmse_rad','xyz_rmse_m','so3_rmse_rad')}
            guard_pairs.append(dict(task=task,before=before,after=after,conditions=len(ids),
                median_paired_delta={k:median(v) for k,v in changes.items()},
                worse_conditions={k:sum(x>0 for x in v) for k,v in changes.items()}))
    guard_table=table(['Task','Command','Bucket','Policy','Outputs','Prefix XYZ mm','Prefix SO3 rad','Full XYZ mm','Full SO3 rad','Wrong direction','REJECT'],guard)
    guard_delta=table(['Task','Before','After','Conditions','Paired prefix XYZ delta mm','Paired prefix SO3 delta rad','XYZ worse conditions'],
        [[r['task'],r['before'],r['after'],r['conditions'],f"{r['median_paired_delta']['prefix_xyz_rmse_m']*1000:.4f}",
          f"{r['median_paired_delta']['prefix_so3_rmse_rad']:.7f}",r['worse_conditions']['prefix_xyz_rmse_m']] for r in guard_pairs])
    banks=read('K8_BANKS192.json')['banks'];assert len(banks)==192
    task_by_id={r['input_id']:r['task'] for r in rows};k8=[]
    for p in ('AGG4096','S4FIT1024'):
        for task in ('M1','M3'):
            own=[b for b in banks if b['point']==p and task_by_id[b['input_id']]==task]
            assert len(own)==48 and all(b['status']=='FINITE_FIXED_K8_DECOMPOSITION' for b in own)
            v=[b['regions']['prefix'] for b in own]
            k8.append([p,task,len(v),f"{median(x['MSE_m2'] for x in v):.9g}",
                f"{100*median(x['empirical_mean_error_squared_m2']/x['MSE_m2'] for x in v):.3f}",
                f"{median(x['within_bank_dispersion_m2'] for x in v):.9g}"])
    trainers=[]
    for arm in ('S4FIT','DIRECT'):
        for step in (512,1024):
            r=json.loads((repo/f'reports/PI0272/archives/{arm}/checkpoint-{step}.full_readback.json').read_text())
            assert r['status']=='PASS_FULL_FILE_READBACK' and len(r['files'])==25
            trainers.append([arm,step,r['revision'],r['tree_sha256']])
    archive=read('RESULT_ARCHIVE_predictions.json')
    resource=json.loads((repo/'reports/PI0272/r1-continuation/TRAINING_RESOURCE_ACCOUNTING.json').read_text())
    assert resource['total_updates']==2080 and resource['total_heads']==24987
    summary=dict(status='MATCHED_FITTING_COMPLETE_LOCAL_DIRECT_ADVANTAGE_NOT_ESTABLISHED',
        primary=read('FITTING_SUMMARY.json')['primary_M3_terminal'],guard_paired=guard_pairs,
        source_tables={name:sha(out/name) for name in cpu['tables']},renderer_sha256=sha(Path(__file__)),
        execution_complete_subject_to_terminal_publication=True,new_task_qualification=False,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',new_model_queries_by_report=0)
    (out/'RESEARCH_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',newline='\n')
    text=f'''# PI0272 R1 — matched direct readout fitting: complete fixed results

## Conclusion

Both registered readouts learn the consumed corrective-reference support. **The predeclared local DIRECT advantage over the matched trained S4 control is not established.** On the primary 24 M3 terminal conditions, both fits remove all opposite-direction outputs under the unchanged reporting guard. DIRECT's median per-condition prefix MSE minus mean S4FIT MSE is **+4.59272529e-7 m2**, with lower MSE on only10/24 conditions. DIRECT is better than old AGG on23/24, but S4FIT also improves strongly. This is counterevidence to attributing the residual repair uniquely to a direct head.

DIRECT has lower marginal terminal prefix/full SO3 and full XYZ medians, while the primary registered paired MSE criterion is worse. Its slightly lower marginal prefix XYZ median must not be substituted for that paired criterion. The primary direction rates tie at zero, rather than favoring DIRECT. Early-support XYZ worsens relative to AGG in both tasks for both fits; DIRECT additionally worsens M1 guard XYZ/SO3 relative to S4FIT. The registered combination of primary position, direction, rotation, gate and guard conditions therefore does not establish superiority.

Execution completion and scientific advantage are separate. These are developed TRAIN-support fitting results, not task completion, blind TEST, generalization, WBC or physical transfer. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.

## Authority, source reuse and actual execution

- Base release `e62e6a469dce1f4d3df5d49b38bad4a3c627e969`, Issue11 canonical5979236384; R1 release `8a2ffbefe0c57536c0a7e340da0616c7987c74f1`, canonical5980876835. Same branch `pi0267/executor-decoupled-recovery`, no new science goal or budget reset.
- Closed PI0271 terminal5978753654 / closeout90489ca reused; AGG4096 donor HF revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`, prefix `arms/PI0270/AGG/checkpoint-4096`, tree `0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66`. Old valid queries and archive receipts were reused.
- Exactly296 designated consumed conditions: M1 Q2/Q3 48/48, terminal32; M3 Q2/Q3 48/48, entry/midarc/terminal24 each. 24 TRAIN families overall; components only two families. Actual per-arm2048 microbatches match the CPU plan:272 conditions seven times and24 six times. NEWFIT192 has zero exact condition/label overlap. No EVAL label entered training.
- Both real continuous8 versus4+full-save/fresh-resume4 canaries PASS:1030 model tensors and1413 optimizer tensors have maximum difference0.0; cursor, scheduler and all saved RNG states match. All4096 main microbatches match source inputs, labels and paired native epsilon/tau packets. Both complete1024 updates from the same donor with fresh optimizer; full512/1024 Trainers retained and read back.
- Training generated queries/Euler states FP32, native epsilon BF16, original mixed parameter storage/autocast; quality model-only BF16 and BF16 queries. DIRECT zero query follows explicit phase feature dtype. Physical loss/decode remain FP32. `.eval()` does not define the phase. Same-phase parity maximum raw/decoded difference0 with identical original gate issues; actual generated gradients cover4 S4 nodes or1 DIRECT node.
- Same471 trainable/559 frozen tensor identities, LR1e-5, warmup10/cosine1024, accumulation2, native loss plus .4 position+.1 increment+.1 rotation+.1 peak-speed. Same PREFIX temporal weights. DIRECT uses the same modules at zero latent/bucket0 once, decoded directly; no Euler, added head, target leakage or label input. S4 remains the original four-step noise generator. The intervention jointly changes query initialization and readout interpretation; it does not isolate stochasticity from depth.
- Actual per-arm exposure is in MAIN_TRAINING_AUDIT.json: M1 each command Q2/Q3 168/168 and terminal108 microbatches; M3 each command Q2/Q3 168/168, entry82, midarc82, terminal80. The full small-source export contains166 exact files with source hashes, including actual logs and kernel binding. Separate per-phase CPU kernel receipts were NOT_RECORDED; startup binding and every actual processed-input hash were checked against the qualified Prescott CPU admission.

## Primary result: consumed M3 terminal

Marginal descriptive medians below use original j0/j1 for stochastic policies and one output per condition for DIRECT. The /48 noisy outputs are not48 independent scenes. All directions are defined under the original1e-9m reporting guard.

{primary_table}

Teacher endpoint displacement median13.077mm, range0.00000326–22.642mm. Some values are almost zero: signs alone are fragile and do not prove a path verdict. Median signed teacher-direction progress is AGG -1.839mm, DIRECT0 -5.362mm, S4FIT +9.218mm, DIRECT1024 +8.493mm. Full continuous vectors/magnitudes and first-point/.4s endpoint/suffix/gripper/gate details remain in ALL_POINTS3504.csv.

Against S4FIT, median paired MSE deltas for family004/command0,005/0,004/1,005/1 are +9.43210e-7,+2.01992e-6,-2.63242e-7,+2.14849e-6 m2. Only one of four family-command groups favors DIRECT. Both commands and both families have zero final wrong-direction outputs in both fits.

{primary_groups}

## All consumed strata and secondary M1 target

Each table entry is the median of DIRECT MSE minus that SAME condition's mean over j0/j1. Finite REJECT remains included; no marginal-median subtraction or noise-as-scene replication.

{pair_table}

Across the96 secondary M1 Q2/Q3 conditions, DIRECT-minus-S4FIT median paired MSE is +1.78785107e-7m2; command0 +9.21381681e-7, command1 -6.20793418e-8. Against AGG it is -1.04804909e-5m2. M1 Q2 prefix XYZ medians are AGG5.121mm, S4FIT3.794mm, DIRECT3.964mm; Q3 j0/j1 are7.401,5.579,5.760mm. These support improved fitting in both arms with mixed direct-versus-S4 differences. M1 terminal, M3 entry and midarc tradeoffs are retained above and in all family/command tables.

DIRECT0 is a distinct, initially poorly calibrated effective policy:18/296 finite original-gate rejections, including M1 Q2 command1 four, Q3 command0 one, Q3 command1 thirteen. After fitting, S4FIT has0/1360 rejects and DIRECT1024 has0/488 distinct evaluated outcomes (plus4 identical repeats). Reused AGG retains its1/1360 finite Q3 j6 angular-speed rejection. No nonfinite/invalid-rotation outputs were removed or retried. DIRECT0 improvement alone is not evidence against S4.

## Early-support spillover guard: all192 retained

These original Q0/Q1 inputs are not a blind holdout or LEGACY qualification. Guard prefix XYZ medians AGG→S4FIT→DIRECT are M1 2.701→4.264→5.026mm, M3 2.120→3.525→3.264mm. M1 wrong directions0→3→1/96; M3 1→1→1/96. All guard policies have zero rejects. Rotation has mixed tradeoffs, explicitly retained below.

Paired guard deltas (after minus before) avoid differences of marginal medians:

{guard_delta}

{guard_table}

## Complete K8 banks, no averaging policy

All192 banks (two policies×96 Q3 inputs) are complete; no smaller K or replacement draw. The reused finite AGG rejection remains in its bank. DIRECT is evaluated once and logically compared to each matched bank, not counted as eight independent draws.

{table(['Policy','Task','Banks','Median prefix MSE m2','Median empirical mean-error share %','Median within-bank dispersion m2'],k8)}

DIRECT-minus-S4FIT completeK8 per-condition median prefix MSE is +3.41596882e-7m2 for M1 Q3 and -1.72637739e-7m2 for M3 Q3. These are finite-bank descriptive mean-error/dispersion identities, not population bias estimates; no averaged trajectory was executed. No K8 claim transfers to terminal components or visited-state A3.

## Compute, correctness and preservation

- One Hopper H200 allocation646325, start2026-10-04 22:35:05SGT, end23:31:21, exit0. Official2048 updates plus32 planned canary updates =2080; zero technical recovery updates. Actual training heads24987/32768, including parity and recomputation. S4FIT main18440 heads /1890.6555s loop; DIRECT main6145 heads /937.9158s. Canaries304/98 heads. Equal updates are not equal FLOPs; loop times exclude model load/archive.
- Exactly2148/2560 quality attempts,6228/7168 heads, zero technical neural retries. S4FIT1168 support+192 guard; DIRECT0 296; DIRECT1024 296+192 guard+4 repeats. All four repeats have exact native output and saved-array bytes, generation leaves RNG unchanged. The1360 reused AGG outputs plus2144 non-repeat new outputs yield3504 metric rows.
- CPU replay PASS for all2148 new predictions: saved S4 Euler and FP32 decode; DIRECT saved raw decode without fictitious Euler or recomputing its neural network. Original gate/metric functions are shared; physical metrics have independent FP64 arithmetic cross-checks, per-condition arithmetic checks, full-bank identities, exact output/source hashes. CPU audit makes zero new neural/gradient/optimizer calls.
- All four complete Trainers have25 files each, fixed HF revisions and full-file readback PASS (model, optimizer, scheduler, RNG, config/processor/statistics, provenance and explicit mode). Local recoveries were not deleted; Corelab only received model/config files.

{table(['Arm','Step','Fixed HF revision','Complete tree SHA256'],trainers)}

HF repo `leoyuer/pi0246-gate-pass-campaign`, prefixes `arms/PI0272/{{arm}}/checkpoint-{{step}}`. DIRECT weights require the explicit DIRECT mode sidecar; stock S4 loading is a different effective policy. Prediction archive revision `{archive['revision']}`, prefix `{archive['prefix']}`, tree `{archive['tree_sha256']}` preserves all4296 output/receipt members and has anonymous full-file/all-member readback. The summary archive and final canonical receipts bind this report after generation.

All source cards, raw budget journals, every condition/family/command metric, full vectors, complete gate reasons, scope audits, four readbacks and166-file training-source export are retained. No new video is required because new episodes/renders/WBC/physics/teacher labels/hardware are all0. No monitor, mail, broad cleanup, PR77 pointer or merge.

## Decision boundary

**PI0272_MATCHED_DIRECT_READOUT_FITTING_COMPLETE — STOP_FOR_FULL_SUPPORT_FEEDBACK_OR_FITTING_DECISION.** The bounded comparison closes with mixed fitting findings and no demonstrated primary DIRECT superiority. Additional same-support fitting repairs the terminal defect in both readouts while losing early-support accuracy. This does not establish that data quantity alone is the cause, that the model cannot represent the task, or that all direct architectures/objectives fail.

A later PI decision may address full-support/feedback coverage while explicitly retaining these guard losses; no such run is authorized here. The separate two-family M1 development/visited-state gap was not reevaluated or solved. PI0270 original/reliable VALID_QUALITY_FAIL, its reliable6/64 and positive LEGACY result, PI0269/68 FAIL and PI0267 WBC PARTIAL remain unchanged. No task threshold or selector was created; selected/deployment remain NONE.
'''
    (out/'FINAL_RESEARCH_RETURN.md').write_text(text,encoding='utf-8',newline='\n')
    print('RESEARCH_RETURN_FROM_VERIFIED_TABLES',sha(out/'FINAL_RESEARCH_RETURN.md'))


if __name__=='__main__':main(Path(__file__).resolve().parents[2])
