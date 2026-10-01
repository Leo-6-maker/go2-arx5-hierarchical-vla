"""Report actual complete CPU tables; no model calls or assumed outcomes."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from pi0264_plan import read,sha,write,AUTHORITY

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0264')
POINTS=('X2000','B2048','L2048')


def rows(name):
    with (ROOT/(name+'.csv')).open(newline='') as stream:return list(csv.DictReader(stream))


def table(fields,values):
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']+
        ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in values])


def main():
    cpu=read(ROOT/'CPU_REPLAY.json');training=read(ROOT/'MAIN_TRAINING_AUDIT.json');q=read(ROOT/'QUALIFICATION.json')
    assert cpu['status']=='PASS_INDEPENDENT_COMPUTATION_PATH_CPU_6336_REPLAY' and cpu['rows']==6336
    assert training['status']=='PASS_BOTH_2048_MATCHED_TRAINING_INTEGRITY'
    assert q['selected_checkpoint']==q['deployment_selected_checkpoint']=='NONE'
    assert [r['point'] for r in q['points']]==list(POINTS)
    lines=['# PI0264 — Matched language diversity and prospectively sealed transfer','',
        'Both authorized2048-update endpoints and all6336 scoring rows are complete. Completion and quality qualification are separate. '
        'Historical PI0263 COMPLETE and F/X valid FAIL are unchanged. Formal selected/deployment remain NONE.','',
        '## Three separate qualifications','',
        table(('Point','Legacy full quality','Old crossed M1','Prospective text/noise','All three'),
            [(r['point'],r['legacy_full_quality_pass'],r['crossed_M1_robustness_pass'],r['prospective_text_noise_pass'],r['combined_pass']) for r in q['points']]),'',
        'Qualified synthetic research endpoints: '+(', '.join(q['synthetic_research_candidates']) or 'NONE')+'. '
        'Both qualifying endpoints are retained; there is no post-hoc winner or deployment selection.','',
        '## What this experiment measures','',
        'B and L both start from fixed PI0263 X2000, use the same256 source units/eight passes,4096 microbatches per arm, '
        'images/state/labels/masks and actual paired native epsilon/tau/free-S4 packets. B uses T0/T1; L uses T0–T7. '
        'Only M1 wording assignment differs. M3 retains its original wording. New B is a continuation control, not historical Base. '
        'L has256 M1 examples per template; B1024 each T0/T1. Diversity at fixed compute also reduces exposure per old phrase.','',
        'Panel A preserves the old1152-row union with680 legacy and640 crossed memberships. Panel B720 rows are new supervised-text fit, '
        'not L language generalization. Panel C240 rows uses Q0–Q3/T0–T1 anchors and20 actual frozen fresh N0/N1 packets. '
        'Q/V never enter supervision/statistics. These are already developed KIN-VAL geometries, not blind new-scene or physical tests. '
        'Foundation-model novelty and independent-binomial trials are not claimed.','',
        '## Original geometry and language groups','',
        table(('Point','Task','Kind','Noise','N','Correct','Reject','Numeric pass','Semantic pass'),
            [(r['point'],r['task'],r['kind'],r['noise'],r['n'],r.get('correct','—'),r['reject'],r.get('numeric_pass','—'),r.get('semantic_pass','—')) for r in q['legacy_groups']]),'']
    fit=rows('TRAIN_NEW_TEXT_ROWS');totals={}
    for point in POINTS:
        vals=[r for r in fit if r['point']==point];assert len(vals)==720
        groups=[r for r in q['prospective_groups'] if r['point']==point];assert len(groups)==12
        totals[point]=dict(fit=sum(r['correct']=='True' for r in vals),
            Q=sum(r['correct'] for r in groups if r['template'].startswith('Q')),
            anchors=sum(r['correct'] for r in groups if r['template'].startswith('T')),
            groups_pass=sum(r['prospective_pass'] for r in groups))
    lines += ['## Added text fit and prospective transfer','',
        table(('Point','New TRAIN texts correct/720','Unsupervised Q correct/160','Anchors correct/80','Prospective groups passing/12'),
            [(p,totals[p]['fit'],totals[p]['Q'],totals[p]['anchors'],totals[p]['groups_pass']) for p in POINTS]),'',
        'These totals summarize the preserved groups; qualification still requires every individual numeric and semantic gate. '
        'A higher count of passing groups cannot replace the mandatory AND across all12.','']
    if not q['synthetic_research_candidates'] and totals['L2048']['fit']>totals['B2048']['fit'] and totals['L2048']['Q']<=totals['B2048']['Q']:
        lines += [f"L improves newly supervised text fit by {totals['L2048']['fit']-totals['B2048']['fit']}/720 versus B, "
            f"while its aggregate unseen-Q difference is {totals['L2048']['Q']-totals['B2048']['Q']}/160. "
            'This single matched run supports learnability of the added training formulations. Prospective transfer remains insufficient '
            'under the registered gates; the result does not establish a stronger aggregate Q advantage for language diversity. '
            'Donor-to-endpoint improvements also include added optimization and the compact common schedule. '
            'Stop at the authorized terminal with the negative results; no additional old-scene template schedule, quality retry or transfer execution follows.','']
    for title,groups,passkey in (('Old crossed M1',q['crossed_groups'],'robustness_pass'),('Prospective text/noise',q['prospective_groups'],'prospective_pass')):
        lines += ['## '+title,'',table(('Point','Split','Template','Noise','Correct/N','c0','c1','Both families','median XYZ mm','p95 XYZ mm','median SO3 rad','Reject','Group pass'),
            [(r['point'],r['split'],r['template'],r['noise'],str(r['correct'])+'/'+str(r['n']),r['c0'],r['c1'],r['both_commands_correct'],
              format(r['median_xyz_m']*1000,'.3f'),format(r['p95_xyz_m']*1000,'.3f'),format(r['median_so3_rad'],'.6f'),r['reject'],r.get(passkey,'TRAIN diagnostic')) for r in groups]),'']
    lines += ['Each legacy language group requires≥10/12; each old crossed KIN-VAL and prospective group requires≥17/20 plus all numeric gates. '
        'Numeric gates: medianXYZ≤.020m, p95≤.035m, medianSO3≤.050rad, zeroREJECT. Own/opposite H40 XYZ margin must be strictly positive; ties are incorrect. '
        'No averaging hides a failed template/noise/command.','']
    scored=rows('M1_ALL_PANELS_ROWS');indexed={(r['point'],r['uid']):r for r in scored}
    paired=defaultdict(list)
    for r in scored:
        if r['point']!='B2048':continue
        other=indexed['L2048',r['uid']]
        assert all(r[k]==other[k] for k in ('panel','split','family','command','color','template','noise'))
        paired[r['panel'],r['split'],r['template'],r['noise']].append((r,other))
    contrast=[]
    for (panel,split,template,noise),values in sorted(paired.items()):
        b=sum(a['correct']=='True' for a,c in values);l=sum(c['correct']=='True' for a,c in values)
        contrast.append((panel,split,template,noise,len(values),b,l,l-b,
            sum(a['correct']=='False' and c['correct']=='True' for a,c in values),
            sum(a['correct']=='True' and c['correct']=='False' for a,c in values)))
    lines += ['## Matched L versus B target choice','',
        table(('Panel','Split','Template','Noise','N','B correct','L correct','L−B','B wrong/L right','B right/L wrong'),contrast),'',
        'The B→L contrast fixes all nonlanguage training support and computation. Donor→endpoint changes also include extra optimization '
        'and the common compact schedule, so they do not isolate diversity. Panel C anchor/Q comparisons fix observation/noise within each family. '
        'Between-panel differences also change wording/noise and are not pure wording effects.','']
    for name,title in (('M1_TEMPLATE_NUMERIC_GROUPS','All M1 template geometry'),('COLOR_COMMAND_GROUPS','Every color/command/noise stratum'),
        ('ORIGINAL_VS_OTHER_BINDINGS','Original versus other TRAIN bindings'),('LEGACY_REGRESSION_GROUPS','M3, geometry, empty language and O64 regression')):
        values=rows(name);keys=list(values[0]);lines += ['## '+title,'',table(keys,[[r[k] for k in keys] for r in values]),'']
    targets=rows('TARGET_CHOICE_DECOMPOSITION');counts=defaultdict(lambda:defaultdict(int))
    for r in targets:counts[r['panel'],r['point'],r['split'],r['template'],r['noise']][r['category']]+=1
    lines += ['## Signed target direction and bias','',
        'Existing normalized s0/s1 decomposition uses g=(s1−s0)/2 and b=(s0+s1)/2. Both commands correct iff s0<0<s1, equivalently g>|b|. '
        'Positive ordering with a wrong target is not semantic success. This describes outputs, not a proven neural cause.','',
        table(('Panel','Point','Split','Template','Noise','Both correct','Positive ordering/bias dominates','Nonpositive ordering'),
            [(*key,c['both_correct'],c['positive_ordering_bias_dominates'],c['nonpositive_ordering']) for key,c in sorted(counts.items())]),'',
        'M1_FAMILY_COMMAND_ROWS.csv retains every signed margin; WORDING_FLIPS.csv retains all within-panel template pairs; '
        'NOISE_BOUNDARY_CROSSINGS.csv retains B0/B1 or N0/N1 direction crossings for every family. Failure rows and unequal color frequencies are retained. '
        'No DTW, alias-floor subtraction, expert-label replacement, output smoothing, seed choice or quality retry is used.','',
        '## Actual training and query accounting','',
        table(('Ledger','Actual','Hard cap'),[(k,training[k],cap) for k,cap in (('raw_completed_updates',4384),('raw_training_heads',110000))]+[
            ('inference attempts',cpu['global_attempts'],5440),('inference heads',cpu['global_heads'],21760),
            ('no-valid technical query reserve',cpu['invalid_only_reserve_used'],256),('exact forward reuses',cpu['new_endpoint_exact_forward_reuses'],'no extra cases')]),'',
        'CPU replay is the same sole owner using an independent computation path, not external reviewer signoff. Actual serialized frozen scope, '
        'full-four-step gradients, model mode and cross-arm packet identities are recorded separately.','',
        '## Four recoverable full Trainer checkpoints','']
    models=[]
    for arm in ('B','L'):
        for step in (1024,2048):
            r=read(ROOT/'evidence'/arm/f'checkpoint-{step}.full_readback.json')
            assert r['status']=='PASS_FULL_FILE_READBACK' and r['complete_trainer'] and len(r['files'])==16
            models.append(dict(arm=arm,step=step,revision=r['revision'],tree_sha256=r['tree_sha256'],files=len(r['files'])))
    lines += [table(('Arm','Step','HF revision','Full Trainer tree','Files'),[(r['arm'],r['step'],r['revision'],r['tree_sha256'],r['files']) for r in models]),'',
        'HF PUBLIC leoyuer/pi0246-gate-pass-campaign; arms/PI0264/{B|L}/checkpoint-{1024|2048}. '
        '1024 is a saved milestone/resume state, never quality screened. Corelab has model-only views, not complete resume Trainers.','',
        '## Fixed-array presentation and evidence','',
        'The sealed display covers old fixed cases, all six new TRAIN texts, Q0–Q3/anchors with fresh noises, and deterministic first-failure coverage. '
        'It uses existing RGB/reference/prediction arrays only and visibly states KINEMATIC RENDERED REFERENCE / OFFLINE PREDICTION / NO PHYSICS OR WBC. '
        'No new scene render, IK, model query or physical success is implied.','',
        'Public archive receipt is separate from this report to avoid a circular hash. Only successful anonymous full-file/every-member readback '
        'confirms the final public seal; preparing files is not upload success.','',
        '## Closed boundary','',
        'PI0264_LANGUAGE_DIVERSITY_TRANSFER_PILOT_COMPLETE — STOP_FOR_NEW_SCENE_OR_PHYSICAL_TRANSFER_ADMISSION follows final deliverable audit. '
        'selected_checkpoint=NONE; deployment_selected_checkpoint=NONE. No further model candidate, extra loss/seed/updates, scene/camera/IK/statistics, '
        'adapter, WBC/physics/TEST/DEV-F/real hardware, monitor/mail, broad cleanup or merge is authorized.','']
    (ROOT/'MODEL_QUALITY_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write(ROOT/'FINAL_CLOSEOUT.json',dict(schema='PI0264_COMPUTE_AND_QUALITY_CLOSEOUT_V1',
        status='PI0264_COMPUTE_AND_QUALITY_COMPLETE_AWAIT_PUBLIC_READBACK',authority_commit=AUTHORITY,
        optimizer_updates=training['optimizer_updates'],raw_completed_updates=training['raw_completed_updates'],raw_training_heads=training['raw_training_heads'],
        rows=cpu['rows'],attempts=cpu['global_attempts'],inference_heads=cpu['global_heads'],checkpoints=models,
        qualification=q['points'],synthetic_research_candidates=q['synthetic_research_candidates'],
        cpu_status=cpu['status'],same_owner_independent_computation_path=True,external_signoff=False,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',stop='STOP_FOR_NEW_SCENE_OR_PHYSICAL_TRANSFER_ADMISSION',
        model_queries_by_report=0,physics_steps=0,WBC_steps=0,TEST=0,DEV_F=0,real_hardware=0,new_scenes=0,new_statistics_fits=0))
    print(json.dumps(dict(status='PASS_REPORT_FROM_ACTUAL_6336_CPU_TABLES',quality=q['points'],model_queries=0)))


if __name__=='__main__':main()
