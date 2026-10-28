"""Predeclared P1 interpretation and final equal-budget research comparison."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0254'
sys.path[:0]=[str(ROOT/'execution'),str(BASE/'PI0253/runtime')]
import pi0253_full_screen_v2 as inherited


def summary(rows):
    import numpy as np
    valid=[r for r in rows if r['status']=='VALID']
    out=dict(fixed_cells=len(rows),valid_cells=len(valid),reject_count=sum(r['gate_decision']=='REJECT' for r in valid),
             expert_reject_count=sum(r.get('expert_reject',False) for r in valid),
             model_only_reject_count=sum(r.get('model_only_reject',False) for r in valid))
    for key in ('xyz_rmse_m','so3_rmse_rad','mean_offset_m','centered_shape_rmse_m','first_xyz_error_m','last_xyz_error_m','first_so3_error_rad','last_so3_error_rad'):
        values=[r['metrics'][key] if key in r['metrics'] else r[key] for r in valid]
        out['median_'+key]=float(np.median(values)) if values else None
        out['p95_'+key]=float(np.quantile(values,.95,method='linear')) if values else None
    return out


def stratify(rows):
    result={}
    strata=[('all_rows',lambda r:True),('expert_nonREJECT',lambda r:not r.get('expert_reject',True))]+[
        (f'task_{task}',lambda r,task=task:r['task']==task) for task in 'abcd']+[
        (f'phase_{phase}',lambda r,phase=phase:r['phase']==phase) for phase in range(4)]
    for name,predicate in strata:result[name]=summary([r for r in rows if predicate(r)])
    result['source_rollouts']={rollout:summary([r for r in rows if r['source_rollout_id']==rollout]) for rollout in sorted({r['source_rollout_id'] for r in rows})}
    return result


def read(point):
    path=ROOT/f'eval/{point}/cpu_replay.json';doc=inherited.read(path)
    assert doc['status']=='PASS_INDEPENDENT_CPU_REPLAY' and doc['point']==point
    return doc['rows'],inherited.sha(path)


def paired_difference(a,b):
    assert a['condition_identity']==b['condition_identity'] and a['noise_source_sha256']==b['noise_source_sha256']
    keys=('xyz_rmse_m','so3_rmse_rad','mean_offset_m','centered_shape_rmse_m','first_xyz_error_m','last_xyz_error_m','first_so3_error_rad','last_so3_error_rad')
    return {key:(a['metrics'][key] if key in a['metrics'] else a[key])-(b['metrics'][key] if key in b['metrics'] else b[key]) for key in keys}


def p1():
    import numpy as np
    g,gsha=read('G2000');full,fsha=read('FULL10000')
    assert len(g)==256 and len(full)==64 and all(r['status']=='VALID' for r in g+full)
    probes={(r['source_episode_id'],r['noise_name']):r for r in g if r['cohort']=='P64'}
    pairs=[]
    for f in full:
        b=probes[f['source_episode_id'],'B0'];assert f['noise_name']=='B0'
        assert f['condition_identity']==b['condition_identity'] and f['noise_source_sha256']==b['noise_source_sha256']
        values=paired_difference(f,b)
        pairs.append(dict(source_episode_id=f['source_episode_id'],task=f['task'],phase=f['phase'],source_rollout_id=f['source_rollout_id'],
                          G=b,FULL=f,differences_FULL_minus_G=values))
    fraction=float(np.mean([r['differences_FULL_minus_G']['xyz_rmse_m']<0 for r in pairs]))
    direction=('NEW_PROBE_IMPROVES_WHILE_OLD_SUPPORT_REGRESSED' if fraction>.5 and summary(full)['median_xyz_rmse_m']<summary([r for r in probes.values() if r['noise_name']=='B0'])['median_xyz_rmse_m']
               else 'BROADER_FITTING_DIFFICULTY_OR_MIXED_TRAIN_STRATA')
    report=dict(schema='PI0254_P1_COMPARISON_V1',status='P1_VALID_BASELINES_COMPLETE',new_queries_valid=320,
                inference_query_attempts=len((ROOT/'query_attempts.jsonl').read_text().splitlines()),
                code_sha256=inherited.sha(Path(__file__)),eval_card_sha256=inherited.sha(ROOT/'eval_card.json'),
                CPU_report_sha256=dict(G2000=gsha,FULL10000=fsha),P1_paired_cells=128,paired_rows=pairs,
                fraction_probe_XYZ_improved=fraction,interpretation=direction,
                G_P64_B0=stratify([r for r in g if r['cohort']=='P64' and r['noise_name']=='B0']),
                FULL_P64_B0=stratify(full),G_P64_B1=stratify([r for r in g if r['cohort']=='P64' and r['noise_name']=='B1']),
                G_E64={n:stratify([r for r in g if r['cohort']=='E64' and r['noise_name']==n]) for n in ('B0','B1')},
                FULL_P64_B1='NOT_QUERIED',valid_mixed_requires_continuation=True,training_design_changed=False,
                limitations=['Repeated TRAIN/development observations; C_T inherited','One training-family contrast; no independent seed replications',
                             'P64 not replayed in this stage, but FULL already trained on it','Centering is decomposition only; no world target statistic'],
                deployment_selected_checkpoint='NONE')
    target=ROOT/'P1_comparison.json';inherited.write(target,report)
    inherited.write(ROOT/'P1_comparison.seal.json',dict(sha256=inherited.sha(target),code_sha256=report['code_sha256'],
                                                    source_CPU_report_sha256=report['CPU_report_sha256']))
    print(json.dumps(dict(status=report['status'],fraction_probe_XYZ_improved=fraction,interpretation=direction)),flush=True)


def qualify(endpoint,baseline):
    import pi0247_m64_screen as m64
    import numpy as np
    assert len(endpoint)==384 and len(baseline)==256
    checks={};all_valid=all(r['status']=='VALID' for r in endpoint)
    for noise in ('B0','B1'):
        o=[r for r in endpoint if r['cohort']=='O64' and r['noise_name']==noise]
        checks['O64_'+noise]=m64.group(o,64,True)['quality_pass']
        checks['old16_'+noise]=m64.group([r for r in o if r['old16']],16,True)['quality_pass']
        for group,ratio in [('E64',.8),('P64',.9)]:
            rows=[r for r in endpoint if r['cohort']==group and r['noise_name']==noise]
            original=[r for r in baseline if r['cohort']==group and r['noise_name']==noise]
            x,y=summary(rows),summary(original)
            checks[group+'_'+noise]=len(rows)==x['valid_cells']==64 and x['median_xyz_rmse_m']<=ratio*y['median_xyz_rmse_m'] and (
                x['p95_xyz_rmse_m']<=y['p95_xyz_rmse_m']+1e-6 and x['median_so3_rmse_rad']<=y['median_so3_rmse_rad']+1e-6 and x['model_only_reject_count']==0)
    worst=max(float(np.mean([summary([r for r in endpoint if r['cohort']=='P64' and r['noise_name']==n and r['task']==t])['median_xyz_rmse_m'] for t in 'abcd'])) for n in ('B0','B1')) if all_valid else None
    return dict(all_fixed_valid=all_valid,checks=checks,promising=all_valid and all(checks.values()),worst_noise_P64_task_macro_median_xyz_m=worst)


def final():
    baseline,gsha=read('G2000');endpoints={};identities={};observations={}
    for arm in ('U576','R576'):
        observations[arm],identities[arm+'_1152']=read(arm+'_1152')
        endpoints[arm],identities[arm+'_4608']=read(arm+'_4608')
    decisions={arm:qualify(rows,baseline) for arm,rows in endpoints.items()}
    eligible=[arm for arm in ('U576','R576') if decisions[arm]['promising']]
    winner=min(eligible,key=lambda arm:(decisions[arm]['worst_noise_P64_task_macro_median_xyz_m'],arm!='U576')) if eligible else 'NONE'
    u={(r['source_episode_id'],r['noise_name']):r for r in endpoints['U576']}
    pairs=[]
    for row in endpoints['R576']:
        peer=u[row['source_episode_id'],row['noise_name']]
        pairs.append(dict(source_episode_id=row['source_episode_id'],noise_name=row['noise_name'],cohort=row['cohort'],
                          task=row['task'],phase=row['phase'],source_rollout_id=row['source_rollout_id'],
                          status='VALID' if row['status']==peer['status']=='VALID' else 'INVALID',
                          differences_R_minus_U=paired_difference(row,peer) if row['status']==peer['status']=='VALID' else None))
    report=dict(schema='PI0254_EQUAL_BUDGET_COMPARISON_V1',status='RESEARCH_COMPARISON_COMPLETE',research_candidate=winner,
         candidate_status='PROMISING_RESEARCH_ONLY' if winner!='NONE' else 'NO_PROMISING_RESEARCH_CANDIDATE',deployment_selected_checkpoint='NONE',
         decisions=decisions,baseline_CPU_sha256=gsha,arm_CPU_sha256=identities,paired_endpoint_rows=pairs,
         endpoint_groups={arm:{group:{noise:stratify([r for r in rows if r['cohort']==group and r['noise_name']==noise])
                            for noise in ('B0','B1')} for group in ('O64','E64','P64')} for arm,rows in endpoints.items()},
         diagnostic_1152={arm:stratify(rows) for arm,rows in observations.items()},
         explanation='R trades 3x old exposure against 25 percent fewer new exposures; shared improvement does not isolate rehearsal.',
         no_deployment_or_sim_claim=True)
    inherited.write(ROOT/'final_comparison.json',report);print(json.dumps(dict(research_candidate=winner,decisions=decisions)),flush=True)


def check():
    sample=dict(status='VALID',cohort='O64',noise_name='B0',old16=False,gate_decision='PASS',expert_reject=False,model_only_reject=False,
                metrics=dict(xyz_rmse_m=.01,so3_rmse_rad=.02),mean_offset_m=.003,centered_shape_rmse_m=.004,
                first_xyz_error_m=.01,last_xyz_error_m=.01,first_so3_error_rad=.02,last_so3_error_rad=.02,task='a',phase=0,source_rollout_id='toy')
    assert summary([sample])['valid_cells']==1 and summary([dict(sample,status='INVALID')])['fixed_cells']==1
    assert summary([dict(sample,expert_reject=True,model_only_reject=False)])['model_only_reject_count']==0
    baseline=[];endpoint=[]
    for cohort in ('O64','E64','P64'):
        for noise in ('B0','B1'):
            for i in range(64):
                row=dict(sample,cohort=cohort,noise_name=noise,old16=i<16,task='abcd'[i%4],phase=i%4)
                endpoint.append(row)
                if cohort!='O64':baseline.append(dict(row,metrics=dict(xyz_rmse_m=.02,so3_rmse_rad=.03)))
    assert qualify(endpoint,baseline)['promising']
    bad=list(endpoint);bad[-1]=dict(bad[-1],model_only_reject=True,gate_decision='REJECT')
    assert not qualify(bad,baseline)['promising']
    bad=list(endpoint);bad[0]=dict(bad[0],status='INVALID')
    assert not qualify(bad,baseline)['promising']
    inherited.write(ROOT/'comparison_cpu_receipt.json',dict(status='PASS_PI0254_DENOMINATOR_AND_ENDPOINT_RULE_CPU',
          code_sha256=inherited.sha(Path(__file__)),new_model_queries=0,new_optimizer_updates=0))
    print('PASS_PI0254_DENOMINATOR_AND_ENDPOINT_RULE_CPU')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--p1',action='store_true');p.add_argument('--final',action='store_true');p.add_argument('--cpu-check',action='store_true');args=p.parse_args()
    if args.cpu_check:check()
    elif args.p1:p1()
    elif args.final:final()
    else:p.error('select --p1, --final or --cpu-check')
