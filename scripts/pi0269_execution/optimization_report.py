"""Join already consumed inputs, losses and preclip norms; no model or gradient call."""
import csv
import json
import math
import statistics
from pathlib import Path
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0269')


def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s]


def joined(arm,plan,samples,s4,full,norms,temporal):
    assert len(samples)==len(s4)==len(full)==2*len(norms)
    assert len(temporal)==len(samples)
    rows=[];source={}
    for i,(sample,rollout,loss) in enumerate(zip(samples,s4,full)):
        unit,micro=i//2+1,i%2;u=plan[unit-1];grad=norms[unit-1]
        family=sample['family']
        if family not in source:source[family]=read(ROOT.with_name('PI0267')/'source/S2'/family/'rows.json')
        raw=source[family][sample['index']]
        assert sample['unit']==grad['unit']==u['unit']==unit and sample['micro']==micro
        assert rollout['forward_index']==loss['step']==i+1
        values=dict(native_loss=rollout['native_loss'],physical_position_loss=rollout['physical_position_loss'],
            physical_increment_loss=rollout['physical_delta_loss'],physical_rotation_loss=loss['rotation'],
            physical_speed_margin_loss=loss['speed_margin'],total_loss=loss['total'],preclip_norm=grad['accumulated_norm'])
        values.update({k:v for k,v in temporal[i].items() if k!='arm'})
        assert temporal[i]['arm']==arm and all(math.isfinite(v) for v in values.values())
        rows.append(dict(arm=arm,unit=unit,micro=micro,quarter=(unit-1)//1024+1,phase_quartile=raw['quartile'],terminal=raw['terminal'],task=sample['task'],family=sample['family'],
            command=sample['command'],slot=sample['slot'],variant=0 if u['slot']=='COMMON_NOMINAL' else u['recovery_variant'],anchor=u['anchor'],
            condition_sha256=sample['condition_sha256'],clip_max_norm=grad['clip_max_norm'],
            clipped=grad['accumulated_norm']>grad['clip_max_norm'],estimated_clip_scale=min(1.,grad['clip_max_norm']/(grad['accumulated_norm']+1e-6)),**values))
    return rows


def main():
    audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status']=='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY'
    plan=read(ROOT/'PAIRING_PLAN.json')['units'][:4096];rows=[];groups=[]
    for arm in ('UNIFORM','PREFIX'):
        folder=ROOT/arm/'main/training'
        samples=lines(folder/'sample_audit.jsonl');s4=lines(folder/'S4_audit.jsonl');full=lines(folder/'full_loss.jsonl');norms=lines(folder/'update_gradient_norms.jsonl')
        assert len(samples)==len(s4)==len(full)==8192 and len(norms)==4096
        rows.extend(joined(arm,plan,samples,s4,full,norms,lines(folder/'temporal_components.jsonl')))
    for arm in ('UNIFORM','PREFIX'):
        own=[r for r in rows if r['arm']==arm]
        keys=[('ALL',None,None,None,None)]+sorted({(r['task'],r['variant'],r['quarter'],r['phase_quartile'],r['terminal']) for r in own})
        for task,variant,quarter,phase,terminal in keys:
            group=own if task=='ALL' else [r for r in own if (r['task'],r['variant'],r['quarter'],r['phase_quartile'],r['terminal'])==(task,variant,quarter,phase,terminal)]
            units={r['unit']:r for r in group};assert len(group)==2*len(units)
            value=dict(arm=arm,task=task,variant=variant,quarter=quarter,phase_quartile=phase,terminal=terminal,microbatches=len(group),updates=len(units),
                clipped_updates=sum(r['clipped'] for r in units.values()),maximum_preclip_norm=max(r['preclip_norm'] for r in units.values()))
            for field in ('native_loss','physical_position_loss','physical_increment_loss','physical_rotation_loss','physical_speed_margin_loss','total_loss','preclip_norm','uniform_position','weighted_position','uniform_increment','weighted_increment','uniform_rotation','weighted_rotation'):
                value['median_'+field]=statistics.median(r[field] for r in group)
            groups.append(value)
    out=ROOT/'evidence/optimization';out.mkdir(exist_ok=True)
    for name,values in [('ACTUAL_MICROBATCH_LOSSES.csv',rows),('TASK_VARIANT_QUARTER.csv',groups)]:
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result=dict(status='PASS_ACTUAL_INPUT_LOSS_GRADIENT_JOIN',microbatches=16384,updates=8192,groups=groups,
        clip_scale='Analytic estimate from recorded preclip norm and 1e-6 denominator; actual scale was not separately logged',
        native_loss_comparability='Same native targets and codec; physical uniform/weighted reductions are different treatments, not task completion scores',
        missing_dense_euler_trajectories='UNOBSERVED: training saves state/velocity hashes and same-pass losses, not dense training rollout arrays',
        model_queries=0,gradient_queries=0,additional_backward_calls=0,selected_checkpoint='NONE')
    (out/'OPTIMIZATION_REPORT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['status'],len(rows),len(groups),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true')
    if parser.parse_args().self_check:
        plan=read(ROOT/'PAIRING_PLAN.json')['units'][:8]
        for arm in ('UNIFORM','PREFIX'):
            folder=ROOT/arm/'continuous/training'
            data=[lines(folder/name) for name in ('sample_audit.jsonl','S4_audit.jsonl','full_loss.jsonl','update_gradient_norms.jsonl','temporal_components.jsonl')]
            rows=joined(arm,plan,*data)
            assert len(rows)==16 and len({r['unit'] for r in rows})==8
        print('PASS_ACTUAL_CANARY_LOSS_INPUT_JOIN_NO_MODEL',flush=True)
    else:main()
