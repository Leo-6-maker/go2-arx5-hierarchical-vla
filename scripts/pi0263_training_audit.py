"""Match actual PI0263 logs to the sealed CPU source/condition plan, read-only."""
import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path

from pi0263_plan import read, write, sha, seed_for

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0263')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def f32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]


def audit(arm,phase,limit):
    path=ROOT/arm/phase/'training'
    expected={(r['unit'],r['micro']):r for r in read(ROOT/'PROCESSED_PLAN_AUDIT.json')['rows'] if r['arm']==arm}
    sample=rows(path/'sample_audit.jsonl');rng=rows(path/'step_rng.jsonl')
    s4=rows(path/'S4_audit.jsonl');loss=rows(path/'full_loss.jsonl');norms=rows(path/'update_gradient_norms.jsonl')
    assert len(sample)==len(rng)==len(s4)==len(loss)==2*limit
    assert len(norms)==limit and [r['unit'] for r in norms]==list(range(1,limit+1))
    for unit in range(1,limit+1):
        packets=[]
        for micro in (0,1):
            i=2*(unit-1)+micro;a,b,c=sample[i],rng[i],s4[i]
            assert (a['unit'],a['micro'])==(unit,micro)
            e=expected[unit,micro]
            for key in e:
                assert a[key]==e[key],(arm,phase,unit,micro,key)
            assert b['unit']==unit and b['micro']==micro and b['condition_sha256']==e['condition_sha256']
            assert b['seed']==seed_for(unit) and b['mode_restored']
            assert c['forward_index']==i+1
            packet=(b['native_epsilon'],b['native_tau'],b['stock_initial_noise'])
            assert packet==(c['native_epsilon'],c['native_tau'],c['stock_initial_noise'])
            packets.append(packet)
        assert packets[0]==packets[1],('WITHIN_UNIT_PACKET',arm,phase,unit)
    checkpoint=path/f'checkpoint-{limit}'
    if checkpoint.exists():
        state=read(checkpoint/'trainer_state.json')
        assert (state['global_step'],state['max_steps'])==(limit,2000)
        assert all((checkpoint/name).is_file() for name in ('optimizer.pt','scheduler.pt','rng_state.pth'))
    else:
        assert phase in ('continuous','split') and limit==8
        assert read(ROOT/arm/'canary_compare.json')['status']=='PASS_PI0263_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
        retirement=read(ROOT/arm/'CANARY_CHECKPOINT_RETIREMENT.json')
        assert any(r['path']==str(checkpoint) and len(r['files'])>=16 for r in retirement['targets'])
    result=dict(status='PASS_PI0263_ACTUAL_SOURCE_TEMPLATE_LABEL_MASK_RNG_LOG_MATCH',arm=arm,phase=phase,
        units=limit,microbatches=2*limit,processed_plan_sha256=sha(ROOT/'PROCESSED_PLAN_AUDIT.json'),
        logs={name:sha(path/name) for name in ('sample_audit.jsonl','step_rng.jsonl','S4_audit.jsonl','full_loss.jsonl','update_gradient_norms.jsonl')})
    write(ROOT/arm/f'{phase}-{limit}.actual_input_audit.json',result)
    print(json.dumps(result))


def paired(phase,limit):
    left=rows(ROOT/'F'/phase/'training/step_rng.jsonl');right=rows(ROOT/'X'/phase/'training/step_rng.jsonl')
    assert len(left)==len(right)==2*limit
    for a,b in zip(left,right):
        assert all(a[k]==b[k] for k in ('unit','micro','command','seed','native_epsilon','native_tau','stock_initial_noise','mode_restored'))
    write(ROOT/f'{phase}-{limit}.paired_actual_rng.json',dict(status='PASS_PI0263_CROSS_ARM_ACTUAL_RANDOM_PACKET_EQUAL',
        units=limit,microbatches=2*limit,phase=phase))


def graph(arm):
    path=ROOT/arm/'main/training'
    s4_rows,loss_rows=rows(path/'S4_audit.jsonl'),rows(path/'full_loss.jsonl')
    assert len(s4_rows)==len(loss_rows)==4000
    for index,(s4,loss) in enumerate(zip(s4_rows,loss_rows),1):
        assert s4['forward_index']==loss['step']==index
        assert (s4['runtime_id'],s4['rollout_steps'],s4['rollout_head_evaluations'])==('S4',4,4)
        assert s4['checkpointed_euler_steps']==s4['backward_recomputation_head_evaluations_expected']==4
        assert not s4['expert_action_in_condition'] and not s4['active_action_modified']
        assert (s4['lambda_position'],s4['lambda_delta'],s4['valid_points'],s4['valid_transitions'])==(.4,.1,40,39)
        assert s4['native_epsilon']['dtype']=='torch.bfloat16' and s4['stock_initial_noise']['dtype']=='torch.float32'
        for key in ('euler_states','euler_velocities'):
            assert len(s4[key])==4 and all(v['dtype']=='torch.float32' and v['shape']==[1,40,132] for v in s4[key])
        assert all(math.isfinite(loss[k]) for k in ('total','rotation','speed_margin'))
        # Replay the source's FP32 operations and grouping, including large native losses.
        base=f32(f32(s4['native_loss']+f32(f32(.4)*s4['physical_position_loss']))+f32(f32(.1)*s4['physical_delta_loss']))
        extra=f32(f32(f32(.1)*loss['rotation'])+f32(f32(.1)*loss['speed_margin']))
        assert base==s4['total_loss'] and f32(base+extra)==loss['total']
    norms=rows(path/'update_gradient_norms.jsonl')
    assert len(norms)==2000 and [r['unit'] for r in norms]==list(range(1,2001))
    assert all(math.isfinite(r['accumulated_norm']) and r['microbatches']==2 and r['clip_max_norm']==1. for r in norms)
    gradients=read(path/'S4_first_update_gradients.json')
    assert set(gradients)=={'1','2','3','4'} and all(math.isfinite(float(v)) and float(v)>0 for v in gradients.values())
    result=dict(status='PASS_PI0263_SAVED_FULL_S4_LOSS_AND_GRADIENT_AUDIT',arm=arm,microbatches=4000,updates=2000,
        loss_coefficients=dict(position=.4,increment=.1,rotation=.1,peak_speed_margin=.1),
        first_update_euler_gradient_norms=gradients,code_sha256=sha(__file__),
        logs={name:sha(path/name) for name in ('S4_audit.jsonl','full_loss.jsonl','update_gradient_norms.jsonl','S4_first_update_gradients.json')},
        model_queries_by_audit=0,gradient_queries_by_audit=0)
    write(ROOT/arm/'main-2000.graph_loss_gradient_audit.json',result);print(json.dumps(result))


def complete():
    for arm in ('F','X'):
        audit(arm,'main',2000);graph(arm)
        assert read(ROOT/arm/'canary_compare.json')['status']=='PASS_PI0263_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
    paired('main',2000)
    updates=rows(ROOT/'raw_completed_updates.jsonl');heads=rows(ROOT/'raw_training_heads.jsonl')
    assert [r['ordinal'] for r in updates]==list(range(1,len(updates)+1))
    assert [r['ordinal'] for r in heads]==list(range(1,len(heads)+1))
    assert len(updates)<=4288 and len(heads)<=100000
    assert all(r['arm'] in ('F','X') and r['phase'] in ('main','continuous','split') for r in updates+heads)
    counts=Counter((r['arm'],r['phase']) for r in updates)
    assert all(counts[arm,phase]>=n for arm in ('F','X') for phase,n in (('main',2000),('continuous',8),('split',8)))
    reserve=len(updates)-4032;assert 0<=reserve<=256
    result=dict(schema='PI0263_MATCHED_MAIN_TRAINING_AUDIT_V1',status='PASS_BOTH_2000_MATCHED_TRAINING_INTEGRITY',
        optimizer_updates={'F':2000,'X':2000},supervised_microbatches={'F':4000,'X':4000},
        discarded_canary_updates=sum(n for (arm,phase),n in counts.items() if phase!='main'),
        actual_completed_by_arm_phase=[dict(arm=a,phase=p,updates=n) for (a,p),n in sorted(counts.items())],
        raw_completed_updates=len(updates),technical_completed_update_reserve_used=reserve,
        raw_training_heads=len(heads),actual_random_packets_matched_within_unit_and_across_arms=True,
        actual_four_step_gradient_and_mode_checks=True,plan_sha256=sha(ROOT/'PAIRING_PLAN.json'),
        completed_updates_sha256=sha(ROOT/'raw_completed_updates.jsonl'),training_heads_sha256=sha(ROOT/'raw_training_heads.jsonl'),
        code_sha256=sha(__file__),model_queries_by_audit=0,optimizer_updates_by_audit=0,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    write(ROOT/'MAIN_TRAINING_AUDIT.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('F','X'));p.add_argument('--phase')
    p.add_argument('--limit',type=int);p.add_argument('--paired',action='store_true');p.add_argument('--complete',action='store_true');a=p.parse_args()
    if a.complete:
        assert not (a.arm or a.phase or a.limit or a.paired);complete()
    else:
        assert a.phase and a.limit and (a.arm or a.paired)
        paired(a.phase,a.limit) if a.paired else audit(a.arm,a.phase,a.limit)
