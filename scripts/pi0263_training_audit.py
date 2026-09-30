"""Match actual PI0263 logs to the sealed CPU source/condition plan, read-only."""
import argparse
import json
from pathlib import Path

from pi0263_plan import read, write, sha, seed_for

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0263')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('F','X'));p.add_argument('--phase',required=True)
    p.add_argument('--limit',type=int,required=True);p.add_argument('--paired',action='store_true');a=p.parse_args()
    paired(a.phase,a.limit) if a.paired else audit(a.arm,a.phase,a.limit)
