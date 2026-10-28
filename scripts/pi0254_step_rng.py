"""Source/visit keys over the existing forward/backward Torch RNG isolation."""
from pathlib import Path
import json
import os
import pi0251_step_rng as inherited

inherited.SEED=25402
inherited.MAX_STEPS=4608
inherited.configure_namespace('PI0254','SUPPORT576')
SEED,MAX_STEPS,SCHEMA=inherited.SEED,inherited.MAX_STEPS,inherited.SCHEMA
schedule=[]


def configure_schedule(rows):
    global schedule
    assert len(rows)==MAX_STEPS and [r['optimizer_step'] for r in rows]==list(range(1,MAX_STEPS+1))
    schedule=rows


def step_key_seed(step):
    if isinstance(step,bool) or not isinstance(step,int) or not 1<=step<=MAX_STEPS or len(schedule)!=MAX_STEPS:
        raise ValueError('PI0254 requires the sealed arm schedule and a valid consumed-update cursor')
    row=schedule[step-1]
    return row['rng_key'],row['rng_seed']


def build_key_table(rows):
    configure_schedule(rows)
    return [dict(optimizer_step=r['optimizer_step'],key=r['rng_key'],seed=r['rng_seed'],
                 fixed_index=r['fixed_index'],source_episode_id=r['source_episode_id'],visit=r['visit']) for r in rows]


inherited.step_key_seed=step_key_seed
active_step_context=inherited.active_step_context
rng_context=inherited.rng_context
verify_identity=inherited.verify_identity
install_training_step_rng=inherited.install_training_step_rng
if os.environ.get('PI0254_RUN_CARD'):
    card=json.loads(Path(os.environ['PI0254_RUN_CARD']).read_text())
    configure_schedule(json.loads(Path(card['schedule']['path']).read_text())['rows'])


def peer_for_visit(current_rows,reference_rows,reference_records):
    assert len(reference_rows)==MAX_STEPS and 0<len(reference_records)<=MAX_STEPS
    lookup={(r['source_episode_id'],r['visit']):record for r,record in zip(reference_rows,reference_records)}
    assert len(lookup)==len(reference_records)
    return [lookup.get((r['source_episode_id'],r['visit'])) for r in current_rows]


def verify_visit(actual,expected):
    fields=('key','seed','model_input_sha256','native_tau','native_tau_identity','native_epsilon',
            'native_epsilon_sha256','stock_initial_noise')
    assert all(k in actual for k in fields)
    if expected is not None:verify_identity(actual,expected,fields)


def cpu_check(design):
    import torch
    u=json.loads((design/'U576_schedule.json').read_text())['rows']
    r=json.loads((design/'R576_schedule.json').read_text())['rows']
    reference=[dict(key=x['rng_key'],seed=x['rng_seed']) for x in u]
    mapped=peer_for_visit(r,u,reference)
    assert sum(x is not None for x in mapped)==3584
    assert all(x is None or (x['key'],x['seed'])==(plan['rng_key'],plan['rng_seed']) for x,plan in zip(mapped,r))
    outer=torch.get_rng_state().clone()
    common=[i for i,x in enumerate(mapped) if x is not None]
    u_index={(x['source_episode_id'],x['visit']):i+1 for i,x in enumerate(u)}
    for ri in (common[0],common[77],common[-1]):
        plan=r[ri];ui=u_index[plan['source_episode_id'],plan['visit']]
        configure_schedule(u)
        with rng_context(ui):a=(torch.distributions.Beta(1.5,1.).sample((1,)),torch.randn(1,40,132))
        configure_schedule(r)
        with rng_context(ri+1):b=(torch.distributions.Beta(1.5,1.).sample((1,)),torch.randn(1,40,132))
        assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert torch.equal(outer,torch.get_rng_state())
    actual=dict(key='k',seed=1,model_input_sha256='m',native_tau=[.5],native_tau_identity={'sha256':'t'},
                native_epsilon={},native_epsilon_sha256='e',stock_initial_noise={'sha256':'n'})
    verify_visit(actual,actual)
    for field in actual:
        poisoned={**actual,field:'wrong'}
        try:verify_visit(poisoned,actual)
        except RuntimeError:pass
        else:raise AssertionError('source/visit poison accepted: '+field)
    return dict(status='PASS_PI0254_SOURCE_VISIT_RNG_CPU',shared_visits=3584,
                schedule_rng_isolated=True,optimizer_step_equality_required=False,new_model_queries=0,new_optimizer_updates=0)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--design',type=Path,required=True);args=p.parse_args()
    print(json.dumps(cpu_check(args.design),sort_keys=True))
