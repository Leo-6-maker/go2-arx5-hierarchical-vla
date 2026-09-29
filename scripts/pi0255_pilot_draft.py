"""CPU-only matched-card preparation. This goal cannot execute model training."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import random
import sys


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def validate(card):
    assert card['status']=='DRAFT_NOT_EXECUTION_AUTHORIZED'
    assert card['stage']==dict(steps=6000,checkpoint_steps=[3000,6000],task_updates=dict(M1=2000,M2=2000,M3=2000))
    assert card['optimizer']==dict(name='AdamW',lr=1e-5,betas=[.9,.999],eps=1e-8,weight_decay=1e-5,max_grad_norm=1.,batch=1,accumulation=1,warmup_steps=10,scheduler='cosine6000',fresh=True)
    assert card['runtime']['S4']==4 and card['runtime']['action_horizon']==40 and card['runtime']['action_dt_s']==.05
    assert card['loss_coefficients']==dict(ell_pos=.4,ell_delta=.1,ell_rot=.1,ell_speed_margin=.1)
    assert card['execution_authorization'] is None and not card['data_ready']
    assert card['selected_checkpoint']=='NONE'


def schedule(rows):
    """Shared draft sampler: task -> scenario -> paired episode -> eligible window."""
    groups={t:{} for t in ('M1','M2','M3')}
    for row in rows:
        assert row['split']=='TRAIN' and row['success'] and row['eligible']
        groups[row['task']].setdefault(row['family_id'],{}).setdefault(row['command'],[]).append(row)
    assert all(groups.values())
    rng=random.Random(25503);visits={};result=[]
    for cycle in range(2000):
        tasks=['M1','M2','M3'];rng.shuffle(tasks)
        for task in tasks:
            families=sorted(f for f,commands in groups[task].items() if set(commands)=={0,1});assert families
            family=rng.choice(families);command=cycle%2
            options=sorted(groups[task][family][command],key=lambda x:(x['episode_id'],x['window_id']))
            episodes=sorted({x['episode_id'] for x in options});episode=rng.choice(episodes)
            row=rng.choice([x for x in options if x['episode_id']==episode]);source=row['window_id']
            visits[source]=visits.get(source,0)+1
            key='PI0255|25503|stage=MOTION3|source=%s|visit=%d'%(source,visits[source])
            seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')&((1<<63)-1)
            result.append(dict(optimizer_step=len(result)+1,task=task,family_id=family,command=command,source_episode_id=episode,
                source_window_id=source,visit=visits[source],rng_key=key,rng_seed=seed))
    assert len(result)==6000 and all(sum(x['task']==t for x in result)==2000 for t in groups)
    return result


def prepare(output):
    common=dict(schema='PI0255_MT_PILOT_CARD_DRAFT_V1',status='DRAFT_NOT_EXECUTION_AUTHORIZED',authority_commit='eac535e7c4016df39821798b111fae7ad79e8253',
        execution_authorization=None,data_ready=False,stage=dict(steps=6000,checkpoint_steps=[3000,6000],task_updates=dict(M1=2000,M2=2000,M3=2000)),
        optimizer=dict(name='AdamW',lr=1e-5,betas=[.9,.999],eps=1e-8,weight_decay=1e-5,max_grad_norm=1.,batch=1,accumulation=1,warmup_steps=10,scheduler='cosine6000',fresh=True),
        runtime=dict(S4=4,action_horizon=40,action_dt_s=.05,backbone='BF16_FROZEN',action_head='F32',tune_llm=False,tune_visual=False,tune_vlln=False,trainable=['projector','diffusion']),
        loss='original native flow matching plus unchanged G full pure-noise four-step physical objective',loss_coefficients=dict(ell_pos=.4,ell_delta=.1,ell_rot=.1,ell_speed_margin=.1),
        processor_proposal='PI0255_C_MT_TRAIN_ONLY_V1',embodiment=dict(name='new_embodiment',existing_slot=10,new_parameter_rows=0,Base='preserve original slot10',G='preserve adapted slot10'),
        sampling='same task/scenario/paired-command/episode/window schedule for both arms; TRAIN successful eligible rows only',
        randomness='source/visit keyed native tau, epsilon, S4 noise AND processor augmentation identity; actual bytes must be recorded',
        historical_cost='6000 NEW updates each; G prior adaptation cost additional and unequal',selected_checkpoint='NONE',
        unresolved=['all three expert admissions','successful TRAIN/MT-VAL episodes','every-window codec/gate and observability','TRAIN-only C_MT numerical values','pinned tokenizer and common actual processor proof','real6000 runner/source-visit input and full Trainer resume binding'])
    models=dict(BASE=dict(repo_id='nvidia/GR00T-N1.7-3B',revision='2fc962b973bccdd5d8ce4f67cc63b264d6886495',index_sha256='407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59'),
        G=dict(repo_id='leoyuer/pi0246-gate-pass-campaign',revision='efecea485ae77000599d19171b02491f8d514f42',prefix='arms/PI0252/r1-speed-margin-20260928/G/checkpoint-2000',Trainer_tree_sha256='cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226'))
    for arm,initializer in models.items():
        card=dict(common,arm='MT-'+arm,initializer=initializer);validate(card);write(output/('MT_'+arm+'_PILOT_CARD_DRAFT.json'),card)
    write(output/'NORMALIZATION_PROPOSAL.json',dict(id=common['processor_proposal'],status='BLOCKED_NO_ADMITTED_TRAIN_DATA',
        numerical_values=None,statistics_source='successful eligible TRAIN only; no pilot/VAL/TEST',proposal='retain verified native XYZ and row-packed rot6d + neutral gripper ABI, original relative-action conversion and padding/mask conventions; derive q01/q99 with numpy linear quantiles from new TRAIN raw support for applicable coordinates, same scales/masks/units for Base/G',
        inherited_C_T_modified=False,required_audits=['actual C_T clipping/outliers','inverse decode physical roundtrip','nonzero SE3','both initializer identical preprocessing','constant channel behavior and phase coverage'],actual_TRAIN_samples=0))


def cpu_check(output):
    import torch
    # Reuse the mature isolation helper, without importing any GR00T model.
    import pi0251_step_rng as rng
    rng.SEED=25503;rng.MAX_STEPS=6000;rng.configure_namespace('PI0255','MOTION3')
    fixtures=[dict(task=t,family_id='CPU-family',command=c,episode_id='CPU-'+t+str(c),window_id='CPU-'+t+str(c)+'-w0',split='TRAIN',success=True,eligible=True) for t in ('M1','M2','M3') for c in (0,1)]
    a=schedule(fixtures);assert a==schedule(fixtures)
    previous=random.getstate();schedule(fixtures);assert previous==random.getstate()
    lookup={x['optimizer_step']:x for x in a}
    rng.step_key_seed=lambda step:(lookup[step]['rng_key'],lookup[step]['rng_seed'])
    outer=torch.get_rng_state().clone()
    for step in (1,149,3000,3001,6000):
        with rng.rng_context(step):x=torch.randn(1,40,132)
        with rng.rng_context(step):y=torch.randn(1,40,132)
        assert torch.equal(x,y)
    assert torch.equal(outer,torch.get_rng_state())
    cards=[json.loads((output/('MT_'+arm+'_PILOT_CARD_DRAFT.json')).read_text()) for arm in ('BASE','G')]
    assert {k:v for k,v in cards[0].items() if k not in ('arm','initializer')}=={k:v for k,v in cards[1].items() if k not in ('arm','initializer')}
    poison=copy.deepcopy(cards[0]);poison['stage']['steps']=2000
    try:validate(poison)
    except AssertionError:pass
    else:raise AssertionError('legacy2000 accepted as6000')
    assert a[3000:][0]['optimizer_step']==3001 and len(a[3000:])==3000
    write(output/'draft_cpu_receipt.json',dict(status='PASS_DRAFT_CARDS_SAMPLER_RNG_BOUNDARY_ONLY',synthetic_schedule_rows=6000,actual_training_schedule_rows=0,
        task_balanced=True,command_balanced=True,paired_draws_equal=True,outer_rng_restored=True,legacy2000_rejected=True,cursor3000_suffix=True,
        real_Trainer_resume='UNRESOLVED_NO_ADMITTED_DATA_OR_EXECUTION_BINDING',processor_randomness_pairing='REQUIRED_NOT_YET_PROVED',new_GR00T_loads=0,new_GR00T_queries=0,new_GR00T_backward=0,new_GR00T_optimizer=0))
    # Reuse the actual qualified S4/mode implementation on a scalar CPU fixture.
    import math
    import tempfile
    import numpy as np
    from pi0247_u_rollout_overlay import _euler_rollout,_module_eval
    lr=lambda k:k/10 if k<10 else .5*(1+math.cos(math.pi*min(1.,(k-10)/5990)))
    assert lr(0)==lr(6000)==0. and lr(10)==1.
    def make():
        model=torch.nn.Linear(1,1).double().train()
        with torch.no_grad():model.weight.fill_(.37);model.bias.fill_(.07)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr);scheduler.last_epoch=2996
        for group in opt.param_groups:group['lr']=1e-5*lr(2996)
        return model,opt,scheduler
    def advance(state,steps):
        model,opt,scheduler=state;grads={};audit={}
        for step in steps:
            opt.zero_grad()
            with rng.rng_context(step):
                value=torch.randn(1,1,dtype=torch.float64,requires_grad=True)
                contexts=lambda:(_module_eval(model,audit,'rollout_forward'),_module_eval(model,audit,'backward_recompute'))
                def hook(index,value):value.register_hook(lambda gradient:grads.update({index:float(gradient.abs().sum())}))
                result,_,_=_euler_rollout(value,lambda current,index:torch.tanh(model(current)),(),steps=4,checkpointed=True,context_fn=contexts,state_hook=hook)
                result.square().sum().backward()
            assert len(grads)==4 and all(x>0 for x in grads.values())
            assert all(m.training for m in model.modules()) and audit.get('phase') is None
            opt.step();scheduler.step()
        return state
    reference=advance(make(),range(2997,3001))
    with tempfile.TemporaryDirectory(dir=output,prefix='CPU-fixture-state-') as folder:
        path=Path(folder)/'full_fixture_state.pt'
        torch.save(dict(model=reference[0].state_dict(),optimizer=reference[1].state_dict(),scheduler=reference[2].state_dict(),
            torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),cursor=3000,steps=6000),path)
        expected=advance(reference,range(3001,3005));restored=make();saved=torch.load(path,weights_only=False,map_location='cpu')
        assert saved['cursor']==3000 and saved['steps']==6000
        for object_,name in zip(restored,('model','optimizer','scheduler')):object_.load_state_dict(saved[name])
        torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        _=torch.rand(11)  # unrelated restore/iterator consumer; keyed model draws remain unchanged
        actual=advance(restored,range(saved['cursor']+1,3005))
        def equal(a,b):
            if isinstance(a,torch.Tensor):return torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        assert all(equal(a.state_dict(),b.state_dict()) for a,b in zip(expected,actual))
    write(output/'resume_s4_cpu_fixture_receipt.json',dict(status='PASS_SCALAR_CPU_FULL_STATE_RESUME_6000_SCHEDULER_AND_QUALIFIED_S4',
        real_GR00T_Trainer_resume=False,fixture_cursor=3000,fixture_last_cursor=3004,all_four_state_gradients=True,modes_restored=True,
        model_optimizer_scheduler_exact=True,restore_interference_rng_draws=11,processor_augmentation_pairing=False,
        actual_training_schedule_rows=0,new_GR00T_loads=0,new_GR00T_queries=0,new_GR00T_backward=0,new_GR00T_optimizer=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-check',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--validate-only',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--training-authorization');args=p.parse_args()
    if args.train:raise SystemExit('PI0255 grants ZERO model training. A later admitted data/card/runner and explicit execution authority are required.')
    if args.prepare:prepare(args.output)
    if args.cpu_check:cpu_check(args.output)
    for arm in ('BASE','G'):validate(json.loads((args.output/('MT_'+arm+'_PILOT_CARD_DRAFT.json')).read_text()))
    print(json.dumps(dict(status='DRAFT_VALIDATED_NO_MODEL_EXECUTION',data_ready=False,training_authorized=False)))
