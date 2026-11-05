"""Train a goal-conditioned KEEP/RELEASE student on actually qualified prefixes."""
import argparse
import gzip
import json
import random
import time
from pathlib import Path


def run(spec, output):
    import numpy as np
    import torch
    from go2arx5_vla.world_model.pilot2_supported_decision import dataset
    from go2arx5_vla.world_model.pilot2_contact_model import replay
    from go2arx5_vla.world_model.pilot1_contact_policy_numpy import logits
    from go2arx5_vla.world_model.actor_sampling import ExactSampler
    from go2arx5_vla.world_model.neural_training import rng_state, restore_rng, tree_error, memory
    from go2arx5_vla.world_model.storage import digest, write_json
    from go2arx5_vla.world_model.runtime import cuda_uuid
    start, cpu = time.monotonic(), time.process_time(); device = torch.device(spec['device'])
    if (spec['kind']!='CONTACT_POLICY' or spec.get('policy_recipe')!='WM02_PILOT2_SUPPORTED_GOAL_POLICY_V1'
            or type(spec.get('endpoint')) is not int or spec['endpoint']<2):
        raise ValueError('FINITE_SUPPORTED_GOAL_POLICY_RECIPE_REQUIRED')
    torch.set_num_threads(len(spec['cpus'])); torch.use_deterministic_algorithms(True)
    if device.type=='cuda':
        props = torch.cuda.get_device_properties(0)
        if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:
            raise PermissionError('ONE_ACTUALLY_REGISTERED_POLICY_GPU_REQUIRED')
        torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory)
        torch.backends.cuda.matmul.allow_tf32=False; torch.cuda.reset_peak_memory_stats()
    random.seed(spec['seed']); np.random.seed(spec['seed']); torch.manual_seed(spec['seed'])
    episodes=[]
    for b in spec['bundles']:
        if digest(b['path'])!=b['sha256']: raise ValueError('ACTUAL_SUPPORTED_POLICY_BUNDLE_CHANGED')
        v=json.loads(gzip.decompress(Path(b['path']).read_bytes()))
        if v['schema']!='WM02_PILOT2_SUPPORTED_ACTUAL_BUNDLE_V1': raise ValueError('ACTUAL_SUPPORTED_BUNDLE_REQUIRED')
        episodes.extend(v['episodes'])
    replay(episodes)  # Independently verify real native ticks, labels and partitions.
    wm=None
    if spec['teacher_method']=='PARTIAL_WM_EXPECTED_EVENT':
        if digest(spec['wm_numpy'])!=spec['wm_numpy_sha256']: raise ValueError('FROZEN_PARTIAL_CONTACT_WM_CHANGED')
        with np.load(spec['wm_numpy'],allow_pickle=False) as z: wm={k:z[k].copy() for k in z.files}
    data=dataset(episodes,teacher=spec['teacher_method'],wm_arrays=wm)
    train,dev=[np.flatnonzero(data['split']==s) for s in ('TRAIN','DEV')]
    if not len(train) or not len(dev): raise ValueError('NONEMPTY_WHOLE_PARENT_SUPPORTED_TRAIN_DEV_REQUIRED')
    t=lambda v:torch.as_tensor(v,dtype=torch.float32,device=device)
    x,labels=t(data['x']),t(data['labels']); mean=x[train].mean(0); scale=x[train].std(0,unbiased=False).clamp_min(1e-4)
    normalization_origin='CURRENT_TRAIN_INPUTS'
    if spec.get('normalizer_reference'):
        if spec.get('resume'):raise ValueError('FRESH_NORMALIZER_CONTROL_MUST_NOT_SILENTLY_CHANGE_RESUMED_INPUTS')
        if digest(spec['normalizer_reference'])!=spec['normalizer_reference_sha256']:
            raise ValueError('FROZEN_TRAIN_NORMALIZER_REFERENCE_CHANGED')
        reference=torch.load(spec['normalizer_reference'],map_location='cpu',weights_only=True)
        if (reference['schema']!='WM02_PILOT2_SUPPORTED_POLICY_CHECKPOINT_V1'
                or reference['mean'].shape!=(x.shape[1],) or reference['scale'].shape!=(x.shape[1],)
                or not torch.isfinite(reference['mean']).all() or not torch.isfinite(reference['scale']).all()
                or not (reference['scale']>0).all()):
            raise ValueError('EXPLICIT_TRAIN_POLICY_NORMALIZER_REFERENCE_REQUIRED')
        mean=reference['mean'].to(device);scale=reference['scale'].to(device)
        normalization_origin='PINNED_PRIOR_TRAIN_INPUTS_FRESH_MODEL_CONTROL'
    def create(): return torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.Tanh(),
        torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).to(device)
    model=create(); opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=1e-4)
    sampler=ExactSampler(len(train),'SEEDED_GLOBAL_SHUFFLE',spec['seed']); previous=0
    identities=[dict(run_id=e['run_id'],parent=e['parent'],split=e['split']) for e in episodes]
    if spec.get('resume'):
        if digest(spec['resume'])!=spec['resume_sha256']: raise ValueError('PRIOR_SUPPORTED_POLICY_CHANGED')
        c=torch.load(spec['resume'],map_location='cpu',weights_only=True)
        if (c['schema']!='WM02_PILOT2_SUPPORTED_POLICY_CHECKPOINT_V1' or c['teacher_method']!=spec['teacher_method']
                or c['seed']!=spec['seed'] or any(e not in identities for e in c['episode_identities'])):
            raise ValueError('POLICY_CONTINUATION_MUST_RETAIN_OLD_EPISODES_PARTITIONS_AND_TEACHER_KIND')
        if c.get('wm_numpy_sha256')!=spec.get('wm_numpy_sha256') and not spec.get('updated_wm_teacher'):
            raise ValueError('WM_TEACHER_UPDATE_MUST_BE_EXPLICIT')
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);mean=c['mean'].to(device);scale=c['scale'].to(device)
        previous=c['updates'];restore_rng(c['rng'])
        normalization_origin='EXPLICIT_FROZEN_RESUMED_TRAIN_NORMALIZER'
        if c['bundles']==spec['bundles']: sampler.load_state_dict(c['sampler'])
        elif spec.get('data_expansion_normalizer')!='FROZEN_PRIOR_TRAIN_NORMALIZER_NEW_SHUFFLE':
            raise ValueError('NEW_POLICY_DATA_NORMALIZER_AND_SAMPLER_MUST_BE_EXPLICIT')
    def update(m,o,s):
        ids=torch.as_tensor(train[s.take(spec['batch_size'])],device=device)
        v=-(labels[ids]*torch.log_softmax(m((x[ids]-mean)/scale),-1)).sum(-1).mean()
        o.zero_grad(set_to_none=True);v.backward()
        if not torch.isfinite(v) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):
            raise FloatingPointError('NONFINITE_SUPPORTED_POLICY_LOSS_OR_GRADIENT')
        o.step();return float(v.detach())
    def save(step):
        torch.save(dict(schema='WM02_PILOT2_SUPPORTED_POLICY_CHECKPOINT_V1',source_sha=spec['source_sha'],
            teacher_method=spec['teacher_method'],wm_numpy_sha256=spec.get('wm_numpy_sha256'),seed=spec['seed'],
            model=model.state_dict(),optimizer=opt.state_dict(),mean=mean,scale=scale,
            normalization_origin=normalization_origin,normalizer_reference_sha256=spec.get('normalizer_reference_sha256'),
            rng=rng_state(),sampler=sampler.state_dict(),updates=step,bundles=spec['bundles'],episode_identities=identities),
            output/f'checkpoint-{step}.pt')
    def measure():
        result={}
        with torch.no_grad(): pred=model((x-mean)/scale);p=pred.softmax(-1);choice=pred.argmax(-1)
        for split,ids in [('TRAIN',train),('DEV',dev)]:
            ix=torch.as_tensor(ids,device=device);values=choice[ix].cpu().numpy()
            result[split]=dict(examples=len(ids),teacher_agreement=float((choice[ix]==labels[ix].argmax(-1)).float().mean()),
                action_counts=np.bincount(values,minlength=2).tolist(),by_goal={})
            for goal in sorted({data['identities'][i]['goal'] for i in ids}):
                selected=np.array([i for i in ids if data['identities'][i]['goal']==goal])
                result[split]['by_goal'][goal]=np.bincount(choice[selected].cpu().numpy(),minlength=2).tolist()
        return result
    write_json(output/'BEFORE_UPDATE.json',measure())
    for step in range(1,spec['endpoint']+1):
        if (time.time()>=spec['deadline_epoch'] or (Path(spec['registered_root'])/'STOP.json').exists()
                or spec.get('campaign_root') and (Path(spec['campaign_root'])/'STOP.json').exists()):
            raise TimeoutError('FINITE_SUPPORTED_POLICY_ENDPOINT_OR_STOP')
        loss=update(model,opt,sampler)
        with(output/'OPTIMIZER_STEPS.jsonl').open('a') as f:f.write(json.dumps(dict(new_update=step,cumulative_update=previous+step,loss=loss))+'\n')
        if step in (spec['endpoint']-1,spec['endpoint']) or step%500==0:save(previous+step)
    expected=rng_state();clone=create();co=torch.optim.AdamW(clone.parameters(),lr=spec['learning_rate'],weight_decay=1e-4)
    c=torch.load(output/f"checkpoint-{previous+spec['endpoint']-1}.pt",map_location='cpu',weights_only=True)
    clone.load_state_dict(c['model']);co.load_state_dict(c['optimizer']);restore_rng(c['rng'])
    cs=ExactSampler(len(train),'SEEDED_GLOBAL_SHUFFLE',spec['seed']);cs.load_state_dict(c['sampler'])
    write_json(output/'RESTORE_PROBE_STARTED.json',dict(optimizer_calls_requested=1));update(clone,co,cs)
    write_json(output/'RESTORE_PROBE_EXECUTED.json',dict(actual_optimizer_calls=1))
    error=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()),
        tree_error(sampler.state_dict(),cs.state_dict()),tree_error(expected,rng_state()),tree_error(mean,c['mean']),tree_error(scale,c['scale']))
    final=output/f"checkpoint-{previous+spec['endpoint']}.pt";c=torch.load(final,map_location='cpu',weights_only=True)
    reload=max(tree_error(model.state_dict(),c['model']),tree_error(opt.state_dict(),c['optimizer']),tree_error(sampler.state_dict(),c['sampler']))
    if error or reload:raise ValueError('SUPPORTED_POLICY_FULL_RESTORE_DIFFERS')
    arrays=dict(mean=mean.cpu().numpy(),scale=scale.cpu().numpy())
    for i in (0,2,4):arrays.update({f'w{i}':model[i].weight.detach().cpu().numpy(),f'b{i}':model[i].bias.detach().cpu().numpy()})
    portable=output/'POLICY_NUMPY.npz'
    with portable.open('xb') as f:np.savez_compressed(f,**arrays)
    with torch.no_grad():expected_logits=model((x[dev]-mean)/scale).cpu().numpy()
    actual=logits(arrays,x[dev].cpu().numpy());numpy_error=float(np.max(np.abs(actual-expected_logits)))
    if numpy_error>5e-5 or not np.array_equal(actual.argmax(-1),expected_logits.argmax(-1)):
        raise ValueError('SUPPORTED_POLICY_NUMPY_TORCH_DIFFERS')
    result=dict(status='SUPPORTED_GOAL_POLICY_TRAINED_NATIVE_EVALUATION_PENDING',source_sha=spec['source_sha'],
        bundles=spec['bundles'],wm_numpy_sha256=spec.get('wm_numpy_sha256'),decision_atom=112,
        teacher_method=spec['teacher_method'],main_Actor_updates=spec['endpoint'],prior_Actor_updates=previous,
        normalization_origin=normalization_origin,normalizer_reference_sha256=spec.get('normalizer_reference_sha256'),
        WM_updates=0,restore_probe_optimizer_calls=1,complete_restore_error=error,endpoint_reload_error=reload,
        checkpoint_sha256=digest(final),portable_policy_sha256=digest(portable),portable_max_logit_error=numpy_error,
        TRAIN_parents=sorted({x['parent'] for x in identities if x['split']=='TRAIN'}),
        DEV_parents=sorted({x['parent'] for x in identities if x['split']=='DEV'}),metrics=measure(),
        goal_conditioning=True,constant_action_gain_NOT_ESTABLISHED=True,native_policy_evaluated=False,
        native_integrations=0,TEST_read=False,GPU_memory=memory(device),CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'RESULT.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.output.name,'wm02-pilot1-neural',a.output,gpu=True) as s:print(json.dumps(run(s,a.output)))
    return 0


if __name__=='__main__':raise SystemExit(main())
