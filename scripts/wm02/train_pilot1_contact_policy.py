"""Distill the empirical TRAIN exit-return model; report its constant-action limit."""
import argparse,gzip,json,time,resource
from pathlib import Path


def dataset(bundles):
    import numpy as np
    from go2arx5_vla.world_model.pilot1_contact_policy_numpy import features
    from go2arx5_vla.world_model.pilot1_contact import exit_score
    from go2arx5_vla.world_model.pilot1_contact_wm import replay
    replay([b['path']for b in bundles])  # Verify every force label, native tick and parent partition.
    records=[]
    for item in bundles:
        for e in json.loads(gzip.decompress(Path(item['path']).read_bytes()))['episodes']:
            if e['split']=='TEST':raise ValueError('TEST_RESERVED_UNTIL_POLICY_FREEZE')
            if len(e['samples'])!=128:continue
            r=e['samples'];offset=e['episode']['withdrawal_xyz_m']
            if offset not in ([0.,0.,0.],[0.,.001,0.]):raise ValueError('QUALIFIED_HOLD_OR_PLUS_Y_ONLY')
            action=int(any(offset));events=[v for row in r for v in row['native_contact_events']]
            score=exit_score(events,r[63]['TCP_m'],r[-1]['TCP_m'],e['score']['actual_table_support_at_end'])
            if score['qualified_contact_exit']!=e['score']['qualified_contact_exit']:raise ValueError('ACTUAL_EXIT_RETURN_DIFFERS')
            before=np.r_[r[63]['TCP_m'],np.asarray(r[63]['object_pose'])[:3,3]]
            curve=np.asarray([np.r_[v['TCP_m'],np.asarray(v['object_pose'])[:3,3]]-before for v in r[64:]])
            records.append(dict(x=features(r[:64]),action=action,success=float(score['qualified_contact_exit']),
                                curve=curve,split=e['split'],parent=e['parent'],run=e['run_id']))
    train=[r for r in records if r['split']=='TRAIN'];dev=[r for r in records if r['split']=='DEV']
    parents=sorted({r['parent']for r in train})
    if len(parents)<2 or not dev or any({r['action']for r in train if r['parent']==p}!={0,1}for p in parents):
        raise ValueError('MATCHED_TWO_ACTIONS_ON_MULTIPLE_TRAIN_PARENTS_REQUIRED')
    returns=np.asarray([np.mean([r['success']for r in train if r['action']==a])for a in (0,1)])
    if np.ptp(returns)<=0:raise ValueError('EMPIRICAL_EXIT_ACTION_SEPARATION_NOT_ESTABLISHED')
    teacher=dict(schema='WM02_PILOT1_EMPIRICAL_CONTACT_RETURN_V1',TRAIN_parents=parents,
                 expected_exit_return=returns.tolist(),preferred_action=int(returns.argmax()),
                 mean_TCP_object_response_curve_m=[np.mean([r['curve']for r in train if r['action']==a],axis=0).tolist()for a in (0,1)],
                 forecast_atoms=64,state_conditional=False,full_simulator=False,
                 limitation='GLOBAL_TWO_ACTION_RETURN_MODEL_MAY_EQUAL_ALWAYS_PLUS_Y_RULE')
    return train,dev,teacher


def run(spec,output):
    import random,numpy as np,torch
    from go2arx5_vla.world_model.storage import digest,write_json
    from go2arx5_vla.world_model.neural_training import rng_state,restore_rng,tree_error,memory
    from go2arx5_vla.world_model.pilot1_contact_policy_numpy import logits
    started=time.monotonic();cpu=time.process_time();device=torch.device(spec['device'])
    if spec['kind']!='CONTACT_POLICY':raise ValueError('REGISTERED_CONTACT_POLICY_PILOT_REQUIRED')
    if device.type=='cuda':
        from go2arx5_vla.world_model.runtime import cuda_uuid
        props=torch.cuda.get_device_properties(0)
        if torch.cuda.device_count()!=1 or cuda_uuid(props.uuid)!=spec['GPU_uuid']:raise PermissionError('ONE_REGISTERED_GPU_REQUIRED')
        torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/props.total_memory);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(len(spec['cpus']));torch.use_deterministic_algorithms(True)
    random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed'])
    for b in spec['bundles']:
        if digest(b['path'])!=b['sha256']:raise ValueError('SEALED_CONTACT_POLICY_DATA_CHANGED')
    train,dev,teacher=dataset(spec['bundles']);write_json(output/'TEACHER.private.json',teacher)
    t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
    x=t([r['x']for r in train]);dx=t([r['x']for r in dev]);mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-4)
    labels=torch.full((len(train),),teacher['preferred_action'],dtype=torch.long,device=device)
    def create():return torch.nn.Sequential(torch.nn.Linear(54,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).to(device)
    model=create();opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=1e-4);cursor=0
    def update(m,o,at):
        ids=torch.arange(at,at+spec['batch_size'],device=device)%len(train);value=torch.nn.functional.cross_entropy(m((x[ids]-mean)/scale),labels[ids])
        o.zero_grad(set_to_none=True);value.backward()
        if not torch.isfinite(value)or any(p.grad is not None and not torch.isfinite(p.grad).all()for p in m.parameters()):raise FloatingPointError('NONFINITE_CONTACT_POLICY')
        o.step();return at+spec['batch_size'],float(value.detach())
    def save(step):
        torch.save(dict(schema='WM02_PILOT1_CONTACT_POLICY_CHECKPOINT_V1',source_sha=spec['source_sha'],model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),cursor=cursor,updates=step,mean=mean,scale=scale,teacher=teacher,bundles=spec['bundles']),output/f'checkpoint-{step}.pt')
    for step in range(1,spec['endpoint']+1):
        if time.time()>=spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():raise TimeoutError('FINITE_CONTACT_POLICY_ENDPOINT_OR_STOP')
        cursor,last=update(model,opt,cursor)
        if step in(1,200,800,spec['endpoint']-1,spec['endpoint']):save(step)
    expected=rng_state();clone=create();co=torch.optim.AdamW(clone.parameters(),lr=spec['learning_rate'],weight_decay=1e-4)
    c=torch.load(output/f"checkpoint-{spec['endpoint']-1}.pt",map_location='cpu',weights_only=True);clone.load_state_dict(c['model']);co.load_state_dict(c['optimizer']);restore_rng(c['rng']);restored_cursor,_=update(clone,co,c['cursor'])
    error=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()),tree_error(expected,rng_state()),tree_error(mean,c['mean']),tree_error(scale,c['scale']))
    final=torch.load(output/f"checkpoint-{spec['endpoint']}.pt",map_location='cpu',weights_only=True);reload=max(tree_error(model.state_dict(),final['model']),tree_error(opt.state_dict(),final['optimizer']))
    if error!=0 or reload!=0 or restored_cursor!=cursor:raise ValueError('CONTACT_POLICY_COMPLETE_RESTORE_DIFFERS')
    arrays=dict(mean=mean.cpu().numpy(),scale=scale.cpu().numpy())
    for i in (0,2,4):arrays.update({f'w{i}':model[i].weight.detach().cpu().numpy(),f'b{i}':model[i].bias.detach().cpu().numpy()})
    np.savez_compressed(output/'POLICY_NUMPY.npz',**arrays)
    with torch.no_grad():expected_logits=model((dx-mean)/scale).cpu().numpy();train_choice=model((x-mean)/scale).argmax(-1).cpu().numpy()
    actual=logits(arrays,dx.cpu().numpy());portable=float(np.max(np.abs(actual-expected_logits)))
    if portable>5e-5 or not np.array_equal(actual.argmax(-1),expected_logits.argmax(-1)):raise ValueError('PORTABLE_CONTACT_POLICY_DIFFERS')
    result=dict(status='CONTACT_EXIT_POLICY_TRAINED_RELOADABLE_NATIVE_EVALUATION_PENDING',source_sha=spec['source_sha'],main_Actor_updates=spec['endpoint'],restore_probe_optimizer_calls=1,WM_optimizer_updates=0,native_integrations=0,
                checkpoint_sha256=digest(output/f"checkpoint-{spec['endpoint']}.pt"),portable_policy_sha256=digest(output/'POLICY_NUMPY.npz'),teacher_sha256=digest(output/'TEACHER.private.json'),complete_restore_error=error,endpoint_reload_error=reload,portable_max_logit_error=portable,
                TRAIN_prefixes=len(train),DEV_prefixes=len(dev),TRAIN_parents=teacher['TRAIN_parents'],expected_exit_return=teacher['expected_exit_return'],TRAIN_action_counts=np.bincount(train_choice,minlength=2).tolist(),DEV_action_counts=np.bincount(actual.argmax(-1),minlength=2).tolist(),
                state_conditioned_action_advantage_established=False,strong_rule_advantage_established=False,full_manipulation_success=False,device=str(device),GPU_memory=memory(device),last_loss=last,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(output/'RESULT.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.output.name,'wm02-pilot1-neural',a.output,gpu=True)as spec:print(json.dumps(run(spec,a.output)))
    return 0


if __name__=='__main__':raise SystemExit(main())
