"""Train/reload matched one-step target contact heads on real qualified episodes."""
import argparse,json,time,random,resource
from pathlib import Path


def run(spec,output):
    import numpy as np,torch
    from go2arx5_vla.world_model.pilot1_contact_wm import replay,ContactWM,metrics
    from go2arx5_vla.world_model.neural_training import rng_state,restore_rng,tree_error
    from go2arx5_vla.world_model.storage import digest,write_json
    start=time.monotonic();cpu=time.process_time();torch.set_num_threads(len(spec['cpus']));torch.use_deterministic_algorithms(True)
    if spec['device']!='cpu' or spec['kind']!='CONTACT_WM':raise ValueError('REGISTERED_LOCAL_CPU_CONTACT_MODEL_ONLY')
    for item in spec['bundles']:
        if digest(item['path'])!=item['sha256']:raise ValueError('ACTUAL_SEALED_CONTACT_BUNDLE_CHANGED')
    data=replay([item['path']for item in spec['bundles']]);train=np.flatnonzero(data['split']=='TRAIN');dev=np.flatnonzero(data['split']=='DEV')
    if not len(train)or not len(dev)or np.any(data['split']=='TEST'):
        raise ValueError('NONEMPTY_WHOLE_PARENT_TRAIN_DEV_WITH_TEST_RESERVED_REQUIRED')
    t=lambda x:torch.as_tensor(x,dtype=torch.float32);y=t(data['delta_mm']);event=t(data['event']);arms={};updates=0
    tensor=t(data['x']);baseline={}
    frequency=float(data['event'][train].mean())
    for name,delta,prob in [('PERSISTENCE',np.zeros_like(data['delta_mm']),data['persistence_event']),
                            ('CONSTANT_VELOCITY',data['cv_mm'],data['persistence_event']),
                            ('TRAIN_EVENT_FREQUENCY',np.zeros_like(data['delta_mm']),np.full_like(data['event'],frequency))]:
        baseline[name]={split:metrics(delta,prob,data,ids)for split,ids in [('TRAIN',train),('DEV',dev)]}
    write_json(output/'BASELINES.json',baseline)
    for arm,field in [('ACTION_CONDITIONED','x'),('NO_NEW_REFERENCE','blind')]:
        random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed']);x=t(data[field])
        model=ContactWM(x.shape[1],spec['hidden']);model.fit_normalizer(tensor[train]);opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=1e-4);cursor=0;folder=output/arm;folder.mkdir()
        def update(m,o,at):
            ids=torch.as_tensor(train[np.arange(at,at+spec['batch_size'])%len(train)])
            prediction=m(x[ids]);value=(prediction[:,:6]-y[ids]).square().mean()+torch.nn.functional.binary_cross_entropy_with_logits(prediction[:,6],event[ids])
            o.zero_grad(set_to_none=True);value.backward()
            if not torch.isfinite(value)or any(p.grad is not None and not torch.isfinite(p.grad).all()for p in m.parameters()):
                raise FloatingPointError('NONFINITE_CONTACT_LOSS_OR_GRADIENT')
            o.step();return at+spec['batch_size'],float(value.detach())
        def save(step):
            torch.save(dict(schema='WM02_PILOT1_CONTACT_CHECKPOINT_V1',source_sha=spec['source_sha'],arm=arm,
                model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),cursor=cursor,updates=step,
                bundles=spec['bundles'],forecast_atoms=1,partial_taskspace_not_complete_state=True),folder/f'checkpoint-{step}.pt')
        for step in range(1,spec['endpoint']+1):
            if time.time()>=spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():raise TimeoutError('FINITE_CONTACT_MODEL_ENDPOINT_OR_STOP')
            cursor,last=update(model,opt,cursor);updates+=1
            if step in(1,200,800,spec['endpoint']-1,spec['endpoint']):save(step)
        saved_rng=rng_state();clone=ContactWM(x.shape[1],spec['hidden']);co=torch.optim.AdamW(clone.parameters(),lr=spec['learning_rate'],weight_decay=1e-4)
        previous=torch.load(folder/f"checkpoint-{spec['endpoint']-1}.pt",weights_only=True);clone.load_state_dict(previous['model']);co.load_state_dict(previous['optimizer']);restore_rng(previous['rng'])
        recovered_cursor,_=update(clone,co,previous['cursor']);error=max(tree_error(model.state_dict(),clone.state_dict()),tree_error(opt.state_dict(),co.state_dict()),tree_error(saved_rng,rng_state()))
        endpoint=torch.load(folder/f"checkpoint-{spec['endpoint']}.pt",weights_only=True);reload=max(tree_error(model.state_dict(),endpoint['model']),tree_error(opt.state_dict(),endpoint['optimizer']))
        if error!=0 or reload!=0 or recovered_cursor!=cursor:raise ValueError('CONTACT_COMPLETE_CHECKPOINT_RESTORE_DIFFERS')
        with torch.no_grad():prediction=model(x);prob=prediction[:,6].sigmoid().numpy();delta=prediction[:,:6].numpy()
        np.savez_compressed(folder/'PREDICTIONS.private.npz',delta_mm=delta,probability=prob)
        arms[arm]=dict(main_WM_updates=spec['endpoint'],restore_probe_optimizer_calls=1,full_restore_error=error,endpoint_reload_error=reload,
                       checkpoint_sha256=digest(folder/f"checkpoint-{spec['endpoint']}.pt"),last_loss=last,
                       metrics={split:metrics(delta,prob,data,ids)for split,ids in [('TRAIN',train),('DEV',dev)]})
    result=dict(status='ACTUAL_CONTACT_ONE_STEP_WM_TRAINED_AND_RESTORED',source_sha=spec['source_sha'],bundles=spec['bundles'],
                main_WM_updates=updates,restore_probe_optimizer_calls=2,Actor_updates=0,native_integrations=0,
                TRAIN_transitions=len(train),DEV_transitions=len(dev),TRAIN_parents=sorted({i['parent']for i,s in zip(data['identities'],data['split'])if s=='TRAIN'}),
                DEV_parents=sorted({i['parent']for i,s in zip(data['identities'],data['split'])if s=='DEV'}),
                arms=arms,baselines=baseline,forecast_atoms=1,free_rollout_qualified=False,full_state_or_Markov_claim=False,
                native_policy_gain_evaluated=False,full_manipulation_success=False,device='cpu',
                CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(output/'RESULT.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.output.name,'wm02-pilot1-offline-cpu',a.output)as spec:print(json.dumps(run(spec,a.output)))
    return 0


if __name__=='__main__':raise SystemExit(main())
