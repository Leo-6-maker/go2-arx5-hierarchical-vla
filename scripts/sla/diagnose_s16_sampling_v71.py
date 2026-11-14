"""Fixed states/noise: saved step0/100/500 and endpoint10/20/40 sampling.

Inference only. Original packages, formal10-step policy, data and safety stay
unchanged. A diagnostic output never grants native execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha

OPERATION='SLA_S16_FIXED_CHECKPOINT_AND_SAMPLER_DIAGNOSTIC_V7_1'
CONFIGURATIONS=[(0,10),(100,10),(500,10),(500,20),(500,40)]


def admission(spec):
    if (spec['operation']!=OPERATION or spec['scope']!='SLA_FIXED_CHECKPOINT_DIAGNOSTICS_ONLY'
            or spec['physics'] is not False or spec['native_execution_admitted'] is not False
            or spec['planned_generations']!=160 or spec['optimizer_updates']!=0
            or spec['configurations']!=[list(v) for v in CONFIGURATIONS]):
        raise PermissionError('FIXED160_INFERENCE_ONLY_NO_TRAINING_OR_NATIVE_REQUIRED')


def restore_active(provider,checkpoint):
    import jax
    from go2arx5_vla.sla_adapter.provider_v1 import active_rows
    m=json.loads((checkpoint/'COMPLETE.json').read_bytes());rows,tree=active_rows(m,provider.active)
    assert m['bindings']==provider.manifest['bindings']
    values=[]
    for row in rows:
        p=checkpoint/row['file'];assert sha(p)==row['sha256']
        a=np.load(p,allow_pickle=False);assert str(a.dtype)==row['dtype'] and list(a.shape)==row['shape']
        values.append(jax.device_put(a))
    provider.active,=jax.tree_util.tree_unflatten(tree,values)
    jax.block_until_ready(provider.active)
    return dict(step=m['step'],cursor=m['cursor'],adapter_arrays=len(rows),optimizer_loaded=False,
        complete_sha256=sha(checkpoint/'COMPLETE.json'))


def run(spec):
    admission(spec)
    from go2arx5_vla.sla_adapter.site_v2 import validate
    validate(spec)
    from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
    from go2arx5_vla.sla_adapter.provider_v3 import Provider
    from go2arx5_vla.sla_adapter.runtime_s16_v7 import S16OfflineEngine
    from go2arx5_vla.sla_adapter.s16_offline_v7 import fixed_indices
    from scripts.sla.train_s16_offline_v7 import prediction_metrics
    from scripts.sla.diagnose_eef_margin_v71 import compare
    import jax
    import flax.nnx as nnx
    root=Path(spec['registered_root']);owner=Path(spec['S16_owner'])
    old=read_pin(spec['S16_result']);assert old['complete500'] and old['step']==500 and old['target_visits']==16000
    original_spec=json.loads((owner/'EXECUTION_SPEC.json').read_bytes())
    assert sha(owner/'EXECUTION_SPEC.json')==spec['S16_spec_sha256']
    assert json.loads((owner/'control/LEASE.json').read_bytes())['status']=='CLOSED'
    e=S16OfflineEngine.__new__(S16OfflineEngine);e.spec=spec
    data=e.dataset_for_spec(spec);indices=fixed_indices(data.manifest,read_pin(spec['S16_selection']))
    baseline=json.loads((owner/'candidate500/work/step500-RGB/MANIFEST.private.json').read_bytes())
    original={(r['index'],r['noise']):r for r in baseline};rows=[];attempts=0;start=time.time()
    provider=Provider(old['package']['path'],manifest_sha256=old['package']['sha256'],
        parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
    ordinary10=provider.infer;restores=[];last=None;functions={10:ordinary10};graph=provider.graph
    noise_hashes={n:hashlib.sha256(np.asarray(jax.random.normal(jax.random.PRNGKey(2026111200+n),(1,40,32))).tobytes()).hexdigest() for n in (0,1)}
    for n in (20,40):
        def make(steps):
            return jax.jit(lambda active,frozen,obs,noise:nnx.merge(graph,active,frozen).sample_actions(
                jax.random.PRNGKey(0),obs,noise=noise,num_steps=steps))
        functions[n]=make(n)
    for checkpoint,steps in CONFIGURATIONS:
        if checkpoint!=last:
            pin=old['checkpoints'][str(checkpoint)]
            assert sha(Path(pin['path'])/'COMPLETE.json')==pin['sha256']
            restores.append(restore_active(provider,Path(pin['path'])));last=checkpoint
        provider.infer=functions[steps];folder=root/f'step{checkpoint:04d}-sampling{steps:02d}';folder.mkdir()
        for index in indices:
            state,_,images,text=data.raw(index);identity=data.identity(index)
            frame=data.frames[identity['episode_id']]['T_policy_from_mount']
            ep,anchor=data.rows[index];labels=data.labels[ep['id']]
            for noise in (0,1):
                if time.time()>=spec['deadline_epoch']:raise TimeoutError('FINITE_DIAGNOSTIC_ENDPOINT_REACHED_NO_RETRY')
                attempts+=1;started=time.time();value=provider.predict(state,images,text,frame,noise_id=noise)
                normalized,physical=value['normalized'],value['physical_H40']
                hashes=dict(normalized_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),physical_sha256=hashlib.sha256(physical.tobytes()).hexdigest())
                if checkpoint==500 and steps==10:
                    assert all(hashes[k]==original[(index,noise)][k] for k in hashes),'ORIGINAL_FIXED10_BASELINE_MUST_BE_BIT_EXACT'
                name=f'{index:04d}-{noise}.npz';np.savez(folder/name,normalized=normalized,physical_H40=physical)
                detail=compare(physical,labels['nominal_policy_H40'][anchor],labels['state_eef10'][anchor])
                row=dict(checkpoint=checkpoint,sampling_steps=steps,index=index,noise=noise,
                    initial_noise_sha256=noise_hashes[noise],
                    file=str((folder/name).relative_to(root)),**hashes,
                    **prediction_metrics(data,index,normalized,physical),boundary_analysis=detail,wall_s=time.time()-started)
                rows.append(row)
                with(root/'GENERATIONS.private.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps(dict(completed=len(rows),planned=160,checkpoint=checkpoint,steps=steps,EEF_pass=row['EEF_full_H40_pass'])),flush=True)
    assert len(rows)==attempts==160
    summary=[]
    for step,n in CONFIGURATIONS:
        group=[r for r in rows if (r['checkpoint'],r['sampling_steps'])==(step,n)]
        summary.append(dict(checkpoint=step,sampling_steps=n,predictions=len(group),EEF_pass=sum(r['EEF_full_H40_pass'] for r in group),
            all_fit_pass=sum(r['EEF_full_H40_pass'] and r['xyz_RMSE_m']<=.002 and r['mean_SO3_rad']<=.01 and r['nominal_grip_MAE']<=.02 for r in group)))
    result=dict(schema=OPERATION,status='FIXED160_GENERATIONS_COMPLETE',exit_code=0,actual_parent_loads=1,
        generation_attempts=attempts,actual_generations=len(rows),configurations=summary,restores=restores,
        original_step500_sampling10_bit_exact=True,optimizer_updates=0,training_visits=0,physical_requests=0,
        physics_integrations=0,formal_gate_or_policy_changed=False,per_window_best_selection=False,
        full_WBC_state_restored=False,native_execution_admitted=False,wall_s=time.time()-start)
    (root/'RESULT.private.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);a=p.parse_args()
    print(json.dumps(run(json.loads(a.spec.read_bytes()))))
