"""Export complete step4 state, then reconstruct ordinary RGB in a new process."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def run(spec, mode):
    from go2arx5_vla.sla_adapter.site_v2 import validate
    validate(spec)
    from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
    from go2arx5_vla.sla_adapter import provider_v3
    root=Path(spec['registered_root']);started=time.time()
    if mode=='export':
        import jax,flax,optax
        from go2arx5_vla.sla_adapter.runtime_v3 import DeviceResidentEngine
        engine=DeviceResidentEngine(spec,root/'export-work')
        engine.bindings['training_site']=dict(GPU_uuid=spec['GPU_uuid'],GPU_name=spec['GPU_name'],
                                             jax=jax.__version__,flax=flax.__version__,optax=optax.__version__)
        engine.resume(spec['resume'])
        if (engine.step,engine.cursor)!=(4,128):raise ValueError('ACTUAL_STEP4_COMPLETE_STATE_REQUIRED')
        audit=Path(spec['action_representation_audit']['path'])
        package=root/'RGB_PACKAGE'
        manifest=provider_v3.export(package,spec['resume']['path'],active_template=engine.active,
            statistics=audit.parent/'STATISTICS.private.json',tokenizer=spec['tokenizer']['path'],
            parent_manifest=spec['parent_manifest'],action_representation_audit=spec['action_representation_audit'])
        engine.close()
        return dict(package=dict(path=str(package),sha256=sha(package/'MANIFEST.private.json')),
                    arrays=len(manifest['adapter_arrays']),model_loads=1,optimizer_updates=0,model_calls=0,wall_s=time.time()-started)
    reference_mode=mode=='reference'
    probes_root=root/'RGB_PROBES'
    if reference_mode:probes_root.mkdir()
    provider=provider_v3.Provider(spec['package']['path'],manifest_sha256=spec['package']['sha256'],
                                  parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
    from go2arx5_vla.sla_adapter.dataset_v2 import data
    from go2arx5_vla.sla_adapter.runtime_v1 import probe_indices
    dataset=data(spec);rows=[]
    for index in probe_indices(dataset):
        state,_,images,text=dataset.original.raw(index)
        identity=dataset.original.identity(index)
        frame=dataset.frames[identity['episode_id']]['T_policy_from_mount']
        for noise in range(2):
            predicted=provider.predict(state,images,text,frame,noise_id=noise)
            rows.append(dict(index=index,noise=noise,
                normalized_sha256=hashlib.sha256(predicted['normalized'].tobytes()).hexdigest(),
                physical_sha256=hashlib.sha256(predicted['physical_H40'].tobytes()).hexdigest()))
            if reference_mode:
                file=probes_root/f'{len(rows)-1:02d}.npz'
                np.savez(file,normalized=predicted['normalized'],physical_H40=predicted['physical_H40'])
                rows[-1]['file']=str(file.relative_to(root))
    if reference_mode:
        if len(rows)!=16:raise ValueError('EXACT_SIXTEEN_ORDINARY_RGB_REQUESTS_REQUIRED')
        records=root/'REFERENCE_RECORDS.private.json'
        with records.open('x')as stream:json.dump(rows,stream,indent=2)
        return dict(schema='SLA_SITE_ORDINARY_RGB_REFERENCE_ACTUAL_V3',
            ordinary_RGB_generations=16,records=rows,reference_records=dict(path=str(records),sha256=sha(records)),
            requires_independent_fresh_process_replay=True,teacher_calls=0,optimizer_updates=0,
            model_loads=1,physical_requests=0,wall_s=time.time()-started)
    reference=json.loads(Path(spec['reference_probes']).read_bytes())
    exact=all(all(a[k]==b[k]for k in ('index','noise','normalized_sha256','physical_sha256'))for a,b in zip(rows,reference))
    if not len(rows)==len(reference)==16:raise ValueError('EXACT_SIXTEEN_ORDINARY_RGB_REQUESTS_REQUIRED')
    if not exact:raise ValueError('FRESH_PACKAGE_RGB_DIFFERS_FROM_TRAIN_ENDPOINT')
    return dict(passes=True,ordinary_RGB_generations=len(rows),records=rows,all_bit_exact=True,
                teacher_calls=0,optimizer_updates=0,model_loads=1,physical_requests=0,wall_s=time.time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);p.add_argument('--mode',choices=['export','reference','replay'],required=True);a=p.parse_args()
    s=json.loads(a.spec.read_bytes());r=run(s,a.mode)
    with (Path(s['registered_root'])/(a.mode.upper()+'_RESULT.private.json')).open('x')as f:json.dump(r,f,indent=2)
