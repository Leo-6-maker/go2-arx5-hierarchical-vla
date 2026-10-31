"""Sixteen frozen TRAIN windows: native loss and matched-noise sampler study."""
import hashlib,json
from pathlib import Path

ANCHORS=(0,80,160,240,320,400,480,600)


def plan():
    return dict(schema='BOOK128_TRAIN16_NATIVE_LOSS_SAMPLER4_VS16_DIAGNOSTIC_V1',
        checkpoint_manifest_sha256='9510982676238e4c2337da660ecf35d579c259121983175eb349f64ba0e7ff47',
        dataset_manifest_sha256='47163b091f60c6d30e49b50699177727f8909e2100533fa4bc4555b8431c010d',
        statistics_sha256='060ff05d4c34205593e234d43a5a76bd5c2305171289489d51fb78b68122e096',
        windows=[dict(id=f'BOOK_SOURCE_SEED{seed:03}',anchor=a,split='TRAIN')for seed in(0,4)for a in ANCHORS],
        denoise_steps=[4,16],generation_queries=32,native_flow_loss_forwards=16,
        same_window_and_initial_noise_across_samplers=True,noise_bank='SHA256_BOOK128_TRAIN16/id/anchor_first8bytes_mod2pow63',
        native_masked_flow_matching_loss=True,optimizer_updates=0,statistics_refit=False,physics_integrations=0,
        DEV_or_TEST_used=False,formal_DEV_conditions=0,original_DEV10_result_unchanged=True,
        purpose='Distinguish TRAIN fit and fixed four-step generation limitations before further optimization or action-side capacity changes.')


def validate_manifest(m,p):
    if p!=plan()or m['checkpoint_manifest_sha256']!=p['checkpoint_manifest_sha256']:
        raise PermissionError('EXACT_FROZEN_BOOK_TRAIN16_DIAGNOSTIC_PLAN')
    for key,value in dict(optimizer_updates=0,physics_integrations=0,generation_queries=32,model_loads=1).items():
        if m['limits'].get(key)!=value:raise PermissionError('BOOK_TRAIN16_FINITE_'+key)
    return p


def run(root,instance,trainer,guard):
    import torch
    import numpy as np
    from scripts.pi0294.native_train import NativeNoiseAudit
    from scripts.pi0294.native_dataset import NativeDataset,ProcessedCollator,sha
    from scripts.pi0294.e1.model import payload
    from scripts.pi0294.e1.artifacts import atomic_npz
    from scripts.pi0294.e1.smoke import put
    from scripts.pi0292.durable import read,write
    from scripts.pi0292.eval_contract import error_metrics,gate_action
    root=Path(root);p=read(root/'TRAIN_PROBE_PLAN.json');validate_manifest(read(root/'RUN_MANIFEST.json'),p)
    bindings=instance.bindings;data=NativeDataset(bindings['dataset'],bindings['source'],bindings['cosmos'],for_optimization=False)
    if data.manifest_sha!=p['dataset_manifest_sha256']or data.statistics_sha!=p['statistics_sha256']:raise PermissionError('BOOK_TRAIN16_ORIGINAL_FROZEN_TRAIN_INPUTS')
    data.initialize();indices={(r['id'],a):i for i,(r,a)in enumerate(data.rows)};results=[]
    for row in p['windows']:
        guard();key=row['id']+'-'+str(row['anchor']).zfill(3);index=indices[row['id'],row['anchor']]
        state,expert,images,text=data.raw(index);request=payload(state,images,text)
        seed=int.from_bytes(hashlib.sha256(f"BOOK128_TRAIN16/{row['id']}/{row['anchor']}".encode()).digest()[:8],'big')%(2**63)
        batch=ProcessedCollator(data.processor)([data[index]])
        batch={k:{n:v.to('cuda')if torch.is_tensor(v)else v for n,v in values.items()}for k,values in batch.items()}
        put(root/(key+'-LOSS.INTENT.json'),dict(window=row,seed=seed,optimizer_updates=0))
        observer=NativeNoiseAudit(instance.model)
        try:
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,_=trainer.compute_loss(instance.model,batch,return_outputs=True)
            loss_value=float(loss.detach());native_noise=observer.record
        finally:observer.close()
        if not np.isfinite(loss_value)or native_noise is None:raise ValueError('BOOK_TRAIN16_ACTUAL_FINITE_NATIVE_LOSS_NOISE')
        put(root/(key+'-LOSS.RESULT.json'),dict(window=row,native_loss=loss_value,native_noise=native_noise,optimizer_updates=0))
        outputs=[];expert_gate=gate_action(expert,state,0.,key+'-EXPERT')
        for steps in p['denoise_steps']:
            guard();label=key+'-D'+str(steps);path=root/('draw-'+label)
            put(path.with_suffix('.INTENT.json'),dict(window=row,denoise_steps=steps,seed=seed,
                observation_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
                actual_image_SHA256=[x['sha256']for x in request['images']],checkpoint_manifest_sha256=p['checkpoint_manifest_sha256']))
            calls=[]
            hook=instance.model.action_head.model.register_forward_pre_hook(lambda module,args:calls.append(1))
            previous=instance.model.action_head.num_inference_timesteps
            try:
                instance.model.action_head.num_inference_timesteps=steps
                value=instance.generate(request,seed)
            finally:
                instance.model.action_head.num_inference_timesteps=previous;hook.remove()
            atomic_npz(path.with_suffix('.npz'),physical_H40=value['physical_H40'],normalized=value['normalized'],
                actual_state10=state,actual_completed_expert_H40=expert)
            record=dict(window=row,denoise_steps=steps,observed_native_DiT_forward_calls=len(calls),seed=seed,noise=value['noise'],
                inference_precision=value['inference_precision'],normalized_outside_unit_components=value['normalized_outside_unit_components'],
                original_gate=gate_action(value['physical_H40'],state,0.,label),errors=error_metrics(value['physical_H40'],expert,state),
                arrays_sha256=sha(path.with_suffix('.npz')),diagnostic_only=True,physics_integrations=0)
            put(path.with_suffix('.RESULT.json'),record)
            if len(calls)!=steps:raise ValueError('BOOK_TRAIN16_ACTUAL_DENOISE_CALLS_MISMATCH')
            outputs.append(record)
        if outputs[0]['noise']!=outputs[1]['noise']:raise ValueError('BOOK_TRAIN16_SAMPLERS_MUST_START_WITH_IDENTICAL_ACTUAL_NOISE')
        results.append(dict(window=row,native_loss=loss_value,expert_gate=expert_gate,matched_initial_noise=True,outputs=outputs))
        write(root/'PROBE_PROGRESS.json',dict(complete_windows=len(results),generation_queries=2*len(results),records=results))
    summary={}
    for steps in p['denoise_steps']:
        values=[next(x for x in r['outputs']if x['denoise_steps']==steps)for r in results]
        summary[str(steps)]=dict(generation_queries=len(values),gate_rejects=sum(x['original_gate']['decision']=='REJECT'for x in values),
            H40_XYZ_mean_m=float(np.mean([x['errors']['H40']['XYZ_euclidean_m']['mean']for x in values])),
            H40_SO3_mean_rad=float(np.mean([x['errors']['H40']['SO3_rad']['mean']for x in values])),
            H40_gripper_MAE=float(np.mean([x['errors']['H40']['gripper_MAE']for x in values])))
    return dict(status='COMPLETE_NATIVE_TRAIN16_MATCHED_NOISE_SAMPLER_DIAGNOSTIC',windows=16,native_loss_mean=float(np.mean([r['native_loss']for r in results])),
        expert_gate_rejects=sum(r['expert_gate']['decision']=='REJECT'for r in results),summary=summary,records=results,
        generation_queries=32,native_flow_loss_forwards=16,optimizer_updates=0,physics_integrations=0,DEV_conditions=0,
        original_DEV10_result_unchanged=True)
