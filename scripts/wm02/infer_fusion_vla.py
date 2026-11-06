"""One real frozen-GR00T query on a hash-bound archived dual-RGB observation.

Uses the original inference primitive/precision and saved processor. This is
offline real inference, never target online feedback or a source-skill pass.
"""
import argparse,json,os,resource,time
from pathlib import Path


def infer(spec,output):
    from go2arx5_vla.world_model.storage import digest,write_json
    import numpy as np
    start=time.monotonic();cpu=time.process_time();checkpoint=Path(spec['checkpoint'])
    # Full bytes, not merely safetensor headers, precede loading. The immutable
    # data remain at the existing authorized asset route; nothing is downloaded.
    for name,pin in spec['checkpoint_files'].items():
        if digest(checkpoint/name)!=pin:raise ValueError('FROZEN_VLA_BYTES_CHANGED:'+name)
    observation=Path(spec['recorded_observation'])
    if digest(observation)!=spec['recorded_observation_sha256']:raise ValueError('ACTUAL_SAVED_OBSERVATION_CHANGED')
    observed=json.loads(observation.read_text());payload=observed['body']['payload']
    from scripts.pi0292.relay_protocol import digest as packet_digest
    if observed['sha256']!=packet_digest(observed['body']):raise ValueError('SAVED_REAL_OBSERVATION_PACKET_HASH_DIFFERS')
    bindings=spec['neural_bindings']
    from scripts.pi0292.train_adapter import official_import
    official_import(bindings['source'])
    import torch
    if torch.cuda.device_count()!=1:raise PermissionError('ONE_ACTUAL_FUSION_GPU_REQUIRED')
    from go2arx5_vla.world_model.runtime import cuda_uuid
    if cuda_uuid(torch.cuda.get_device_properties(0).uuid)!=spec['GPU_uuid']:raise PermissionError('ACTUAL_GPU_UUID_CHANGED')
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/total,0)
    torch.cuda.reset_peak_memory_stats()
    from transformers import AutoModel
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from scripts.pi0294.local_backbone import local_model_name
    alias=local_model_name(spec['registered_root'],bindings)
    kwargs=dict(local_files_only=True,trust_remote_code=True,cache_dir=bindings['cosmos_hub'],
                revision='9ce19a195e423419c349abfc86fd07178b230561')
    write_json(output/'MODEL_LOAD_INTENT.private.json',dict(checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256'],numeric_mode=spec['numeric_mode'],model_loads_requested=1))
    model,loading=AutoModel.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=True,
        output_loading_info=True,model_name=alias,transformers_loading_kwargs=kwargs,
        tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,tune_vlln=True,
        state_dropout_prob=0.,backbone_trainable_params_fp32=True,load_bf16=False)
    if any(loading.get(k)for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise ValueError('FULL_FROZEN_GR00T_LOAD_MISMATCH')
    model.requires_grad_(False);model.to('cuda').eval()
    if model.action_head.num_inference_timesteps!=4:raise ValueError('FROZEN_FOUR_STEP_INFERENCE_REQUIRED')
    processor=Gr00tN1d7Processor.from_pretrained(checkpoint/'processor',model_name=alias,
        local_files_only=True,transformers_loading_kwargs=kwargs)
    processor.eval()
    if processor.embodiment_id_mapping['new_embodiment']!=10:raise ValueError('ACTUAL_SOURCE_EMBODIMENT_CHANGED')
    # Reuse generate(), not the legacy allocation-specific constructor. This
    # invocation has its own actual FUSION registry/lease/runtime admission.
    from scripts.pi0294.resident import ResidentModel
    from scripts.pi0294.e1.model import original_inference_precision
    class Frozen:
        def __init__(self):
            self.model,self.processor,self.identity=model,processor,dict(checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256'],files=spec['checkpoint_files'])
    instance=Frozen();write_json(output/'MODEL_LOADED.private.json',dict(checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256'],all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),parameter_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),source_embodiment='new_embodiment_CATEGORY10_FIXED_BASE_ARX5',target_embodiment_qualified=False))
    if time.time()>=spec['deadline_epoch']or(Path(spec['campaign_root'])/'STOP.json').exists():raise TimeoutError('NO_TIME_FOR_FROZEN_VLA_QUERY')
    write_json(output/'GENERATION_INTENT.private.json',dict(actual_queries_requested=1,noise_seed=spec['seed'],observation_sha256=spec['recorded_observation_sha256'],inference_mode='OFFLINE_REAL_VLA_INFERENCE'))
    tick=time.monotonic();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
    with original_inference_precision(torch)as precision:value=ResidentModel.generate(instance,payload,spec['seed'])
    end.record();torch.cuda.synchronize();query_wall=time.monotonic()-tick
    action_path=output/'REAL_VLA_ACTION.npz'
    with action_path.open('xb')as f:np.savez_compressed(f,physical_H40=value['physical_H40'],normalized=value['normalized'])
    write_json(output/'RESULT.private.json',dict(status='REAL_FROZEN_VLA_OFFLINE_INFERENCE_DONE',source_sha=spec['source_sha'],
        checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256'],recorded_observation_sha256=spec['recorded_observation_sha256'],
        action_sha256=digest(action_path),actual_model_loads=1,actual_VLA_inferences=1,WM_updates=0,Actor_updates=0,GR00T_updates=0,native_integrations=0,
        inference_mode='OFFLINE_REAL_VLA_INFERENCE',numeric_mode=spec['numeric_mode'],precision=precision,noise=value['noise'],
        physical_H40_shape=list(value['physical_H40'].shape),normalized_outside_unit_components=value['normalized_outside_unit_components'],
        source_frame=payload['frame'],source_TCP=payload['tcp'],desired_outcome_provenance='SEMANTICS_UNRESOLVED',
        REAL_VLA_INFERENCE_DONE=True,FROZEN_VLA_NATIVE_COMPARISON=False,TASK_TRANSFER_GAIN=False,
        target_online_images=False,source_task_success='NOT_ESTABLISHED_BY_THIS_QUERY',
        query_wall_s=query_wall,GPU_event_s=begin.elapsed_time(end)/1000,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.output.name,'wm02-fusion-vla-shadow',a.output,gpu=True)as spec:infer(spec,a.output)
    return 0


if __name__=='__main__':raise SystemExit(main())
