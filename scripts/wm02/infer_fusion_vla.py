"""One real frozen-GR00T query on a hash-bound archived dual-RGB observation.

Uses the original inference primitive/precision and saved processor. This is
offline real inference, never target online feedback or a source-skill pass.
"""
import argparse,json,os,resource,time
from pathlib import Path


def recorded_payload(spec):
    """Read source packets or the complete, sealed target capture sidecar.

    A saved target view remains a shadow diagnostic. Its mapped source ABI does
    not turn target cameras, task language or embodiment into qualified inputs.
    """
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.relay_protocol import unseal, validate_observation
    if spec.get('observation_mode') == 'FROZEN_PROMPT_CONTROL_VIEW_V1':
        from go2arx5_vla.world_model.fusion_prompt_control import prompt_control_payload
        return prompt_control_payload(Path(spec['recorded_observation']).read_bytes(),
            view_sha256=spec['recorded_observation_sha256'], query_spec=spec)
    observation = Path(spec['recorded_observation'])
    if digest(observation) != spec['recorded_observation_sha256']:
        raise ValueError('ACTUAL_SAVED_OBSERVATION_CHANGED')
    observed = json.loads(observation.read_text())
    mode = spec.get('observation_mode', 'ARCHIVED_SOURCE_RELAY_PACKET')
    if mode == 'ARCHIVED_SOURCE_RELAY_PACKET':
        payload = unseal(observed)['payload']
        validate_observation(payload)
        return payload, dict(observation_mode=mode, actual_saved_target_images=False,
            target_online_images=False, target_embodiment_qualified=False)
    if mode != 'SAVED_TARGET_NATIVE_REPLAY_SHADOW':
        raise ValueError('EXPLICIT_FUSION_OBSERVATION_MODE_REQUIRED')
    from go2arx5_vla.world_model.fusion_observation import shadow_inference_payload, transform_hash
    from scripts.pi0294.task_state import parse_state
    result_path = Path(spec['target_capture_result'])
    if (result_path.name != 'RESULT.private.json' or result_path.parent.name != 'fit-01'
            or digest(result_path) != spec['target_capture_result_sha256']):
        raise ValueError('EXACT_SAVED_TARGET_CAPTURE_RESULT_REQUIRED')
    output, root = result_path.parent, result_path.parent.parent
    result = json.loads(result_path.read_text())
    lease = json.loads((root/'control/LEASE.json').read_text())
    worker = json.loads((root/'WORKER_EXIT.private.json').read_text())
    capture_spec = json.loads((root/'EXECUTION_SPEC.json').read_text())
    if (lease.get('status') != 'CLOSED' or lease.get('worker_exit_code') != 0
            or worker.get('exit_code') != 0 or lease.get('spec_sha256') != digest(root/'EXECUTION_SPEC.json')
            or result.get('schema') != 'WM02_FUSION_TARGET_OBSERVATION_REPLAY_CAPTURE_V1'
            or result.get('status') != 'SAVED_TARGET_OBSERVATIONS_CAPTURED_SHADOW_ONLY'
            or result.get('failure') is not None or result.get('native_integrations') != 0
            or result.get('prohibited_integration_calls') != 0
            or result.get('actual_renderer_saves') != 3 or result.get('bound_observations') != 3
            or result.get('actual_dual_camera_PNGs') != 6
            or [row['native_step'] for row in result.get('captures', [])] != [3503, 3703, 4103]
            or result.get('source_sha') != capture_spec.get('source_sha')):
        raise PermissionError('SUCCESSFULLY_CLOSED_THREE_VIEW_TARGET_CAPTURE_REQUIRED')
    step = spec['target_capture_step']
    if step != 3703 or observation != output/f'BOUND_OBSERVATION_{step}.private.json':
        raise PermissionError('PREDECLARED_DECISION_STEP3703_NO_VIEW_SELECTION')
    rows = [row for row in result['captures'] if row['native_step'] == step]
    if len(rows) != 1:
        raise ValueError('EXACT_TARGET_CAPTURE_ROW_REQUIRED')
    row = rows[0]
    index = 1  # Middle of the preregistered 3503/3703/4103 capture sequence.
    exact_files = {
        observation.name: spec['recorded_observation_sha256'],
        f'STATE_{step}.private.bin': row['native_state_sha256'],
        f'images/{index:06d}-calibration.json': row['calibration_sha256'],
        **{f'images/{index:06d}-{camera}.png': row['image_sha256'][camera]
           for camera in ('body_rgb', 'wrist_rgb')}}
    if row['files'] != exact_files:
        raise ValueError('EXACT_FIVE_CAPTURE_MEMBERS_MUST_MATCH_STATE_CALIBRATION_AND_IMAGES')
    for name, pin in row['files'].items():
        relative = Path(name)
        path = output/relative
        if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(output.resolve()) or digest(path) != pin):
            raise ValueError('ORIGINAL_TARGET_CAPTURE_MEMBER_CHANGED')
    body = unseal(observed)
    raw = parse_state((output/f'STATE_{step}.private.bin').read_bytes())
    original, frame_map = capture_spec['original'], capture_spec['frame_map']
    view = body['source_policy_view']
    if (body['native_step'] != step or body['native_time_s'] != row['native_time_s']
            or raw['step'] != step or raw['time_s'] != body['native_time_s']
            or abs(raw['time_s']-step*.002) > 1e-8
            or body['capture'] != row['capture']
            or row['capture'] != dict(index=index, native_step=step, time_s=raw['time_s'])
            or body['target_observation']['scene_sha256'] != result['scene_sha256']
            or result['scene_sha256'] != original['scene']['sha256']
            or body['target_observation']['target_runtime_sha256'] != original['spec']['sha256']
            or result['original_native_result_sha256'] != original['native_result']['sha256']
            or result['original_journal_sha256'] != original['journal']['sha256']
            or result['original_native_ticks'] != 4103
            or view is None or view['transform_sha256'] != frame_map['sha256']
            or transform_hash(frame_map['T_source_from_target']) != frame_map['sha256']
            or view['T_source_from_target'] != frame_map['T_source_from_target']
            or body['target_observation']['native_state_sha256'] != row['native_state_sha256']
            or body['target_observation']['calibration_sha256'] != row['calibration_sha256']
            or body['target_observation']['image_sha256'] != row['image_sha256']
            or capture_spec.get('task_prompt_matches_target_task') is not False):
        raise ValueError('SAME_TARGET_SNAPSHOT_AND_EXPLICIT_UNQUALIFIED_LANGUAGE_REQUIRED')
    return shadow_inference_payload(observed), dict(observation_mode=mode,
        actual_saved_target_images=True, target_online_images=False,
        target_capture_result_sha256=spec['target_capture_result_sha256'],
        native_step=step, native_time_s=body['native_time_s'],
        target_provenance=body['provenance'],
        language_provenance=capture_spec['language_provenance'],
        task_prompt_matches_target_task=False, target_embodiment_qualified=False,
        target_task_native_qualified=False, input_mode='SEALED_TARGET_NATIVE_REPLAY_NOT_ONLINE')


def infer(spec,output):
    from go2arx5_vla.world_model.storage import digest,write_json
    import numpy as np
    start=time.monotonic();cpu=time.process_time();checkpoint=Path(spec['checkpoint'])
    # Full bytes, not merely safetensor headers, precede loading. The immutable
    # data remain at the existing authorized asset route; nothing is downloaded.
    for name,pin in spec['checkpoint_files'].items():
        if digest(checkpoint/name)!=pin:raise ValueError('FROZEN_VLA_BYTES_CHANGED:'+name)
    payload, observation_provenance = recorded_payload(spec)
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
        observation_provenance=observation_provenance,
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
