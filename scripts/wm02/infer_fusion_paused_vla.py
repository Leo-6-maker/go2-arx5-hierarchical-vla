"""One frozen draw from a sealed paused capture, with no native authority.

Both exact-snapshot and pre-registered single-snapshot bindings use the original
saved processor and ResidentModel.generate; no second decode or action clipping.
"""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from go2arx5_vla.world_model.fusion_paused_inference import (
    CHECKPOINT_INVENTORY, MODE, PROFILE, RESULT_SCHEMA, need, read_payload, validate_spec,
    LOAD_AFTER_SNAPSHOT,PRELOAD_BEFORE_SNAPSHOT)
from go2arx5_vla.world_model.storage import digest, write_json


def _load_frozen(spec, output, *, cpu_model_initializer=None):
    """Original loader/precision ABI, after admission and checkpoint checks.

    An explicitly paired preload may precede snapshot creation. Generation still
    requires all original input, capture, task and actual-pause checks.
    """
    if cpu_model_initializer is not None:
        need(callable(cpu_model_initializer)and(spec.get('profile')=='CORE_C500_NEW_LIVE_LOADER_PARITY_V3'
            or spec.get('core_local_protocol')in('CORE_LOCAL_C500_EXPLICIT_CONIC_DIAGNOSTIC_NOT_DEV_TEST_V3',
                'CORE_LOCAL_C500_ENDPOINT_BASIS_DIAGNOSTIC_NOT_DEV_TEST_V3',
                'CORE_MATCHED_WEIGHTS_PROCESSOR_SMOOTH_TRAIN_NOT_DEV_TEST_V3_1')),
            'EXPLICIT_NEW_C500_CPU_INITIALIZATION_NOT_ORIGINAL_BASELINE')
    from scripts.pi0292.train_adapter import official_import
    bindings = spec['neural_bindings']
    official_import(bindings['source'])
    import torch
    from go2arx5_vla.world_model.runtime import cuda_uuid
    need(torch.cuda.device_count() == 1
         and cuda_uuid(torch.cuda.get_device_properties(0).uuid) == spec['GPU_uuid'],
         'ONE_ACTUAL_PINNED_GPU')
    total = torch.cuda.get_device_properties(0).total_memory
    # A pre-registered worker can wait while native physics reaches its pause.
    # Recheck live headroom now, in addition to original runtime admission.
    free, _ = torch.cuda.mem_get_info(0)
    need(0 < spec['allocator_limit_bytes'] <= total
         and spec['allocator_limit_bytes'] < free-1024**3, 'ACTUAL_ALLOCATOR_FRACTION_AND_FRESH_HEADROOM')
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/total, 0)
    torch.cuda.reset_peak_memory_stats()
    from transformers import AutoModel
    import gr00t.model.gr00t_n1d7.gr00t_n1d7  # noqa: F401; original model registration
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from scripts.pi0294.local_backbone import local_model_name
    alias = local_model_name(spec['registered_root'], bindings)
    kwargs = dict(local_files_only=True, trust_remote_code=True, cache_dir=bindings['cosmos_hub'],
                  revision='9ce19a195e423419c349abfc86fd07178b230561')
    checkpoint = Path(spec['checkpoint'])
    write_json(output/'MODEL_LOAD_INTENT.private.json', dict(
        checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256'],
        numeric_mode=spec['numeric_mode'], model_loads_requested=1))
    model, loading = AutoModel.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True,
        output_loading_info=True, model_name=alias, transformers_loading_kwargs=kwargs,
        tune_llm=False, tune_visual=False, tune_projector=True, tune_diffusion_model=True, tune_vlln=True,
        state_dropout_prob=0., backbone_trainable_params_fp32=True, load_bf16=False)
    write_json(output/'MODEL_LOADING_INFO.private.json', loading)
    need(not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')),
         'FULL_FROZEN_GR00T_LOAD_MISMATCH')
    model.requires_grad_(False)
    parameters = list(model.parameters())
    parameter_bytes = sum(p.numel()*p.element_size() for p in parameters)
    need(parameters and all(not p.requires_grad and p.dtype == torch.float32 for p in parameters)
         and parameter_bytes == spec['checkpoint_parameter_bytes'], 'FROZEN_FP32_PARAMETER_IDENTITY')
    initialized=None
    if cpu_model_initializer is not None:
        # The qualified ordinary C500 mapper validates CPU tensors before copying.
        # Install once before the original CUDA move; never swap a live model.
        initialized=cpu_model_initializer(model)
        need(all(not p.requires_grad and p.dtype==torch.float32 and p.device.type=='cpu'for p in model.parameters())
            and sum(p.numel()*p.element_size()for p in model.parameters())==parameter_bytes,
            'C500_INITIALIZED_SAME_FROZEN_FP32_ARCHITECTURE_BEFORE_CUDA')
    model.to('cuda').eval()
    need(model.action_head.num_inference_timesteps == 4, 'ORIGINAL_FOUR_STEP_SAMPLER')
    processor = Gr00tN1d7Processor.from_pretrained(checkpoint/'processor', model_name=alias,
        local_files_only=True, transformers_loading_kwargs=kwargs)
    processor.eval()
    need(processor.embodiment_id_mapping['new_embodiment'] == 10, 'UNCHANGED_SOURCE_CATEGORY10')
    instance = SimpleNamespace(model=model, processor=processor,
        identity=dict(checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256'], files=spec['checkpoint_files']))
    write_json(output/'MODEL_LOADED.private.json', dict(
        checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256'], parameter_bytes=parameter_bytes,
        all_parameters_frozen=True, all_parameter_dtypes='torch.float32', num_inference_timesteps=4,
        **(dict(cpu_candidate_initialization=initialized)if initialized is not None else{}),
        source_embodiment='new_embodiment_CATEGORY10_FIXED_BASE_ARX5', target_embodiment_qualified=False))
    return instance, torch


def _generate_once(instance, payload, seed, torch):
    from scripts.pi0294.resident import ResidentModel
    from scripts.pi0294.e1.model import original_inference_precision
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    tick = time.monotonic()
    begin.record()
    with original_inference_precision(torch) as precision:
        value = ResidentModel.generate(instance, payload, seed)
    end.record()
    torch.cuda.synchronize()
    return value, precision, dict(query_wall_s=time.monotonic()-tick,
                                 GPU_event_s=begin.elapsed_time(end)/1000)


def _deadline(spec):
    need(time.time() < spec['deadline_epoch']
         and not (Path(spec['registered_root'])/'STOP.json').exists()
         and not (Path(spec['campaign_root'])/'STOP.json').exists(), 'FINITE_QUERY_DEADLINE_AND_ORIGINAL_STOP')


def infer(spec, output):
    """Call only inside admitted_run; an intent makes every failure nonretryable."""
    output = Path(output)
    # admitted_run creates a new output; the exclusive intent also protects this
    # callable against a second draw on the same output after a lost response.
    write_json(output/'QUERY_ATTEMPT.private.json', dict(
        query_uuid=spec.get('query_uuid'), retry_authorized=False, started_epoch=time.time()))
    start, cpu = time.monotonic(), time.process_time()
    stage, queries, completed = 'INPUT_VALIDATION', 0, 0
    load_calls,loads_completed=0,0
    check_wall=load_wall=wait_wall=None

    def load_verified():
        nonlocal stage,load_calls,loads_completed,check_wall,load_wall
        stage='CHECKPOINT_INTEGRITY';_deadline(spec);tick=time.monotonic()
        checkpoint=Path(spec['checkpoint'])
        try:
            for name,pin in spec['checkpoint_files'].items():
                need(digest(checkpoint/name)==pin,'FROZEN_WEIGHT_OR_PROCESSOR_BYTES_CHANGED:'+name)
        finally:check_wall=time.monotonic()-tick
        _deadline(spec);stage='MODEL_LOAD';tick=time.monotonic();load_calls+=1
        instance,torch=_load_frozen(spec,output)
        load_wall=time.monotonic()-tick;loads_completed+=1;_deadline(spec)
        return instance,torch

    try:
        validate_spec(spec)
        schedule=spec.get('frozen_model_load_schedule',LOAD_AFTER_SNAPSHOT)
        if schedule==PRELOAD_BEFORE_SNAPSHOT:
            instance,torch=load_verified()
        from go2arx5_vla.world_model.fusion_paused_query_bridge import wait_for_snapshot
        stage='SNAPSHOT_WAIT';tick=time.monotonic()
        try:wait_for_snapshot(spec)
        finally:wait_wall=time.monotonic()-tick
        stage='ACTUAL_INPUT_VALIDATION'
        payload, identity, _ = read_payload(spec)
        if schedule==LOAD_AFTER_SNAPSHOT:
            instance,torch=load_verified()
        _deadline(spec)
        stage = 'ONE_GENERATION'
        write_json(output/'GENERATION_INTENT.private.json', dict(
            query_uuid=spec['query_uuid'], actual_queries_requested=1, noise_seed=spec['seed'],
            observation_sha256=identity['observation_sha256'], inference_mode=MODE,
            checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256']))
        queries = 1
        value, precision, timing = _generate_once(instance, payload, spec['seed'], torch)
        completed = 1
        stage = 'RETAIN_COMPLETE_RAW_OUTPUT'
        action = output/'REAL_VLA_ACTION.npz'
        with action.open('xb') as stream:
            np.savez_compressed(stream, physical_H40=value['physical_H40'], normalized=value['normalized'])
            stream.flush()
            os.fsync(stream.fileno())
        physical, normalized, noise = value['physical_H40'], value['normalized'], value['noise']
        need(physical.shape == (40,10) and np.isfinite(physical).all()
             and normalized.shape == (1,40,132) and np.isfinite(normalized).all()
             and noise.get('shape') == [1,40,132] and noise.get('dtype') == 'torch.float32'
             and isinstance(noise.get('sha256'), str) and len(noise['sha256']) == 64
             and value['identity'] == instance.identity and value['extra_correction'] is False
             and value['physics_integrations'] == 0, 'COMPLETE_NATIVE_DRAW_IDENTITY_AND_FINITE_RAW_ARRAYS')
        # The GPU cannot inspect the actual native process or grant motion. Its
        # owner must revalidate the retained capture against that process.
        result = dict(schema=RESULT_SCHEMA, status='REAL_FROZEN_VLA_LIVE_PAUSED_QUERY_DONE',
            source_sha=spec['source_sha'], query_uuid=spec['query_uuid'], seed=spec['seed'],
            execution_spec_sha256=digest(Path(spec['registered_root'])/'EXECUTION_SPEC.json'),
            pair_sha256=spec.get('paused_query_pair_sha256'),
            producer_native_spec_sha256=spec.get('producer_native_spec_sha256'),
            checkpoint_manifest_sha256=spec['checkpoint_manifest_sha256'],
            checkpoint_file_inventory_sha256=CHECKPOINT_INVENTORY,
            recorded_observation_sha256=identity['observation_sha256'], observation_binding=identity,
            action_sha256=digest(action), actual_model_loads=1, actual_VLA_inferences=1,
            WM_updates=0, Actor_updates=0, GR00T_updates=0, native_integrations=0,
            inference_mode=MODE, numeric_mode=spec['numeric_mode'], precision=precision, noise=noise,
            physical_H40_shape=list(physical.shape), normalized_shape=list(normalized.shape),
            normalized_outside_unit_components=value['normalized_outside_unit_components'],
            source_frame=payload['frame'], source_TCP=payload['tcp'],
            desired_outcome_provenance='SEMANTICS_UNRESOLVED',
            native_execution_authorized=False, native_caller_must_verify_still_paused=True,
            online_closed_loop=False, target_embodiment_qualified=False, target_task_qualified=False,
            snapshot_wait_wall_s=wait_wall, model_cold_load_wall_s=load_wall,
            frozen_model_load_schedule=schedule,checkpoint_integrity_check_wall_s=check_wall,
            wall_s=time.monotonic()-start, CPU_s=time.process_time()-cpu,
            simulated_time_advanced_s=0., reference_TTL_rebased=False,
            returned_action_is_live_reference=False,
            reference_TTL_status='NOT_GRANTED_QUERY_WALL_TIME_DOES_NOT_RENEW_ORIGINAL_TTL',
            latency_distribution='N1_SINGLE_OBSERVATION_NOT_P50_P95',
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), **timing)
        if 'producer_phase' in identity:
            result.update(producer_phase=identity['producer_phase'],
                physical_recipe_phase=identity['physical_recipe_phase'],
                capture_condition=identity['capture_condition'],
                capture_camera_recipe=identity['capture_camera_recipe'],
                source_camera_identity_sha256=identity['source_camera_identity_sha256'],
                returned_VLA_action_installations=0,camera_equivalence=False)
        write_json(output/'RESULT.private.json', result)
        return result
    except BaseException as error:
        write_json(output/'FAILED.private.json', dict(schema=RESULT_SCHEMA,
            query_uuid=spec.get('query_uuid'), stage=stage, failure=type(error).__name__+': '+str(error),
            generation_invocations=queries, completed_queries=completed,
            model_load_calls=load_calls,completed_model_loads=loads_completed,
            model_load_completion_unknown=load_calls>loads_completed,
            frozen_model_load_schedule=spec.get('frozen_model_load_schedule',LOAD_AFTER_SNAPSHOT),
            checkpoint_integrity_check_wall_s=check_wall,model_cold_load_wall_s=load_wall,
            snapshot_wait_wall_s=wait_wall,
            completed_query_unknown_after_intent=queries == 1 and completed == 0,
            retry_authorized=False, native_integrations=0, native_execution_authorized=False,
            retained_raw_action=(output/'REAL_VLA_ACTION.npz').exists(),
            wall_s=time.monotonic()-start, CPU_s=time.process_time()-cpu))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec, args.output.name, PROFILE, args.output, gpu=True) as spec:
        infer(spec, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
