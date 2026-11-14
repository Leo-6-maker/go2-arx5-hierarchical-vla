"""JAX SLA provider in the original finite native/neural file handoff."""
import base64
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from . import task_learning_grasp_v1 as task
from .stable_task_scene_v1 import pinned
from .storage import digest, write_json


def apply_inference_spec(spec, qualification):
    q = qualification
    runtime = q['runtime']
    spec.update(python=runtime['python'], SLA_runtime=runtime, GPU_uuid=q['GPU_uuid'],
        checkpoint=q['parent_params'], checkpoint_files={},
        checkpoint_manifest_sha256=q['parent_manifest']['sha256'],
        checkpoint_manifest_path=q['parent_manifest']['path'],
        checkpoint_parameter_bytes=q['ordinary_parameter_bytes'],
        allocator_limit_bytes=q['allocator_limit_bytes'],
        numeric_mode='SLA_PINNED_JAX_BF16_DETERMINISTIC_V1',
        memory_execution='SLA_ORDINARY_RGB_JAX_SINGLE_CORE_V1',
        memory_execution_reference=q['ordinary_RGB_reconstruction'])
    if q.get('schema') in ('SLA_COMPLETE_DENSE3K_CORE_RGB16_NATIVE_ADMISSION_V5',
                           'SLA_FRESH_N20_FM3K_CORE_RGB16_NATIVE_ADMISSION_V5',
                           'SLA_REFERENCE_N20_FM3K_CORE_RGB16_NATIVE_ADMISSION_V6'):
        spec['neural_owner_lock'] = q['neural_owner_lock']


def inference_environment(spec):
    runtime = spec['SLA_runtime']
    rule = pinned(runtime['initial_function_rule'])
    return dict(JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(runtime['JAX_memory_fraction']), XLA_FLAGS=rule['XLA_FLAGS'],
        PYTHONPATH=str(Path(spec['source_root'])/'src')+':'+spec['source_root']+':'
            +runtime['upstream_root']+'/src:'+runtime['upstream_root']+'/packages/openpi-client/src:'
            +os.environ.get('PYTHONPATH', ''), TOKENIZERS_PARALLELISM='false')


def source_custody(spec):
    """Additional source requirements apply only to the separate SLA identity."""
    names = ('sla_native_v1.py', 'sla_native_policy_v1.py', 'sla_live_v1.py', 'sla_population_v1.py')
    if spec.get('core_local_protocol') == task.SLA_DENSE_ROLE:
        names += ('sla_dense_native_v5.py','sla_dense_policy_v5.py','sla_dev48_v5.py','sla_reference_native_v6.py')
    source = Path(spec['source_root'])
    for name in names:
        key = 'src/go2arx5_vla/world_model/'+name
        task.need(key in spec['source_files'] and digest(source/key) == spec['source_files'][key],
                  'SLA_ACTUAL_SEALED_NATIVE_NEURAL_SOURCE')
    for p in (source/'src/go2arx5_vla/sla_adapter').glob('*.py'):
        key = str(p.relative_to(source))
        task.need(key in spec['source_files'] and digest(p) == spec['source_files'][key],
                  'SLA_ACTUAL_SEALED_PROVIDER_CODEC_SOURCE')


def infer(spec, output):
    from PIL import Image
    from scripts.pi0292.durable import write
    from scripts.pi0294.e1.artifacts import atomic_npz
    from scripts.pi0294.e1.journal import Journal
    from .fusion_paused_query_bridge import _registered, _live
    from .task_learning_live_pair_v1 import (validate_spec, wait_request, read_request,
                                             native_terminal, SCHEMA, query_count)
    import jax
    root = Path(output)
    producer = validate_spec(spec)
    source_custody(spec)
    task.need(os.environ.get('XLA_FLAGS') == pinned(spec['SLA_runtime']['initial_function_rule'])['XLA_FLAGS']
        and os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION') == str(spec['SLA_runtime']['JAX_memory_fraction'])
        and os.environ.get('CUDA_VISIBLE_DEVICES') == spec['GPU_uuid'], 'SLA_OWN_QUALIFIED_JAX_ENVIRONMENT')
    jax.config.update('jax_compilation_cache_dir', spec['SLA_runtime']['compilation_cache_dir'])
    task.need(len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu', 'SLA_SINGLE_ACTUAL_GPU')
    started, cpu = time.monotonic(), time.process_time()
    records, loads, attempted, terminal = [], 0, 0, None
    try:
        _registered(producer, spec['producer_native_spec']['path'])
        _live(spec, producer)
        write(root/'MODEL_LOAD_INTENT.private.json', dict(policy=spec['target_gr00t_policy'], retries=False))
        package = spec['target_gr00t_policy']['package']
        manifest = pinned(dict(path=str(Path(package['path'])/'MANIFEST.private.json'),sha256=package['sha256']))
        if manifest['schema'] == 'SLA_GRIP_LOGIT_ORDINARY_RGB_INFERENCE_PACKAGE_V3':
            from ..sla_adapter.provider_v3 import Provider
            provider = Provider(package['path'], manifest_sha256=package['sha256'],
                parent_params=spec['checkpoint'], parent_manifest_path=spec['checkpoint_manifest_path'])
        else:
            from ..sla_adapter.provider_v1 import Provider
            provider = Provider(package['path'], manifest_sha256=package['sha256'], parent_params=spec['checkpoint'])
        loads = 1
        write(root/'MODEL_LOADED.private.json', dict(actual_SLA_parent_loads=1, optimizer_loaded=False,
            teacher_loaded=False, training_cache_used=False, deployed_LoRA_retained=True))
        for index in range(query_count(producer)):
            path, terminal = wait_request(spec, producer, index)
            if path is None:
                break
            _registered(producer, spec['producer_native_spec']['path'])
            _live(spec, producer)
            request, payload = read_request(spec, producer, index)
            frame = pinned(request['sla_task_frame'])
            task.need(frame['native_spec_sha256'] == spec['producer_native_spec']['sha256']
                and frame['scene_sha256'] == digest(producer['model_path']) and frame['native_step'] == task.B
                and frame['fixed_for_episode'] and not frame['object_truth_used'], 'SLA_FIXED_OWN_POST_HOLD_FRAME')
            images = []
            task.need([r['camera'] for r in payload['images']] == ['external_rgb', 'wrist_rgb'],
                      'SLA_BODY_AND_WRIST_ORDER')
            for row in payload['images']:
                raw = base64.b64decode(row['png_base64'], validate=True)
                task.need(__import__('hashlib').sha256(raw).hexdigest() == row['sha256'], 'SLA_CURRENT_RGB_BYTES')
                with Image.open(io.BytesIO(raw)) as image:
                    task.need(image.mode == 'RGB' and image.size == (256, 256), 'SLA_CURRENT_RGB256')
                    images.append(np.asarray(image).copy())
            folder = root/'queries'/f'{index:02d}'
            folder.mkdir(parents=True, exist_ok=False)
            journal = Journal(folder/'journal', folder.name)
            def generate():
                nonlocal attempted
                attempted += 1
                wall = time.monotonic()
                seed32 = request['seed'] % 2**32
                value = provider.predict(payload['state_eef10'], images, payload['language'],
                    frame['T_policy_from_mount'], noise_id=seed32-2026111200)
                action = folder/'ACTION.npz'
                atomic_npz(action, physical_H40=value['physical_H40'], normalized=value['normalized'])
                return dict(schema=SCHEMA+'_RESULT', index=index, pair_id=spec['target_gr00t_pair_id'],
                    request_sha256=digest(path), observation_sha256=request['observation']['sha256'],
                    seed=request['seed'], model_policy=spec['target_gr00t_policy'],
                    noise=dict(shape=[1,40,32], dtype='jax.float32', key_seed_u32=seed32,
                               generator='PINNED_JAX_PRNGKey_NORMAL'),
                    GPU_spec_sha256=digest(Path(spec['registered_root'])/'EXECUTION_SPEC.json'),
                    action_sha256=digest(action), Processor_decode_calls=1, SLA_codec_decode_calls=1,
                    native_integrations=0, GR00T_updates=0, WM_updates=0, Actor_updates=0, SLA_updates=0,
                    teacher_calls=0, precision='PINNED_JAX_BF16_MODEL_FLOAT32_ACTION_NOISE',
                    generation_wall_s=time.monotonic()-wall, normalized_container_dimensions=32,
                    task_frame=request['sla_task_frame'])
            result = journal.once('GENERATE', dict(request_sha256=digest(path), policy=spec['target_gr00t_policy']), generate)
            records.append(result)
            write(folder/'RESULT.json', result)
        write_json(root/'RESULT.private.json', dict(schema=SCHEMA,
            status='ACTUAL_CONTINUOUS_LIVE_QUERIES_TERMINAL', actual_model_loads=loads,
            actual_SLA_parent_loads=loads, actual_SLA_queries=len(records), actual_GR00T_queries=0,
            actual_VLA_inferences=len(records), Processor_decode_calls=len(records), SLA_codec_decode_calls=len(records),
            requested_query_slots=query_count(producer), unused_query_slots=query_count(producer)-len(records), actual_early_native_terminal=terminal,
            GR00T_updates=0, WM_updates=0, Actor_updates=0, SLA_updates=0, native_integrations=0,
            records=records, GPU_memory=jax.devices()[0].memory_stats(),
            wall_s=time.monotonic()-started, CPU_s=time.process_time()-cpu,
            native_task_success='NOT_ESTABLISHED_BY_GPU'))
    except BaseException as exc:
        write_json(root/'FAILED.private.json', dict(failure=type(exc).__name__+': '+str(exc),
            completed_model_loads=loads, generation_invocations=attempted,
            completed_query_records=len(records), query_completion_unknown_after_intent=attempted>len(records),
            native_integrations=0, optimizer_updates=0, retry_authorized=False,
            wall_s=time.monotonic()-started, CPU_s=time.process_time()-cpu))
        raise
