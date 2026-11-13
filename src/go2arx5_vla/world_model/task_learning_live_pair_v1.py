"""Fifteen finite continuous-task live-observation queries in the original registered owner system.

Requests live under the native producer; replies live under its paired neural
worker. There is no submission, new scheduler, physics authority or retry here.
One original GR00T model and Processor remain resident for all current observations.
"""
import json,os,time
from pathlib import Path
import numpy as np

from . import task_learning_grasp_v1 as task
from . import target_gr00t_profile_v1 as profile
from .storage import digest,write_json
from .fusion_paused_inference import CHECKPOINT_INVENTORY,SOURCE_MEMBERS as ORIGINAL_SOURCE_MEMBERS

PROFILE='wm02-task-learning-live-query-v1'
KIND='TASK_LEARNING_LIVE_QUERY_V1'
SCHEMA='TASK_LEARNING_CONTINUOUS_LIVE_QUERY_V1'
need=profile.need
SOURCE_MEMBERS=tuple(dict.fromkeys(ORIGINAL_SOURCE_MEMBERS+(
    'src/go2arx5_vla/world_model/native_arx5_diagnostic_v1.py',
    'src/go2arx5_vla/world_model/native_a500_diagnostic_v1.py',
    'src/go2arx5_vla/world_model/native_arx5_endpoint_v1.py',
    'src/go2arx5_vla/world_model/native_arx5_training_profile_v1.py',
    'src/go2arx5_vla/world_model/native_arx5_training_runtime_v1.py',
    'scripts/wm02/native_a500_train4_delivery_v1.py',
    'experiments/native_arx5_transfer/a_target500_train4_delivery_v1.json',
    'src/go2arx5_vla/world_model/native_body_camera_capture_v1.py',
    'src/go2arx5_vla/world_model/native_camera_transfer_v1.py',
    'scripts/wm02/native_body_train4_diagnostic_v1.py',
    'experiments/native_arx5_transfer/sft2000_native_body_diagnostic_v1.json',
    'src/go2arx5_vla/world_model/target_gr00t_live_pair_v1.py',
    'src/go2arx5_vla/world_model/task_learning_live_pair_v1.py',
    'src/go2arx5_vla/world_model/task_learning_student_v1.py',
    'src/go2arx5_vla/world_model/task_learning_local_policy_v1.py',
    'src/go2arx5_vla/world_model/task_learning_local_analysis_v1.py',
    'src/go2arx5_vla/world_model/task_learning_conic_native_v3.py',
    'src/go2arx5_vla/world_model/task_learning_endpoint_native_v3.py',
    'src/go2arx5_vla/world_model/task_learning_matched_native_v3_1.py',
    'src/go2arx5_vla/world_model/task_learning_matched_policy_v3_1.py',
    'scripts/wm02/task_learning_core_matched_panel_v3_1.py',
    'src/go2arx5_vla/world_model/task_learning_endpoint_policy_v3.py',
    'src/go2arx5_vla/world_model/task_learning_endpoint_basis_v3.py',
    'scripts/wm02/task_learning_core_endpoint_panel_v3.py',
    'src/go2arx5_vla/world_model/task_learning_conic_policy_v3.py',
    'src/go2arx5_vla/world_model/task_learning_conic_decoder_v3.py',
    'src/go2arx5_vla/world_model/task_learning_conic_qualification_v3.py',
    'src/go2arx5_vla/world_model/task_learning_feasible_decoder_v3.py',
    'src/go2arx5_vla/world_model/task_learning_features_v1.py',
    'src/go2arx5_vla/world_model/task_learning_student_batch_v1.py',
    'src/go2arx5_vla/world_model/task_learning_student_registration_v2.py',
    'scripts/wm02/task_learning_core_conic_panel_v3.py',
    'scripts/wm02/task_learning_core_local_panel_v1.py',
    'src/go2arx5_vla/world_model/task_learning_grasp_v1.py',
    'src/go2arx5_vla/world_model/task_learning_core_inference_v1.py',
    'src/go2arx5_vla/world_model/target_gr00t_prefix_v1.py',
    'src/go2arx5_vla/world_model/target_gr00t_dataset_v1.py',
    'scripts/wm02/infer_task_learning_live_v1.py',
    'scripts/wm02/infer_fusion_paused_vla.py',
    'scripts/pi0294/resident.py','scripts/pi0294/e1/model.py',
    'scripts/pi0294/numerics.py')))


def policy_contract(value,checkpoint_files,*,verify_package=True,expected_global_step=2000):
    need(set(value)=={'arm','parent_manifest_sha256','package','Core_qualification'}
        and value['arm']in('B0','B1')and value['parent_manifest_sha256']==profile.PARENT,
        'EXPLICIT_ORIGINAL_B0_OR_TRAINED_INTERNAL_GR00T_B1')
    if value['arm']=='B0':
        need(value['package']is None and value['Core_qualification']is None,
             'B0_ORIGINAL_WEIGHTS_AND_ORIGINAL_PROCESSOR')
        return
    need(type(expected_global_step)is int and expected_global_step in(2000,4000,8000),
         'DECLARED_PRODUCTION_ENDPOINT')
    from .stable_task_scene_v1 import pinned
    from .target_gr00t_inference_v1 import CROSS_SITE_ATOL
    proof=pinned(value['Core_qualification']);package=value['package']
    check=proof.get('numerical')or{}
    need(proof.get('schema')=='TASK_LEARNING_CORE_PACKAGE_QUALIFICATION_V1'
        and proof.get('package')==package
        and check.get('status')=='PASS_ACTUAL_HOPPER_CORE_MATCHED_INPUT_NOISE_NUMERICAL_GATE'
        and check.get('absolute_component_tolerances')==CROSS_SITE_ATOL
        and check.get('actual_matched_queries')==40
        and all(k in check.get('actual_maximum_absolute_error',{})
            and np.isfinite(check['actual_maximum_absolute_error'][k])
            and 0<=check['actual_maximum_absolute_error'][k]<=limit
            for k,limit in CROSS_SITE_ATOL.items()),'REAL_CORE_PACKAGE_NUMERICAL_QUALIFICATION')
    spec_path=Path(proof['Core_spec']['path'])
    need(digest(spec_path)==proof['Core_spec']['sha256'],'EXACT_CORE_RECONSTRUCTION_SPEC')
    s=json.loads(spec_path.read_bytes())
    need(s.get('diagnostic_only')is not True and s.get('numerical_diagnostic')is None
         and s.get('tensor_trace')is None and s.get('backbone_trace')is None
         and s.get('inference_precision_diagnostic')is None,
         'DIAGNOSTIC_NUMERICAL_ABLATION_IS_NOT_NATIVE_POLICY_QUALIFICATION')
    root=Path(s['registered_root'])
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    result=Path(proof['Core_result']['path'])
    need(result==root/'fit-01/RESULT.private.json'
        and digest(result)==proof['Core_result']['sha256']
        and lease.get('status')=='CLOSED'and lease.get('spec_sha256')==digest(spec_path)
        and lease.get('worker_exit_code')==worker.get('exit_code')==0
        and worker.get('failure')is None and s.get('candidate_package')==package
        and s.get('profile')=='wm02-task-learning-package-core-v1'
        and s.get('kind')=='TASK_LEARNING_PACKAGE_CORE_RECONSTRUCTION_V1'
        and s.get('target_inference_mode')=='CONTINUOUS_TASK_PACKAGE_RECONSTRUCTION_TRAIN40',
        'ACTUAL_CLOSED0_CORE_MODEL_RECONSTRUCTION')
    r=json.loads(result.read_bytes())
    need(r.get('actual_model_loads')==1 and r.get('actual_VLA_queries')==40
        and r.get('Processor_decode_calls')==40
        and r.get('target_native_execution')is False
        and (r.get('model_identity')or{}).get('candidate_package')==package
        and (r.get('model_identity')or{}).get('global_step')==expected_global_step
        and r.get('GR00T_updates')==r.get('WM_updates')==r.get('Actor_updates')==0,
        'ACTUAL_CORE_FORTY_QUERY_RECONSTRUCTION')
    if verify_package:
        from .target_gr00t_delivery_v1 import verify
        m=verify(package['path'],package['sha256'],checkpoint_files)
        need(m.get('profile')=='TASK_LEARNING_GR00T_V1'
            and m.get('training_site')=='HOPPER_SINGLE_GPU'
            and m['recipe']['dataset_sha256']==s['target_dataset_sha256']
            and m['recipe']['statistics_sha256']==s['target_statistics_sha256']
            and m.get('global_step')==expected_global_step,
            'QUALIFIED_SAME_TARGET_PROCESSOR_AND_MODEL_PACKAGE')


def validate_spec(s,*,verify_model=True):
    if task.native_body_role(s.get('core_local_protocol')):
        return task.native_body_profile(s['core_local_protocol']).validate_live(s,verify_model=verify_model)
    if s.get('core_local_protocol')==task.MATCHED_ROLE:
        from .task_learning_matched_native_v3_1 import validate_live
        return validate_live(s,verify_model=verify_model)
    if s.get('core_local_protocol')==task.ENDPOINT_ROLE:
        from .task_learning_endpoint_native_v3 import validate_live
        return validate_live(s,verify_model=verify_model)
    if s.get('core_local_protocol')==task.CONIC_ROLE:
        from .task_learning_conic_native_v3 import validate_live
        return validate_live(s,verify_model=verify_model)
    if s.get('core_local_protocol') is not None:
        from .task_learning_local_policy_v1 import validate_live
        return validate_live(s,verify_model=verify_model)
    from scripts.pi0292.relay_protocol import digest as document_hash
    from scripts.pi0294.numerics import MODE
    need(s.get('profile')==PROFILE and s.get('kind')==KIND
        and s.get('execution_site')=='CORE_FUSION_NEURAL'and s.get('physics')is False
        and s.get('device')=='cuda'and s.get('max_queries')==task.QUERIES
        and s.get('GR00T_updates')==s.get('WM_updates')==s.get('Actor_updates')==0
        and s.get('numeric_mode')==MODE and s.get('baseline_changed')is False
        and s.get('checkpoint_manifest_sha256')==profile.PARENT
        and s.get('task_learning_grasp')==task.protocol(task.STUDENT_DEV_ROLE),
        'FINITE_FIFTEEN_QUERY_MODEL_ONLY_ORIGINAL_NUMERICS')
    need(isinstance(s.get('checkpoint_files'),dict)
        and document_hash(s['checkpoint_files'])==CHECKPOINT_INVENTORY,
        'EXACT_ORIGINAL_PARENT_WEIGHTS_AND_PROCESSOR_INVENTORY')
    descriptor=s['producer_native_spec'];path=Path(descriptor['path'])
    need(path.is_absolute()and path.resolve()==path and not path.is_symlink()
        and path.name=='EXECUTION_SPEC.json'and digest(path)==descriptor['sha256'],
        'EXACT_REGISTERED_NATIVE_PRODUCER_SPEC')
    producer=json.loads(path.read_bytes());task.declared(producer)
    need(producer.get('task_learning_role')==task.STUDENT_DEV_ROLE,'ACTUAL_NEW_STUDENT_DEV_NATIVE_PRODUCER')
    need(path==Path(producer['registered_root'])/'EXECUTION_SPEC.json'
        and producer.get('target_gr00t_gpu_root')==s['registered_root']
        and producer.get('source_sha')==s['source_sha']
        and producer.get('target_gr00t_policy')==s.get('target_gr00t_policy')
        and producer.get('target_gr00t_B1_global_step',2000)==s.get('target_gr00t_B1_global_step',2000)
        and type(s.get('target_gr00t_pair_id'))is str and bool(s['target_gr00t_pair_id'])
        and producer.get('target_gr00t_pair_id')==s['target_gr00t_pair_id']
        and producer.get('target_gr00t_query_seeds')==s.get('target_gr00t_query_seeds')
        and isinstance(s.get('target_gr00t_query_seeds'),list)
        and len(s['target_gr00t_query_seeds'])==task.QUERIES
        and all(type(v)is int and 0<=v<2**63 for v in s['target_gr00t_query_seeds'])
        and len(set(s['target_gr00t_query_seeds']))==task.QUERIES,
        'FIFTEEN_PREREGISTERED_SAME_MODEL_NATIVE_GPU_QUERY_SLOTS')
    policy_contract(s['target_gr00t_policy'],s['checkpoint_files'],verify_package=verify_model,
        expected_global_step=s.get('target_gr00t_B1_global_step',2000))
    memory_contract(s)
    return producer


def memory_contract(s):
    from .task_learning_core_inference_v1 import validate_memory
    return validate_memory(s)


def preflight(native):
    from .fusion_paused_query_bridge import _registered,_live
    path=Path(native['target_gr00t_gpu_root'])/'EXECUTION_SPEC.json'
    s=json.loads(path.read_bytes());producer=validate_spec(s)
    need(producer==native,'ACTUAL_SAME_NATIVE_SPEC_BEFORE_COMPUTE')
    _registered(native,Path(native['registered_root'])/'EXECUTION_SPEC.json')
    _registered(s,path);_live(s,native)
    need(not (Path(native['registered_root'])/'native-01/GR00T_REQUESTS').exists()
        and not (Path(s['registered_root'])/'fit-01/queries').exists(),
        'NEW_PAIR_WITHOUT_ANY_PREVIOUS_QUERY_ATTEMPT')
    return s


def validate_gpu(s,actual):
    from .fusion_scope import validate
    from .pilot1_scope import GPU_UUIDS
    from .pilot1_storage import local_path
    validate_spec(s)
    if s.get('core_local_protocol')==task.SLA_ROLE:
        from .sla_live_v1 import source_custody
        source_custody(s)
    need(s.get('GPU_uuid')in GPU_UUIDS and actual['uuid']==s['GPU_uuid']
        and os.environ.get('CUDA_VISIBLE_DEVICES')==s['GPU_uuid']
        and type(s.get('checkpoint_parameter_bytes'))is int and s['checkpoint_parameter_bytes']>0
        and s['checkpoint_parameter_bytes']<=s.get('allocator_limit_bytes',0)<actual['free_bytes']-1024**3,
        'ORIGINAL_SHARED_GPU_UUID_ALLOCATOR_AND_HEADROOM')
    validate(s);local_path(s,s['checkpoint'])
    if s['target_gr00t_policy']['package']:local_path(s,s['target_gr00t_policy']['package']['path'])
    # Reuse the actual original first calibration requirement, not a new grant.
    p=Path(s['calibration_result']);r=json.loads(p.read_bytes())
    l=json.loads((p.parent/'control/LEASE.json').read_bytes())
    need(digest(p)==s['calibration_result_sha256']and l.get('status')=='CLOSED'
        and r.get('schema')=='WM02_FUSION_FIRST_ACTUAL_CALIBRATION_GROUP_V1'
        and r.get('exit_code')==0 and r.get('actual_method_evaluations')==4
        and r.get('actual_native_ticks')==5*4103,'ORIGINAL_ACTUAL_CALIBRATION_BEFORE_QUERY')
    source=Path(s['source_root'])
    need(all(n in s['source_files']and digest(source/n)==s['source_files'][n]for n in SOURCE_MEMBERS),
         'EXACT_LIVE_MODEL_AND_ORIGINAL_SINGLE_DECODE_SOURCE')


def request_path(producer,index):
    need(type(index)is int and 0<=index<task.QUERIES,'FINITE_REQUEST_INDEX')
    return Path(producer['registered_root'])/'native-01/GR00T_REQUESTS'/f'{index:02d}.json'


def request_body(s,observation,index):
    from .stable_task_prefix_v1 import observation_steps
    need(type(index)is int and 0<=index<task.QUERIES,'FINITE_REQUEST_INDEX')
    producer=validate_spec(s,verify_model=False);step=observation_steps(producer)[index]
    need(observation.get('native_step')==step
        and abs(observation['native_time_s']-step*.002)<=1e-8,
        'FRESH_OBSERVATION_AT_EXACT_CURRENT_CYCLE')
    extra={}
    if producer.get('task_learning_role')==task.SLA_ROLE:
        frame=Path(producer['registered_root'])/'native-01/SLA_TASK_FRAME.private.json'
        from .stable_task_scene_v1 import pinned
        extra['sla_task_frame']=dict(path=str(frame),sha256=digest(frame))
        f=pinned(extra['sla_task_frame'])
        need(f['native_spec_sha256']==s['producer_native_spec']['sha256'] and f['native_step']==task.B
             and f['fixed_for_episode'] and not f['object_truth_used'],'SLA_EPISODE_FIXED_MEASURED_FRAME')
        if index>0:
            first_path=request_path(producer,0)
            first=json.loads(first_path.read_bytes())
            completed=json.loads((Path(s['registered_root'])/'fit-01/queries/00/RESULT.json').read_bytes())
            need(first['sla_task_frame']==extra['sla_task_frame']
                 and completed['request_sha256']==digest(first_path),'SLA_SAME_FIRST_CAPTURE_FRAME_ALL_CYCLES')
    return dict(schema=SCHEMA+'_REQUEST',pair_id=s['target_gr00t_pair_id'],index=index,**extra,
        producer_native_spec_sha256=s['producer_native_spec']['sha256'],
        seed=s['target_gr00t_query_seeds'][index],observation=observation,
        expected_native_step=step,model_policy=s['target_gr00t_policy'])


def read_request(s,producer,index):
    path=request_path(producer,index);r=json.loads(path.read_bytes())
    need(r==request_body(s,r['observation'],index),'EXACT_FRESH_QUERY_REQUEST_BINDING')
    from scripts.pi0292.relay_protocol import unseal
    from .fusion_observation import shadow_inference_payload
    observation=r['observation']
    capture=Path(observation['path']);native=Path(producer['registered_root'])/'native-01'
    need(capture.resolve()==capture and capture.is_relative_to(native/'PAUSED_OBSERVATIONS')
        and digest(capture)==observation['sha256'],'OWN_REAL_HASH_BOUND_CURRENT_CAPTURE')
    body=unseal(json.loads(capture.read_bytes()));shadow=unseal(body['original_shadow_ABI'])
    need(body['native_step']==observation['native_step']==r['expected_native_step']
        and body['world']['execution_spec_sha256']==s['producer_native_spec']['sha256']
        and body['world']['scene_sha256']==digest(producer['model_path'])
        and body['world']['native_output']==str(native)
        and body['task']['language']==task.LANGUAGE
        and shadow['native_step']==body['native_step']
        and shadow['target_observation']['native_state_sha256']==body['files']['STATE.private.bin'],
        'SAME_ACTUAL_NATIVE_STATE_WORLD_TASK_AND_POLICY_VIEW')
    for name,pin in body['files'].items():
        member=capture.parent/name
        need(not Path(name).is_absolute()and '..'not in Path(name).parts
            and member.resolve()==member and digest(member)==pin,'REAL_CAPTURE_MEMBER_BYTES')
    payload=shadow_inference_payload(body['original_shadow_ABI'])
    return r,payload


def response(s,index,request,result,folder):
    need(result.get('schema')==SCHEMA+'_RESULT'and result.get('index')==index
        and result.get('pair_id')==s['target_gr00t_pair_id']
        and result.get('request_sha256')==digest(request_path(validate_spec(s,verify_model=False),index))
        and result.get('observation_sha256')==request['observation']['sha256']
        and result.get('seed')==s['target_gr00t_query_seeds'][index]
        and result.get('model_policy')==s['target_gr00t_policy']
        and result.get('GPU_spec_sha256')==digest(Path(s['registered_root'])/'EXECUTION_SPEC.json')
        and result.get('Processor_decode_calls')==1 and result.get('native_integrations')==0
        and result.get('GR00T_updates')==result.get('WM_updates')==result.get('Actor_updates')==0,
        'EXACT_ACTUAL_MODEL_RETURN_FOR_THIS_OBSERVATION')
    action=Path(folder)/'ACTION.npz'
    need(digest(action)==result['action_sha256'],'COMPLETE_RETURNED_ACTION_BYTES')
    return action


def query_from_native(adapter,observation,index):
    from scripts.pi0292.durable import write
    from .fusion_paused_query_bridge import _registered,_live
    from .fusion_paused_observation import verify_still_paused
    s_path=Path(adapter.spec['target_gr00t_gpu_root'])/'EXECUTION_SPEC.json'
    s=json.loads(s_path.read_bytes());producer=validate_spec(s,verify_model=False)
    need(producer==adapter.spec,'SAME_ACTUAL_NATIVE_OWNER')
    _registered(s,s_path);_live(s,producer)
    request=request_body(s,observation,index);path=request_path(producer,index)
    if producer.get('task_learning_role')==task.SLA_ROLE and index==0:
        from scripts.pi0294.task_state import read_state
        from .stable_task_scene_v1 import pinned
        import hashlib
        capture=Path(observation['path'])
        packet=read_state(capture.parent/'STATE.private.bin')
        frame=pinned(request['sla_task_frame'])
        need(packet['step']==task.B and np.array_equal(packet['qpos'],adapter.observe()['qpos'])
             and hashlib.sha256(np.asarray(packet['qpos'],dtype='<f8').tobytes()).hexdigest()==frame['initial_qpos_sha256'],
             'SLA_FIXED_FRAME_BOUND_TO_FIRST_ACTUAL_CAPTURE_PACKET')
    need(not path.exists(),'QUERY_ATTEMPT_NEVER_REPLAYED')
    path.parent.mkdir(exist_ok=True)
    write_json(path.with_suffix('.intent.json'),dict(request=request,retry_authorized=False))
    write(path,request) # Atomic publication, never a partly written request.
    folder=Path(s['registered_root'])/'fit-01/queries'/f'{index:02d}'
    result_path=folder/'RESULT.json';started=time.monotonic()
    while not result_path.exists():
        adapter.guard();_live(s,producer)
        need(not (Path(s['registered_root'])/'fit-01/FAILED.private.json').exists(),
             'PAIRED_GPU_FAILED_NO_TEACHER_OR_WM_FALLBACK')
        try:_registered(s,s_path)
        except (ValueError,PermissionError):
            if result_path.exists():break
            raise
        adapter.spin(.15)
    result=json.loads(result_path.read_bytes());action=response(s,index,request,result,folder)
    record=json.loads(Path(observation['path']).read_bytes())
    pause=verify_still_paused(adapter,record)
    return action,result['action_sha256'],dict(result=result,
        result_sha256=digest(result_path),request_sha256=digest(path),
        actual_pause_after_inference=pause,query_wait_wall_s=time.monotonic()-started)


def _load(s,root):
    if task.native_body_role(s.get('core_local_protocol')):
        return task.native_body_profile(s['core_local_protocol']).load(s,root)
    if s.get('core_local_protocol')==task.MATCHED_ROLE:
        from .task_learning_matched_native_v3_1 import load
        return load(s,root)
    if s.get('core_local_protocol')==task.ENDPOINT_ROLE:
        from .task_learning_endpoint_native_v3 import load
        return load(s,root)
    if s.get('core_local_protocol')==task.CONIC_ROLE:
        from .task_learning_conic_native_v3 import load
        return load(s,root)
    from scripts.wm02.infer_fusion_paused_vla import _load_frozen
    instance,torch=_load_frozen(s,root)
    from .target_gr00t_core_v1 import trim_unused_logits
    trim_unused_logits(instance.model)
    policy=s['target_gr00t_policy']
    if policy['arm']=='B1':
        from .target_gr00t_delivery_v1 import install
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
        from scripts.pi0294.local_backbone import local_model_name
        from scripts.pi0294.neural import COSMOS_REVISION
        package=policy['package'];install(instance.model,package['path'],package['sha256'],s['checkpoint_files'])
        bindings=s['neural_bindings']
        instance.processor=Gr00tN1d7Processor.from_pretrained(Path(package['path'])/'processor',
            model_name=local_model_name(s['registered_root'],bindings),local_files_only=True,
            transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
                cache_dir=bindings['cosmos_hub'],revision=COSMOS_REVISION))
        instance.processor.eval()
        need(instance.processor.embodiment_id_mapping['new_embodiment']==10,
             'SAVED_TARGET_PROCESSOR_ORIGINAL_CATEGORY10')
    instance.identity=dict(policy=policy,numeric_mode=s['numeric_mode'])
    write_json(root/'TARGET_POLICY_LOADED.private.json',dict(identity=instance.identity,
        actual_model_loads=1,original_B0_Processor=policy['arm']=='B0',
        ordinary_internal_GR00T_weights=True,external_inference_adapter=False))
    return instance,torch


def native_terminal(producer):
    """An early native rejection is retained; do not wait for unused query slots."""
    root=Path(producer['registered_root']);lease_path=root/'control/LEASE.json'
    try:
        lease = json.loads(lease_path.read_bytes())
    except FileNotFoundError:
        return None
    if lease.get('status')!='CLOSED':return None
    # CLOSED0 is intentionally not required: a rejected student proposal can
    # terminate the native worker nonzero without a model-serving failure.
    spec_path=root/'EXECUTION_SPEC.json'
    need(json.loads(spec_path.read_bytes())==producer
        and lease.get('spec_sha256')==digest(spec_path)
        and lease.get('run_uuid')==root.name,'SAME_CLOSED_NATIVE_PRODUCER')
    worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    result=root/'native-01/RESULT.json'
    need(lease.get('worker_exit_code')==worker.get('exit_code')and result.is_file(),
        'ACTUAL_TERMINAL_NATIVE_RESULT_AND_WORKER')
    return dict(worker_exit_code=worker['exit_code'],worker_sha256=digest(root/'WORKER_EXIT.private.json'),
        result_sha256=digest(result),lease_sha256=digest(lease_path))


def wait_request(s,producer,index):
    from .fusion_paused_query_bridge import _registered,_live
    p=request_path(producer,index)
    lease_path = Path(producer['registered_root']) / 'control/LEASE.json'
    while True:
        terminal = native_terminal(producer)
        if terminal is not None:return None,terminal
        _live(s,producer)
        try:_registered(producer,s['producer_native_spec']['path'])
        except FileNotFoundError as error:
            if error.filename != str(lease_path):
                raise
            # The local filesystem can briefly hide the atomically replaced
            # lease. Read again under the original deadline before accepting
            # a request or marking the producer terminal.
            time.sleep(.15)
            continue
        except (ValueError,PermissionError):
            terminal = native_terminal(producer)
            if terminal is not None:return None,terminal
            raise
        if p.exists():return p,None
        time.sleep(.15)


def infer(s,output):
    if s.get('core_local_protocol')==task.SLA_ROLE:
        from .sla_live_v1 import infer as infer_sla
        return infer_sla(s,output)
    from scripts.pi0294.numerics import prepare_environment,configure_torch
    from scripts.wm02.infer_fusion_paused_vla import _generate_once
    from scripts.pi0294.e1.artifacts import atomic_npz
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0292.durable import write
    from .fusion_paused_query_bridge import _registered,_live
    root=Path(output);producer=validate_spec(s);start=time.monotonic();cpu=time.process_time()
    write_json(root/'ATTEMPT.private.json',dict(pair_id=s['target_gr00t_pair_id'],retry_authorized=False))
    records=[];loads=0;attempted=0;terminal=None
    try:
        _registered(producer,s['producer_native_spec']['path']);_live(s,producer)
        if s.get('core_local_protocol')in(task.NATIVE_A500_ROLE,task.SLA_G_ROLE):
            from .native_arx5_training_runtime_v1 import checked_parent
            from .stable_task_scene_v1 import pinned
            checked_parent(s['checkpoint'],pinned(s['official_parent_audit']))
        else:
            for n,pin in s['checkpoint_files'].items():
                need(digest(Path(s['checkpoint'])/n)==pin,'FROZEN_PARENT_FILE:'+n)
        prepare_environment(s['numeric_mode'])
        import torch
        configure_torch(torch,s['numeric_mode']);torch.set_num_threads(4)
        instance,torch=_load(s,root);loads=1
        for index in range(task.QUERIES):
            p,terminal=wait_request(s,producer,index)
            if p is None:break
            try:_registered(producer,s['producer_native_spec']['path'])
            except (ValueError,PermissionError):
                terminal=native_terminal(producer)
                if terminal is not None:break
                raise
            _live(s,producer)
            request,payload=read_request(s,producer,index)
            folder=root/'queries'/f'{index:02d}';folder.mkdir(parents=True,exist_ok=False)
            journal=Journal(folder/'journal',folder.name)
            def generate():
                nonlocal attempted
                attempted+=1
                value,precision,timing=_generate_once(instance,payload,request['seed'],torch)
                need(value['identity']==instance.identity and value['physics_integrations']==0
                    and value['extra_correction']is False,'ACTUAL_ORIGINAL_GENERATION')
                need(value['physical_H40'].shape==(40,10)and value['normalized'].shape==(1,40,132)
                    and np.isfinite(value['physical_H40']).all()and np.isfinite(value['normalized']).all()
                    and value['noise'].get('shape')==[1,40,132]
                    and value['noise'].get('dtype')=='torch.float32',
                    'ACTUAL_COMPLETE_FINITE_H40_AND_ORIGINAL_NOISE')
                action=folder/'ACTION.npz'
                packet=dict(physical_H40=value['physical_H40'],normalized=value['normalized'])
                if s.get('core_local_protocol')==task.CONIC_ROLE:
                    from .task_learning_conic_policy_v3 import action_packet
                    packet=action_packet(value,s['target_gr00t_policy']['arm'])
                atomic_npz(action,**packet)
                diagnostics={}
                if s.get('core_local_protocol')==task.CONIC_ROLE:
                    from .task_learning_local_analysis_v1 import geometry
                    diagnostics['raw_original_envelope']=geometry(value['physical_H40'],payload['state_eef10'])
                    diagnostics['decoded_original_envelope']=(geometry(packet['explicit_conic_H40'],payload['state_eef10'])
                        if 'explicit_conic_H40'in packet else None)
                    diagnostics['explicit_policy_method']=s['target_gr00t_policy']['arm']
                return dict(schema=SCHEMA+'_RESULT',index=index,pair_id=s['target_gr00t_pair_id'],
                    request_sha256=digest(p),observation_sha256=request['observation']['sha256'],
                    seed=request['seed'],model_policy=s['target_gr00t_policy'],noise=value['noise'],
                    GPU_spec_sha256=digest(Path(s['registered_root'])/'EXECUTION_SPEC.json'),
                    action_sha256=digest(action),Processor_decode_calls=1,native_integrations=0,
                    GR00T_updates=0,WM_updates=0,Actor_updates=0,precision=precision,**timing,**diagnostics)
            result=journal.once('GENERATE',dict(request_sha256=digest(p),policy=s['target_gr00t_policy']),generate)
            records.append(result);write(folder/'RESULT.json',result)
        write_json(root/'RESULT.private.json',dict(schema=SCHEMA,status='ACTUAL_CONTINUOUS_LIVE_QUERIES_TERMINAL',
            actual_model_loads=loads,actual_VLA_inferences=len(records),Processor_decode_calls=len(records),
            requested_query_slots=task.QUERIES,unused_query_slots=task.QUERIES-len(records),
            actual_early_native_terminal=terminal,
            GR00T_updates=0,WM_updates=0,Actor_updates=0,native_integrations=0,records=records,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,native_task_success='NOT_ESTABLISHED_BY_GPU'))
    except BaseException as error:
        write_json(root/'FAILED.private.json',dict(schema=SCHEMA,failure=type(error).__name__+': '+str(error),
            completed_model_loads=loads,model_load_completion_unknown=(root/'MODEL_LOAD_INTENT.private.json').exists()and loads==0,
            generation_invocations=attempted,completed_query_records=len(records),
            query_completion_unknown_after_intent=attempted>len(records),retry_authorized=False,
            native_integrations=0,GR00T_updates=0,WM_updates=0,Actor_updates=0,
            wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu))
        raise
