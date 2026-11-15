"""Separately scoped matched PD4 student TRAIN4 evaluation, frozen WBC.

No legacy3K admission is relaxed. Every new500 candidate must have real save,
fresh RGB32 and qualified new statistics before these eight fixed conditions.
"""
from contextlib import contextmanager
import json,os,time
from pathlib import Path
from .storage import digest
from .stable_task_scene_v1 import pinned
from . import task_learning_grasp_v1 as task

KEY='pd4_student_v71'
VERSION='SLA_PD4_MATCHED500_RAW_P10_STUDENT_V7_1'
PROTOCOL='SLA_PD4_MATCHED500_RAW_P10_NATIVE_EVALUATION_V71'
NATIVE_SCHEMA=VERSION+'_NATIVE_ACTUAL'
EXPERIMENT=VERSION+'_TRAIN4_NOT_DEV_TEST'
SCOPE=VERSION+'_SIM_ONLY'
GPU_SITE='CORE_SLA_PD4_STUDENT_RGB_V71'
NATIVE_SITE='CORE_SLA_PD4_STUDENT_NATIVE_V71'
ENDPOINT=21703
QUERIES=60
LAYOUTS=('TRAIN01','TRAIN04','TRAIN07','TRAIN10')


def profile(value):
    """Resolve explicit system identity; do not rename old raw experiments."""
    import sys
    role=value.get('core_local_protocol') or value.get('task_learning_role') or value.get('protocol',{}).get('stage')
    if role==task.SLA_JOIN_ROLE:
        from . import pd4_join_v72
        return pd4_join_v72
    if role not in (None,PROTOCOL):raise PermissionError('EXPLICIT_PD4_RAW_OR_JOIN_SYSTEM_IDENTITY')
    if value.get('reference_join10_v72')is not None:raise PermissionError('JOIN10_CANNOT_MODIFY_RAW_IDENTITY')
    return sys.modules[__name__]


def protocol():
    from .sla_dense_native_v5 import protocol as dense
    return dict(dense(),stage=PROTOCOL,scene_family='PD4_MATCHED_TRAIN4_NEW_RESTARTS_NOT_GENERALIZATION',
        policy='SLA_MATCHED500_RAW_H40_WITHOUT_ENDPOINT_BASIS')


def policy_contract(value,checkpoint_files=None,*,verify_package=True,expected_global_step=500):
    from ..sla_adapter.pd4_config_v71 import MANIFESTS,STATISTICS,OPERATION
    from ..sla_adapter.provider_v3 import contract
    if set(value)!={'arm','package','qualification'}or value['arm']not in MANIFESTS or expected_global_step!=500:
        raise PermissionError('PD4_ONLY_TWO_FIXED500_RAW_POLICIES')
    pin=value['qualification'];v=pinned(pin);root=Path(pin['path']).parent
    spec=pinned(dict(path=str(root/'EXECUTION_SPEC.json'),sha256=json.loads((root/'control/LEASE.json').read_bytes())['spec_sha256']))
    lease=json.loads((root/'control/LEASE.json').read_bytes());worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    if (lease['status']!='CLOSED' or lease['worker_exit_code']!=worker['exit_code']
            or worker['exit_code']!=0 or v['schema']!='SLA_PD4_SINGLE_MATCHED_ARM_ACTUAL_V7_1'
            or not v['complete500']or not v['initial87_bit_exact']or not v['fresh_RGB32_bit_exact']
            or(v['step'],v['target_visits'])!=(500,16000)or v['actual']['optimizer_updates']!=500
            or not v['full_restore']['bit_exact']or v['full_restore']['arrays']!=87
            or not v['frozen_fingerprint_unchanged']or v['PD4_view']!=value['arm']
            or v['package']!=value['package']or spec['PD4_view']!=value['arm']):
        raise PermissionError('REAL_TERMINAL500_SAVE_AND_FRESH_RGB32_REQUIRED')
    p=Path(value['package']['path']);m=pinned(dict(path=str(p/'MANIFEST.private.json'),sha256=value['package']['sha256']));contract(m);b=m['bindings']
    if (m['step']!=500 or m['cursor']!=16000 or m['qualification_only']
            or b['recipe']!=OPERATION or b['PD4_view']!=value['arm']
            or b['dense_manifest_sha256']!=MANIFESTS[value['arm']]
            or m['statistics_sha256']!=STATISTICS[value['arm']]
            or b['supervision_mask']!='ALL_TRUE_ORIGINAL_FM32' or b['consumed_mask_used_for_loss']is not False
            or m['sampler_steps']!=10 or m['action_horizon']!=40 or m['label_interval_s']!=.05):
        raise PermissionError('EXACT_MATCHED500_RAW_FP32_LOGIT_PACKAGE')
    if verify_package:
        for row in m['adapter_arrays']:
            if digest(p/row['file'])!=row['sha256']:raise PermissionError('ORIGINAL_TRAINED_PD4_ADAPTER_BYTES')
        if digest(p/'STATISTICS.json')!=m['statistics_sha256']:raise PermissionError('FROZEN_PD4_STATISTICS_BYTES')
    return dict(result=v,spec=spec,manifest=m,root=root)


def panel_contract(panel):
    from ..sla_adapter.pd4_config_v71 import MANIFESTS
    p=profile(panel);rows=panel['assignments']
    if (panel['schema']!=p.VERSION+'_PANEL'or panel['B1_global_step']!=500
            or panel['protocol']!=p.protocol()or len(rows)!=8 or len({r['native_run']for r in rows})!=8
            or {(r['method'],r['layout'])for r in rows}!={(v,l)for v in MANIFESTS for l in LAYOUTS}
            or panel['old_RP8_passed']is not False or panel['automatic_retry']is not False):
        raise PermissionError('EIGHT_MATCHED_PD4_STUDENT_CONDITIONS_NOT_OLD_POPULATIONS')
    for layout in LAYOUTS:
        a,b=[r for r in rows if r['layout']==layout]
        if a['query_seeds']!=b['query_seeds']or a['restart']!=b['restart']or len(set(a['query_seeds']))!=60:
            raise PermissionError('SAME_FIXED_SCENE_RESTART_AND_SIXTY_NOISE_SLOTS')
        if any(type(v)is not int or not 0<=v<2**63 for v in a['query_seeds']):
            raise PermissionError('EXACT_PREREGISTERED_INTEGER_NOISE_SLOTS')
    return rows


def configuration(spec):
    p=profile(spec);c=spec[KEY]
    if p.PROTOCOL==task.SLA_JOIN_ROLE:
        from .sla_reference_join_v72 import configuration as join
        if spec.get('reference_join10_v72')!=join():raise PermissionError('EXACT_FIXED_JOIN10_SYSTEM')
    if (c['version']!=p.VERSION or spec['scope']!=p.SCOPE
            or spec.get('core_local_protocol')!=p.PROTOCOL
            or spec['task_learning_grasp']!=p.protocol()or c['old_RP8_passed']is not False
            or any(spec.get(k)for k in ('pilot1_binding','fusion_binding','night1_binding','next1_binding','next2_binding'))):
        raise PermissionError('EXPLICIT_FRESH_PD4_STUDENT_SCOPE')
    return c


def _original(spec):
    c=configuration(spec);old=pinned(c['original_spec'])
    from .sla_label_replay_v61 import _archived_scene
    proof,builder=_archived_scene(c['original_spec']['path'],c['original_spec']['sha256'])
    for k in ('native_binary','solver_binary','observer_library','outer_library','trace_library'):
        if old[k]!=spec[k]or digest(spec[k])!=old['frozen_files'][old[k]]:raise PermissionError('UNCHANGED_NATIVE_RUNTIME:'+k)
    from .pd4_prospective_v71 import CONTROLLER
    if digest(spec['controller_binary'])!=CONTROLLER or old['frozen_files'][old['controller_binary']]!=CONTROLLER:
        raise PermissionError('ORIGINAL_CONTROLLER_BYTES')
    return old,proof,builder


def native_scene_admission(spec):
    configuration(spec);p=profile(spec);old,proof,builder=_original(spec)
    from .task_learning_scene_v1 import validate_static_provenance
    validate_static_provenance(spec,proof,expected_role=p.PROTOCOL,verified_builder_sha256=builder)
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']));matches=[r for r in panel_contract(panel)if r['native_run']==Path(spec['registered_root']).name]
    if len(matches)!=1:raise PermissionError('UNIQUE_FIXED_PD4_STUDENT_CONDITION')
    r=matches[0]
    if (r['layout']!=proof['layout']or spec['target_gr00t_policy']!=panel['policies'][r['method']]
            or spec['target_gr00t_query_seeds']!=r['query_seeds']or spec['target_gr00t_B1_global_step']!=500
            or spec['target_gr00t_pair_id']!=r['native_run']or spec['data_role']!='TRAIN'
            or spec.get('task_learning_teacher_mode')is not None or spec.get('task_learning_control')is not None
            or spec.get('expert_eef_stream_v2')is not None):
        raise PermissionError('CURRENT_RGB_RAW_STUDENT_WITHOUT_TEACHER_OR_RECONSTRUCTOR')
    policy_contract(spec['target_gr00t_policy'],verify_package=False)
    return proof


def validate_scope(spec):
    c=configuration(spec);p=profile(spec);binding=pinned(c['binding']);G=Path(binding['registered_root']);lease=json.loads((G/'control/LEASE.json').read_bytes());now=time.time()
    if (spec['host']!=os.uname().nodename or spec['host']!='corelab-4080s'or spec['uid']!=2009 or os.getuid()!=2009
            or spec['owner']!='EXISTING_LOCAL_PC'or binding['schema']!=p.VERSION+'_BINDING'
            or binding['scope']!=p.SCOPE or binding['source_sha']!=spec['source_sha']
            or not binding['activated_epoch']<=now<spec['deadline_epoch']<=binding['deadline_epoch']
            or binding['maximum_conditions']!=8 or binding['deadline_epoch']-binding['activated_epoch']>14400
            or str(Path(spec['registered_root']))not in binding['roots']
            or lease['status']!='ACTIVE'or lease['binding_sha256']!=c['binding']['sha256']
            or any((r/'STOP.json').exists()for r in (G,Path(spec['registered_root']),Path(spec['campaign_root'])))):
        raise PermissionError('FRESH_REGISTERED_FINITE_PD4_STUDENT_OWNER')
    reg=json.loads(Path(spec['registry']).read_bytes())
    if not any(v.get('path')==spec['registered_root']and v.get('scope')==p.SCOPE and v.get('source_sha')==spec['source_sha']and v.get('owner')==spec['owner']for v in reg['entries']):raise PermissionError('ACTUAL_PD4_STUDENT_REGISTRY')
    return binding


def recipe_config(spec,episode,*,actual_phase,actual_kind,source_members):
    configuration(spec);native_scene_admission(spec);p=profile(spec)
    if (actual_phase!=task.PHASE or actual_kind!=task.KIND or spec['task_learning_role']!=p.PROTOCOL
            or spec['max_ticks']!=ENDPOINT or spec['physics_dt_s']!=.002 or spec['WBC_changes']is not False
            or spec['wall_limit_s']!=3600 or spec['reference_binding_version']!='MEASURED_EEF_AND_BASE_POSE_HOLD_V2'):
        raise PermissionError('UNCHANGED_PD4_STUDENT_NATIVE_RECIPE')
    if not all(digest(Path(spec['source_root'])/n)==spec['source_files'][n]for n in source_members):raise PermissionError('SEALED_SHARED_NATIVE_SOURCE')
    for n,pin in spec['frozen_files'].items():
        if digest(n)!=pin:raise PermissionError('FROZEN_PD4_NATIVE_BYTES:'+n)
    catalog=pinned(dict(path=spec['scene_allowlist'],sha256=spec['scene_allowlist_sha256']))
    matches=[r for r in catalog['scenes']if r['scene_id']==spec['scene_id']]
    if (len(matches)!=1 or matches[0]['parent']!=spec['data_parent']or matches[0]['split']!='TRAIN'
            or matches[0]['model_sha256']!=digest(spec['model_path'])or matches[0]['model_path']!=spec['model_path']
            or catalog['native_binary_sha256']!=digest(spec['native_binary'])):
        raise PermissionError('SAME_SCENE_AND_RUNTIME_ALLOWLIST')
    from .stable_task_scene_v1 import prefix_episode
    view=prefix_episode(pinned(dict(path=spec['episode_manifest'],sha256=spec['episode_manifest_sha256'])),spec)
    if episode is not None and view!=episode:raise PermissionError('ORIGINAL_TASK_LOCAL_PREPARATION')
    from .fusion_observation import rigid,transform_hash
    from .fusion_task_phase_initialization import FRAME_MAP
    frame=rigid(spec['task_phase_source_from_target'])
    if transform_hash(frame)!=spec['task_phase_frame_map_sha256']or spec['task_phase_frame_map_sha256']!=FRAME_MAP:raise PermissionError('EXACT_FROZEN_POLICY_FRAME')
    return view,dict(T_source_from_target=frame,scene_sha256=digest(spec['model_path']),source_pose_alignment=False)


def audit_prefix_predecessor(spec,prior):
    old,_,_=_original(spec);native=Path(old['registered_root'])/'native-01/RESULT.json';v=json.loads(native.read_bytes())
    from ..sla_adapter.s16_offline_v7 import DATA_SHA
    m=pinned(spec[KEY]['frozen_N20_manifest']);ep=next(e for e in m['episodes']if e['mount']['source_spec']==spec[KEY]['original_spec'])
    if (spec[KEY]['frozen_N20_manifest']['sha256']!=DATA_SHA or digest(native)!=ep['source_native_result_sha256']
            or v['status']!='NATIVE_STAGE_PASSED'or not v['reference_qualification']['task_local_preparation']['original_steady_prefix_qualified']
            or digest(spec['previous_stage'])!=spec['previous_stage_sha256']or prior!=json.loads(Path(spec['previous_stage']).read_bytes())):
        raise PermissionError('SEALED_SAME_RUNTIME_TASK_LOCAL_PREFIX_EVIDENCE')


@contextmanager
def admitted(spec_path,run_id,output):
    from .auto1_native import phase_limit,validate_parent_loader
    from .runtime import admitted_run
    from .corelab_world_owner_v3 import world_owner
    s=json.loads(Path(spec_path).read_bytes());validate_scope(s);p=profile(s)
    if s['execution_site']!=p.NATIVE_SITE or s['physics']is not True or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('PD4_NATIVE_CPU_SCOPE_ONLY')
    validate_parent_loader(s['native_environment']['LD_LIBRARY_PATH'],os.environ.get('LD_LIBRARY_PATH'))
    if phase_limit(s['phase'],json.loads(Path(s['previous_stage']).read_bytes()),s)!=ENDPOINT:raise PermissionError('EXACT_PD4_STUDENT_ENDPOINT')
    with world_owner(s,json.loads(Path(s['registry']).read_bytes())):
        with admitted_run(spec_path,run_id,'wm02-auto1-target',output,physics=True):yield s


def validate_live(spec,*,verify_model=True):
    from .task_learning_live_pair_v1 import PROFILE,KIND
    configuration(spec);validate_scope(spec);p=profile(spec)
    if (spec['profile']!=PROFILE or spec['kind']!=KIND or spec['execution_site']!=p.GPU_SITE
            or spec['physics']is not False or spec['max_queries']!=60 or spec['device']!='cuda'
            or spec['GR00T_updates']!=0 or spec['WM_updates']!=0 or spec['Actor_updates']!=0
            or spec['baseline_changed']is not False
            or spec['numeric_mode']!='SLA_PINNED_JAX_BF16_DETERMINISTIC_V1'):
        raise PermissionError('PD4_FINITE_GPU_LIVE_QUERY_SCOPE')
    producer=pinned(spec['producer_native_spec']);native_scene_admission(producer)
    if (producer['target_gr00t_gpu_root']!=spec['registered_root']or producer['source_sha']!=spec['source_sha']
            or Path(spec['producer_native_spec']['path'])!=Path(producer['registered_root'])/'EXECUTION_SPEC.json'
            or any(producer[k]!=spec[k]for k in ('target_gr00t_policy','target_gr00t_query_seeds',
                'target_gr00t_pair_id','target_gr00t_B1_global_step','observation_camera_profile'))):
        raise PermissionError('EXACT_PD4_NATIVE_GPU_PAIR')
    q=policy_contract(spec['target_gr00t_policy'],verify_package=verify_model)
    if (spec['GPU_uuid']!=q['spec']['GPU_uuid']or Path(spec['python']).resolve()!=Path(q['spec']['python']).resolve()
            or spec['SLA_runtime']['initial_function_rule']!=q['spec']['initial_function_rule']
            or spec['SLA_runtime']['upstream_sha']!=q['spec']['upstream_sha']
            or spec['SLA_runtime']['upstream_root']!=q['spec']['upstream_root']
            or spec['SLA_runtime']['JAX_memory_fraction']!=q['spec']['JAX_memory_fraction']
            or spec['allocator_limit_bytes']!=q['spec']['allocator_limit_bytes']
            or spec['neural_owner_lock']!=q['spec']['neural_owner_lock']
            or spec['checkpoint']!=q['spec']['parent_params']
            or spec['checkpoint_manifest_path']!=q['spec']['parent_manifest']['path']):
        raise PermissionError('SAME_ACTUAL_CANDIDATE_RGB_DEVICE_PARENT_AND_RUNTIME')
    return producer


def validate_gpu(spec,actual):
    from .pilot1_storage import local_path
    validate_live(spec)
    if (spec['GPU_uuid']!=actual['uuid']or os.environ.get('CUDA_VISIBLE_DEVICES')!=spec['GPU_uuid']
            or not spec['checkpoint_parameter_bytes']<=spec['allocator_limit_bytes']<actual['free_bytes']-1024**3):raise PermissionError('REAL_PD4_SHARED_GPU_HEADROOM')
    for k in ('source_root','registered_root','checkpoint','cache','logs','tmp','data'):local_path(spec,spec[k])
    from .sla_live_v1 import source_custody
    source_custody(spec)
