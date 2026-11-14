"""Fresh finite PD4-statistics native diagnostics, independent of failed RP8.

Only ownership and codec-statistic identity are new. The exact frozen runtime,
raw compiler, full-H40 certifier, native consumption and safety are retained.
No historical role/lease is reopened or original result changed.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from .storage import digest

KEY='pd4_statistics_prospective_v71'
VERSION='SLA_PD4_FROZEN_STATISTICS_PROSPECTIVE_V7_1'
SITE='CORE_SLA_PD4_PROSPECTIVE_NATIVE_V71'
SCOPE='SLA_PD4_PROSPECTIVE_SIM_ONLY_V71'
STATS={'PD4_EXECUTED_REF':'66ff4bdf39bfbf1db3d9ff394dad8a0a0384be1b4e22c0b80c1be2304cf6fe37',
       'PD4_FINAL_PLAN':'31462fbeb99e907605eb203d359c78dedaf8f356f643adb725dc637c9a52fcbf'}
LAYOUTS=('TRAIN01','TRAIN04','TRAIN07','TRAIN10')
CONTROLLER='9e8ce46837d09bc39ae04d4b49f1ce4dd8c8c2d31bea39eac56c5806d108c1f3'


def configuration(spec):
    c=spec[KEY];value=spec['expert_plan_capture_v61'];pin=value['transport_statistics']
    if (c['version']!=VERSION or c['view']not in STATS or c['old_RP8_passed']is not False
            or c['full_WBC_state_restored']is not False or spec['execution_site']!=SITE
            or spec['scope']!=SCOPE or spec['phase']!='collect-target-task-learning-grasp-v1'
            or spec['task_learning_role']!='CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3'
            or spec['task_learning_teacher_mode']!='NORMAL_TEACHER_V7'
            or spec['expert_eef_stream_v2']!=dict(version='EEF_STREAM_V2',prefix_atoms=10)
            or value['version']!=VERSION or pin['sha256']!=STATS[c['view']]
            or digest(pin['path'])!=pin['sha256']
            or value['codec_module']!='scripts.sla.pd4_plan_transport_v71'
            or any(spec.get(k)for k in ('target_gr00t_policy','task_learning_control','sla_label_replay_v61',
                'native_callback_replay_v61','sla_expert_epsilon_v3','sla_pick_place_v5','sla_five_object_hold_v5'))):
        raise PermissionError('EXACT_FROZEN_PD4_STATS_FRESH_ORACLE_ONLY')
    return value


def panel_contract(panel):
    from .task_learning_teacher_panel_v1 import FROZEN_TEACHER,teacher_fingerprint
    rows=panel['assignments']
    if (panel['schema']!=VERSION+'_PANEL' or len(rows)!=8
            or {(v['view'],v['layout'])for v in rows}!={(v,l)for v in STATS for l in LAYOUTS}
            or len({v['native_run']for v in rows})!=8
            or panel['Teacher_implementation_source_sections_sha256']!=FROZEN_TEACHER
            or teacher_fingerprint()!=FROZEN_TEACHER or panel['old_RP8_passed']is not False
            or panel['full_WBC_state_restored']is not False or panel['maximum_parallel_worlds']!=2
            or panel['automatic_retry']is not False):
        raise PermissionError('EIGHT_FIXED_PROSPECTIVE_STATS_CONDITIONS_NO_RETRY')
    return rows


def native_scene_admission(spec):
    from .stable_task_scene_v1 import pinned
    from .sla_label_replay_v61 import _archived_scene
    from .task_learning_scene_v1 import validate_static_provenance
    configuration(spec);panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    rows=panel_contract(panel);matches=[r for r in rows if r['native_run']==Path(spec['registered_root']).name]
    if len(matches)!=1 or panel['source_sha']!=spec['source_sha']:raise PermissionError('UNIQUE_FRESH_PD4_CONDITION')
    row=matches[0];old=pinned(row['original_spec']);proof,builder=_archived_scene(row['original_spec']['path'],row['original_spec']['sha256'])
    validate_static_provenance(spec,proof,expected_role=old['task_learning_role'],verified_builder_sha256=builder)
    if (row['layout']!=proof['layout'] or row['view']!=spec[KEY]['view']
            or old['model_path']!=spec['model_path'] or old['task_phase_source_from_target']!=spec['task_phase_source_from_target']
            or digest(spec['controller_binary'])!=CONTROLLER or old['frozen_files'][old['controller_binary']]!=CONTROLLER):
        raise PermissionError('ORIGINAL_SCENE_FRAME_AND_CONTROLLER_BYTES')
    for k in ('native_binary','solver_binary','observer_library','outer_library','trace_library'):
        if old[k]!=spec[k] or digest(spec[k])!=old['frozen_files'][old[k]]:raise PermissionError('UNCHANGED_NATIVE_RUNTIME:'+k)
    return proof


def validate_scope(spec):
    """Validate an explicitly new local grant; never repair an expired grant."""
    from .stable_task_scene_v1 import pinned
    configuration(spec);now=time.time();root=Path(spec['registered_root']);binding=pinned(spec[KEY]['binding'])
    group=Path(binding['registered_root']);lease=json.loads((group/'control/LEASE.json').read_bytes())
    if (spec['host']!=os.uname().nodename or spec['host']!='corelab-4080s' or spec['uid']!=2009 or os.getuid()!=2009
            or spec['owner']!='EXISTING_LOCAL_PC' or spec['profile']!='wm02-auto1-target'
            or spec['site_profile']!='4080-simulation' or spec['physics']is not True
            or spec['WBC_changes']is not False or spec['physics_dt_s']!=.002
            or spec['max_ticks']!=21703 or spec['wall_limit_s']!=3600
            or binding['schema']!=VERSION+'_BINDING' or binding['scope']!=SCOPE
            or binding['owner']!=spec['owner'] or binding['source_sha']!=spec['source_sha']
            or binding['maximum_requests']!=8 or binding['maximum_parallel_worlds']!=2
            or not binding['activated_epoch']<=now<spec['deadline_epoch']<=binding['deadline_epoch']
            or binding['deadline_epoch']-binding['activated_epoch']>7200
            or str(root)not in binding['native_roots'] or lease['status']!='ACTIVE'
            or lease['binding_sha256']!=spec[KEY]['binding']['sha256']
            or lease['owner']!=spec['owner'] or lease['scope']!=SCOPE
            or (group/'STOP.json').exists() or (Path(spec['campaign_root'])/'STOP.json').exists()
            or any(spec.get(k)for k in ('pilot1_binding','night1_binding','next1_binding','next2_binding'))):
        raise PermissionError('FRESH_FINITE_PD4_SITE_OWNER_REQUIRED')
    registry=json.loads(Path(spec['registry']).read_bytes())
    for r in (group,root):
        entry=next((v for v in registry['entries']if v.get('path')==str(r)),None)
        if not entry or any(entry.get(k)!=v for k,v in dict(owner=spec['owner'],scope=SCOPE,source_sha=spec['source_sha']).items()):
            raise PermissionError('ACTUAL_PD4_REGISTRY_REQUIRED')
    for name,pin in spec['frozen_files'].items():
        if digest(name)!=pin:raise PermissionError('UNCHANGED_FROZEN_RUNTIME_REQUIRED:'+name)
    if digest(spec['controller_binary'])!=CONTROLLER:raise PermissionError('ORIGINAL_CONTROLLER_BYTES_REQUIRED')
    for k in ('native_binary','solver_binary','controller_binary','observer_library','outer_library','trace_library','xvfb_run'):
        if spec[k]not in spec['frozen_files']:raise PermissionError('ALL_LIVE_RUNTIME_INPUTS_PINNED')
    if digest(Path(spec['source_root'])/'scripts/pi0294/task_path_model.py')!='edfb96620f91b6e2e583168bb1e157e8b8a9b5dab884917985af3d68d8a82e93':
        raise PermissionError('ORIGINAL_CERTIFIER_UNCHANGED')
    return binding


@contextmanager
def admitted(spec_path,run_id,output):
    from .auto1_native import phase_limit,validate_parent_loader
    from .runtime import admitted_run
    from .corelab_world_owner_v3 import world_owner
    spec=json.loads(Path(spec_path).read_bytes());validate_scope(spec)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU_ONLY_NATIVE_DIAGNOSTIC')
    validate_parent_loader(spec['native_environment']['LD_LIBRARY_PATH'],os.environ.get('LD_LIBRARY_PATH'))
    prior=json.loads(Path(spec['previous_stage']).read_bytes())
    if digest(spec['previous_stage'])!=spec['previous_stage_sha256'] or phase_limit(spec['phase'],prior,spec)!=21703:
        raise PermissionError('UNCHANGED_ORIGINAL_COLD_START_AND_EXACT_ENDPOINT')
    from .pilot1_storage import local_path
    for k in ('source_root','registered_root','logs','tmp','cache','data'):local_path(spec,spec[k])
    with world_owner(spec,json.loads(Path(spec['registry']).read_bytes())):
        with admitted_run(spec_path,run_id,'wm02-auto1-target',output,physics=True):yield spec


def recipe_config(spec,episode,*,actual_phase,actual_kind,source_members):
    """Fresh runtime custody instead of unavailable historical build paths.

    This is an independent diagnostic recipe, never a claim that the old
    expired P1/build lineage has been restored. Dynamic checks stay shared.
    """
    from .stable_task_scene_v1 import pinned,prefix_episode
    from .task_learning_grasp_v1 import CALIBRATION_B,PHASE,KIND
    from .fusion_task_phase_initialization_contract import _source
    configuration(spec);binding=pinned(spec[KEY]['binding']);root=Path(spec['registered_root'])
    if (actual_phase!=PHASE or actual_kind!=KIND or spec['profile']!='wm02-auto1-target'
            or spec['domain']!='WM02_AUTO1_TARGET_COMMISSION_V1' or spec['physics']is not True
            or spec['WBC_changes']is not False or spec['physics_dt_s']!=.002
            or spec['wall_limit_s']!=3600 or spec['max_ticks']!=21703
            or spec['live_skill_start_step']!=CALIBRATION_B
            or binding['schema']!=VERSION+'_BINDING' or binding['scope']!=SCOPE
            or binding['source_sha']!=spec['source_sha'] or str(root)not in binding['native_roots']
            or spec['reference_binding_version']!='MEASURED_EEF_AND_BASE_POSE_HOLD_V2'
            or any(spec.get(k)for k in ('pilot1_binding','fusion_binding','night1_binding','next1_binding',
                'next2_binding','parallel_batch_spec','standby_shutdown_diagnostic','reference_readiness_probe','diagnostic_debugger'))):
        raise PermissionError('EXACT_PD4_FRESH_NATIVE_RECIPE_NOT_OLD_P1_GRANT')
    source=Path(spec['source_root']);files=spec['source_files']
    if not all(n in files and digest(source/n)==files[n]for n in source_members):
        raise PermissionError('ALL_SHARED_NATIVE_SOURCE_CHECKS_RETAINED')
    proof=native_scene_admission(spec)
    view=prefix_episode(pinned(dict(path=spec['episode_manifest'],sha256=spec['episode_manifest_sha256'])),spec)
    if episode is not None and episode!=view:raise PermissionError('EXACT_SAME_COLD_START_RECIPE')
    if spec['trajectory_trace_capacity']!=524288:raise PermissionError('UNCHANGED_TRACE_CAPACITY')
    for n,pin in spec['frozen_files'].items():
        if digest(n)!=pin:raise PermissionError('FROZEN_RUNTIME_CUSTODY_CHANGED:'+n)
    # Reuse the sealed successful native producer, with exact current runtime
    # bytes, not the disappeared historical build executable/expired grant.
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    row=next(r for r in panel['assignments']if r['native_run']==root.name)
    mpath=Path(spec[KEY]['frozen_N20_manifest']['path']);mpin=spec[KEY]['frozen_N20_manifest']['sha256']
    from ..sla_adapter.s16_offline_v7 import DATA_SHA
    if mpin!=DATA_SHA or digest(mpath)!=DATA_SHA:raise PermissionError('ORIGINAL_CLOSED_PRODUCER_CUSTODY')
    m=json.loads(mpath.read_bytes());ep=next(e for e in m['episodes']if e['mount']['source_spec']==row['original_spec'])
    old=pinned(row['original_spec']);native=Path(old['registered_root'])/'native-01/RESULT.json';v=json.loads(native.read_bytes())
    if (digest(native)!=ep['source_native_result_sha256'] or v['status']!='NATIVE_STAGE_PASSED'
            or v['actual_target_integrations']!=21703 or not v['shutdown_clean']
            or v['physics_stops'] or v['original_joint_or_warning_samples']
            or v['native_binary_sha256']!=digest(spec['native_binary'])
            or v['scene_sha256']!=proof['compiled']['scene_sha256']):
        raise PermissionError('SEALED_COMPLETE_NATIVE_PRODUCER_WITH_SAME_RUNTIME')
    catalog=pinned(dict(path=spec['scene_allowlist'],sha256=spec['scene_allowlist_sha256']))
    matches=[r for r in catalog['scenes']if r['scene_id']==spec['scene_id']]
    if (len(matches)!=1 or matches[0]['parent']!=spec['data_parent'] or matches[0]['split']!='TRAIN'
            or matches[0]['model_sha256']!=digest(spec['model_path'])
            or matches[0]['model_path']!=spec['model_path']
            or catalog['native_binary_sha256']!=digest(spec['native_binary'])):
        raise PermissionError('SAME_SCENE_AND_RUNTIME_ALLOWLIST')
    return view,_source(spec)
