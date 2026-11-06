"""One full known-posture program, then one query in that same paused world.

The physical ancestor is the unchanged V1 calibration recipe. This V3 keeps its
real spec/episode identity; no returned VLA action is certified or installed.
Source-pose closeness is an engineering diagnostic, never a native safety gate
or evidence of camera equivalence, expert actions, or task success.
"""
from pathlib import Path
import time

import numpy as np

from . import fusion_source_posture_qualification_contract as posture
from . import fusion_live_proposal_qualification_contract as common
from .fusion_observation import TargetShadowBinding, transform_hash
from .fusion_short_skill import document_hash, score_contract, task_document
from .storage import digest, write_json

PHASE = 'collect-target-fusion-source-posture-query-v3'
SCHEMA = 'WM02_FUSION_SOURCE_POSTURE_QUERY_EPISODE_V3'
REPORT_SCHEMA = 'WM02_NATIVE_SOURCE_POSTURE_PAUSED_QUERY_V3'
PAIR_SCHEMA = 'WM02_FUSION_SOURCE_POSTURE_PAUSED_QUERY_PAIR_V2'
PAIR_MODE = 'PRE_REGISTERED_SOURCE_POSTURE_SNAPSHOT_V2'
CAMERA_RECIPE = 'ORIGINAL_TARGET_NATIVE_DUAL_CAMERA_V1'
SOURCE_CAMERA_RECIPE = 'SIM_EXTERNAL_WORLD_SOURCE_RIG'
SOURCE_CAMERA_IDENTITY = '0fe9090e469e7b864213b23c3b98b82bd2bd0162a2c4afc4cea7820536091858'
SOURCE_CAMERA_MEMBERS = ('src/go2arx5_vla/world_model/fusion_source_camera_capture.py',
                         'src/go2arx5_vla/world_model/fusion_source_camera_rig.py')
SOURCE_MEMBERS = tuple(dict.fromkeys((*posture.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_source_posture_paused_query_v3.py',
    'src/go2arx5_vla/world_model/fusion_paused_query_bridge.py',
    'src/go2arx5_vla/world_model/fusion_paused_inference.py',
    'src/go2arx5_vla/world_model/fusion_paused_observation.py',
    'scripts/wm02/infer_fusion_paused_vla.py',
    'scripts/wm02/fusion_source_posture_paused_query_check.py')))
PAIR_EXTRA_KEYS = {'producer_phase', 'physical_recipe_phase', 'source_posture_episode_sha256',
                   'posture_source_observation_sha256', 'geometry_scale', 'time_stretch_factor',
                   'returned_VLA_action_installations','camera_recipe','source_camera_identity_sha256'}


def need(value, reason):
    if not value:
        raise ValueError('SOURCE_POSTURE_QUERY_V3_'+reason)


def _fixed_episode(camera_recipe=CAMERA_RECIPE):
    need(camera_recipe in (CAMERA_RECIPE,SOURCE_CAMERA_RECIPE),'KNOWN_VERSIONED_CAMERA_RECIPE')
    return dict(posture._fixed_episode(), schema=SCHEMA, phase=PHASE,
        physical_recipe_phase=posture.PHASE, physical_recipe_schema=posture.SCHEMA,
        physical_recipe_rule=posture.RECIPE, max_queries=1, snapshot_binding_mode=PAIR_MODE,
        capture_camera_recipe=camera_recipe,
        source_camera_identity_sha256=(SOURCE_CAMERA_IDENTITY if camera_recipe==SOURCE_CAMERA_RECIPE else None),
        query_after_operator_cancel_and_trace_seal=True, returned_VLA_action_installations=0)


def episode_contract(episode):
    need(isinstance(episode,dict),'EXACT_ACTUAL_V3_EPISODE')
    fixed = _fixed_episode(episode.get('capture_camera_recipe'))
    need(isinstance(episode, dict) and set(episode) == set(fixed)|{'run','geometry_scale','planned_end_atom'}
         and all(episode.get(k) == v for k,v in fixed.items()), 'EXACT_ACTUAL_V3_EPISODE')
    need(type(episode['geometry_scale']) in (int,float) and episode['geometry_scale'] == 1.
         and type(episode['time_stretch_factor']) is int and episode['time_stretch_factor'] == 2
         and isinstance(episode['run'],str) and Path(episode['run']).is_absolute(),
         'ONLY_FULL1_SOURCE_POSTURE_TIME2')
    endpoint(episode)
    return episode


def endpoint(episode):
    return posture.endpoint(episode)


def parent_recipe_view(episode):
    """Explicit in-memory ancestor view, never an execution spec or manifest."""
    episode_contract(episode)
    keys = set(posture._fixed_episode())|{'run','geometry_scale','planned_end_atom'}
    view = {key:episode[key] for key in keys}
    view.update(phase=posture.PHASE,schema=posture.SCHEMA)
    return posture.episode_contract(view)


def make_program(inputs,episode,initial_tcp,decision_tcp,base):
    return posture.make_program(inputs,parent_recipe_view(episode),initial_tcp,decision_tcp,base)


def _previous(spec,episode,previous=None):
    path = Path(spec['previous_stage'])
    actual = posture._json(path,spec['previous_stage_sha256'])
    need(previous is None or previous == actual,'EXACT_PINNED_PREDECESSOR_RESULT')
    prior_spec,_ = common._closed(path.parent.parent)
    report = actual.get('reference_qualification') or {}
    prior = report.get('episode') or {}
    need(posture._json(prior_spec['episode_manifest'],prior_spec['episode_manifest_sha256']) == prior,
         'PREDECESSOR_SPEC_MANIFEST_REPORT_JOIN')
    posture.episode_contract(prior)
    need(actual.get('phase') == prior_spec.get('phase') == posture.PHASE
         and prior['geometry_scale'] == .5 and prior['time_stretch_factor'] == 2
         and prior['query_pins'] == episode['query_pins']
         and prior['posture_source_observation_sha256'] == episode['posture_source_observation_sha256']
         and prior['scene_id'] == episode['scene_id'] and prior['parent'] == episode['parent'],
         'IMMEDIATE_COMPLETE_V1_SOURCE_POSTURE_HALF_SCALE_REQUIRED')
    common._complete_previous(spec,actual,prior_spec,report,posture.endpoint(prior))


def recipe_config(spec):
    """All immutable recipe checks, without recursive live-pair preflight."""
    need(spec.get('phase') == PHASE,'ACTUAL_V3_NATIVE_SPEC_REQUIRED')
    episode = episode_contract(posture._json(spec['episode_manifest'],spec['episode_manifest_sha256']))
    need(spec.get('posture_query_camera_recipe') == episode['capture_camera_recipe']
         and not any(spec.get(key) for key in ('camera_override','paused_camera_override',
             'posture_camera_override','source_camera_rig')),
         'EXPLICIT_CAMERA_RECIPE_NO_UNQUALIFIED_OVERRIDE')
    # Keep the actual spec untouched while explicitly reusing only the physical
    # ancestor recipe. The pinned on-disk episode above remains the real V3.
    _,inputs = posture.recipe_config(spec,parent_recipe_view(episode),actual_phase=PHASE)
    source,files = Path(spec['source_root']),spec.get('source_files',{})
    need(all(name in files and digest(source/name) == files[name] for name in SOURCE_MEMBERS),
         'EXACT_V3_AND_ORIGINAL_SHARED_SOURCE_BYTES')
    _previous(spec,episode)
    task = posture._json(spec['paused_target_task'],spec['paused_target_task_file_sha256'])
    contract = posture._json(spec['paused_score_contract'],spec['paused_score_contract_file_sha256'])
    need(contract == score_contract() and task == task_document(digest(spec['model_path']))
         and document_hash(task) == spec['paused_target_task_sha256']
         and task['score_contract_sha256'] == document_hash(contract), 'ORIGINAL_TARGET_TASK_AND_SCORE')
    need(transform_hash(spec['paused_source_from_target']) == spec['paused_frame_map_sha256']
         == posture.PINS['frame_map_sha256']
         and np.array_equal(np.asarray(spec['paused_source_from_target']),inputs['frame_map']),
         'UNCHANGED_EXPLICIT_RIGID_FRAME_NOT_CAMERA_EQUIVALENCE')
    if episode['capture_camera_recipe']==SOURCE_CAMERA_RECIPE:
        need(all(name in files and digest(source/name)==files[name] for name in SOURCE_CAMERA_MEMBERS),
             'EXACT_SOURCE_RIG_AND_CAPTURE_IMPLEMENTATION_BYTES')
        from .fusion_source_camera_capture import SourceCameraCapture
        path=Path(spec['source_camera_identity'])
        need(path.is_absolute() and not path.is_symlink()
             and path.resolve()==path and spec.get('source_camera_identity_sha256')==SOURCE_CAMERA_IDENTITY,
             'EXACT_PINNED_SOURCE_CAMERA_IDENTITY_PATH')
        raw=path.read_bytes()
        import hashlib
        need(hashlib.sha256(raw).hexdigest()==SOURCE_CAMERA_IDENTITY,'SOURCE_CAMERA_IDENTITY_BYTES')
        inputs=dict(inputs,capture_camera_recipe=SourceCameraCapture(raw,inputs['frame_map']))
    else:
        need(spec.get('source_camera_identity') is None and spec.get('source_camera_identity_sha256') is None,
             'ORIGINAL_CAMERA_HAS_NO_SILENT_SOURCE_RIG_OVERRIDE')
    return episode,inputs,task


def episode_config(spec):
    episode,inputs,task = recipe_config(spec)
    from .fusion_paused_query_bridge import preflight_pair
    preflight_pair(Path(spec['paused_query_gpu_spec']),spec)
    return episode,inputs,task


def episode_limit(spec,previous):
    episode,_,_ = episode_config(spec)
    _previous(spec,episode,previous)
    return endpoint(episode)


def validate_pair_spec(spec):
    """Strict V2 only: derive the pause from the actual pinned full recipe."""
    from . import fusion_paused_query_bridge as bridge
    pair = posture._json(spec['paused_query_pair'],spec['paused_query_pair_sha256'])
    need(pair.get('schema') == PAIR_SCHEMA and spec.get('snapshot_binding_mode') == PAIR_MODE,
         'EXPLICIT_POSTURE_PAIR_V2_SCHEMA_AND_MODE')
    producer = posture._json(spec['producer_native_spec'],spec['producer_native_spec_sha256'])
    episode,_,_ = recipe_config(producer)
    boundary = endpoint(episode)
    need(pair.get('producer_phase') == PHASE
         and type(pair.get('capture_index')) is int and pair['capture_index'] == 0
         and pair.get('physical_recipe_phase') == posture.PHASE
         and pair.get('camera_recipe') == episode['capture_camera_recipe']
         and pair.get('source_camera_identity_sha256') == episode['source_camera_identity_sha256']
         and pair.get('source_posture_episode_sha256') == producer['episode_manifest_sha256']
         and pair.get('posture_source_observation_sha256') == posture.POSTURE_OBSERVATION
         and type(pair.get('geometry_scale')) in (int,float) and pair['geometry_scale'] == 1.
         and type(pair.get('time_stretch_factor')) is int and pair['time_stretch_factor'] == 2
         and type(pair.get('returned_VLA_action_installations')) is int
         and pair['returned_VLA_action_installations'] == 0,
         'EXACT_FULL_SOURCE_POSTURE_BOUNDARY_RECIPE_NO_RETURNED_ACTION')
    return bridge._validate_pair_spec(spec,pair_schema=PAIR_SCHEMA,binding_mode=PAIR_MODE,
        boundary_step=boundary,pair_keys=bridge.KEYS|PAIR_EXTRA_KEYS)


def validate_capture_camera(pair,shadow):
    sensor=(shadow.get('target_observation') or {}).get('sensor_adapter')
    if pair['camera_recipe']==SOURCE_CAMERA_RECIPE:
        need(isinstance(sensor,dict) and sensor.get('schema')=='WM02_FUSION_SOURCE_CAMERA_CAPTURE_V1'
             and sensor.get('status')=='ACTUAL_TARGET_PACKET_CPU_RENDERED'
             and sensor.get('rig')==SOURCE_CAMERA_RECIPE
             and sensor.get('source_identity_audit_sha256')==pair['source_camera_identity_sha256']==SOURCE_CAMERA_IDENTITY
             and sensor.get('frame_map_sha256')==pair['frame_map_sha256'],
             'CAPTURE_MUST_RETAIN_EXACT_DECLARED_SOURCE_RIG_NOT_ORIGINAL_CAMERA')
    else:
        need(sensor is None,'ORIGINAL_CAMERA_PAIR_CANNOT_SILENTLY_USE_SOURCE_SENSOR')


def run(adapter):
    """Motion is complete before this wrapper asks for its one fresh query."""
    from .fusion_source_posture_qualification_native import run_posture_program
    from .fusion_paused_query_bridge import capture_and_wait_query
    episode,inputs,task = episode_config(adapter.spec)
    original_spec_hash = document_hash(adapter.spec)
    report = run_posture_program(adapter,episode,inputs,program_factory=make_program,
        report_context=dict(schema=REPORT_SCHEMA,actual_spec_phase=PHASE,
            physical_recipe_phase=posture.PHASE,physical_recipe_schema=posture.SCHEMA,
            physical_recipe_rule=posture.RECIPE,query_stage='NOT_YET_CALLED',
            capture_camera_recipe=episode['capture_camera_recipe'],
            source_camera_identity_sha256=episode['source_camera_identity_sha256']))
    end = endpoint(episode)
    path = adapter.output/'TRAJECTORY_QUALIFICATION.private.json'
    need(document_hash(adapter.spec) == original_spec_hash and adapter.observe()['step'] == end
         and adapter.trace_sealed is True and report['native_endpoint_step'] == end
         and report['all_40_knots_consumed'] is True
         and report['operator_cancel'].get('kind') == 'operator_cancel'
         and report['operator_cancel'].get('mode') == 'HOLD'
         and report['operator_cancel'].get('native_step') == end
         and report['actual_VLA_commands'] == report['VLA_commands'] == 0
         and 'source_posture_diagnostic' in report and posture._json(path) == report,
         'COMPLETE_SAVED_MEASURED_POSTURE_CANCEL_HOLD_TRACE_SEAL_BEFORE_QUERY')
    physical_pin = digest(path)
    binding = TargetShadowBinding(target_world_frame='MUJOCO_GO2_ARX5_WORLD',
        scene_sha256=digest(adapter.spec['model_path']),
        target_runtime_sha256=digest(Path(adapter.spec['registered_root'])/'EXECUTION_SPEC.json'),
        T_source_from_target=adapter.spec['paused_source_from_target'],
        expected_transform_sha256=adapter.spec['paused_frame_map_sha256'])
    start,cpu = time.monotonic(),time.process_time()
    try:
        camera_kwargs=({'camera_recipe':inputs['capture_camera_recipe']}
                       if episode['capture_camera_recipe']==SOURCE_CAMERA_RECIPE else {})
        query = capture_and_wait_query(adapter,binding=binding,task=task,
            task_sha256=document_hash(task),gpu_spec_path=Path(adapter.spec['paused_query_gpu_spec']),
            capture_index=0,**camera_kwargs)
        need(document_hash(adapter.spec) == original_spec_hash and adapter.observe()['step'] == end
             and digest(path) == physical_pin,'NO_SPEC_OR_PHYSICS_OR_PHYSICAL_REPORT_CHANGE_DURING_QUERY')
        complete = dict(report,query_stage='ONE_FROZEN_GPU_QUERY_RETURNED_AND_PAUSE_REVALIDATED',
            physical_report_path=str(path),physical_report_sha256=physical_pin,paused_query=query,
            actual_VLA_inferences=1,VLA_inferences=1,VLA_queries=1,VLA_action_installations=0,
            native_process_VLA_inferences=0,native_process_Processor_decodes=0,
            GPU_action_decode_contract='ONE_ORIGINAL_PROCESSOR_DECODE_INSIDE_RESIDENT_GENERATE',
            additional_action_decodes=0,
            query_wall_s=time.monotonic()-start,query_CPU_s=time.process_time()-cpu,
            camera_equivalence=False,full_policy_input_distribution_qualified=False,
            returned_action_is_live_reference=False,online_VLA=False,online_closed_loop=False,
            native_integrations_by_query=0,native_stage_RESULT_json_required=True)
        # The retained physical report's zero counters describe the native
        # process. The successful GPU invocation uses the original one-decode
        # primitive; do not relabel its contract as an instrumented GPU counter.
        complete.pop('original_Processor_decodes',None)
        complete.pop('Processor_decodes',None)
        write_json(adapter.output/'SOURCE_POSTURE_PAUSED_QUERY.private.json',complete)
        adapter.reference_qualification = complete
        return complete
    except BaseException as error:
        write_json(adapter.output/'SOURCE_POSTURE_QUERY_FAILED.private.json',dict(
            phase=PHASE,physical_report_sha256=physical_pin,failure=type(error).__name__+': '+str(error),
            retry_authorized=False,returned_VLA_action_installations=0,
            query_wall_s=time.monotonic()-start,query_CPU_s=time.process_time()-cpu))
        raise
