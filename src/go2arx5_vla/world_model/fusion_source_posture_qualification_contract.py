"""Known source-observed posture commissioning, not VLA/Teacher supervision.

Query02 supplies only the frozen observation anchor and target frame map. The
40 identical raw goal rows are generated from source observation 71558, as a
declared CALIBRATION_INTENT. Original AUTO1/P1 and native safety remain required.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import unseal, validate_observation
from . import fusion_live_proposal_qualification_contract as common
from . import fusion_trajectory_qualification_contract as geometry
from .fusion_trajectory_adapter import TrajectoryAdaptation
from .fusion_paused_query_archive import read_archive
from .storage import digest

PHASE = 'collect-target-fusion-source-posture-qualification-v1'
SCHEMA = 'WM02_FUSION_SOURCE_POSTURE_TRAJECTORY_EPISODE_V1'
RECIPE = common.RECIPE
PURPOSE = 'VLA_INPUT_SOURCE_POSTURE_ALIGNMENT_DIAGNOSTIC'
ROLE = 'NON_VLA_CALIBRATION'
SCALES = (.03, .1, .25, .5, 1.)
INIT_STEP, DECISION_STEP, TAIL_ATOMS = 2503, 3703, 8
SCENE_ID, PARENT = common.SCENE_ID, common.PARENT
PINS = dict(common.PINS)
FIRST_PREDECESSOR = common.FIRST_PREDECESSOR
FIRST_PREDECESSOR_SPEC = common.FIRST_PREDECESSOR_SPEC
FIRST_PREDECESSOR_SOURCE = common.FIRST_PREDECESSOR_SOURCE
TRACE_BUILD = common.TRACE_BUILD
POSTURE_OBSERVATION = '71558ff5bcbc03e83eef9fbe79c868e75a2fece5c712f708682c8f23a090a594'
POSTURE_BODY = '2d052b104e2a780e6f0cadba7797f64396c5c2d883d3da2c8775003ee9b58af2'
SOURCE_MEMBERS = tuple(dict.fromkeys((*common.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_source_posture_qualification_contract.py',
    'src/go2arx5_vla/world_model/fusion_source_posture_qualification_native.py',
    'scripts/wm02/fusion_source_posture_qualification.py')))


def need(value, reason):
    if not value:
        raise ValueError('SOURCE_POSTURE_QUALIFICATION_'+reason)


def _json(path, pin=None):
    path=Path(path);raw=path.read_bytes()
    need(not path.is_symlink() and (pin is None or hashlib.sha256(raw).hexdigest()==pin),
         'EXACT_JSON_FILE_BYTES:'+path.name)
    return json.loads(raw)


def _fixed_episode():
    return dict(schema=SCHEMA,phase=PHASE,purpose=PURPOSE,experiment_role=ROLE,split='TRAIN',
        reference_rule=RECIPE,reference_anchor_mode='INIT_TCP_2503_REFERENCE_V1',
        prehold_atoms=40,screen_atom=48,screen=geometry.SCREEN,tail_hold_atoms=TAIL_ATOMS,
        desired_outcome_provenance='CALIBRATION_INTENT',physics_dt_s=.002,
        base_program='MEASURED_INIT_BASE_POSE_HOLD',time_stretch_factor=2,
        query_pins=dict(PINS),posture_source_observation_sha256=POSTURE_OBSERVATION,
        calibration_goal='REPEATED_SOURCE_OBSERVED_TCP_AND_GRIPPER_POSE',raw_generated_H40=True,
        source_observed_pose_is_expert_action_label=False,VLA_action_installations=0,
        full_task_success_claim=False,zero_shot_claim=False,online_closed_loop_claim=False,
        scene_id=SCENE_ID,parent=PARENT,GR00T_updates=0,WM_updates=0,Actor_updates=0)


def episode_contract(episode):
    exact=_fixed_episode()
    need(isinstance(episode,dict) and set(episode)==set(exact)|{'run','geometry_scale','planned_end_atom'}
         and all(episode.get(k)==v for k,v in exact.items()),'EXACT_CALIBRATION_SOURCE_POSTURE_EPISODE')
    need(type(episode['time_stretch_factor']) is int
         and type(episode['geometry_scale']) in (int,float) and episode['geometry_scale'] in SCALES
         and isinstance(episode['run'],str) and Path(episode['run']).is_absolute(),
         'PREDECLARED_GRADED_SOURCE_POSTURE_COMMISSION')
    endpoint(episode)
    return episode


def endpoint(episode):
    n=episode['planned_end_atom']
    need(type(n) is int and n>=40,'FULL40_INTEGER_CALIBRATION_CLOCK')
    end=DECISION_STEP+25*(n+TAIL_ATOMS)
    need(end<=33000,'FINITE_SOURCE_POSTURE_ENDPOINT_AT_MOST_33000')
    return end


def load_query(spec):
    """Read query02 anchor and a separately pinned source posture, no new draw."""
    need(spec.get('source_posture_query_pins')==PINS,'EXACT_14_QUERY02_ANCHOR_PINS')
    archive=read_archive(Path(spec['source_posture_native_spec']),Path(spec['source_posture_gpu_spec']),pins=dict(PINS))
    need(archive.get('status')=='PAIR_COMPLETED_AND_PAUSE_REVALIDATED'
         and (archive.get('paired_return')or{}).get('established') is True
         and archive.get('external_pins')==PINS and archive.get('historical_read') is True
         and archive.get('current_native_pause_checked') is False
         and archive.get('native_execution_authorized') is False,'GENUINE_ARCHIVED_QUERY02_ANCHOR_NOT_A_LIVE_GRANT')
    body=unseal(archive['observation']);view=unseal(body['original_shadow_ABI'])['source_policy_view']
    need(body['world']['scene_sha256']==digest(spec['model_path']) and body['task_sha256']==PINS['task_sha256'],
         'SAME_T06_QUERY_TASK_AND_SCENE')
    need(spec.get('posture_source_observation_sha256')==POSTURE_OBSERVATION,'FIXED_ORIGINAL_71558_OBSERVATION')
    source_record=_json(spec['posture_source_observation'],POSTURE_OBSERVATION)
    source=unseal(source_record)
    need(source_record.get('sha256')==POSTURE_BODY,'ORIGINAL_SOURCE_OBSERVATION_BODY_SEAL')
    payload=source['payload'];validate_observation(payload)
    need(payload['frame']==archive['payload']['frame'] and payload['tcp']==archive['payload']['tcp'],
         'EXPLICIT_SAME_SOURCE_POLICY_FRAME_AND_TCP_NOT_CAMERA_EQUIVALENCE')
    goal=np.asarray(payload['state_eef10'],dtype=np.float64)
    goal_tcp=matrix10(goal)
    need(goal.shape==(10,) and np.isfinite(goal).all() and 0<=goal[9]<=1,
         'ACTUAL_SOURCE_OBSERVED_POSE_AND_MECHANICAL_GRIPPER_FRACTION')
    repeated=np.repeat(goal[None],40,axis=0)
    repeated=np.frombuffer(repeated.tobytes(),dtype=repeated.dtype).reshape(40,10)
    return dict(raw=repeated,raw_generated_H40=True,raw_is_generated_calibration_reference=True,
        query_tcp=matrix10(archive['payload']['state_eef10']),frame_map=np.asarray(view['T_source_from_target'],float),
        archive=archive,task=body['task'],task_sha256=PINS['task_sha256'],
        source_observation=source_record,source_observation_payload=payload,source_goal_TCP_source=goal_tcp,
        source_observation_TCP_goal=goal_tcp,
        source_observation_state_goal=goal.copy(),
        provenance=dict(query_pins=dict(PINS),archive_status=archive['status'],historical_read=True,
            original_live_query_completed=True,current_native_pause_checked=False,native_execution_authorized=False,
            posture_source_observation_sha256=POSTURE_OBSERVATION,posture_source_body_sha256=POSTURE_BODY,
            calibration_goal='REPEATED_SOURCE_OBSERVED_TCP_AND_GRIPPER_POSE',raw_generated_H40=True,
            raw_is_generated_calibration_reference=True,
            raw_H40_origin='GENERATED_40_IDENTICAL_SOURCE_OBSERVATION_STATE_ROWS',
            query02_VLA_action_used_as_program=False,source_observed_pose_is_expert_action_label=False,
            source_outcomes_supplied=False,VLA_action_installations=0,desired_outcome_provenance='CALIBRATION_INTENT',
            execution_condition=PURPOSE,task_intent_sufficient_for_task_success=False,
            camera_equivalence=False,target_task_success=False,WM_updates=0))


def make_program(inputs,episode,initial_tcp,decision_tcp,base):
    episode_contract(episode)
    goal=np.asarray(inputs['source_observation_state_goal'],float)
    need(inputs.get('raw_generated_H40') is True and goal.shape==(10,)
         and np.asarray(inputs['raw']).shape==(40,10)
         and np.array_equal(inputs['raw'],np.repeat(goal[None],40,axis=0)),
         'ONLY_GENERATED_SOURCE_POSTURE_GOAL_NOT_ARCHIVED_VLA_ACTION')
    program=TrajectoryAdaptation(inputs['raw'],inputs['query_tcp'],inputs['frame_map'],decision_tcp,initial_tcp,
        base,episode['geometry_scale'],time_stretch_factor=2)
    need(program.end_atom==episode['planned_end_atom'],'ACTUAL_RETIME_LENGTH_EQUALS_PREDECLARED_END')
    return program


def episode_config(spec):
    need(spec.get('phase')==PHASE and spec.get('profile')=='wm02-auto1-target'
         and spec.get('domain')=='WM02_AUTO1_TARGET_COMMISSION_V1'
         and spec.get('site_profile')=='4080-simulation' and spec.get('physics') is True
         and spec.get('scope')=='WM02_PILOT1_SIM_ONLY' and bool(spec.get('pilot1_binding')),
         'ORIGINAL_AUTO1_P1_NATIVE_BACKEND_NOT_A_GRANT')
    need(not any(spec.get(k)for k in('fusion_binding','night1_binding','next1_binding','next2_binding',
        'parallel_batch_spec','standby_shutdown_diagnostic','reference_readiness_probe','diagnostic_debugger')),
         'SINGLE_WORLD_NO_OLD_FUSION_OR_SCOPE_BYPASS')
    need(spec.get('WBC_changes') is False and spec.get('physics_dt_s')==.002
         and type(spec.get('wall_limit_s')) in (int,float) and 0<spec['wall_limit_s']<=1200,
         'FROZEN_WBC_2MS_FINITE_WALL')
    source,files=Path(spec['source_root']),spec.get('source_files',{})
    need(all(name in files and digest(source/name)==files[name]for name in SOURCE_MEMBERS),
         'NEW_SOURCE_POSTURE_CAPSULE_AND_SHARED_ORIGINAL_HELPERS')
    episode=episode_contract(_json(spec['episode_manifest'],spec['episode_manifest_sha256']))
    need(episode['run']==spec['registered_root'] and type(spec.get('max_ticks')) is int
         and spec['max_ticks']==endpoint(episode)
         and spec.get('reference_binding_version')=='MEASURED_EEF_AND_BASE_POSE_HOLD_V2',
         'EXACT_REGISTERED_CLOCK_AND_INIT_REFERENCE_BINDING')
    catalog=_json(spec['scene_allowlist'],spec['scene_allowlist_sha256'])
    scenes=[s for s in catalog['scenes']if s['scene_id']==SCENE_ID]
    need(len(scenes)==1 and scenes[0]['parent']==PARENT and scenes[0]['split']=='TRAIN'
         and scenes[0]['model_path']==spec['model_path'] and scenes[0]['model_sha256']==digest(spec['model_path'])
         and scenes[0].get('static_check')=='PASSED_NO_INTEGRATION'
         and catalog['native_binary_sha256']==digest(spec['native_binary']),'ORIGINAL_STATIC_T06_AND_BINARY')
    need(spec.get('trajectory_trace_capacity')==524288 and type(spec.get('trajectory_trace_capacity')) is int
         and spec.get('trajectory_trace_capacity_build_sha256')==TRACE_BUILD,'FROZEN_BUILD03_TRACE524_RECEIPT')
    geometry.validate_trace_capacity_failure(spec)
    build_path=Path(spec['trajectory_trace_capacity_build']);build=_json(build_path,TRACE_BUILD)
    need(build.get('status')=='TRACE_ONLY_524288_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
         and build.get('trace_capacity')==524288 and build.get('new_target_integrations')==0
         and build.get('trace_library_sha256')==digest(spec['trace_library'])
         and build.get('native_binary_sha256')==digest(spec['native_binary'])
         and build.get('observer_sha256')==digest(spec['observer_library'])
         and all(build.get('source_files',{}).get(name)==files[name]for name in(
             'src/go2arx5_vla/target_mujoco/init_trace.cpp','src/go2arx5_vla/target_mujoco/init_trace.hpp')),
         'ACTUAL_TRACE_ONLY_BUILD03_UNCHANGED_NATIVE_BYTES')
    common._closed(build_path.parent.parent)
    inputs=load_query(spec)
    mapped=np.linalg.inv(inputs['frame_map'])@inputs['query_tcp']
    make_program(inputs,episode,mapped,mapped,[0.,0.,0.,0.,0.,0.,1.])
    return episode,inputs


def episode_limit(spec,previous):
    episode,_=episode_config(spec)
    path=Path(spec['previous_stage'])
    need(_json(path,spec['previous_stage_sha256'])==previous,'ACTUAL_PINNED_PREDECESSOR_RESULT')
    prior_spec,prior_pin=common._closed(path.parent.parent)
    report=previous.get('reference_qualification')or{};prior=report.get('episode')or{}
    need(_json(prior_spec['episode_manifest'],prior_spec['episode_manifest_sha256'])==prior,
         'PREDECESSOR_SPEC_EPISODE_REPORT_JOIN')
    index=SCALES.index(episode['geometry_scale'])
    if index==0:
        need(spec['previous_stage_sha256']==FIRST_PREDECESSOR and prior_pin==FIRST_PREDECESSOR_SPEC
             and prior_spec.get('source_sha')==FIRST_PREDECESSOR_SOURCE
             and previous.get('phase')==prior_spec.get('phase')==geometry.SLOW_PHASE,
             'FIRST_SOURCE_POSTURE_GRADE_REQUIRES_EXACT_67_FULL1_BOOTSTRAP')
        geometry.episode_contract(prior)
        need(prior['geometry_scale']==1. and prior['time_stretch_factor']==2,'COMPLETE_OLD_FULL1_BOOTSTRAP_ONLY')
        expected=geometry.endpoint(prior)
    else:
        episode_contract(prior)
        need(previous.get('phase')==prior_spec.get('phase')==PHASE
             and prior['geometry_scale']==SCALES[index-1]
             and prior['query_pins']==episode['query_pins']
             and prior['posture_source_observation_sha256']==episode['posture_source_observation_sha256'],
             'IMMEDIATE_SAME_SOURCE_POSTURE_FAMILY_PREVIOUS_GRADE')
        expected=endpoint(prior)
    need(prior['scene_id']==SCENE_ID and prior['parent']==PARENT,'SAME_T06_PHYSICAL_PARENT')
    common._complete_previous(spec,previous,prior_spec,report,expected)
    return endpoint(episode)
