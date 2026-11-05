"""PILOT2 supported contact contracts; no controller or physical-law changes."""
from pathlib import Path
import json
from .storage import digest

PHASE='collect-target-pilot2-supported'
PLAN='6d82fa23c892f167001abe743dd5eadfca6fd777'
MODES=('CLOSE_KEEP','CLOSE_RELEASE','OPEN_ONLY')
GOALS=('SUPPORTED_GRASP_HOLD','RELEASE_FROM_SUPPORTED_GRASP')


def stage_for(atom,episode):
    close=episode['close_start_atom']
    if atom<close:return 'OPEN_STEADY'
    if episode['command_program']=='OPEN_ONLY':return 'OPEN_HOLD'
    if atom<close+32:return 'CLOSE'
    if episode['command_program']=='CLOSE_KEEP'or atom<close+56:return 'SUPPORTED_HOLD'
    return 'RELEASE'


def contract(episode):
    expected=dict(schema='WM02_PILOT2_SUPPORTED_EPISODE_V1',phase=PHASE,
        reference_rule='FIXED_EEF_HOLD40_PILOT2_PREDECLARED_GRIPPER_PROGRAM_V1',
        task_atoms=168,prehold_atoms=40,screen_atom=48,close_logical_opening=.4,
        required_supported_grasp_native_ticks=500,required_released_native_ticks=100,
        minimum_open_finger_q_m=.040,contact_force_threshold_N=.05,EEF_offset_xyz_m=[0.,0.,0.])
    if (any(episode.get(k)!=v for k,v in expected.items())
            or type(episode.get('close_start_atom'))is not int or episode.get('close_start_atom')not in (56,64,72)
            or episode.get('command_program')not in MODES or episode.get('task_goal')not in GOALS
            or episode.get('purpose')not in ('SUPPORTED_CONTACT_COLLECTION','SUPPORTED_POLICY_EVALUATION')
            or episode.get('split')not in ('TRAIN','DEV','TEST')):
        raise PermissionError('EXACT_PREDECLARED_SUPPORTED_CONTACT_PROGRAM_REQUIRED')
    return episode


def validate(spec,previous):
    path=Path(spec['episode_manifest']);episode=json.loads(path.read_text())
    if episode.get('schema')=='WM02_PILOT2_SUPPORTED_POLICY_EPISODE_V1':
        from .pilot2_supported_policy_runtime import contract as policy_contract
        policy_contract(episode)
    else:contract(episode)
    parallel=bool(spec.get('parallel_batch_spec'))
    if parallel:
        if (episode.get('schema')!='WM02_PILOT2_SUPPORTED_POLICY_EPISODE_V1'
                or spec.get('supported_policy_parallel_schema')!='WM02_PILOT2_TWO_SUPPORTED_POLICY_BATCH_V1'):
            raise PermissionError('NEW_CONTACT_SINGLE_FIRST_OR_MATCHING_QUALIFIED_POLICY_BATCH_REQUIRED')
        from .pilot2_supported_policy_runtime import validate_single_qualification
        validate_single_qualification(spec,episode)
    if (digest(path)!=spec['episode_manifest_sha256']or episode['run']!=spec['registered_root']
            or spec.get('pilot2_plan_sha')!=PLAN or not spec.get('pilot1_binding')):
        raise PermissionError('FRESH_SINGLE_PILOT2_CONTACT_SCOPE_AND_PROGRAM_REQUIRED')
    assets=json.loads(Path(spec['scene_allowlist']).read_text());scenes=[s for s in assets['scenes']if s['scene_id']==episode['scene_id']]
    if (digest(spec['scene_allowlist'])!=spec['scene_allowlist_sha256']or len(scenes)!=1
            or scenes[0]['parent']!=episode['parent']or scenes[0]['split']!=episode['split']
            or scenes[0]['model_path']!=spec['model_path']or scenes[0]['model_sha256']!=digest(spec['model_path'])
            or assets['native_binary_sha256']!=digest(spec['native_binary'])):
        raise PermissionError('FROZEN_CONTACT_PARENT_SPLIT_AND_ORIGINAL_NATIVE_REQUIRED')
    p=Path(spec['scene_qualification']);native=json.loads(p.read_text());prefix=json.loads((p.parent/'STEADY_PREFIX_RESULT.private.json').read_text())
    if (digest(p)!=spec['scene_qualification_sha256']or native.get('status')!='NATIVE_STAGE_PASSED'
            or not native.get('shutdown_clean')or native.get('physics_stops')or not prefix.get('qualified')
            or native.get('scene_sha256')!=digest(spec['model_path'])
            or native.get('native_binary_sha256')!=digest(spec['native_binary'])
            or not previous or previous.get('status')!='NATIVE_STAGE_PASSED'or not previous.get('shutdown_clean')
            or previous.get('native_binary_sha256')!=digest(spec['native_binary'])):
        raise PermissionError('ACTUAL_NEW_SCENE_PREFIX_QUALIFICATION_REQUIRED')
    if episode.get('schema')=='WM02_PILOT2_SUPPORTED_POLICY_EPISODE_V1':
        from .pilot2_supported_policy_runtime import validate_qualifications
        validate_qualifications(spec,episode)
    return 6703


def score(events,episode):
    contract(episode)
    if [e['native_step']for e in events]!=list(range(2504,6704)):
        raise ValueError('EVERY_ACTUAL_CONTACT_TICK_REQUIRED')
    longest=hold=terminal_hold=released=0
    for e in events:
        supported=e['native_grasp']and e['table_support']
        hold=hold+1 if supported and e['stage']=='SUPPORTED_HOLD'else 0
        longest=max(longest,hold)
        terminal_hold=terminal_hold+1 if supported and e['stage']=='SUPPORTED_HOLD'else 0
        open_supported=not e['contact_positive']and min(e['finger_q'])>=.040 and e['table_support']
        released=released+1 if open_supported and e['stage']=='RELEASE'else 0
    held=terminal_hold>=500
    # An already-open cube is not evidence of releasing a supported grasp.
    release=longest>=500 and released>=100
    return dict(schema='WM02_PILOT2_SUPPORTED_GOAL_SCORE_V1',task_goal=episode['task_goal'],
        command_program=episode['command_program'],close_start_atom=episode['close_start_atom'],
        qualified_supported_grasp_hold=longest>=500,
        longest_continuous_supported_grasp_native_ticks=longest,
        longest_continuous_supported_grasp_s=longest*.002,
        terminal_supported_grasp_native_ticks=terminal_hold,
        terminal_released_native_ticks=released,terminal_released_s=released*.002,
        qualified_release=release,qualified_supported_close_hold_release=release,
        goal_success=held if episode['task_goal']==GOALS[0]else release,
        full_manipulation_success=False,original_native_grasp_predicate=True)
