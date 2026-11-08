"""Adjacent grades require actual qualified feedback and every rolling join."""
from pathlib import Path

from . import fusion_task_phase_xyz_feedback_continuation_v5 as recipe
from . import fusion_task_phase_xyz_feedback_v4 as first
from . import fusion_task_phase_xyz_feedback_contract_v4 as first_contract
from . import fusion_task_phase_initialization_contract as operational
from . import fusion_live_proposal_qualification_contract as common
from .fusion_task_phase_xyz_feedback_evidence import audit_completed_feedback

PHASE = recipe.PHASE
KIND = 'FUSION_TASK_PHASE_XYZ_FEEDBACK_CONTINUATION_V5'
FIRST_PREDECESSOR_SOURCE = '59db1a50e006417eab16ae2e828116b5d7775448'
FIRST_PREDECESSOR_SPEC = '1688fa86537129306fdd74051f69a5ca095b4ba0189472abf8d79410d264f2dc'
FIRST_PREDECESSOR = '01b6e25762b86c809bdb4cd679bc38de31549212f3ea76e52cfa598207b98a61'
SOURCE_MEMBERS = tuple(dict.fromkeys((*first_contract.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_task_phase_xyz_feedback_evidence.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_xyz_feedback_continuation_v5.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_xyz_feedback_continuation_contract_v5.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_xyz_feedback_continuation_native_v5.py',
    'scripts/wm02/fusion_task_phase_xyz_feedback_continuation_check_v5.py')))


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_XYZ_FEEDBACK_CONTINUATION_NATIVE_V5_' + reason)


def episode_config(spec):
    episode = recipe.episode_contract(common._json(spec['episode_manifest'],spec['episode_manifest_sha256']))
    return operational.recipe_config(spec,episode,actual_phase=PHASE,actual_kind=KIND,
                                     source_members=SOURCE_MEMBERS)


def episode_limit(spec, previous):
    episode,_ = episode_config(spec)
    path = Path(spec['previous_stage'])
    actual = common._json(path,spec['previous_stage_sha256'])
    need(actual == previous,'EXACT_PINNED_PREDECESSOR_RESULT')
    prior_spec,prior_pin = common._closed(path.parent.parent)
    report = actual.get('reference_qualification') or {}
    prior = report.get('episode') or {}
    need(common._json(prior_spec['episode_manifest'],prior_spec['episode_manifest_sha256']) == prior,
         'PREDECESSOR_SPEC_EPISODE_REPORT_JOIN')
    index = recipe.SCALES.index(episode['geometry_scale'])
    if index == 0:
        need(spec['previous_stage_sha256'] == FIRST_PREDECESSOR and prior_pin == FIRST_PREDECESSOR_SPEC
             and prior_spec.get('source_sha') == FIRST_PREDECESSOR_SOURCE
             and actual.get('phase') == prior_spec.get('phase') == first.PHASE
             and prior_spec.get('kind') == first_contract.KIND,
             'FIRST_POINT25_EXACT_QUALIFIED_CLOSED_V4_POINT1_NOT_DEFERRED_V3')
        first.episode_contract(prior)
        need(prior['geometry_scale'] == .1,'EXACT_FIRST_POINT1_FEEDBACK_CONDITION')
        predecessor_recipe = first
    else:
        recipe.episode_contract(prior)
        need(actual.get('phase') == prior_spec.get('phase') == PHASE
             and prior_spec.get('kind') == KIND
             and prior['geometry_scale'] == recipe.SCALES[index-1]
             and prior_spec.get('source_sha') == spec.get('source_sha')
             and prior_spec.get('source_root') == spec.get('source_root')
             and all(prior_spec.get('source_files',{}).get(name) == spec['source_files'][name]
                     for name in SOURCE_MEMBERS),
             'IMMEDIATE_SAME_V5_SOURCE_AND_GRADE_ONLY')
        predecessor_recipe = recipe
    need(prior['source_observation_sha256'] == episode['source_observation_sha256']
         and prior['frame_map_sha256'] == episode['frame_map_sha256']
         and prior['scene_sha256'] == episode['scene_sha256'], 'UNCHANGED_SOURCE_FRAME_AND_NATIVE_SCENE')
    evidence = audit_completed_feedback(prior_spec,actual,recipe=predecessor_recipe)
    need(evidence['all_nine_non_full_grade_predicates_passed'] is True,
         'ALL_NINE_ACTUAL_MEASUREMENT_PREDICATES_REQUIRED_BEFORE_AMPLITUDE_ADVANCE')
    return recipe.endpoint(episode)
