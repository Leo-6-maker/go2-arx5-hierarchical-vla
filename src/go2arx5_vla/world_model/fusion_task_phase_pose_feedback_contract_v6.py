"""Same-grade failed V5 repair, then only qualified adjacent V6 grades."""
from pathlib import Path

from . import fusion_task_phase_pose_feedback_v6 as recipe
from . import fusion_task_phase_xyz_feedback_continuation_v5 as previous_recipe
from . import fusion_task_phase_xyz_feedback_continuation_contract_v5 as previous_contract
from . import fusion_task_phase_initialization_contract as operational
from . import fusion_live_proposal_qualification_contract as common
from .fusion_task_phase_xyz_feedback_evidence import audit_completed_feedback

PHASE = recipe.PHASE
KIND = 'FUSION_TASK_PHASE_POSE_FEEDBACK_V6'
FIRST_PREDECESSOR_SOURCE = 'e5272f8fb90de2d910bb52c0575771b7a2125ee7'
FIRST_PREDECESSOR_SPEC = '90f5aa95552180dbe2c447608948643ca6e6988580521335310dab7d357184a4'
FIRST_PREDECESSOR = 'f6d6913a17490381c65946361cff5d8b6c47ca91c84ab5e9a2783a74b50cd023'
SOURCE_MEMBERS = tuple(dict.fromkeys((*previous_contract.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_task_phase_pose_feedback_v6.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_pose_feedback_contract_v6.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_pose_feedback_native_v6.py',
    'scripts/wm02/fusion_task_phase_pose_feedback_check_v6.py')))


def need(ok,reason):
    if not ok:
        raise ValueError('TASK_PHASE_POSE_FEEDBACK_NATIVE_V6_'+reason)


def episode_config(spec):
    episode = recipe.episode_contract(common._json(spec['episode_manifest'],spec['episode_manifest_sha256']))
    return operational.recipe_config(spec,episode,actual_phase=PHASE,actual_kind=KIND,
                                     source_members=SOURCE_MEMBERS)


def episode_limit(spec,previous):
    episode,_ = episode_config(spec)
    path = Path(spec['previous_stage'])
    actual = common._json(path,spec['previous_stage_sha256'])
    need(actual == previous,'EXACT_PINNED_PREDECESSOR_RESULT')
    prior_spec,prior_pin = common._closed(path.parent.parent)
    prior = (actual.get('reference_qualification') or {}).get('episode') or {}
    need(common._json(prior_spec['episode_manifest'],prior_spec['episode_manifest_sha256']) == prior,
         'PREDECESSOR_SPEC_EPISODE_REPORT_JOIN')
    index = recipe.SCALES.index(episode['geometry_scale'])
    if index == 0:
        need(spec['previous_stage_sha256'] == FIRST_PREDECESSOR and prior_pin == FIRST_PREDECESSOR_SPEC
             and prior_spec.get('source_sha') == FIRST_PREDECESSOR_SOURCE
             and actual.get('phase') == prior_spec.get('phase') == previous_recipe.PHASE
             and prior_spec.get('kind') == previous_contract.KIND,
             'FIRST_SAME_POINT25_EXACT_CLOSED_FAILED_V5_NOT_AMPLITUDE_ADVANCE')
        previous_recipe.episode_contract(prior)
        need(prior['geometry_scale'] == .25,'EXACT_FAILED_POINT25_BASELINE')
        predecessor_recipe = previous_recipe
    else:
        recipe.episode_contract(prior)
        need(actual.get('phase') == prior_spec.get('phase') == PHASE
             and prior_spec.get('kind') == KIND
             and prior['geometry_scale'] == recipe.SCALES[index-1]
             and prior_spec.get('source_sha') == spec.get('source_sha')
             and prior_spec.get('source_root') == spec.get('source_root')
             and all(prior_spec.get('source_files',{}).get(name) == spec['source_files'][name]
                     for name in SOURCE_MEMBERS),
             'IMMEDIATE_SAME_V6_SOURCE_AND_QUALIFIED_GRADE_ONLY')
        predecessor_recipe = recipe
    need(prior['source_observation_sha256'] == episode['source_observation_sha256']
         and prior['frame_map_sha256'] == episode['frame_map_sha256']
         and prior['scene_sha256'] == episode['scene_sha256'],'UNCHANGED_SOURCE_FRAME_AND_NATIVE_SCENE')
    evidence = audit_completed_feedback(prior_spec,actual,recipe=predecessor_recipe)
    predicates = evidence['readiness']['predicates']
    if index == 0:
        failures = {'full_grade','position','lateral','orientation'}
        need(set(predicates) == set(original_predicates())
             and all(type(value) is bool for value in predicates.values())
             and all(predicates[key] is False for key in failures)
             and all(predicates[key] is True for key in set(predicates)-failures)
             and evidence['all_nine_non_full_grade_predicates_passed'] is False,
             'RETAIN_POSITION_LATERAL_ORIENTATION_FAILURE_AND_OTHER_SIX_TRUE')
    else:
        need(evidence['all_nine_non_full_grade_predicates_passed'] is True,
             'ALL_NINE_ACTUAL_MEASURED_PREDICATES_BEFORE_AMPLITUDE_ADVANCE')
    return recipe.endpoint(episode)


def original_predicates():
    return ('full_grade','position','lateral','orientation','actual_cube_distance',
        'continuous_relative_hold','actual_open','cube_still','no_contact','original_safety')
