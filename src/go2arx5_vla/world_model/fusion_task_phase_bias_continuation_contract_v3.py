"""Adjacent continuation requires complete native and measured-hold evidence."""
from pathlib import Path

from . import fusion_task_phase_bias_continuation_v3 as recipe
from . import fusion_task_phase_bias_calibration_v2 as first
from . import fusion_task_phase_bias_calibration_contract_v2 as first_contract
from . import fusion_task_phase_initialization as original
from . import fusion_task_phase_initialization_contract as operational
from . import fusion_live_proposal_qualification_contract as common
from .storage import digest

PHASE = recipe.PHASE
KIND = 'FUSION_TASK_PHASE_BIAS_CONTINUATION_V3'
FIRST_PREDECESSOR_SPEC = 'd244df6e81cc57ef4c6feefa3914c016de8593a47789b630989ce0d711466628'
FIRST_PREDECESSOR = '5c9d9cbfed522e9deaed2d957e04f8141785b696d91a022d901808af1292e262'
FIRST_PREDECESSOR_SOURCE = 'efead413525c17bd290c2514bbaf82ea75ba37c3'
READINESS_PREDICATES = {'full_grade', 'position', 'lateral', 'orientation', 'actual_cube_distance',
    'continuous_relative_hold', 'actual_open', 'cube_still', 'no_contact', 'original_safety'}
SOURCE_MEMBERS = tuple(dict.fromkeys((*first_contract.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_task_phase_bias_continuation_v3.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_bias_continuation_contract_v3.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_bias_continuation_native_v3.py',
    'scripts/wm02/fusion_task_phase_bias_continuation_check_v3.py')))


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_BIAS_CONTINUATION_NATIVE_V3_' + reason)


def episode_config(spec):
    episode = recipe.episode_contract(common._json(spec['episode_manifest'], spec['episode_manifest_sha256']))
    return operational.recipe_config(spec, episode, actual_phase=PHASE, actual_kind=KIND,
                                     source_members=SOURCE_MEMBERS)


def _qualified_hold(prior_spec, report, prior_episode, source, result_path, math_view):
    """Recheck saved actual samples without loading a model, FK or physics."""
    binding = report.get('task_phase_initialization') or {}
    need(binding.get('experiment_role') == 'NON_VLA_CALIBRATION'
         and binding.get('actual_VLA_commands') == binding.get('VLA_queries') == 0
         and binding.get('object_and_goal_are_policy_inputs') is False,
         'ORIGINAL_NON_VLA_CALIBRATION_ROLE')
    artifacts = {}
    for key in ('goal_binding', 'readiness', 'hold_samples'):
        path = Path(binding[key+'_path'])
        need(path.parent == result_path.parent and digest(path) == binding.get(key+'_sha256'),
             'EXACT_PRIOR_NATIVE_ARTIFACT_FILE:' + key)
        artifacts[key] = common._json(path, binding[key+'_sha256'])
    goal = artifacts['goal_binding']
    need(goal.get('actual_spec_phase') == prior_spec['phase']
         and goal.get('actual_spec_source_sha') == prior_spec['source_sha']
         and goal.get('episode_sha256') == prior_spec['episode_manifest_sha256']
         and goal.get('decision_step') == 3703
         and goal.get('frozen_before_generated_reference_install') is True,
         'ACTUAL_PRIOR_VERSION_AND_PREINSTALL_GOAL_JOIN')
    packet_path, state_path = Path(goal['saved_packet_path']), Path(goal['measured_state_path'])
    need(packet_path.parent == state_path.parent == result_path.parent
         and not packet_path.is_symlink() and digest(packet_path) == goal['saved_packet_sha256'],
         'EXACT_PRIOR_3703_PACKET_NOT_CURRENT_RESTART')
    measured = common._json(state_path, goal['measured_state_sha256'])
    inputs = original.load_inputs(**source, decision_packet_bytes=packet_path.read_bytes(),
                                  measured_decision_state=measured)
    need(inputs['frozen'] == goal['frozen_input'], 'RECOMPUTED_FROZEN_GOAL_FROM_EXACT_PRIOR_MEASUREMENT')
    readiness = original.query_readiness(artifacts['hold_samples'], math_view(prior_episode), inputs)
    need(readiness == artifacts['readiness'] == binding.get('readiness')
         and readiness['actual_sample_steps'] == binding.get('readiness_sample_native_steps'),
         'SAVED_NINE_ACTUAL_SAMPLES_RECOMPUTE_EXACT_PRIOR_READINESS')
    predicates = readiness['predicates']
    need(set(predicates) == READINESS_PREDICATES and all(type(x) is bool for x in predicates.values())
         and all(value for key, value in predicates.items() if key != 'full_grade')
         and predicates['full_grade'] is (prior_episode['geometry_scale'] == 1.)
         and readiness['query_ready'] is all(predicates.values())
         and readiness['native_execution_authorized'] is False
         and readiness['GPU_query_authorized'] is False,
         'ALL_NINE_NON_FULL_GRADE_MEASUREMENT_PREDICATES_REQUIRED_NOT_A_GRANT')


def episode_limit(spec, previous):
    episode, source = episode_config(spec)
    path = Path(spec['previous_stage'])
    actual = common._json(path, spec['previous_stage_sha256'])
    need(actual == previous, 'EXACT_PINNED_PREDECESSOR_RESULT')
    prior_spec, prior_pin = common._closed(path.parent.parent)
    report = actual.get('reference_qualification') or {}
    prior = report.get('episode') or {}
    need(common._json(prior_spec['episode_manifest'], prior_spec['episode_manifest_sha256']) == prior,
         'PREDECESSOR_ACTUAL_SPEC_EPISODE_RESULT_JOIN')
    index = recipe.SCALES.index(episode['geometry_scale'])
    if index == 0:
        need(spec['previous_stage_sha256'] == FIRST_PREDECESSOR and prior_pin == FIRST_PREDECESSOR_SPEC
             and prior_spec.get('source_sha') == FIRST_PREDECESSOR_SOURCE
             and actual.get('phase') == prior_spec.get('phase') == first.PHASE
             and prior_spec.get('kind') == first_contract.KIND,
             'FIRST_POINT1_REQUIRES_EXACT_CLOSED_QUALIFIED_V2_POINT03')
        first.episode_contract(prior)
        need(prior['geometry_scale'] == .03, 'EXACT_V2_POINT03_NOT_ANOTHER_GRADE')
        expected, math_view = first.endpoint(prior), first.parent_recipe_view
    else:
        recipe.episode_contract(prior)
        need(actual.get('phase') == prior_spec.get('phase') == PHASE
             and prior_spec.get('kind') == KIND
             and prior['geometry_scale'] == recipe.SCALES[index-1]
             and prior_spec.get('source_sha') == spec.get('source_sha')
             and prior_spec.get('source_root') == spec.get('source_root')
             and all(prior_spec.get('source_files', {}).get(name) == spec['source_files'][name]
                     for name in SOURCE_MEMBERS),
             'IMMEDIATELY_PRECEDING_SAME_V3_SOURCE_AND_GRADE_ONLY')
        expected, math_view = recipe.endpoint(prior), recipe.parent_recipe_view
    need(prior['source_observation_sha256'] == episode['source_observation_sha256']
         and prior['frame_map_sha256'] == episode['frame_map_sha256']
         and prior['scene_sha256'] == episode['scene_sha256'], 'UNCHANGED_SOURCE_FRAME_SCENE')
    common._complete_previous(spec, actual, prior_spec, report, expected)
    _qualified_hold(prior_spec, report, prior, source, path, math_view)
    return recipe.endpoint(episode)
