"""One point1 feedback commissioning condition; no amplitude advancement."""
from pathlib import Path

from . import fusion_task_phase_xyz_feedback_v4 as recipe
from . import fusion_task_phase_bias_continuation_v3 as prior_recipe
from . import fusion_task_phase_bias_continuation_contract_v3 as prior_contract
from . import fusion_task_phase_initialization_contract as operational
from . import fusion_task_phase_initialization as original
from . import fusion_live_proposal_qualification_contract as common
from .storage import digest

PHASE = recipe.PHASE
KIND = 'FUSION_TASK_PHASE_XYZ_FEEDBACK_V4'
FIRST_PREDECESSOR_SPEC = '550e4f4bec3de66c12a24fd911cc5691c49db901461ec88bedfcbcd9c7030e33'
FIRST_PREDECESSOR = 'cab1653ee05fa261c520bc01dc05ea17a861466e5c4570995fa2882d02494339'
FIRST_PREDECESSOR_SOURCE = '94b6b789cc4789fb1c9202d1147b1bee56c79895'
SOURCE_MEMBERS = tuple(dict.fromkeys((*prior_contract.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_task_phase_xyz_feedback_v4.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_xyz_feedback_contract_v4.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_xyz_feedback_native_v4.py',
    'scripts/wm02/fusion_task_phase_xyz_feedback_check_v4.py')))


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_XYZ_FEEDBACK_NATIVE_V4_' + reason)


def episode_config(spec):
    episode = recipe.episode_contract(common._json(spec['episode_manifest'], spec['episode_manifest_sha256']))
    return operational.recipe_config(spec, episode, actual_phase=PHASE, actual_kind=KIND,
                                     source_members=SOURCE_MEMBERS)


def episode_limit(spec, previous):
    episode, source = episode_config(spec)
    path = Path(spec['previous_stage'])
    actual = common._json(path, spec['previous_stage_sha256'])
    prior_spec, prior_pin = common._closed(path.parent.parent)
    need(actual == previous and spec['previous_stage_sha256'] == FIRST_PREDECESSOR
         and prior_pin == FIRST_PREDECESSOR_SPEC
         and prior_spec.get('source_sha') == FIRST_PREDECESSOR_SOURCE
         and actual.get('phase') == prior_spec.get('phase') == prior_recipe.PHASE
         and prior_spec.get('kind') == prior_contract.KIND,
         'EXACT_CLOSED_V3_POINT1_NATIVE_PASS_WITH_RETAINED_GOAL_FAILURE')
    report = actual.get('reference_qualification') or {}
    prior = prior_recipe.episode_contract(report.get('episode'))
    need(prior['geometry_scale'] == .1 and prior_recipe.endpoint(prior) == 10303
         and common._json(prior_spec['episode_manifest'], prior_spec['episode_manifest_sha256']) == prior,
         'SAME_POINT1_NOT_POINT25_OR_DIFFERENT_RECIPE')
    common._complete_previous(spec, actual, prior_spec, report, 10303)
    binding = report.get('task_phase_initialization') or {}
    need(binding.get('experiment_role') == 'NON_VLA_CALIBRATION'
         and binding.get('actual_VLA_commands') == binding.get('VLA_queries') == 0
         and binding.get('object_and_goal_are_policy_inputs') is False, 'ORIGINAL_NON_VLA_ROLE')
    artifacts = {}
    for key in ('goal_binding', 'readiness', 'hold_samples'):
        artifact = Path(binding[key+'_path'])
        need(artifact.parent == path.parent, 'EXACT_PRIOR_NATIVE_ARTIFACT_LOCATION')
        artifacts[key] = common._json(artifact, binding[key+'_sha256'])
    goal = artifacts['goal_binding']
    need(goal.get('actual_spec_phase') == prior_recipe.PHASE
         and goal.get('actual_spec_source_sha') == FIRST_PREDECESSOR_SOURCE
         and goal.get('episode_sha256') == prior_spec['episode_manifest_sha256']
         and goal.get('decision_step') == 3703
         and goal.get('frozen_before_generated_reference_install') is True,
         'EXACT_PRIOR_PREINSTALL_GOAL_IDENTITY')
    packet, measured = Path(goal['saved_packet_path']), Path(goal['measured_state_path'])
    need(packet.parent == measured.parent == path.parent and not packet.is_symlink()
         and digest(packet) == goal['saved_packet_sha256'], 'EXACT_PRIOR_PACKET_NOT_NEW_RESTART')
    inputs = original.load_inputs(**source, decision_packet_bytes=packet.read_bytes(),
        measured_decision_state=common._json(measured, goal['measured_state_sha256']))
    need(inputs['frozen'] == goal['frozen_input'], 'SAVED_FROZEN_PRIOR_GOAL_RECOMPUTES')
    readiness = prior_recipe.query_readiness(artifacts['hold_samples'], prior, inputs)
    need(readiness == artifacts['readiness'] == binding.get('readiness')
         and readiness['actual_sample_steps'] == binding.get('readiness_sample_native_steps'),
         'EXACT_SAVED_NINE_ACTUAL_SAMPLES_AND_READINESS')
    predicates = readiness['predicates']
    required = prior_contract.READINESS_PREDICATES-{'full_grade', 'position', 'lateral'}
    need(set(predicates) == prior_contract.READINESS_PREDICATES
         and all(type(value) is bool for value in predicates.values())
         and predicates['full_grade'] is predicates['position'] is predicates['lateral'] is False
         and all(predicates[key] for key in required) and readiness['query_ready'] is False
         and readiness['native_execution_authorized'] is readiness['GPU_query_authorized'] is False,
         'RETAIN_POSITION_LATERAL_FAILURE_AND_REQUIRE_OTHER_SEVEN_MEASURED_PREDICATES')
    return recipe.endpoint(episode)
