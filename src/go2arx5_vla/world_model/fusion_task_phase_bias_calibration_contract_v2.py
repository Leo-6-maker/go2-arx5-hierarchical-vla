"""Finite .03 INIT calibration using its own frozen predecision measurements."""
from pathlib import Path

from . import fusion_task_phase_bias_calibration_v2 as recipe
from . import fusion_task_phase_initialization as absolute
from . import fusion_task_phase_initialization_contract as operational
from . import fusion_live_proposal_qualification_contract as common
from .storage import digest

PHASE = recipe.PHASE
KIND = 'FUSION_TASK_PHASE_BIAS_CALIBRATION_V2'
FIRST_PREDECESSOR_SPEC = 'fd15364dbe76753a677865842adb6c5d65bf62998249cbb19a5a661b241a97ec'
FIRST_PREDECESSOR = '9c71b1d4d69d6c27c7d49d211794237f3f12f5acc3b54d48e86af2629b6cd8b4'
FIRST_PREDECESSOR_SOURCE = 'abc116e90d9eea5ca1a602de8aa848439996cbae'
SOURCE_MEMBERS = tuple(dict.fromkeys((*operational.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_task_phase_bias_calibration_v2.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_bias_calibration_contract_v2.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_bias_calibration_native_v2.py',
    'scripts/wm02/fusion_task_phase_bias_calibration_check_v2.py')))


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_BIAS_CALIBRATION_NATIVE_V2_' + reason)


def episode_config(spec):
    episode = recipe.episode_contract(common._json(spec['episode_manifest'], spec['episode_manifest_sha256']))
    return operational.recipe_config(spec, episode, actual_phase=PHASE, actual_kind=KIND,
                                     source_members=SOURCE_MEMBERS)


def episode_limit(spec, previous):
    episode, _ = episode_config(spec)
    path = Path(spec['previous_stage'])
    actual = common._json(path, spec['previous_stage_sha256'])
    prior_spec, pin = common._closed(path.parent.parent)
    report = actual.get('reference_qualification') or {}
    prior = report.get('episode') or {}
    need(actual == previous and spec['previous_stage_sha256'] == FIRST_PREDECESSOR
         and pin == FIRST_PREDECESSOR_SPEC and prior_spec.get('source_sha') == FIRST_PREDECESSOR_SOURCE
         and actual.get('phase') == prior_spec.get('phase') == operational.PHASE,
         'EXACT_CLOSED_ABSOLUTE_INIT_POINT03_PREDECESSOR_NOT_A_NEW_RESULT')
    absolute.episode_contract(prior)
    need(common._json(prior_spec['episode_manifest'], prior_spec['episode_manifest_sha256']) == prior
         and prior['geometry_scale'] == episode['geometry_scale'] == .03
         and prior['ramp_atoms'] == prior['planned_end_atom'] == episode['planned_end_atom'] == 128
         and absolute.endpoint(prior) == recipe.endpoint(episode) == 7103
         and prior['source_observation_sha256'] == episode['source_observation_sha256']
         and prior['frame_map_sha256'] == episode['frame_map_sha256']
         and prior['scene_sha256'] == episode['scene_sha256'],
         'SAME_FIXED_DESIRED_RECIPE_GRADE_SCENE_AND_CLOCK')
    common._complete_previous(spec, actual, prior_spec, report, 7103)
    binding = report.get('task_phase_initialization') or {}
    need(binding.get('actual_VLA_commands') == binding.get('VLA_queries') == 0
         and binding.get('object_and_goal_are_policy_inputs') is False
         and all(digest(binding[name+'_path']) == binding.get(name+'_sha256')
                 for name in ('goal_binding', 'readiness', 'hold_samples')),
         'ACTUAL_ABSOLUTE_INIT_RUNTIME_GOAL_AND_NINE_SAMPLES_RETAINED')
    return 7103
