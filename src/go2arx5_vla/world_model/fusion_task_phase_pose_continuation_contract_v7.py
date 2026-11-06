"""Qualified V6 point25 to finite V7 point5, then qualified adjacent full1."""
from pathlib import Path
from . import fusion_task_phase_pose_continuation_v7 as recipe
from . import fusion_task_phase_pose_feedback_v6 as previous_recipe
from . import fusion_task_phase_pose_feedback_contract_v6 as previous_contract
from . import fusion_task_phase_initialization_contract as operational
from . import fusion_live_proposal_qualification_contract as common
from .fusion_task_phase_xyz_feedback_evidence import audit_completed_feedback

PHASE=recipe.PHASE
KIND='FUSION_TASK_PHASE_POSE_CONTINUATION_V7'
FIRST_PREDECESSOR_SOURCE='da177fe856fb184c50c9d7763132ec626ecfc172'
FIRST_PREDECESSOR_SPEC='3dc59acc133d2d6dc4de15e07b1ee7bea13abba49348327a628db300b48d2857'
FIRST_PREDECESSOR='c0a113d1426f0f460fcd4d4b9e9c3859f52f9b28a1a856567124ed1bfb4679ba'
SOURCE_MEMBERS=tuple(dict.fromkeys((*previous_contract.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/planner_event_log.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_pose_continuation_v7.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_pose_continuation_contract_v7.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_pose_continuation_native_v7.py',
    'scripts/wm02/fusion_task_phase_pose_continuation_check_v7.py')))


def need(ok,why):
    if not ok:raise ValueError('TASK_PHASE_POSE_CONTINUATION_NATIVE_V7_'+why)


def episode_config(spec):
    episode=recipe.episode_contract(common._json(spec['episode_manifest'],spec['episode_manifest_sha256']))
    return operational.recipe_config(spec,episode,actual_phase=PHASE,actual_kind=KIND,source_members=SOURCE_MEMBERS)


def episode_limit(spec,previous):
    episode,_=episode_config(spec);path=Path(spec['previous_stage'])
    actual=common._json(path,spec['previous_stage_sha256']);need(actual==previous,'EXACT_PREDECESSOR_RESULT')
    prior_spec,prior_pin=common._closed(path.parent.parent)
    prior=(actual.get('reference_qualification')or{}).get('episode')or{}
    need(common._json(prior_spec['episode_manifest'],prior_spec['episode_manifest_sha256'])==prior,'PREDECESSOR_SPEC_EPISODE_JOIN')
    if episode['geometry_scale']==.5:
        need(spec['previous_stage_sha256']==FIRST_PREDECESSOR and prior_pin==FIRST_PREDECESSOR_SPEC
             and prior_spec.get('source_sha')==FIRST_PREDECESSOR_SOURCE
             and actual.get('phase')==prior_spec.get('phase')==previous_recipe.PHASE
             and prior_spec.get('kind')==previous_contract.KIND,'EXACT_QUALIFIED_V6_PREDECESSOR')
        previous_recipe.episode_contract(prior);need(prior['geometry_scale']==.25,'POINT25_BEFORE_POINT5')
        ancestor=previous_recipe
    else:
        recipe.episode_contract(prior)
        need(actual.get('phase')==prior_spec.get('phase')==PHASE and prior_spec.get('kind')==KIND
             and prior['geometry_scale']==.5 and prior_spec.get('source_sha')==spec.get('source_sha')
             and prior_spec.get('source_root')==spec.get('source_root')
             and all(prior_spec.get('source_files',{}).get(name)==spec['source_files'][name]for name in SOURCE_MEMBERS),
             'QUALIFIED_ADJACENT_SAME_V7_SOURCE_ONLY')
        ancestor=recipe
    need(all(prior[key]==episode[key]for key in ('source_observation_sha256','frame_map_sha256','scene_sha256')),
         'UNCHANGED_SOURCE_FRAME_SCENE')
    evidence=audit_completed_feedback(prior_spec,actual,recipe=ancestor)
    need(evidence['all_nine_non_full_grade_predicates_passed']is True,'ALL_NINE_MEASURED_PREDICATES_REQUIRED')
    return recipe.endpoint(episode)
