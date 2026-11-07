"""Exact qualified half-grade plus a separately retained failed-full1 repair.

Historical FAILED/CLOSED1 evidence is never passed to a success validator and
does not authorize a new native world. Original AUTO1/P1 admission is required.
"""
import hashlib
from pathlib import Path
import subprocess

from . import fusion_task_phase_bias_fade_v8 as recipe
from . import fusion_task_phase_pose_continuation_v7 as previous_recipe
from . import fusion_task_phase_pose_continuation_contract_v7 as previous_contract
from . import fusion_task_phase_initialization_contract as operational
from . import fusion_live_proposal_qualification_contract as common
from .fusion_task_phase_xyz_feedback_evidence import audit_completed_feedback
from .storage import digest

PHASE = recipe.PHASE
KIND = 'FUSION_TASK_PHASE_BIAS_FADE_V8'
FIRST_PREDECESSOR_SOURCE = 'cdced494d46bd4df1930142317244b2da1ba9229'
FIRST_PREDECESSOR_SPEC = '79e5159de909c1d128bb14adc02ba61a2e0a83a142c1fb42d4b7e90c32d9ddad'
FIRST_PREDECESSOR = '1e1b0c75626cefeaf89f417cb0c1d276ec14fbc4eb473cf2b64fc238581581ae'
FAILED_SOURCE = FIRST_PREDECESSOR_SOURCE
FAILED_SPEC = 'f7c79e28cba447f72d005d5be2e39cb1d0a73a1354d840aeaae23a939ce8ff3a'
FAILED_RESULT = '595e05e9538918990c068227eee32e5009aaf7c1c6220189e754d31f89d85b79'
FAILED_LEASE = '240d9d957eff2d8ea77050201e3df106d519d436a4678741c353167570641ecd'
FAILED_WORKER = 'a9b0cbc3ae383730c269904f87bd4fcafe6aafeed8c85ae6ca90639dde254b13'
FAILED_COMMANDS_SHA = '13bcf983733c038f958eb9a4ba05e6fcbba0437452bf1449f85670878b1b5666'
FAILED_REASON = 'ORIGINAL_NATIVE_JOINT_LIMIT_REJECT_NO_CLIP'
SOURCE_MEMBERS = tuple(dict.fromkeys((*previous_contract.SOURCE_MEMBERS,
    'src/go2arx5_vla/world_model/fusion_task_phase_bias_fade_v8.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_bias_fade_contract_v8.py',
    'src/go2arx5_vla/world_model/fusion_task_phase_bias_fade_native_v8.py',
    'scripts/wm02/fusion_task_phase_bias_fade_check_v8.py')))


def need(ok, why):
    if not ok:
        raise ValueError('TASK_PHASE_BIAS_FADE_NATIVE_V8_' + why)


def _unit_status(root):
    result = subprocess.run(['systemctl', '--user', 'show', 'wm02-'+root.name,
        '-p', 'MainPID,ExecMainStatus'], capture_output=True, text=True, timeout=10)
    need(result.returncode == 0, 'READ_ONLY_TERMINAL_UNIT_STATUS_REQUIRED')
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def episode_config(spec):
    episode = recipe.episode_contract(common._json(spec['episode_manifest'], spec['episode_manifest_sha256']))
    return operational.recipe_config(spec, episode, actual_phase=PHASE, actual_kind=KIND,
                                     source_members=SOURCE_MEMBERS)


def _failed_full1_supplement(spec, prior_spec):
    """Only the exact CLOSED1 failed scope; deliberately never ``_closed``."""
    path = Path(spec['repaired_full1_failed_result'])
    root = path.parent.parent
    need(path == root/'native-01/RESULT.json'
         and Path(spec['repaired_full1_failed_spec']) == root/'EXECUTION_SPEC.json'
         and spec['repaired_full1_failed_spec_sha256'] == FAILED_SPEC
         and spec['repaired_full1_failed_result_sha256'] == FAILED_RESULT,
         'EXPLICIT_EXACT_RETAINED_FAILED_FULL1_SPEC_AND_RESULT_PINS')
    failed_spec = common._json(root/'EXECUTION_SPEC.json', FAILED_SPEC)
    failed = common._json(path, FAILED_RESULT)
    lease = common._json(root/'control/LEASE.json', FAILED_LEASE)
    worker = common._json(root/'WORKER_EXIT.private.json', FAILED_WORKER)
    unit = _unit_status(root)
    need(unit.get('MainPID') == '0' and unit.get('ExecMainStatus') == '1'
         and lease.get('status') == 'CLOSED' and lease.get('worker_exit_code') == worker.get('exit_code') == 1
         and lease.get('spec_sha256') == FAILED_SPEC and lease.get('source_sha') == FAILED_SOURCE
         and lease.get('run_uuid') == root.name and worker.get('failure') is None,
         'ACTUAL_CLOSED1_PID0_FAILED_SCOPE_NOT_A_REOPENED_PASS')
    need(failed_spec.get('registered_root') == str(root)
         and failed_spec.get('source_sha') == failed.get('source_sha') == FAILED_SOURCE
         and failed_spec.get('source_root') == prior_spec.get('source_root')
         and failed_spec.get('phase') == failed.get('phase') == previous_recipe.PHASE
         and failed_spec.get('kind') == previous_contract.KIND
         and failed_spec.get('previous_stage_sha256') == FIRST_PREDECESSOR
         and all(failed_spec.get('source_files', {}).get(name) == prior_spec.get('source_files', {}).get(name)
                 for name in previous_contract.SOURCE_MEMBERS),
         'EXACT_OLD_FULL1_AND_QUALIFIED_HALF_SOURCE_LINEAGE')
    old_episode = previous_recipe.episode_contract(common._json(
        failed_spec['episode_manifest'], failed_spec['episode_manifest_sha256']))
    need(old_episode['geometry_scale'] == 1. and previous_recipe.endpoint(old_episode) == 46303
         and failed_spec.get('max_ticks') == 46303, 'RETAINED_ORIGINAL_FULL1_CLOCK')
    need(failed.get('status') == 'NATIVE_STAGE_FAILED_PRESERVED'
         and failed.get('actual_target_integrations') == 20503
         and failed.get('complete_atomic_transitions') == 820
         and failed.get('shutdown_clean') is True and failed.get('reached_exact_pause') is False
         and failed.get('physics_stops') == [] and failed.get('original_joint_or_warning_samples') == 0
         and failed.get('native_outer_guard_rejection_count') == 0
         and failed.get('trace_audit_error') is None and failed.get('reference_qualification') is None
         and FAILED_REASON in str(failed.get('failure'))
         and len(failed.get('owned_process_exits', [])) == 4
         and {p.get('name') for p in failed['owned_process_exits']} == {'sim', 'controller', 'solver', 'planner'}
         and all(p.get('exit_code') == 0 for p in failed['owned_process_exits']),
         'PRE_FEEDBACK_UNPUBLISHED_ORIGINAL_REJECTION_WITH_CLEAN_SHUTDOWN')
    need(all(failed_spec.get(key) == prior_spec.get(key) == spec.get(key)
             for key in ('campaign_root', 'candidate_index', 'model_path', 'native_binary', 'observer_library'))
         and all(failed.get(key) == digest(spec[member]) for key, member in (
             ('scene_sha256', 'model_path'), ('native_binary_sha256', 'native_binary'),
             ('observer_sha256', 'observer_library'))), 'UNCHANGED_FROZEN_NATIVE_SCENE_AND_OWNER_LINEAGE')
    commands_path = root/'native-01/COMMANDS.private.jsonl'
    need(not commands_path.is_symlink(), 'EXACT_FAILED_COMMAND_ARTIFACT')
    raw = commands_path.read_bytes()
    need(len(raw) <= 64*1024**2 and hashlib.sha256(raw).hexdigest() == FAILED_COMMANDS_SHA
         and raw.endswith(b'\n'), 'PINNED_SEALED_FAILED_COMMAND_ARTIFACT')
    import json
    rejected = [json.loads(line) for line in raw.splitlines()
                if b'"reference-certificate-rejected"' in line]
    need(len(rejected) == 1 and rejected[0].get('native_step') == 20503
         and rejected[0].get('reason') == FAILED_REASON
         and rejected[0].get('submitted_or_executed') is False,
         'EXACT_UNPUBLISHED_REJECTION_NOT_EXECUTED_OR_RELABELLED')


def episode_limit(spec, previous):
    episode, _ = episode_config(spec)
    path = Path(spec['previous_stage'])
    actual = common._json(path, spec['previous_stage_sha256'])
    need(actual == previous and spec['previous_stage_sha256'] == FIRST_PREDECESSOR,
         'EXACT_QUALIFIED_HALF_RESULT')
    prior_spec, pin = common._closed(path.parent.parent)
    unit = _unit_status(path.parent.parent)
    need(pin == FIRST_PREDECESSOR_SPEC and prior_spec.get('source_sha') == FIRST_PREDECESSOR_SOURCE
         and unit.get('MainPID') == '0' and unit.get('ExecMainStatus') == '0'
         and prior_spec.get('phase') == actual.get('phase') == previous_recipe.PHASE
         and prior_spec.get('kind') == previous_contract.KIND, 'EXACT_CLOSED0_PID0_QUALIFIED_V7_HALF')
    prior = previous_recipe.episode_contract(common._json(
        prior_spec['episode_manifest'], prior_spec['episode_manifest_sha256']))
    need(prior['geometry_scale'] == .5 and prior == actual['reference_qualification']['episode']
         and all(prior[key] == episode[key] for key in (
             'source_observation_sha256', 'frame_map_sha256', 'scene_sha256')),
         'SAME_QUALIFIED_HALF_SOURCE_FRAME_SCENE')
    proof = audit_completed_feedback(prior_spec, actual, recipe=previous_recipe)
    need(proof['all_nine_non_full_grade_predicates_passed'] is True,
         'ALL_NINE_ORIGINAL_MEASURED_HALF_PREDICATES_REQUIRED')
    _failed_full1_supplement(spec, prior_spec)
    return recipe.endpoint(episode)
