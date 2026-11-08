"""Real new cube layouts, with a separately declared unchanged T06 prefix.

Static geometry is evidence, never native admission. Original scopes, worker
locks, frozen native files and every command certificate remain mandatory.
"""
import json
from pathlib import Path
from .storage import digest
from . import stable_task_layout_static_v1 as layouts

PHASE = 'collect-target-stable-task-data-v1'
KIND = 'STABLE_TASK_DATA_V1'
B,E = 34303,38703
BUILDER_SOURCE = 'afb9a8f63f373d63d0f2a915cad3c65671df8c72'
JOURNAL = '331f73cd8d0898092fbbf553e94440baaa5d1b5a1f8fe698a5349a2feb57b54c'


def need(ok, reason):
    if not ok:
        raise ValueError('STABLE_TASK_SCENE_' + reason)


def pinned(descriptor):
    need(set(descriptor) == {'path','sha256'}, 'EXACT_DESCRIPTOR')
    p = Path(descriptor['path'])
    need(p.is_absolute() and p.resolve() == p and not p.is_symlink()
         and p.stat().st_size < 16*1024**2 and digest(p) == descriptor['sha256'],
         'CANONICAL_PINNED_METADATA')
    return json.loads(p.read_bytes())


def reference_geometry(reference, scene_sha256):
    """Read an explicit static layout and the old nominal calibration cube."""
    from .fusion_source_phase_c01 import nominal_reference
    need(set(reference) == {'static_result','nominal_calibration_reference'}, 'EXACT_SCENE_REFERENCE')
    proof = pinned(reference['static_result']); compiled = proof.get('compiled') or {}
    geometry = proof.get('dense_saved_geometry') or {}; changes = compiled.get('changes') or {}
    layout = proof.get('layout')
    need(layout in layouts.LAYOUTS and proof.get('schema') == 'WM02_STABLE_TASK_LAYOUT_STATIC_V1'
         and proof.get('status') == 'STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED'
         and proof.get('source_sha') == BUILDER_SOURCE and proof.get('original_journal_sha256') == JOURNAL
         and compiled.get('scene_sha256') == scene_sha256
         and compiled.get('base_scene_sha256') == layouts.BASE_SCENE
         and compiled.get('layout') == layout
         and changes.get('changed_model_arrays') == ['key_qpos']
         and changes.get('offset_xy_m') == list(layouts.LAYOUTS[layout])
         and changes.get('robot_startup_joints_changed') is False
         and changes.get('robot_dynamics_changed') is False
         and proof.get('WBC_changes') is False and proof.get('physics_dt_s') == .002
         and proof.get('native_integrations') == proof.get('new_transitions') == 0
         and proof.get('native_execution_authorized') is False
         and geometry.get('saved_pose_FK_calls') == 38703
         and geometry.get('contact_free_at_saved_poses') is True
         and geometry.get('integrations') == 0, 'EXACT_NEW_STATIC_LAYOUT_NOT_DYNAMIC_PERMISSION')
    nominal = nominal_reference(reference['nominal_calibration_reference'])
    return proof,nominal


def scene(spec):
    vla = spec.get('target_gr00t_collection') is not None
    target_loop=spec.get('phase')=='collect-target-gr00t-prefix-loop-v1'
    evaluated=spec.get('phase')=='collect-target-stable-task-eval-v1'
    closed=spec.get('phase')=='collect-target-stable-task-closed-loop-v1'
    expected_kind='TARGET_GR00T_PREFIX_LOOP_V1'if target_loop else 'STABLE_TASK_CLOSED_LOOP_V1'if closed else 'STABLE_TASK_EVAL_V1'if evaluated else KIND
    need(spec.get('phase')in(PHASE,'collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1','collect-target-gr00t-prefix-loop-v1')and spec.get('kind')==expected_kind
         and (not(evaluated or closed)or spec.get('data_role')=='DEV')
         and spec.get('data_role') in ('TRAIN','DEV')
         and spec.get('live_skill_start_step') == B and spec.get('max_ticks') == (40503 if target_loop else 43503 if closed else E),
         'NEW_REAL_SCENE_TYPED_PHASE_AND_ROLE')
    model = Path(spec['model_path'])
    need(model.is_absolute() and model.resolve() == model and not model.is_symlink(), 'EXACT_NEW_MODEL_PATH')
    proof,_ = reference_geometry(spec['stable_task_scene_reference'],digest(model))
    need(proof['split'] == spec['data_role'] and proof['parent'] == spec['data_parent']
         and spec['scene_id'] == 'STABLE_TASK_' + proof['layout'], 'REAL_PHYSICAL_PARENT_AND_SPLIT')
    panel = pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    need(not vla or not(evaluated or closed or target_loop), 'VLA_COLLECTION_NOT_ADAPTER_EVALUATION')
    need(panel.get('schema') == ('WM02_TARGET_GR00T_NATIVE_PREFIX_PANEL_V1'if target_loop else 'WM02_TARGET_GR00T_DEMONSTRATION_PANEL_V1'if vla else 'WM02_STABLE_TASK_CLOSED_LOOP_PANEL_V1'if closed else 'WM02_STABLE_TASK_ADAPTER_NATIVE_PANEL_V1'if evaluated else 'WM02_STABLE_TASK_DATA_PANEL_V1'), 'PREDECLARED_NEW_DATA_PANEL')
    rows = panel['assignments']; need(len({r['native_run'] for r in rows}) == len(rows), 'UNIQUE_NEW_RUNS')
    roles = {}
    for row in rows:
        need(row['split'] in ('TRAIN','DEV') and row['command'] in (('APPROACH_RED_4MM_OPEN','RETREAT_RED_4MM_OPEN')if vla or target_loop else ('HOLD_OPEN','APPROACH_RED_4MM_OPEN')),
             'ACTUAL_TASK_INSTRUCTION_NOT_TEACHER_PHASE')
        need(row['parent'] not in roles or roles[row['parent']] == row['split'], 'NO_PARENT_CROSS_SPLIT')
        roles[row['parent']] = row['split']
    match = [r for r in rows if r['native_run'] == Path(spec['registered_root']).name]
    need(len(match) == 1 and match[0]['parent'] == proof['parent'] and match[0]['split'] == proof['split']
         and match[0]['model_sha256'] == digest(model)
         and match[0]['static_sha256'] == spec['stable_task_scene_reference']['static_result']['sha256']
         and match[0]['command'] == spec['task_command'], 'FROZEN_LAYOUT_ROLE_GOAL_RUN_JOIN')
    return proof


def episode(spec):
    from .fusion_task_phase_pose_slow_feedback_v9 import make_episode
    return dict(schema='WM02_STABLE_TASK_ACTUAL_SCENE_CALIBRATION_PREFIX_V1',run=spec['registered_root'],
        actual_phase=spec['phase'],actual_kind=spec['kind'],
        actual_scene_id=spec['scene_id'],actual_scene_sha256=digest(spec['model_path']),
        actual_parent=spec['data_parent'],data_role=spec['data_role'],task_command=spec['task_command'],
        stable_task_scene_reference=spec['stable_task_scene_reference'],
        purpose='NEW_REAL_LAYOUT_WITH_EXPLICIT_UNCHANGED_T06_CALIBRATION_REFERENCE',
        calibration_prefix_recipe=make_episode(run=spec['registered_root'],geometry_scale=.5))


def prefix_episode(document,spec):
    from .fusion_task_phase_pose_slow_feedback_v9 import episode_contract
    need(document == episode(spec), 'ACTUAL_SCENE_WRAPPER_NOT_RELABELLED_T06_MANIFEST')
    return episode_contract(document['calibration_prefix_recipe'])
