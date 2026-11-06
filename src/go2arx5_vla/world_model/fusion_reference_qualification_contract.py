"""One versioned reference qualification, separate from frozen FUSION XY4.

NumPy-only program generation/validation. These functions never register a
scope, advance physics, supply a native certificate or grant execution.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from scripts.pi0294.base_intent import mount_pose
from scripts.pi0294.task_gripper_contract import require_working_target, teacher_physical_fraction
from ..task_data.codec import pose10
from .storage import digest
from .target_reference import require_install_clock, require_live_window

SCHEMA = 'WM02_FUSION_REFERENCE_QUALIFICATION_EPISODE_V1'
PHASE = 'collect-target-fusion-reference-qualification-v1'
RULE = 'FIXED_HOLD40_BURN8_DECIDE_RAMP8_REFERENCE_QUALIFICATION_V1'
INIT_ANCHOR = 'INIT_TCP_2503_REFERENCE_V1'
DECISION_ANCHOR = 'DECISION_TCP_3703_REANCHOR_V1'
INIT_STEP, DECISION_STEP, END_STEP = 2503, 3703, 4103
SCREEN = dict(window_s=.4, relative_TCP_diameter_m=.002, max_relative_TCP_speed_m_s=.025)
# This is a finite offline catalogue, not an admitted action set or a sweep.
CANDIDATES = {
    'Z_PLUS_0P5MM': (0., 0., .0005), 'Z_MINUS_0P5MM': (0., 0., -.0005),
    'Z_PLUS_1MM': (0., 0., .001), 'Z_MINUS_1MM': (0., 0., -.001),
    'X_PLUS_1P25MM': (.00125, 0., 0.), 'X_MINUS_1P25MM': (-.00125, 0., 0.),
    'Y_PLUS_1P25MM': (0., .00125, 0.), 'Y_MINUS_1P25MM': (0., -.00125, 0.),
}
FIRST_NATIVE_CANDIDATE = 'Z_PLUS_0P5MM'
SOURCE_MEMBERS = (
    'src/go2arx5_vla/world_model/fusion_reference_qualification_contract.py',
    'src/go2arx5_vla/world_model/fusion_reference_qualification_native.py',
    'src/go2arx5_vla/world_model/auto1_native.py',
    'src/go2arx5_vla/world_model/target_sequential.py',
    'scripts/wm02/fusion_reference_qualification_check.py',
    'scripts/wm02/pilot1_native_worker.py',
    'scripts/wm02/commission_target.py',
)


def need(condition, message):
    if not condition:
        raise ValueError('REFERENCE_QUALIFICATION_' + message)


def episode_contract(episode, *, native=False):
    """Validate a single predeclared TRAIN diagnostic; no target-domain grant."""
    expected = dict(schema=SCHEMA, phase=PHASE, purpose='REFERENCE_QUALIFICATION',
                    experiment_role='NON_VLA_CALIBRATION',
                    split='TRAIN', reference_rule=RULE, task_atoms=64,
                    prehold_atoms=40, screen_atom=48, screen=SCREEN,
                    desired_outcome_provenance='CALIBRATION_INTENT',
                    rotation_program='HOLD_ANCHOR_ORIENTATION',
                    gripper_program='HOLD_ORIGINAL_WORKING_OPEN',
                    base_program='MEASURED_INIT_BASE_POSE_HOLD', physics_dt_s=.002)
    need(isinstance(episode, dict) and all(episode.get(k) == v for k, v in expected.items()),
         'EXACT_VERSIONED_EPISODE_AND_ORIGINAL_CLOCK')
    need(isinstance(episode.get('run'), str) and bool(episode['run'])
         and isinstance(episode.get('scene_id'), str) and bool(episode['scene_id'])
         and isinstance(episode.get('parent'), str) and bool(episode['parent']), 'EXACT_RUN_SCENE_PARENT')
    need(episode.get('candidate_id') in CANDIDATES
         and episode.get('reference_anchor_mode') in (INIT_ANCHOR, DECISION_ANCHOR),
         'EXPLICIT_CANDIDATE_AND_ANCHOR_MODE')
    forbidden = {'offset_xyz_m', 'rotation_delta', 'gripper', 'rescale', 'clip',
                 'skip_certificate', 'safety_override', 'native_execution_authorized'}
    need(not (forbidden & set(episode)), 'NO_ARBITRARY_PROGRAM_OR_GRANT_FIELDS')
    if native:
        need(episode['candidate_id'] == FIRST_NATIVE_CANDIDATE,
             'V1_NATIVE_ENTRY_ONLY_FIRST_Z_PLUS_0P5MM_OTHER_CANDIDATES_OFFLINE_ONLY')
    return episode


@dataclass(frozen=True)
class ReferenceProgram:
    """Immutable initial reference and decision-derived desired pose are distinct."""
    candidate_id: str
    anchor_mode: str
    initial_tcp: object
    decision_tcp: object
    base_pose_hold: object

    def __post_init__(self):
        need(self.candidate_id in CANDIDATES and self.anchor_mode in (INIT_ANCHOR, DECISION_ANCHOR),
             'FINITE_CANDIDATE_AND_VERSIONED_ANCHOR')
        for name in ('initial_tcp', 'decision_tcp'):
            value = mount_pose(getattr(self, name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        base = np.asarray(self.base_pose_hold, float).copy()
        need(base.shape == (7,) and np.isfinite(base).all()
             and abs(np.linalg.norm(base[3:])-1) <= 1e-6, 'MEASURED_INITIAL_BASE_POSE7')
        base.setflags(write=False)
        object.__setattr__(self, 'base_pose_hold', base)

    @property
    def anchor(self):
        return (self.initial_tcp if self.anchor_mode == INIT_ANCHOR else self.decision_tcp).copy()

    @property
    def reference_goal(self):
        result = self.anchor
        result[:3, 3] += CANDIDATES[self.candidate_id]
        return result

    @property
    def desired_goal(self):
        result = self.decision_tcp.copy()
        result[:3, 3] += CANDIDATES[self.candidate_id]
        return result

    def preview(self, elapsed_atoms=0):
        """Continue the originally planned ramp; renewal never reanchors to motion."""
        need(type(elapsed_atoms) is int and elapsed_atoms in (0, 7, 14),
             'ONLY_ORIGINAL_350MS_RENEWALS_WITHIN_H16')
        from .target_reference import fixed_goal_poses
        # At atom7 the remaining original eighth ramp point is the fixed goal.
        poses = fixed_goal_poses(self.anchor, self.reference_goal, renewed=elapsed_atoms > 0)
        return np.stack([pose10(p, float(teacher_physical_fraction(1.))) for p in poses])

    def evidence(self):
        return dict(candidate_id=self.candidate_id, reference_anchor_mode=self.anchor_mode,
            offset_xyz_m=list(CANDIDATES[self.candidate_id]),
            init_native_step=INIT_STEP, decision_native_step=DECISION_STEP,
            initial_reference_TCP=self.initial_tcp.tolist(), actual_decision_TCP=self.decision_tcp.tolist(),
            reference_anchor_TCP=self.anchor.tolist(), nominal_reference_goal_TCP=self.reference_goal.tolist(),
            desired_actual_goal_TCP=self.desired_goal.tolist(),
            reference_minus_desired_xyz_m=(self.reference_goal[:3, 3]-self.desired_goal[:3, 3]).tolist(),
            measured_initial_BASE_reference=self.base_pose_hold.tolist(),
            reanchor_is_new_explicit_mode=self.anchor_mode == DECISION_ANCHOR,
            subsequent_achieved_state_reanchor=False, desired_outcome_provenance='CALIBRATION_INTENT',
            experiment_role='NON_VLA_CALIBRATION',
            VLA_inferences=0, full_task_success=False, native_execution_authorized=False,
            original_reference_history_certificate_still_required=True)


def hold_preview(initial_tcp):
    pose = mount_pose(initial_tcp)
    return np.stack([pose10(pose, float(teacher_physical_fraction(1.)))] * 40)


def validate_install_preview(action, *, initial_tcp, program, native_step, base_pose_hold,
                             frozen_base_pose_hold):
    """Exact full-H40 contract before the ORIGINAL native certifier and receiver."""
    need(type(native_step) is int and INIT_STEP <= native_step < END_STEP
         and (native_step-INIT_STEP) % 25 == 0, 'ACTUAL_ORIGINAL_ATOMIC_BOUNDARY')
    base, frozen = np.asarray(base_pose_hold, float), np.asarray(frozen_base_pose_hold, float)
    need(base.shape == frozen.shape == (7,) and np.isfinite([base, frozen]).all()
         and np.allclose(base, frozen, rtol=0, atol=1e-10), 'UNCHANGED_MEASURED_BASE_REFERENCE')
    action = np.asarray(action, float)
    need(action.shape == (40, 10) and np.isfinite(action).all(), 'FINITE_FULL_H40')
    require_working_target(action[:, 9])  # Raw mechanical fractions; no rescaling.
    if native_step < DECISION_STEP:
        need(program is None, 'NO_CANDIDATE_BEFORE_REAL_DECISION')
        expected = hold_preview(initial_tcp)
    else:
        need(isinstance(program, ReferenceProgram), 'REAL_DECISION_PROGRAM_REQUIRED')
        need(program.candidate_id == FIRST_NATIVE_CANDIDATE,
             'V1_NATIVE_INSTALL_ONLY_FIRST_Z_PLUS_0P5MM')
        need(np.allclose(program.initial_tcp, initial_tcp, rtol=0, atol=1e-12)
             and np.allclose(program.base_pose_hold, frozen, rtol=0, atol=1e-10),
             'ORIGINAL_INITIAL_AND_BASE_REFERENCE_RETAINED')
        expected = program.preview((native_step-DECISION_STEP)//25)
    need(np.allclose(action, expected, rtol=0, atol=1e-12), 'ENTIRE_H40_DIFFERS_FROM_FIXED_NEW_PROGRAM')
    return dict(status='EXACT_NEW_PROGRAM_PREVIEW_CHECKED_NOT_NATIVE_QUALIFIED',
                native_step=native_step, points_checked=40, physics_advance_authorized=False,
                original_certify_chunk_required=True, original_install_ACK_required=True)


def check_atomic_window(source_s, native_time_s):
    require_install_clock(source_s, native_time_s)
    return require_live_window(source_s, native_time_s, 25)


def episode_config(spec):
    """Read-only recipe check; real AUTO1/PILOT1 validation remains independent."""
    need(spec.get('phase') == PHASE and spec.get('profile') == 'wm02-auto1-target'
         and spec.get('domain') == 'WM02_AUTO1_TARGET_COMMISSION_V1'
         and spec.get('physics') is True and bool(spec.get('pilot1_binding')),
         'ORIGINAL_AUTO1_PILOT1_TARGET_ROUTE_REQUIRED')
    need(not any(spec.get(k) for k in ('fusion_binding', 'night1_binding', 'next1_binding', 'next2_binding')),
         'NEW_PROGRAM_CANNOT_BORROW_FROZEN_FUSION_XY_BINDING')
    need(not any(spec.get(k) for k in ('parallel_batch_spec', 'standby_shutdown_diagnostic',
             'reference_readiness_probe', 'diagnostic_debugger')),
         'FIRST_NEW_ACTION_SINGLE_WORLD_WITHOUT_DIAGNOSTIC_BYPASSES')
    need(spec.get('WBC_changes') is False and spec.get('physics_dt_s') == .002,
         'FROZEN_WBC_AND_ORIGINAL_2MS_CLOCK')
    source = Path(spec['source_root'])
    pins = spec.get('source_files', {})
    need(all(name in pins and digest(source/name) == pins[name] for name in SOURCE_MEMBERS),
         'NEW_CAPSULE_NATIVE_ENTRY_AND_ADAPTER_SOURCE_BYTES')
    path = Path(spec['episode_manifest'])
    need(digest(path) == spec['episode_manifest_sha256'], 'EXACT_NEW_EPISODE_BYTES')
    episode = episode_contract(json.loads(path.read_text()), native=True)
    need(episode['run'] == spec['registered_root'] and spec.get('max_ticks') == END_STEP,
         'EXACT_FRESH_RUN_AND_FINITE_4103_ENDPOINT')
    need(spec.get('reference_binding_version') == 'MEASURED_EEF_AND_BASE_POSE_HOLD_V2',
         'ORIGINAL_STANDING_BASE_HOLD_REPAIR')
    catalog_path = Path(spec['scene_allowlist'])
    need(digest(catalog_path) == spec['scene_allowlist_sha256'], 'ORIGINAL_SCENE_CATALOG_BYTES')
    catalog = json.loads(catalog_path.read_text())
    scenes = [s for s in catalog['scenes'] if s['scene_id'] == episode['scene_id']]
    need(len(scenes) == 1, 'ONE_EXISTING_TRAIN_SCENE')
    scene = scenes[0]
    need(scene['split'] == 'TRAIN' and scene['parent'] == episode['parent']
         and scene['model_path'] == spec['model_path']
         and scene.get('static_check') == 'PASSED_NO_INTEGRATION'
         and scene['model_sha256'] == digest(spec['model_path'])
         and catalog['native_binary_sha256'] == digest(spec['native_binary']),
         'UNCHANGED_SCENE_AND_NATIVE_EXECUTOR_BYTES')
    return episode


def episode_limit(spec, previous):
    episode_config(spec)
    need(isinstance(previous, dict) and previous.get('status') == 'NATIVE_STAGE_PASSED'
         and previous.get('shutdown_clean') is True and not previous.get('physics_stops')
         and bool(previous.get('owned_process_exits'))
         and all(p.get('exit_code') == 0 for p in previous['owned_process_exits'])
         and (previous.get('trace_audit') or {}).get('status') == 'RECORDED_TRACE_CHECKS_PASS'
         and previous.get('candidate_index') == spec['candidate_index']
         and previous.get('campaign_uuid') == Path(spec['campaign_root']).name
         and previous.get('actual_target_integrations', 0) >= INIT_STEP
         and previous.get('native_binary_sha256') == digest(spec['native_binary'])
         and previous.get('observer_sha256') == digest(spec['observer_library'])
         and previous.get('scene_sha256') == digest(spec['model_path'])
         and bool(previous.get('reference_qualification')), 'REUSE_ACTUAL_SAME_CANDIDATE_NATIVE_PREDECESSOR_NO_G1_RERUN')
    return END_STEP
