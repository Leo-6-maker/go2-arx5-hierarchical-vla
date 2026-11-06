"""Pure contracts/programs for a fresh, one-direction response identification.

No admission, native execution, model fitting, or qualification is granted here.
The old trajectory-qualification phase and every old dataset remain unchanged.
"""
from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np

from .fusion_trajectory_adapter import (
    ATOM_DT_S, NATIVE_DT_S, NATIVE_TICKS_PER_ATOM,
    TrajectoryAdaptation, WORKING_OPEN_FRACTION, _geometry, _identity,
    _immutable, _rigid, _rows,
)

PHASE = 'collect-target-fusion-trajectory-response-v1'
PANEL_SCHEMA = 'WM02_TRAJECTORY_RESPONSE_PANEL_V1'
EPISODE_SCHEMA = 'WM02_TRAJECTORY_RESPONSE_EPISODE_V1'
RECIPE = 'FULL_H40_X_RESPONSE_CONTEXT_ROUNDTRIP_PULSE_8_16_24_V1'
NOMINAL_RECIPE = 'FULL_H40_RIGID_RELOCATE_SCALE_WORKING_GRIP_INTEGER_RETIME_2X_V2'
CONTEXT_IDS = ('C0', 'C1')
THETA_LEVELS = (-1, 0, 1)
REPEAT_INDICES = (0, 1)
INIT_STEP, DECISION_STEP = 2503, 3703
PREHOLD_ATOMS, TAIL_ATOMS = 48, 8
EPSILON_M = .0005
DIRECTION = np.array([1., 0., 0.])


def need(condition, message):
    if not condition:
        raise ValueError('TRAJECTORY_RESPONSE_' + message)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def _fixed_panel_fields():
    return dict(schema=PANEL_SCHEMA, phase=PHASE,
        purpose='STATE_CONDITIONAL_CORRECTION_IDENTIFICATION', reference_rule=RECIPE,
        nominal_recipe=NOMINAL_RECIPE, time_stretch_factor=2, geometry_scale=1.,
        context_ids=list(CONTEXT_IDS), theta_levels=list(THETA_LEVELS),
        restart_indices=list(REPEAT_INDICES), epsilon_m=EPSILON_M,
        direction_xyz_target_world=DIRECTION.tolist(),
        context_rise_atoms=8, context_return_start_atom=40, context_return_end_atom=48,
        pulse_rise_atoms=8, pulse_hold_until_atom=16, pulse_return_end_atom=24,
        probe_atom=0, history_atoms=8, forecast_atoms=16, expected_episodes=72,
        atom_dt_s=ATOM_DT_S, native_dt_s=NATIVE_DT_S,
        native_ticks_per_atom=NATIVE_TICKS_PER_ATOM,
        init_native_step=INIT_STEP, decision_native_step=DECISION_STEP,
        forecast_conditioning='PREDECLARED_PROGRAM_CONDITIONAL')


def make_panel(*, physical_parents, source_sha, query_pins):
    panel = dict(_fixed_panel_fields(), physical_parents=dict(physical_parents),
                 source_sha=source_sha, query_pins=dict(query_pins))
    return validate_panel(panel)


def validate_panel(panel):
    """The prospective 72-run panel; a 12-run pilot is not a training cohort."""
    exact = _fixed_panel_fields()
    need(isinstance(panel, dict) and all(panel.get(k) == v for k, v in exact.items()),
         'EXACT_NEW_PANEL_RECIPE')
    need(set(panel) == set(exact) | {'physical_parents', 'source_sha', 'query_pins'},
         'EXACT_PANEL_KEYS_NO_HIDDEN_SELECTION')
    parents = panel['physical_parents']
    need(isinstance(parents, dict) and len(parents) == 6
         and all(isinstance(k, str) and k for k in parents)
         and list(parents.values()).count('TRAIN') == 4
         and list(parents.values()).count('DEV') == 2, 'FOUR_TRAIN_TWO_DEV_WHOLE_PARENTS')
    need(_sha(panel['source_sha'], 40), 'FULL_SOURCE_SHA')
    pins = panel['query_pins']
    need(isinstance(pins, dict) and set(pins) == {
        'checkpoint', 'spec', 'result', 'action', 'observation', 'frame_map'}
        and all(_sha(v) for v in pins.values()), 'EXPLICIT_FULL_QUERY_PINS')
    return panel


def validate_episode(episode, panel):
    validate_panel(panel)
    exact = dict(schema=EPISODE_SCHEMA, phase=PHASE,
                 purpose='TRAJECTORY_RESPONSE_IDENTIFICATION', reference_rule=RECIPE,
                 panel_sha256=canonical_sha256(panel), geometry_scale=1., query_pins=panel['query_pins'])
    keys = {'run', 'scene_id', 'parent', 'split', 'context_id', 'theta_level', 'restart_index', 'planned_end_atom'}
    need(isinstance(episode, dict) and set(episode) == set(exact) | keys
         and all(episode.get(k) == v for k, v in exact.items()), 'EXACT_EPISODE_MANIFEST')
    need(isinstance(episode['run'], str) and bool(episode['run'])
         and isinstance(episode['scene_id'], str) and bool(episode['scene_id'])
         and episode['parent'] in panel['physical_parents']
         and episode['split'] == panel['physical_parents'][episode['parent']], 'REGISTERED_RUN_WHOLE_PARENT_SPLIT')
    need(episode['context_id'] in CONTEXT_IDS
         and type(episode['theta_level']) is int and episode['theta_level'] in THETA_LEVELS
         and type(episode['restart_index']) is int and episode['restart_index'] in REPEAT_INDICES,
         'ONE_DECLARED_CONTEXT_LEVEL_RESTART_CELL')
    need(type(episode['planned_end_atom']) is int and episode['planned_end_atom'] >= 40,
         'FINITE_COMPLETE_COMMON_PROGRAM_ENDPOINT')
    return episode


def _quintic(u):
    return u ** 3 * (10 + u * (-15 + 6 * u))


def pulse_basis(atoms):
    """Same clock in all arms: rise8, hold to16, return to zero by24."""
    value = np.asarray(atoms, dtype=np.float64)
    need(np.isfinite(value).all() and np.all(value >= 0), 'FINITE_NONNEGATIVE_PROGRAM_ATOMS')
    rise = _quintic(np.minimum(value / 8., 1.))
    fall = 1. - _quintic(np.minimum(np.maximum((value - 16.) / 8., 0.), 1.))
    return rise * fall


def _preview(path, elapsed):
    need(type(elapsed) is int and elapsed >= 0, 'NONNEGATIVE_PYTHON_INTEGER_CLOCK')
    start = min(elapsed, len(path) - 1)
    return _immutable(path[np.minimum(start + np.arange(40), len(path) - 1)])


@dataclass(frozen=True, eq=False)
class ContextProgram:
    initial_tcp: object
    base_pose_hold: object
    context_id: str

    def __post_init__(self):
        need(self.context_id in CONTEXT_IDS, 'DECLARED_CONTEXT')
        initial = _rigid(self.initial_tcp, 'initial_tcp')
        base = np.asarray(self.base_pose_hold, dtype=np.float64)
        need(base.shape == (7,) and np.isfinite(base).all()
             and abs(np.linalg.norm(base[3:]) - 1) <= 1e-6, 'MEASURED_FIXED_BASE7')
        path = _rows(np.repeat(initial[None], PREHOLD_ATOMS + 1, axis=0),
                     np.full(PREHOLD_ATOMS + 1, WORKING_OPEN_FRACTION))
        if self.context_id == 'C1':
            t = np.arange(PREHOLD_ATOMS + 1, dtype=float)
            progress = _quintic(np.minimum(t / 8., 1.))
            progress *= 1. - _quintic(np.maximum((t - 40.) / 8., 0.))
            path[:, :3] += EPSILON_M * progress[:, None] * DIRECTION
        object.__setattr__(self, 'initial_tcp', initial)
        object.__setattr__(self, 'base_pose_hold', _immutable(base))
        object.__setattr__(self, 'reference_path', _immutable(path))

    @property
    def end_TCP(self):
        return self.initial_tcp

    @property
    def reference_anchor_TCP(self):
        return self.initial_tcp

    def preview(self, global_atom=0):
        return _preview(self.reference_path, global_atom)

    def evidence(self):
        return dict(schema='WM02_TRAJECTORY_RESPONSE_CONTEXT_V1', context_id=self.context_id,
                    reference_path=_identity(self.reference_path), initial_TCP=self.initial_tcp.tolist(),
                    planned_decision_reference_TCP=self.initial_tcp.tolist(),
                    actual_decision_TCP='MUST_BE_MEASURED_SEPARATELY',
                    rise_atoms=8, return_start_atom=40, return_end_atom=48,
                    peak_offset_xyz_m=(EPSILON_M * DIRECTION).tolist() if self.context_id == 'C1' else [0., 0., 0.],
                    absolute_native_clock_origin=INIT_STEP, theta_published=False,
                    native_execution_authorized=False)


@dataclass(frozen=True, eq=False)
class ResponseProgram:
    base_program: TrajectoryAdaptation
    theta_level: int
    panel: dict

    def __post_init__(self):
        validate_panel(self.panel)
        need(isinstance(self.base_program, TrajectoryAdaptation)
             and self.base_program.geometry_scale == 1.
             and getattr(self.base_program, 'time_stretch_factor', None) == 2,
             'FULL_GEOMETRY_EXPLICIT_2X_V2_BASE_PROGRAM')
        need(type(self.theta_level) is int and self.theta_level in THETA_LEVELS, 'DECLARED_SIGNED_THETA')
        panel = json.loads(json.dumps(self.panel))
        object.__setattr__(self, 'panel', panel)
        nominal = self.base_program.reference_path
        basis = pulse_basis(np.arange(len(nominal)))
        corrected = nominal.copy()
        corrected[:, :3] += self.theta_level * EPSILON_M * basis[:, None] * DIRECTION
        object.__setattr__(self, 'correction_basis', _immutable(basis))
        object.__setattr__(self, 'reference_path', _immutable(corrected))

    @property
    def nominal_reference_path(self):
        return self.base_program.reference_path

    @property
    def desired_path(self):
        return self.base_program.desired_path

    @property
    def initial_reference_TCP_target(self):
        return self.base_program.initial_reference_TCP_target

    @property
    def base_pose_hold(self):
        return self.base_program.base_pose_hold

    @property
    def end_atom(self):
        return self.base_program.end_atom

    @property
    def knot_indices(self):
        return self.base_program.knot_indices

    def preview(self, elapsed_atoms=0):
        return _preview(self.reference_path, elapsed_atoms)

    def nominal_preview(self, elapsed_atoms=0):
        return self.base_program.preview(elapsed_atoms)

    def desired_at(self, elapsed_atoms):
        return self.base_program.desired_at(elapsed_atoms)

    def evidence(self):
        bounds = {}
        for theta in THETA_LEVELS:
            candidate = self.nominal_reference_path.copy()
            candidate[:, :3] += theta * EPSILON_M * self.correction_basis[:, None] * DIRECTION
            bounds[str(theta)] = _geometry(candidate, self.initial_reference_TCP_target)
        return dict(schema='WM02_TRAJECTORY_RESPONSE_PROGRAM_V1', reference_rule=RECIPE,
                    panel_sha256=canonical_sha256(self.panel), theta_level=self.theta_level,
                    theta_amplitude_m=self.theta_level * EPSILON_M,
                    direction_xyz_target_world=DIRECTION.tolist(),
                    nominal_adaptation=self.base_program.evidence(),
                    nominal_reference_path=_identity(self.nominal_reference_path),
                    corrected_reference_path=_identity(self.reference_path),
                    desired_path=_identity(self.desired_path),
                    correction_basis=_identity(self.correction_basis),
                    pulse_rise_atoms=8, pulse_hold_until_atom=16, pulse_return_end_atom=24,
                    end_atom=self.end_atom, knot_indices=list(self.knot_indices),
                    same_clock_for_all_theta=True, theta_specific_retiming=False,
                    correction_changes_desired_outcome=False,
                    whole_candidate_path_diagnostics=bounds,
                    original_full_H40_native_certifier_still_required=True,
                    qualification='NOT_ESTABLISHED_BY_PURE_PROGRAM', native_execution_authorized=False)
