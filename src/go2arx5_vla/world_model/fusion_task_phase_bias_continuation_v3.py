"""Graded continuation of the qualified simple predecision-bias initialization.

This pure recipe supplies no admission. Every fresh restart measures its own
INIT/DEC bias; the contract separately gates adjacent grades by actual evidence.
"""
import copy
from dataclasses import dataclass

from scripts.pi0292.relay_protocol import digest as document_hash
from . import fusion_task_phase_initialization as original
from . import fusion_task_phase_bias_calibration_v2 as fixed

PHASE = 'collect-target-fusion-task-phase-bias-continuation-v3'
SCHEMA = 'WM02_FUSION_TASK_PHASE_BIAS_CONTINUATION_EPISODE_V3'
RECIPE = 'FROZEN_PREDECISION_XYZ_SO3_BIAS_REFERENCE_GRADED_QUINTIC_V3'
CONDITION = 'INIT_FROZEN_PREDECISION_BIAS_GRADED_V3'
BASELINE_CONDITION = fixed.BASELINE_CONDITION
SCALES = (.1, .25, .5, 1.)
DECISION_STEP, INIT_STEP = original.DECISION_STEP, original.INIT_STEP
TICKS_PER_ATOM, TAIL_ATOMS = original.TICKS_PER_ATOM, original.TAIL_ATOMS
load_inputs = original.load_inputs


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_BIAS_CONTINUATION_V3_' + reason)


def _fixed():
    return dict(fixed._fixed(), schema=SCHEMA, phase=PHASE, reference_rule=RECIPE,
        initialization_condition=CONDITION, continuation='ADJACENT_ACTUAL_QUALIFIED_GRADES_ONLY',
        predecessor_non_full_grade_readiness_required=True)


def make_episode(*, run, geometry_scale):
    need(type(geometry_scale) in (int, float) and geometry_scale in SCALES,
         'ONLY_REGISTERED_POINT1_POINT25_POINT5_FULL1')
    return episode_contract(dict(original.make_episode(run=run, geometry_scale=geometry_scale), **_fixed()))


def episode_contract(episode):
    fixed_fields = _fixed()
    need(isinstance(episode, dict) and set(episode) == set(fixed_fields) | {'run', 'geometry_scale', 'ramp_atoms', 'planned_end_atom'}
         and all(document_hash(episode[key]) == document_hash(value) for key, value in fixed_fields.items()),
         'EXACT_ACTUAL_V3_MANIFEST')
    need(type(episode['geometry_scale']) in (int, float) and episode['geometry_scale'] in SCALES,
         'ONLY_REGISTERED_CONTINUATION_GRADE')
    original.episode_contract(parent_recipe_view(episode, checked=True))
    return episode


def parent_recipe_view(episode, *, checked=False):
    """Original desired-intent math only; never relabel an execution manifest."""
    if not checked:
        episode_contract(episode)
    return original.episode_contract(dict(original._fixed(), **{key: copy.deepcopy(episode[key])
        for key in ('run', 'geometry_scale', 'ramp_atoms', 'planned_end_atom')}))


def endpoint(episode):
    episode_contract(episode)
    return original.endpoint(parent_recipe_view(episode, checked=True))


@dataclass(frozen=True, init=False, eq=False)
class GradedFrozenBiasProgram(fixed.FrozenBiasProgram):
    def __init__(self, inputs, episode, initial_tcp, decision_tcp, base):
        self._initialize(inputs, episode, initial_tcp, decision_tcp, base,
            episode_validator=episode_contract, ancestor_view=parent_recipe_view,
            schema=SCHEMA, recipe_name=RECIPE, condition=CONDITION,
            binding_schema='WM02_FROZEN_PREDECISION_REFERENCE_BINDING_V3')


def make_program(inputs, episode, initial_tcp, decision_tcp, base):
    return GradedFrozenBiasProgram(inputs, episode, initial_tcp, decision_tcp, base)


def validate_install_preview(action, *, initial_tcp, program, step, base, frozen_base, end_step):
    return fixed._validate_install_preview(action, initial_tcp=initial_tcp, program=program, step=step,
        base=base, frozen_base=frozen_base, end_step=end_step, program_type=GradedFrozenBiasProgram,
        allowed_endpoints=(10303, 16703, 23103, 29503),
        status='EXACT_GRADED_FROZEN_BIAS_SUFFIX_NOT_NATIVE_QUALIFIED')


def query_readiness(samples, episode, inputs):
    return original.query_readiness(samples, parent_recipe_view(episode), inputs)
