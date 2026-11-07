"""Adjacent graded continuation of the same finite ordinary XYZ feedback law."""
from dataclasses import dataclass

from scripts.pi0292.relay_protocol import digest as document_hash
from . import fusion_task_phase_initialization as original
from . import fusion_task_phase_xyz_feedback_v4 as previous

PHASE = 'collect-target-fusion-task-phase-xyz-feedback-continuation-v5'
SCHEMA = 'WM02_FUSION_TASK_PHASE_XYZ_FEEDBACK_CONTINUATION_EPISODE_V5'
RECIPE = 'FROZEN_BIAS_FINITE_XYZ_FEEDBACK_ADJACENT_GRADED_QUINTIC_V5'
CONDITION = 'INIT_FROZEN_BIAS_ORDINARY_XYZ_FEEDBACK_GRADED_V5'
BASELINE_CONDITION = previous.CONDITION
SCALES = (.25, .5, 1.)
INIT_STEP, DECISION_STEP = original.INIT_STEP, original.DECISION_STEP
TICKS_PER_ATOM, TAIL_ATOMS = 25, 8
GAIN, STEP_CAP_M, DEADBAND_M, CUMULATIVE_CAP_M = .5, .004, .005, .016
MOVE_ATOMS, SETTLE_ATOMS = 20, 8
load_inputs, fixed = original.load_inputs, previous.fixed


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_XYZ_FEEDBACK_CONTINUATION_V5_' + reason)


def _fixed():
    fields = dict(previous._fixed(), schema=SCHEMA, phase=PHASE, reference_rule=RECIPE,
        initialization_condition=CONDITION, baseline_condition=BASELINE_CONDITION,
        continuation='ADJACENT_COMPLETE_NATIVE_AND_NINE_MEASURED_PREDICATES_ONLY')
    del fields['feedback_decision_native_steps']
    fields['feedback_goal'] = 'UNCHANGED_CURRENT_RESTART_REGISTERED_GRADED_DESIRED_ENDPOINT'
    return fields


def feedback_steps(episode):
    start = DECISION_STEP+25*(episode['ramp_atoms']+8)
    return tuple(start+700*i for i in range(4))


def make_episode(*, run, geometry_scale):
    need(type(geometry_scale) in (int, float) and geometry_scale in SCALES,
         'ONLY_POINT25_POINT5_FULL1_NO_REPEAT_POINT1')
    ancestor = original.make_episode(run=run, geometry_scale=geometry_scale)
    return episode_contract(dict(ancestor, **_fixed(),
        feedback_decision_native_steps=list(feedback_steps(ancestor)),
        planned_end_atom=ancestor['ramp_atoms']+112))


def episode_contract(episode):
    fields = _fixed()
    need(isinstance(episode, dict) and set(episode) == set(fields) | {
        'run','geometry_scale','ramp_atoms','planned_end_atom','feedback_decision_native_steps'}
        and all(document_hash(episode[key]) == document_hash(value) for key,value in fields.items()),
        'EXACT_ACTUAL_V5_MANIFEST_SAME_FIXED_FEEDBACK_LAW')
    need(type(episode['geometry_scale']) in (int,float) and episode['geometry_scale'] in SCALES
         and type(episode['ramp_atoms']) is int
         and episode['ramp_atoms'] == original.RAMPS[episode['geometry_scale']]
         and type(episode['planned_end_atom']) is int
         and episode['planned_end_atom'] == episode['ramp_atoms']+112
         and episode['feedback_decision_native_steps'] == list(feedback_steps(episode)),
         'ORIGINAL_GRADED_RAMP_PLUS_EXACT_FOUR_FINITE_SEGMENTS')
    original.episode_contract(parent_recipe_view(episode, checked=True))
    need(DECISION_STEP+25*(episode['planned_end_atom']+8) <= 33000,
         'ORIGINAL_FINITE_ENDPOINT_AT_MOST_33000')
    return episode


def parent_recipe_view(episode, *, checked=False):
    """Original desired math only; actual V5 manifests/specs stay V5."""
    if not checked:
        episode_contract(episode)
    return original.make_episode(run=episode['run'], geometry_scale=episode['geometry_scale'])


def endpoint(episode):
    episode_contract(episode)
    return DECISION_STEP+25*(episode['planned_end_atom']+8)


@dataclass(frozen=True, init=False, eq=False)
class FeedbackProgram(previous.FeedbackProgram):
    def __init__(self, inputs, episode, initial_tcp, decision_tcp, base):
        self._initialize_feedback(inputs, episode, initial_tcp, decision_tcp, base,
            episode_validator=episode_contract, ancestor_view=parent_recipe_view,
            schema=SCHEMA, recipe_name=RECIPE, condition=CONDITION, baseline=BASELINE_CONDITION,
            binding_schema='WM02_FROZEN_PREDECISION_REFERENCE_BINDING_V5',
            decision_schema='WM02_FINITE_CAUSAL_XYZ_FEEDBACK_DECISION_V5',
            final_step=endpoint(episode), final_move_atom=episode['planned_end_atom'],
            ramp_atoms=episode['ramp_atoms'], feedback_steps=feedback_steps(episode))


def make_program(inputs, episode, initial_tcp, decision_tcp, base):
    return FeedbackProgram(inputs, episode, initial_tcp, decision_tcp, base)


def validate_install_preview(action, *, initial_tcp, program, step, base, frozen_base, end_step):
    if step >= DECISION_STEP:
        need(type(program) is FeedbackProgram, 'EXACT_NEW_TYPED_GRADED_FEEDBACK_PROGRAM')
        program.require_decisions_through(step)
    return fixed._validate_install_preview(action, initial_tcp=initial_tcp, program=program,
        step=step, base=base, frozen_base=frozen_base, end_step=end_step,
        program_type=FeedbackProgram, allowed_endpoints=(19503,25903,32303),
        status='EXACT_GRADED_FINITE_XYZ_FEEDBACK_SUFFIX_NOT_NATIVE_QUALIFIED')


def query_readiness(samples, episode, inputs):
    episode_contract(episode)
    value = original._query_readiness_at_endpoint(samples, parent_recipe_view(episode), inputs, end=endpoint(episode))
    value.update(schema='WM02_FUSION_TASK_PHASE_XYZ_FEEDBACK_READINESS_V5',
        actual_episode_schema=SCHEMA, ordinary_feedback_not_model_benefit=True)
    return value
