"""Finite larger-grade ordinary pose feedback; no learned response extrapolation."""
from dataclasses import dataclass
from . import fusion_task_phase_pose_feedback_v6 as previous

original,engine,fixed = previous.original,previous.engine,previous.fixed
load_inputs = previous.load_inputs
PHASE='collect-target-fusion-task-phase-pose-continuation-v7'
SCHEMA='WM02_FUSION_TASK_PHASE_POSE_CONTINUATION_EPISODE_V7'
RECIPE='FINITE_GRADED_WORLD_XYZ_LEFT_SO3_CONTINUATION_V7'
CONDITION='INIT_FROZEN_BIAS_FINITE_XYZ_SO3_CONTINUATION_V7'
BASELINE_CONDITION=previous.CONDITION
SCALES=(.5,1.)
SLOTS={.5:16,1.:24}
INIT_STEP,DECISION_STEP=2503,3703
TICKS_PER_ATOM,TAIL_ATOMS,MOVE_ATOMS,SETTLE_ATOMS=25,8,20,8
GAIN,STEP_CAP_M,DEADBAND_M,CUMULATIVE_CAP_M=.5,.008,.005,.192
ANGULAR_GAIN,ANGULAR_STEP_CAP_RAD,ANGULAR_DEADBAND_RAD,ANGULAR_CUMULATIVE_CAP_RAD=.5,.01,.005,.24
MAX_ENDPOINT=46303


def need(ok,why):
    if not ok:raise ValueError('TASK_PHASE_POSE_CONTINUATION_V7_'+why)


def _fixed(grade):
    count=SLOTS[grade]
    return dict(previous._fixed(),schema=SCHEMA,phase=PHASE,reference_rule=RECIPE,
        initialization_condition=CONDITION,baseline_condition=BASELINE_CONDITION,
        feedback_step_norm_cap_m=STEP_CAP_M,feedback_cumulative_path_length_cap_m=count*STEP_CAP_M,
        feedback_angular_cumulative_path_length_cap_rad=count*ANGULAR_STEP_CAP_RAD,
        finite_feedback_decisions=count,finite_native_endpoint_maximum=MAX_ENDPOINT,
        continuation='EXACT_QUALIFIED_V6_POINT25_THEN_ADJACENT_SAME_V7_SOURCE',
        response_gain_assumed_from_point25=False,
        body_motion_is_part_of_actual_feedback=True)


def feedback_steps(episode):
    start=DECISION_STEP+25*(episode['ramp_atoms']+8)
    return tuple(start+700*i for i in range(SLOTS[episode['geometry_scale']]))


def make_episode(*,run,geometry_scale):
    need(type(geometry_scale)in(int,float) and geometry_scale in SCALES,'ONLY_POINT5_OR_FULL1')
    ancestor=original.make_episode(run=run,geometry_scale=geometry_scale)
    return episode_contract(dict(ancestor,**_fixed(geometry_scale),
        feedback_decision_native_steps=list(feedback_steps(ancestor)),
        planned_end_atom=ancestor['ramp_atoms']+SLOTS[geometry_scale]*28))


def episode_contract(episode):
    need(isinstance(episode,dict) and type(episode.get('geometry_scale'))in(int,float)
         and episode['geometry_scale']in SCALES,'EXACT_SUPPORTED_GRADE')
    grade=episode['geometry_scale'];fields=_fixed(grade)
    need(set(episode)==set(fields)|{'run','geometry_scale','ramp_atoms','planned_end_atom','feedback_decision_native_steps'}
         and all(previous.document_hash(episode[k])==previous.document_hash(v)for k,v in fields.items()),
         'EXACT_VERSIONED_FINITE_LAW')
    need(type(episode['ramp_atoms'])is int and episode['ramp_atoms']==original.RAMPS[grade]
         and type(episode['planned_end_atom'])is int
         and episode['planned_end_atom']==episode['ramp_atoms']+SLOTS[grade]*28
         and episode['feedback_decision_native_steps']==list(feedback_steps(episode)),
         'ORIGINAL_RAMP_AND_PREDECLARED_FINITE_SLOTS')
    original.episode_contract(parent_recipe_view(episode,checked=True))
    return episode


def parent_recipe_view(episode,*,checked=False):
    if not checked:episode_contract(episode)
    return original.make_episode(run=episode['run'],geometry_scale=episode['geometry_scale'])


def endpoint(episode):
    episode_contract(episode)
    return DECISION_STEP+25*(episode['planned_end_atom']+8)


@dataclass(frozen=True,init=False,eq=False)
class FeedbackProgram(previous.FeedbackProgram):
    def __init__(self,inputs,episode,initial_tcp,decision_tcp,base):
        self._initialize_feedback(inputs,episode,initial_tcp,decision_tcp,base,
            episode_validator=episode_contract,ancestor_view=parent_recipe_view,
            schema=SCHEMA,recipe_name=RECIPE,condition=CONDITION,baseline=BASELINE_CONDITION,
            binding_schema='WM02_FROZEN_PREDECISION_REFERENCE_BINDING_V7',
            decision_schema='WM02_FINITE_CAUSAL_POSE_FEEDBACK_DECISION_V7',
            final_step=endpoint(episode),final_move_atom=episode['planned_end_atom'],
            ramp_atoms=episode['ramp_atoms'],feedback_steps=feedback_steps(episode))
        object.__setattr__(self,'_cumulative_feedback_rad',0.)

    def _law_parameters(self):
        count=len(self._feedback_steps)
        return (GAIN,STEP_CAP_M,DEADBAND_M,count*STEP_CAP_M,
                ANGULAR_GAIN,ANGULAR_STEP_CAP_RAD,ANGULAR_DEADBAND_RAD,count*ANGULAR_STEP_CAP_RAD)


def make_program(inputs,episode,initial_tcp,decision_tcp,base):
    return FeedbackProgram(inputs,episode,initial_tcp,decision_tcp,base)


def validate_install_preview(action,*,initial_tcp,program,step,base,frozen_base,end_step):
    if step>=DECISION_STEP:
        need(type(program)is FeedbackProgram,'EXACT_TYPED_NEW_PROGRAM')
        program.require_decisions_through(step)
    return fixed._validate_install_preview(action,initial_tcp=initial_tcp,program=program,step=step,
        base=base,frozen_base=frozen_base,end_step=end_step,program_type=FeedbackProgram,
        allowed_endpoints=(34303,46303),status='EXACT_FINITE_GRADED_POSE_SUFFIX_NOT_NATIVE_QUALIFIED')


def query_readiness(samples,episode,inputs):
    episode_contract(episode)
    value=original._query_readiness_at_endpoint(samples,parent_recipe_view(episode),inputs,
        end=endpoint(episode),maximum_endpoint=MAX_ENDPOINT)
    value.update(schema='WM02_FUSION_TASK_PHASE_POSE_CONTINUATION_READINESS_V7',actual_episode_schema=SCHEMA,
        ordinary_feedback_not_model_benefit=True)
    return value
