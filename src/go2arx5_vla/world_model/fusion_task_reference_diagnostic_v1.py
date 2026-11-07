"""Finite privileged task-intent diagnostic; never a GR00T/WM action label.

This only constructs an explicit 5mm noncontact geometric reference. Original
native certificates, consumption and the unchanged measured task score remain
necessary. No executor, physical owner, GPU, training or source substitution.
"""
import json
from pathlib import Path
import numpy as np
from ..trajectory.so3 import rotation_matrix_to_rot6d_rows
from .fusion_trajectory_adapter import TrajectoryAdaptation
from .fusion_bounded_reference_tracking_v1 import screen
from .fusion_graded_short_skill_v1 import screen_proposal

SCHEMA='WM02_PRIVILEGED_TASK_REFERENCE_DIAGNOSTIC_V1'
PRODUCER='PRIVILEGED_CURRENT_SCENE_GEOMETRY_NOT_GR00T_NOT_WM'
DISTANCE_M=.005
RAW_KNOTS=40
TASK_CONDITION='PRIVILEGED_GEOMETRIC_SHORT_REFERENCE_DIAGNOSTIC_AT34303_V1'
CONFIG=dict(schema=SCHEMA,producer=PRODUCER,step_m=DISTANCE_M,
            policy='ALWAYS_GEOMETRIC_REFERENCE_NOT_REJECTION_FALLBACK',
            data_role='DIAGNOSTIC_ONLY_NO_AUTO_TRAIN',GR00T_proposal='GENERATE_SAVE_BUT_DO_NOT_EXECUTE')
SHORT_TASK_CONDITION='PRIVILEGED_GEOMETRIC_SHORT_DURATION_DIAGNOSTIC_AT34303_V2'
SHORT_SUFFIX_ATOMS=176
SHORT_BEGIN=34303
SHORT_CONFIG=dict(CONFIG,schema='WM02_PRIVILEGED_TASK_REFERENCE_DIAGNOSTIC_V2',
    suffix_atoms=SHORT_SUFFIX_ATOMS,standing_guard='UNCHANGED_ORIGINAL_SHORT_TASK_BASE_POSE_BOUNDS')


def endpoint(condition):
    return SHORT_BEGIN+25*SHORT_SUFFIX_ATOMS if condition==SHORT_TASK_CONDITION else 66503


def require_standing(sample,initial):
    """New short diagnostic stops at the original task's standing bounds."""
    from .fusion_graded_short_skill_v1 import score_contract
    contract=score_contract();base=np.asarray(sample['qpos'][:7],float);before=np.asarray(initial['qpos'][:7],float)
    if (base.shape!=(7,) or before.shape!=(7,) or not np.isfinite(base).all()or not np.isfinite(before).all()
            or abs(np.linalg.norm(base[3:])-1)>1e-6 or abs(np.linalg.norm(before[3:])-1)>1e-6):
        raise ValueError('TASK_REFERENCE_ACTUAL_FINITE_UNIT_BASE_POSES_REQUIRED')
    displacement=float(np.linalg.norm(base[:3]-before[:3]))
    angle=float(2*np.arccos(np.clip(abs(np.dot(base[3:],before[3:])),0.,1.)))
    if displacement>contract['maximum_base_displacement_m'] or angle>contract['maximum_base_rotation_rad']:
        raise ValueError('TASK_REFERENCE_ORIGINAL_STANDING_BOUND_EXCEEDED:'+repr(dict(native_step=sample['native_step'],base_displacement_m=displacement,base_rotation_rad=angle)))


def validate_registration(spec):
    """Bind this diagnostic to a distinct finite panel; supply no site grant."""
    config=spec.get('task_reference_diagnostic')
    condition=spec.get('task_condition')
    if config is None and condition not in(TASK_CONDITION,SHORT_TASK_CONDITION):return None
    from .fusion_live_program_replay_v1 import assignment
    from .storage import digest
    a=assignment(spec)
    expected=SHORT_CONFIG if condition==SHORT_TASK_CONDITION else CONFIG
    if (config!=expected or condition not in(TASK_CONDITION,SHORT_TASK_CONDITION)
            or spec.get('phase')!='collect-target-fusion-task-phase-live-skill-v9'
            or spec.get('kind')!='FUSION_TASK_PHASE_LIVE_SKILL_V9'
            or spec.get('live_skill_geometry_scale')!=1.
            or spec.get('live_skill_time_stretch_factor')!=2
            or a['split']!='DIAGNOSTIC'
            or any(spec.get(k)is not None for k in ('program_response_forecast','program_response_counterfactual','program_reference_selection','program_reference_intervention'))):
        raise ValueError('EXPLICIT_DIAGNOSTIC_NOT_VLA_TRANSFER_OR_XY_MODEL_SCOPE')
    if condition==SHORT_TASK_CONDITION and not(spec.get('live_skill_start_step')==SHORT_BEGIN and spec.get('max_ticks')==endpoint(condition)):
        raise ValueError('EXACT_PREREGISTERED_COMPLETE_SHORT_TEACHER_HORIZON')
    path=Path(spec['response_learning_panel'])
    if path.is_symlink() or digest(path)!=a['panel_sha256'] or digest(path)!=spec['response_learning_panel_sha256']:
        raise ValueError('EXACT_PRE_EFFECT_TASK_REFERENCE_DIAGNOSTIC_PANEL')
    rows=[r for r in json.loads(path.read_text())['assignments'] if r.get('native_run')==Path(spec['registered_root']).name]
    if len(rows)!=1 or rows[0].get('parent')!=a['parent'] or rows[0].get('split')!='DIAGNOSTIC' or rows[0].get('task_reference_diagnostic')!=expected:
        raise ValueError('ORIGINAL_DISTINCT_TASK_REFERENCE_DIAGNOSTIC_ROW')
    return config


def build(*,actual_TCP,initial_reference_TCP,base_pose_hold,actual_object_pose):
    """Use one actual decision-time scene observation, with no future states.

    The fixed magnitude is a separately declared Teacher intent, not clipping
    a VLA proposal or extrapolating the local XY response model. Deployable
    policy equivalence and dynamic feasibility are deliberately not supplied.
    """
    tcp=np.asarray(actual_TCP,float);obj=np.asarray(actual_object_pose,float)
    if (tcp.shape!=(4,4) or obj.shape!=(4,4) or not np.isfinite(tcp).all()
            or not np.isfinite(obj).all()):
        raise ValueError('TASK_REFERENCE_ACTUAL_FINITE_TCP_AND_OBJECT_POSES')
    delta=obj[:3,3]-tcp[:3,3];distance=float(np.linalg.norm(delta))
    if distance<=.14+DISTANCE_M:
        raise ValueError('TASK_REFERENCE_ORIGINAL_NONCONTACT_ENVELOPE_NO_SHORTENING')
    direction=delta/distance
    u=np.linspace(0.,1.,RAW_KNOTS);weight=u**3*(10+u*(-15+6*u))
    raw=np.empty((RAW_KNOTS,10));raw[:,:3]=tcp[:3,3]+weight[:,None]*DISTANCE_M*direction
    raw[:,3:9]=rotation_matrix_to_rot6d_rows(tcp[:3,:3]);raw[:,9]=1.
    program=TrajectoryAdaptation(raw,tcp,np.eye(4),tcp,initial_reference_TCP,
                                 base_pose_hold,1.,time_stretch_factor=2)
    domain=screen(program,obj);intent=screen_proposal(program,obj)
    if not domain['eligible_for_original_native_certificate'] or not intent['qualified']:
        raise ValueError('TASK_REFERENCE_FIXED_WHOLE_PROGRAM_FAILS_ORIGINAL_DOMAIN_OR_INTENT')
    return program,dict(schema=SCHEMA,producer=PRODUCER,privileged_geometry_use='DIAGNOSTIC_AND_EXPLICIT_TEACHER_ONLY',
        desired_step_m=DISTANCE_M,desired_direction_target_world=direction,
        actual_decision_TCP=tcp,actual_decision_object_pose=obj,
        original_physical_H40=raw,adaptation=program.evidence(),
        original_domain_screen=domain,original_short_intent_screen=intent,
        GR00T_queries=0,Processor_decodes=0,WM_evaluations=0,
        future_achieved_state_labels=False,actual_consumed_commands=0,
        expert_training_eligibility='REQUIRES_REAL_CONSUMPTION_AND_SEPARATE_TASK_OUTCOME',
        native_execution_authorized=False,dynamic_qualification=False,
        source_or_target_task_transfer=False)
