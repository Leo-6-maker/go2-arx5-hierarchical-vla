"""Read-only metrics after the original DIRECT report-serialization failure.

The caller authenticates original CLOSED specs/artifacts. Original failure is
never upgraded. Missing per-sample safety flags are not invented: only original
geometry metrics/predicates are exposed, with safety evidence kept separate.
No dynamics, FK, model inference, training, action or state restore occurs.
"""
import numpy as np
from . import fusion_graded_short_skill_v1 as short

SOURCE='93ad6aab3a251c55f11c372bf9e86f1c64c453aa'
B,E=34303,66503


def _need(ok,why):
    if not ok:raise ValueError('RECOVER_ORIGINAL_DIRECT_'+why)


def _pose(values):
    value=np.asarray(values,float);pose=np.eye(4)
    pose[:3,3]=value[:3];pose[:3,0]=value[3:6];pose[:3,1]=value[6:9]
    pose[:3,2]=np.cross(pose[:3,0],pose[:3,1])
    return pose


def recover(spec,prefix,decision,segment,native):
    """Preserve the native FAIL; derive only retained, observed quantities."""
    _need(spec['source_sha']==native['source_sha']==SOURCE
          and spec['program_reference_selection']['method']=='DIRECT'
          and native['status']=='NATIVE_STAGE_FAILED_PRESERVED'
          and str(native.get('failure','')).strip().endswith('TypeError: Object of type ndarray is not JSON serializable'),
          'EXACT_RECORDED_SERIALIZATION_FAILURE_NOT_ARBITRARY_RUN_RESCORING')
    audit=native.get('trace_audit')or{}
    _need(native['actual_target_integrations']==E and native['complete_atomic_transitions']==2660
          and native['shutdown_clean']is True and native.get('trace_audit_error')is None
          and native['original_joint_or_warning_samples']==native['native_outer_guard_rejection_count']==0
          and not native.get('physics_stops')
          and len(native['owned_process_exits'])==4 and all(x['exit_code']==0 for x in native['owned_process_exits'])
          and audit.get('status')=='RECORDED_TRACE_CHECKS_PASS' and not audit.get('errors')
          and audit.get('writer',{}).get('dropped')==0,'ACTUAL_NATIVE_TRACE_AND_CONTROLLED_SHUTDOWN')
    rows=segment['windows'];atoms=(E-B)//25
    _need(spec['max_ticks']==decision['end_step']==segment['native_endpoint_step']==E
          and spec['live_skill_start_step']==decision['start_step']==segment['start_step']==B
          and prefix['native_endpoint_step']==B and segment['all_40_knots_consumed']is True
          and segment['raw_knot_consumed']==[True]*40
          and len(rows)==segment['complete_suffix_atoms']==segment['solver_target_confirmed_atoms']==segment['gripper_confirmed_atoms']==atoms,
          'COMPLETE_ORIGINAL_SUFFIX_NOT_A_FUTURE_OR_REPLAYED_STATE')
    first=np.asarray(decision['actual_history']['actual_state_history'][-1],float)
    samples=[];previous=first
    def sample(vector,step,tcp=None):
        _need(vector.shape==(106,) and np.isfinite(vector).all(),'ORIGINAL_106_STATE_ABI')
        return dict(native_step=step,time_s=step*.002,tcp=_pose(vector[86:95])if tcp is None else np.asarray(tcp,float),
            object_pose=_pose(vector[95:104]),qpos=vector[:34],finger_q=vector[104:106],
            safety_valid=None,native_abort=None,robot_object_contact=None)
    samples.append(sample(first,B,prefix['windows'][-1]['actual_TCP']))
    tracking=[]
    for i,row in enumerate(rows):
        current=np.asarray(row['actual_state_vector'],float)
        _need(row['native_start_step']==B+25*i and row['native_end_step']==B+25*(i+1)
              and row['actual_ticks']==25 and row['stage']=='ACTUAL_CONSUMPTION_CONFIRMED'
              and row['gripper']['command_consumed']is True and bool(row['solver_policies_with_exact_installed_target'])
              and np.array_equal(row['actual_history_before_atom']['actual_state_history'][-1],previous)
              and np.array_equal(current[86:89],np.asarray(row['actual_TCP'])[:3,3])
              and np.array_equal(current[104:106],row['gripper']['actual_finger_q_m']),
              'CONTIGUOUS_ACTUAL_STATE_TCP_FINGER_AND_CONSUMPTION_JOIN')
        samples.append(sample(current,row['native_end_step'],row['actual_TCP']));previous=current
        tracking.append(float(np.linalg.norm(current[86:89]-np.asarray(row['desired_interval_end_EEF10'])[:3])))
    # None marks unavailable original booleans. Only geometric results of the
    # unchanged scorer are retained; its safety/overall verdict is not reused.
    scored=short.score(samples,short.score_contract())
    predicates={k:v for k,v in scored['predicates'].items()if k not in('original_native_safety','no_robot_object_contact')}
    fail=any(v is False for v in predicates.values())
    geometry=dict(schema='WM02_RECOVERED_ORIGINAL_SHORT_GEOMETRY_ONLY_V1',metrics=scored['metrics'],predicates=predicates,
        approach_step_success=False if fail else None,geometry_failure_proves_task_failure=fail,
        original_complete_strict_score_recovered=False,unavailable_sample_flags=['safety_valid','native_abort','robot_object_contact'],
        unavailable_flags_not_fabricated=True,full_task_success=False,score_contract_sha256=scored['score_contract_sha256'])
    return dict(schema='WM02_ORIGINAL_DIRECT_OBSERVED_OUTCOME_RECOVERY_V1',original_native_exit_code=1,
        original_failure_unchanged=True,confirmed_suffix_atoms=atoms,all40_actual_consumption=True,
        H16_actual_minus_desired_L2_m=tracking[15],final_actual_minus_desired_L2_m=tracking[-1],
        observed_task_geometry=geometry,new_model_evaluations=0,new_physics=0,new_updates=0,
        native_safety_evidence_is_separate_from_missing_sample_flags=True)
