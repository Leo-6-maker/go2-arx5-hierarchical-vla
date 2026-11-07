"""Read-only recovery retains FAIL and never invents missing safety samples."""
import copy
import numpy as np
import pytest
from go2arx5_vla.world_model.fusion_live_skill_outcome_recovery_v1 import recover,SOURCE,B,E,_pose


def fixture():
    vector=np.zeros(106);vector[2]=.3;vector[3]=1.
    vector[86:95]=[.3,0,.3,1,0,0,0,1,0]
    vector[95:104]=[.51,0,.3,1,0,0,0,1,0];vector[104:106]=.042
    spec=dict(source_sha=SOURCE,program_reference_selection=dict(method='DIRECT'),max_ticks=E,live_skill_start_step=B)
    decision=dict(start_step=B,end_step=E,actual_history=dict(actual_state_history=[vector.tolist()]))
    prefix=dict(native_endpoint_step=B,windows=[dict(actual_TCP=_pose(vector[86:95]).tolist())]);rows=[];previous=vector.copy();n=(E-B)//25
    for i in range(n):
        current=vector.copy();current[86]+=.005*(i+1)/n;current[87]+=.006*(i+1)/n
        rows.append(dict(native_start_step=B+25*i,native_end_step=B+25*(i+1),actual_ticks=25,stage='ACTUAL_CONSUMPTION_CONFIRMED',
            actual_history_before_atom=dict(actual_state_history=[previous.tolist()]),actual_state_vector=current.tolist(),actual_TCP=_pose(current[86:95]).tolist(),
            gripper=dict(command_consumed=True,actual_finger_q_m=current[104:106].tolist()),solver_policies_with_exact_installed_target=[1],desired_interval_end_EEF10=current[86:89].tolist()))
        previous=current
    segment=dict(start_step=B,native_endpoint_step=E,all_40_knots_consumed=True,raw_knot_consumed=[True]*40,windows=rows,complete_suffix_atoms=n,solver_target_confirmed_atoms=n,gripper_confirmed_atoms=n)
    native=dict(source_sha=SOURCE,status='NATIVE_STAGE_FAILED_PRESERVED',failure='TypeError: Object of type ndarray is not JSON serializable',
        actual_target_integrations=E,complete_atomic_transitions=2660,shutdown_clean=True,trace_audit_error=None,original_joint_or_warning_samples=0,native_outer_guard_rejection_count=0,physics_stops=[],owned_process_exits=[dict(exit_code=0)]*4,
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS',errors=[],writer=dict(dropped=0)))
    return spec,prefix,decision,segment,native


def test_actual_geometry_can_rule_out_task_success_without_inventing_safety_flags():
    values=fixture();before=copy.deepcopy(values)
    result=recover(*values);g=result['observed_task_geometry']
    assert result['original_native_exit_code']==1 and result['original_failure_unchanged']is True
    assert result['all40_actual_consumption']is True and result['confirmed_suffix_atoms']==1288
    assert result['H16_actual_minus_desired_L2_m']==0
    assert g['metrics']['maximum_lateral_drift_m']==pytest.approx(.006)
    assert g['geometry_failure_proves_task_failure']is True and g['approach_step_success']is False
    assert g['original_complete_strict_score_recovered']is False and g['unavailable_flags_not_fabricated']is True
    assert 'original_native_safety'not in g['predicates']
    assert result['new_physics']==result['new_model_evaluations']==result['new_updates']==0
    assert values==before


@pytest.mark.parametrize('fault',['other_failure','source','warning','trace','partial','state_join','unconfirmed'])
def test_recovery_cannot_hide_an_unrelated_failure_or_make_up_physical_coverage(fault):
    values=fixture();s,p,d,g,n=values
    if fault=='other_failure':n['failure']='A real native safety abort'
    elif fault=='source':s['source_sha']='0'*40
    elif fault=='warning':n['original_joint_or_warning_samples']=1
    elif fault=='trace':n['trace_audit']['writer']['dropped']=1
    elif fault=='partial':g['windows'].pop()
    elif fault=='state_join':g['windows'][2]['actual_state_vector'][86]+=.001
    else:g['windows'][0]['stage']='BEFORE_EFFECT'
    with pytest.raises(ValueError):recover(*values)
