import numpy as np
import pytest
from go2arx5_vla.sla_adapter.hybrid_eef_grip_v1 import compose,coordination
from scripts.sla.hybrid_saved96_v1 import summarize


def test_composition_has_fixed_sources_even_if_eef_gripper_is_illegal():
    eef=np.zeros((40,10));eef[:,3:9]=[1,0,0,0,1,0];eef[:,9]=np.arange(40)%2
    grip=eef.copy();grip[:,:3]=999.;grip[:,9]=np.linspace(1,0,40)
    value=compose(eef,grip)
    assert np.array_equal(value[:,:9],eef[:,:9])
    assert np.array_equal(value[:,9],grip[:,9])
    assert np.array_equal(eef[:,9],np.arange(40)%2)
    with pytest.raises(ValueError):compose(eef[:-1],grip)
    wrong=grip.copy();wrong[0,0]=np.nan
    with pytest.raises(ValueError):compose(eef,wrong)


def test_coordination_exposes_lift_before_close_and_command_seam():
    a=np.zeros((40,10));a[:,3:9]=[1,0,0,0,1,0];a[:,9]=1.
    a[20:,9]=0.;a[5:,2]=.003
    state=a[0].copy();context=dict(width_m=.042,source_native_s=10.,query_native_s=10.05)
    r=coordination(a,a,state,context,np.eye(4))
    assert r['EEF_raise2mm_k']==5 and r['near_predicted_terminal_width_k']==20
    assert r['lift_minus_near_close_s']==-.75
    assert not r['student_closed_loop_continuity_proven']
    assert not r['object_grasp_or_contact_inferred']


def test_full_spatial_failure_cannot_be_hidden_by_finger_first_rejection():
    phases=['PREGRASP']*16+['DESCEND']*20+['CLOSE']*20+['LIFT']*20+['HOLD']*20
    rows=[]
    for phase in phases:
        channels={k:{'exceedance_indices':[]}for k in ('linear_velocity','angular_velocity','linear_acceleration','angular_acceleration')}
        rows.append(dict(phase=phase,width=dict(expert_crossing_in_window=10 if phase=='CLOSE'else None,
            crossing_error_s=0.,terminal_width_error_m=0.),envelope=dict(original_full_H40_pass=True,
            paths={k:{'channels':{n:dict(v)for n,v in channels.items()}}for k in ('nominal','reference')}),
            coordination=dict(previous_actual_command_to_first_speed_m_s=0.)))
    assert summarize(rows)['offline_candidate_admitted']
    rows[40]['envelope']['paths']['nominal']['channels']['angular_acceleration']['exceedance_indices']=[30]
    assert summarize(rows)['full_spatial_channel_failures']==1
    assert not summarize(rows)['offline_candidate_admitted']
