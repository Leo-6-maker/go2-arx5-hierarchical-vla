import numpy as np
from scripts.sla.diagnose_eef_margin_v71 import series,compare,position

def rows():
    return np.tile([0.,0.,.7,1.,0.,0.,0.,1.,0.,.5],(40,1))

def test_physical_time_curvature_and_endpoint_index_convention():
    value=rows();value[:3,0]=[.0001,-.0001,.0001]
    _,a=series(value)
    assert np.isclose(a[1,0],.16)
    assert position(0,'acceleration')['kind']=='START_ZERO_VELOCITY_BOUNDARY'
    assert position(39,'acceleration')['kind']=='END_ZERO_VELOCITY_BOUNDARY'
    assert position(1,'acceleration')['H40_knots']==[0,1,2]

def test_boundary_failure_is_not_reported_as_interior_and_original_gate_decides():
    label=rows();pred=label.copy();pred[:,0]=np.arange(40)*.0007
    r=compare(pred,label,label[0]);points=r['metrics']['maximum_discrete_translation_acceleration_m_s2']['exceeded_points']
    assert not r['EEF_pass'] and r['label_EEF_pass']
    assert [p['kind'] for p in points]==['START_ZERO_VELOCITY_BOUNDARY','END_ZERO_VELOCITY_BOUNDARY']
    assert np.isclose(r['metrics']['maximum_discrete_translation_acceleration_m_s2']['label_peak_margin'],.25)
