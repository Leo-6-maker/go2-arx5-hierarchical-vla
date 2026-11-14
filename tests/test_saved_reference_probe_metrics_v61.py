import numpy as np
from scripts.sla.audit_saved_reference_probes_v61 import preview_metrics
from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram


def test_saved_probe_geometry_has_original_units_and_full_suffix_limits():
    state=np.array([0.,0.,.7,1.,0.,0.,0.,1.,0.,1.]);raw=np.tile(state,(40,1))
    metrics,failed=preview_metrics(raw,state)
    assert not failed and metrics['maximum_finger_speed_m_s']==0
    raw[-1,0]=.01
    metrics,failed=preview_metrics(raw,state)
    assert 'maximum_translation_speed_m_s'in failed
    assert 'maximum_discrete_translation_acceleration_m_s2'in failed
    assert np.isclose(metrics['maximum_translation_speed_m_s'],.2)
    actual=np.eye(4);actual[:3,3]=state[:3]
    import pytest
    with pytest.raises(ValueError,match='ORIGINAL_FULL_PREVIEW'):FiftyMsProgram(raw,actual,actual,[0,0,0,1,0,0,0])
