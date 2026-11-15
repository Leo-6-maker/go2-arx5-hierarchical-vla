import numpy as np
from scripts.sla.derive_first4_views_v61 import reference_labels
from scripts.pi0291_arx5_data.native_dataset import pose10
from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram


def test_explicit_previous_reference_not_rounded_first_installed_command():
    actual=np.eye(4);actual[:3,3]=[.1,.2,.6]
    previous=np.eye(4);previous[:3,3]=[.11,.19,.61]
    angle=.1;previous[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    raw=np.repeat(pose10(actual,.6)[None],40,axis=0)
    raw[:,0]+=2e-9+.001*np.sin(np.linspace(0,np.pi,40))**2
    base=np.array([0.,0.,.3,1.,0.,0.,0.]);program=FiftyMsProgram(raw,actual,previous,base)
    derived=reference_labels(actual,previous,program.reference_path,raw[:,9])
    again=FiftyMsProgram(derived,actual,previous,base)
    np.testing.assert_allclose(again.reference_path,program.reference_path,rtol=0,atol=1e-15)
    assert abs(derived[0,0]-actual[0,3]-2e-9)<1e-16
    # Assuming installed[0] was the old anchor would erase this real offset.
    assert not np.array_equal(derived[0,:3],actual[:3,3])
