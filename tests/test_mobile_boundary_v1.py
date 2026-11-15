import numpy as np
import pytest
from go2arx5_vla.world_model.mobile_wbc_reference_v1 import AXES,m0_reference,document,full_eef_gate


@pytest.mark.parametrize('axis',AXES)
@pytest.mark.parametrize('mode',['WORLD_HOLD','SMOOTH_5CM_TRANSLATION'])
def test_predeclared_motion_and_unchanged_h40_gate(axis,mode):
    tcp=np.eye(4);tcp[:3,3]=[.4,0,.5]
    mount=np.eye(4);mount[:3,3]=[.05,0,.4]
    for cursor in range(0,120,7):
        action,base=m0_reference(tcp,mount,axis,mode,cursor)
        full_eef_gate(action,tcp)
        assert action.shape==(40,10)
        assert len(document(action,mode='M_ASSIST',base_poses=base)['base']['waypoints'])==41
    action,base=m0_reference(tcp,mount,axis,mode,120)
    assert np.isclose(action[0,0],.45 if mode=='SMOOTH_5CM_TRANSLATION'else .4)
    if axis.endswith('yaw'):
        assert np.isclose(np.arctan2(base[0,1,0],base[0,0,0]),np.deg2rad(10)*(1 if axis[0]=='+'else -1))
    else:
        offset=base[0,:3,3]-mount[:3,3]
        assert np.isclose(offset[0 if axis.endswith('x')else 1],.1*(1 if axis[0]=='+'else -1))
    assert 'base'not in document(action,mode='M_FREE')
    with pytest.raises(ValueError,match='OMIT_BASE'):document(action,mode='M_FREE',base_poses=base)


def test_illegal_suffix_still_rejected():
    action,_=m0_reference(np.eye(4),np.eye(4),'+x','WORLD_HOLD',0)
    action[30,0]+=.1
    with pytest.raises(ValueError,match='FULL40_EEF_GATE'):full_eef_gate(action,np.eye(4))


def test_gripper_sequence_uses_current_native_clock_not_live_plan_clock(tmp_path):
    import struct
    from types import SimpleNamespace
    from scripts.wm02.mobile_boundary_runner_v1 import write_mobile_gripper
    adapter=SimpleNamespace(output=tmp_path,active_reference=dict(source_mpc_s=5.006))
    row=np.zeros(10);row[9]=.042/.044
    write_mobile_gripper(adapter,dict(time_s=5.056),1,row)
    magic,sequence,source,opening,until=struct.unpack('<8sQddd',(tmp_path/'gripper.bin').read_bytes())
    assert magic==b'GRTGRIP1'and sequence==1 and source==5.056 and until==5.456


def test_renewal_checks_original_full_program_without_fictitious_velocity_reset():
    from types import SimpleNamespace
    from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import validate_preview,quintic
    raw,_=m0_reference(np.eye(4),np.eye(4),'+x','WORLD_HOLD',0)
    raw[:,0]=.05*quintic(np.arange(40)/39);raw[:,9]=1.
    program=FiftyMsProgram(raw,np.eye(4),np.eye(4),[0,0,0,0,0,0,1])
    adapter=SimpleNamespace(spec=dict(mobile_stage='L1'),mobile_program=program,
        mobile_program_start_step=5703,observe=lambda:dict(step=5878))
    action=program.preview(7)
    with pytest.raises(ValueError):full_eef_gate(action,np.eye(4))
    assert validate_preview(adapter,action,np.eye(4))['original_full40_desired_and_reference_gate']
    action[30,0]+=.1
    with pytest.raises(ValueError,match='EXACT_PREDECLARED'):validate_preview(adapter,action,np.eye(4))
