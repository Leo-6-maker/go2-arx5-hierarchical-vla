import numpy as np
import pytest

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0291_arx5_data.native_dataset import pose10
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows
from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram
from go2arx5_vla.sla_adapter.reference_labels_v6 import derive


def rz(a):
    return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])


def four_actual_replans():
    actual=np.eye(4);actual[:3,3]=[.28,-.02,.68];actual[:3,:3]=rz(.08)
    state=pose10(actual,.93);reference=actual.copy();reference[:3,3]+=[.01,.01,.01];reference[:3,:3]=rz(.1)
    w=np.linspace(0,1,10);w=w**3*(10+w*(-15+6*w));nominal=[];installed=[];grip=1.
    for q,delta in enumerate([[.004,0,.001],[0,.003,.001],[-.003,0,.001],[0,-.002,0]]):
        measured=actual.copy();measured[:3,3]+=[q*.0005,0,q*.0002];measured[:3,:3]=rz(.08+q*.001)
        raw=np.empty((40,10));raw[:10,:3]=measured[:3,3]+w[:,None]*delta
        raw[:10,3:9]=rotation_matrix_to_rot6d_rows(np.array([measured[:3,:3]@rz(t*.008)for t in w]))
        raw[:10,9]=grip-.1*w;raw[10:]=raw[9];grip-=.1
        program=FiftyMsProgram(raw,measured,reference,np.array([0,0,.5,1,0,0,0.]))
        nominal.extend(raw[:10]);installed.extend(program.reference_path[:10]);reference=matrix10(program.reference_path[9])
    return state,np.array(nominal),np.array(installed),5703+np.arange(40)*25


def test_cross_replan_nominal_failure_becomes_exact_original_consumed_reference():
    state,old,installed,steps=four_actual_replans();actual=matrix10(state);reference=matrix10(installed[0])
    with pytest.raises(ValueError,match='RATE_AND_ACCELERATION'):
        FiftyMsProgram(old,actual,reference,np.array([0,0,.5,1,0,0,0.]))
    new,proof=derive(state,old,installed,anchor_native_step=5703,command_native_steps=steps)
    program=FiftyMsProgram(new,actual,reference,np.array([0,0,.5,1,0,0,0.]))
    np.testing.assert_allclose(program.reference_path,installed,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(new[:,9],old[:,9])
    assert proof['reconstructed_reference_max_error']<1e-12 and not proof['native_execution_authorized']
    assert np.linalg.norm(new[19,:3]-new[0,:3])>np.linalg.norm(new[-1,:3]-new[0,:3])
    with pytest.raises(ValueError):new[0,0]=0


def test_equivalence_is_rigid_policy_frame_invariant():
    state,old,installed,steps=four_actual_replans()
    new,_=derive(state,old,installed,anchor_native_step=5703,command_native_steps=steps)
    frame=np.eye(4);frame[:3,:3]=rz(.7);frame[:3,3]=[.2,-.3,.5]
    move=lambda rows:pose10(frame@matrix10(rows),rows[...,9])
    transformed,_=derive(move(state),move(old),move(installed),anchor_native_step=5703,command_native_steps=steps)
    np.testing.assert_allclose(transformed,move(new),rtol=0,atol=1e-12)


@pytest.mark.parametrize('fault',['missing','clock','anchor','jump','grip','nan'])
def test_missing_or_unconsumed_or_invalid_commands_cannot_be_padded_or_repaired(fault):
    state,old,installed,steps=four_actual_replans()
    if fault=='missing':installed=installed[:-1];steps=steps[:-1]
    elif fault=='clock':steps[20]+=25
    elif fault=='anchor':state[0]+=.001
    elif fault=='jump':installed[20,0]+=.1
    elif fault=='grip':old[10,9]=1.1
    else:installed[10,0]=np.nan
    with pytest.raises(ValueError):derive(state,old,installed,anchor_native_step=5703,command_native_steps=steps)


def test_fp32_logit_codec_keeps_the_corrected_curve_inside_the_original_envelope():
    from types import SimpleNamespace
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles,pad32,roundtrip_error
    from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
    from go2arx5_vla.world_model.task_learning_grasp_v1 import WORKING_CLOSED_M,WORKING_OPEN_M,MECHANICAL_TRAVEL_M
    state,old,installed,steps=four_actual_replans();new,_=derive(state,old,installed,anchor_native_step=5703,command_native_steps=steps)
    codec=GripLogitCodec(np.eye(4));_,latent=codec.encode(state,new);q=Quantiles.fit(latent)
    decoded=codec.decode(state,pad32(q.unnormalize(pad32(q.normalize(latent)).astype(np.float32)[:,:7])))
    assert roundtrip_error(new,decoded)['xyz_m']<1e-6
    desired=decoded.copy();desired[:,9]=(WORKING_CLOSED_M+(WORKING_OPEN_M-WORKING_CLOSED_M)*decoded[:,9])/MECHANICAL_TRAVEL_M
    FiftyMsProgram.validate_envelope(SimpleNamespace(desired_path=desired,decision_TCP_target=matrix10(state),reference_path=desired,initial_reference_TCP_target=matrix10(state)))


def test_original_teacher_and_transport_source_sections_are_still_frozen():
    from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER
    assert teacher_fingerprint()==FROZEN_TEACHER
