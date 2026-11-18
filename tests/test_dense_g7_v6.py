import numpy as np
import pytest
from go2arx5_vla.sla_adapter.dense_g7_v6 import grip_speed_loss,previous_command_context,label_margin_report


def sigmoid(x):return 1/(1+np.exp(-x))


def test_grip_objective_includes_real_previous_command_not_only_interior():
    normalized=np.zeros((1,40,32));q01=np.zeros(7);q99=np.ones(7)
    # Constant generated trajectory has no interior motion but a large seam.
    result,detail=grip_speed_loss(normalized,q01,q99,np.asarray([.002]),xp=np,sigmoid=sigmoid)
    assert result>0 and detail['first_seam_speed']>.04
    assert detail['interior_max_speed']==0
    width=.002+.04*sigmoid(.5*(1+1e-6))
    assert grip_speed_loss(normalized,q01,q99,np.asarray([width]),xp=np,sigmoid=sigmoid)[0]==0


def test_dense_g7_can_represent_multiple_open_close_changes_without_block_average():
    from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
    from scripts.pi0291_arx5_data.native_dataset import pose10
    codec=GripLogitCodec(np.eye(4));state=pose10(np.eye(4),.5)
    target=np.repeat(state[None],40,axis=0);target[:,9]=.5+.4*np.sin(np.arange(40)*2*np.pi/20)
    _,latent=codec.encode(state,target);padded=np.pad(latent,[(0,0),(0,25)])
    decoded=codec.decode(state,padded)
    assert np.max(abs(decoded[:,9]-target[:,9]))<1e-12
    assert np.count_nonzero(np.diff(np.sign(np.diff(decoded[:,9]))))>=3


def test_context_rejects_synthetic_clock_and_reports_original_label_margin():
    context=dict(width_m=.042,source_native_s=10.,query_native_s=10.05)
    assert previous_command_context(context)==.042
    with pytest.raises(ValueError):previous_command_context(dict(context,source_native_s=10.01))
    with pytest.raises(ValueError):previous_command_context(dict(context,source_native_s=None))
    target=np.zeros((40,10));target[:,9]=1
    assert not label_margin_report(target,context)['training_margin_exceeded']
    target[1:,9]=0
    assert label_margin_report(target,context)['original_execution_speed_exceeded']
