import numpy as np
import pytest
from go2arx5_vla.sla_adapter.source_batch_v6 import SourceTokenizer,host_batch
from go2arx5_vla.sla_adapter.source_evaluation_v6 import errors,transfer_gate
from go2arx5_vla.sla_adapter.source_native_v6 import Quantiles14,CAMERAS


def test_source_tokens_keep_second_arm_and_drop_only_segment():
    class Delegate:
        def tokenize(self,prompt,state,**kw):return state,kw
    token=SourceTokenizer.__new__(SourceTokenizer);token.delegate=Delegate()
    state=np.r_[np.arange(14),np.zeros(18)]
    values,_=token('full instruction',state)
    assert np.array_equal(values,np.arange(14))
    assert token('full instruction',state,drop_state=True)[0]is None
    with pytest.raises(ValueError):token('x',np.ones(32))


def test_source_batch_h50_is_separate_and_rejects_nonzero_padding():
    item=dict(image={k:np.zeros((224,224,3),np.uint8)for k in CAMERAS},
        image_mask={k:True for k in CAMERAS},image_padding_mask={k:np.ones((224,224),bool)for k in CAMERAS},
        state=np.zeros(32,np.float32),actions=np.zeros((50,32),np.float32),prompt='both arms task')
    class Dataset:
        def __getitem__(self,index):return item
    tokenizer=lambda *a,**k:(np.zeros(200,np.int32),np.ones(200,bool))
    batch=host_batch(Dataset(),tokenizer,np.arange(32),0,4)
    assert batch['actions'].shape==(4,50,32)
    assert all(batch['observation']['image_mask'][k].all()for k in CAMERAS)
    item['actions'][40,15]=1
    with pytest.raises(ValueError,match='PADDING18'):host_batch(Dataset(),tokenizer,np.arange(32),0,4)


def test_reference_metrics_preserve_raw_gripper_and_persistence():
    stats={'actions':Quantiles14(dict(q01=np.zeros(14),q99=np.ones(14)))}
    state=np.zeros(14);state[[6,13]]=[-3,4]
    target=np.zeros((50,14));target[:,6]=-3;target[:,13]=4;target[:,0]=.2
    value=errors(target.copy(),target,state,stats)
    assert value['model']['normalized_active14_MAE']==0
    assert value['model']['0.5s'][0]['gripper_native_error']==0
    assert value['persistence']['normalized_active14_MAE']>0
    assert value['persistence']['0.5s'][1]['gripper_native_error']==0


def test_source_gate_requires_general_task_gain_and_motion():
    s0=dict(dynamic=dict(normalized_reference_error=1),tasks={str(i):dict(normalized_reference_error=1)for i in range(6)})
    s1=dict(dynamic=dict(normalized_reference_error=.8,better_than_persistence=True),tasks={str(i):dict(normalized_reference_error=.8 if i<4 else 1.1)for i in range(6)})
    assert transfer_gate(s0,s1,reloadable=True)['passed']
    assert not transfer_gate(s0,s1,reloadable=False)['passed']
    s1['dynamic']['better_than_persistence']=False
    assert not transfer_gate(s0,s1,reloadable=True)['passed']
