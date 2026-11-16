import copy
import numpy as np
import pytest
from go2arx5_vla.sla_adapter.hybrid_request_v1 import predict_request


@pytest.mark.parametrize('fault', ['stop','cancelled','expiry','partial','native_advance','request_change'])
def test_mid_inference_fault_returns_no_separate_gripper_action(fault):
    live=dict(stop=False,cancelled=False,request_id='one',consumed_atoms=0,query_native_s=1.05)
    calls=[];clock=[1.]
    action=np.zeros((40,10));action[:,3:9]=[1,0,0,0,1,0]
    class EEF:
        def predict(self,*args,**kwargs):
            calls.append('EEF')
            if fault in ('stop','cancelled'):live[fault]=True
            elif fault=='expiry':clock[0]=3.
            elif fault=='partial':live['consumed_atoms']=3
            elif fault=='native_advance':live['query_native_s']+=.05
            else:live['request_id']='other'
            return {'physical_H40':action}
    class Grip:
        def predict(self,*args,**kwargs):calls.append('GRIP');return {'physical_H40':action}
    with pytest.raises(RuntimeError):
        predict_request(EEF(),Grip(),state=np.zeros(10),images=[],language='pick',frame=np.eye(4),
            noise_id=0,previous_reference=dict(width_m=.042,source_native_s=1.,query_native_s=1.05),
            request_id='one',deadline_monotonic=2.,now=lambda:clock[0],live_state=lambda:live)
    assert calls==['EEF']
