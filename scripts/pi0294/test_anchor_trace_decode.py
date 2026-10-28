from types import SimpleNamespace
import unittest
import numpy as np
from .mp1.anchor_trace_decode import component_differences


class TraceDecodeTests(unittest.TestCase):
    def fixture(self):
        flags=dict(mjSTATE_TIME=1,mjSTATE_QPOS=2,mjSTATE_WARMSTART=4,mjSTATE_INTEGRATION=7)
        def size(model,kind):return 3 if kind==7 else 1
        def set_state(model,data,value,kind):data.state=np.array(value)
        def get_state(model,data,result,kind):result[:]=data.state[{1:0,2:1,4:2}[kind]]
        return SimpleNamespace(mjtState=SimpleNamespace(mjSTATE_INTEGRATION=7,__members__=flags),
            mj_stateSize=size,MjData=lambda model:SimpleNamespace(),mj_setState=set_state,mj_getState=get_state)

    def test_api_component_decode_distinguishes_warmstart_from_position_and_ik(self):
        reference=np.zeros((3,8));actual=reference.copy();actual[1,2]=3.4e-9
        value=component_differences(object(),actual,reference,mj=self.fixture())
        self.assertEqual(value['live']['mjSTATE_WARMSTART']['maximum_absolute_difference'],3.4e-9)
        self.assertEqual(value['live']['mjSTATE_QPOS']['maximum_absolute_difference'],0)
        self.assertTrue(all(x['maximum_absolute_difference']==0 for x in value['IK'].values()))
        self.assertTrue(value['steps_and_safety_exact']);self.assertFalse(value['tolerance_changed'])
        self.assertEqual(value['scalar_tolerance'],1e-10)

    def test_decoder_rejects_bad_shape_nonfinite_and_incomplete_api_layout(self):
        for bad in [np.zeros((2,8)),np.full((3,8),np.nan),np.zeros((3,3))]:
            with self.subTest(shape=bad.shape),self.assertRaises(ValueError):
                component_differences(object(),bad,np.zeros((3,8)),mj=self.fixture())
        mj=self.fixture();mj.mjtState.__members__.pop('mjSTATE_WARMSTART')
        with self.assertRaises(ValueError):component_differences(object(),np.zeros((3,8)),np.zeros((3,8)),mj=mj)


if __name__=='__main__':unittest.main()
