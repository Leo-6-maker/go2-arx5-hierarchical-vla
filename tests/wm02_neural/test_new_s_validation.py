import unittest
import numpy as np
from go2arx5_vla.world_model.new_s_validation import actual_quality,selectors,forecast_inputs
from go2arx5_vla.world_model.replay import SLICES

def state():
    return dict(arm_q=np.zeros(6),arm_qd=np.zeros(6),tcp=np.eye(4),object_pose=np.eye(4),tool_linear_velocity=np.zeros(3),tool_angular_velocity=np.zeros(3),object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),gripper_fraction=.43,gripper_speed=0.,time_s=1.,native_grasp=True,supported=True)

class NewValidation(unittest.TestCase):
    def test_incomplete_never_scores_as_safe_release_and_support_is_actual(self):
        s=state();s['native_grasp']=False;s['gripper_fraction']=1.
        self.assertEqual(actual_quality('supported_release',s,np.zeros(5),False),-1e6)
        self.assertEqual(actual_quality('supported_release',s,np.zeros(5),True),1.)
        s['supported']=False;self.assertEqual(actual_quality('supported_release',s,np.zeros(5),True),0.)
    def test_predictor_has_only_burn8_and_current_no_future_state_event_label(self):
        history=[dict(state_before=state(),reference_compiled=np.zeros((40,10)),t0=i*.05,native_progress=dict(point=i))for i in range(8)]
        b=forecast_inputs(history,state(),np.zeros(5),np.zeros((5,8,7)))
        self.assertEqual(set(b),{'state','reference','context','clock','raw'});self.assertEqual(tuple(b['state'].shape),(5,9,50));self.assertEqual(tuple(b['reference'].shape),(5,8,40,10));self.assertTrue(np.array_equal(b['clock'][:,8].numpy(),np.tile([1./32,0],(5,1))))
    def test_method_fallback_is_explicit_and_raw_command_order_invariant(self):
        s=state();context=np.array([.1,0,.46,.44,.04]);raw=np.zeros((5,8,7));raw[:,:,6]=.05;raw[:,0,0]=[0,.001,-.001,.002,-.002];original=raw.copy();pred=np.zeros((5,16,50));events=np.zeros((5,16,4));events[:,:,2]=1.;pred[:,-1,SLICES['object_position']]=np.c_[np.arange(5)*.001,np.zeros((5,2))]
        coef=np.zeros((4,96));a=selectors('transport',s,context,raw,pred,events,coef);order=np.array([4,2,0,3,1]);b=selectors('transport',s,context,raw[order],pred[order],events[order],coef)
        for method in a:self.assertTrue(np.array_equal(raw[a[method]['index']],raw[order[b[method]['index']]]))
        self.assertTrue(a['hybrid']['fallback']);self.assertFalse(a['OLS_transport_rule_contact']['fallback']);self.assertTrue(np.array_equal(original,raw))

if __name__=='__main__':unittest.main()
