import copy,unittest
import numpy as np
from go2arx5_vla.world_model.contact_identification import cards,candidates,compare_repeats,LO,HI

class ContactIdentification(unittest.TestCase):
    def test_predeclared_physical_layouts_and_whole_parent_partitions(self):
        c=cards();self.assertEqual(len(c),24)
        self.assertEqual([sum(x['ident_partition']==p for x in c)for p in ('FIT','DIAGNOSTIC_DEV','ONCE_HOLDOUT')],[16,4,4])
        self.assertTrue(all(x['split']=='TRAIN'for x in c));self.assertEqual(len({x['parent_id']for x in c}),24)
        self.assertEqual(len({tuple([*x['source_xy'],*x['goal_xy'],x['cube_yaw']])for x in c}),24)
    def test_candidate_units_timing_and_no_unsupported_open(self):
        raw=np.zeros((8,7));raw[:,2]=.001;raw[:,6]=0
        out=candidates(raw,1.,{'native_grasp':False})
        self.assertEqual(set(out),{'hold','nominal','left','right','grip_timing'})
        np.testing.assert_allclose(out['left'][:,0],-.0005)
        np.testing.assert_allclose(out['grip_timing'][:4,6],HI);np.testing.assert_allclose(out['grip_timing'][4:,6],LO)
        out=candidates(raw,0.,{'native_grasp':True})
        self.assertTrue((out['grip_timing'][:,6]==.25).all())
        self.assertTrue(all((x[:,6]<=HI).all()and(x[:,6]>=LO).all()for x in out.values()))
    def test_physical_match_without_control_or_full_state_is_not_restore_qualification(self):
        r=dict(native_after={'integration':np.zeros(5)},state_after={'contacts':[]},terminal=False,truncated=False,safety_reason=None,controller_memory_after={'previous_action':np.zeros(2)})
        a=[copy.deepcopy(r)for _ in range(16)];b=copy.deepcopy(a);state={'integration':np.zeros(3),'rng':{'state':1}}
        self.assertFalse(compare_repeats(a,b)['qualified']);self.assertTrue(compare_repeats(a,b,state,state)['qualified'])
        b[-1]['controller_memory_after']['previous_action'][0]=1
        self.assertFalse(compare_repeats(a,b,state,state)['qualified'])

if __name__=='__main__':unittest.main()
