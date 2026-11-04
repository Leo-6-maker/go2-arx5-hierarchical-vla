import unittest
import numpy as np
from go2arx5_vla.world_model.decision_native import proposals,intent,LO,HI
from go2arx5_vla.world_model.replay import SLICES


class DecisionNative(unittest.TestCase):
    def test_candidate_domains_units_and_known_intent_without_privileged_events(self):
        s=np.zeros(50);s[SLICES['object_position']]=[.4,0,.5];c=np.array([.6,0,.46,.44,.04])
        for kind in ('preclose','transport','supported_release'):
            names,raw=proposals(kind,s,c,0.,16)
            self.assertEqual(raw.shape,(5,8,7));self.assertEqual(len(set(names)),5)
            self.assertTrue((np.linalg.norm(raw[:,:,:3],axis=-1)<=.01).all())
            self.assertTrue((raw[:,:,6]>=LO).all()and(raw[:,:,6]<=HI).all())
        names,raw=proposals('transport',s,c,LO,16)
        self.assertGreater(raw[names.index('toward_goal'),:,0].sum(),0)
        self.assertLess(raw[names.index('away'),:,0].sum(),0)
    def test_working_intent_preserves_new_commands_and_only_maps_original_endpoints(self):
        self.assertEqual(intent(0),LO);self.assertEqual(intent(1),HI);self.assertEqual(intent(.25),.25)
        with self.assertRaises(ValueError):intent(.001)
    def test_candidate_permutation_is_fixed_and_not_always_nominal_first(self):
        s=np.zeros(50);c=np.array([.5,0,.46,.44,.04])
        orders=[proposals('preclose',s,c,1.,scene)[0]for scene in range(16,20)]
        self.assertGreater(len(set(tuple(x)for x in orders)),1)
        self.assertEqual(orders[0],proposals('preclose',s,c,1.,16)[0])

if __name__=='__main__':unittest.main()
