import unittest
import numpy as np
from go2arx5_vla.world_model.next2_native_events import contact_outcomes
class WithinAtomEvents(unittest.TestCase):
    def test_brief_native_contact_is_retained_when_atom_endpoint_has_none(self):
        rows=[dict(contacts=[])for _ in range(25)];rows[12]=dict(contacts=[dict(geom0=2,geom1=4,wrench=[.1,0,0,0,0,0],normal=np.array([0,1,0])),dict(geom0=3,geom1=4,wrench=[.2,0,0,0,0,0],normal=np.array([0,-1,0]))]);side,peak,opposing,ticks=contact_outcomes(rows,[2,3],4);np.testing.assert_array_equal(side,[True,True]);np.testing.assert_allclose(peak,[.1,.2]);self.assertTrue(opposing);self.assertEqual(ticks,1);self.assertFalse(rows[-1]['contacts'])
if __name__=='__main__':unittest.main()
