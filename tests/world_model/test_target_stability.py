import unittest
import numpy as np
from go2arx5_vla.world_model.target_stability import motion,screen,rotation_wxyz


class TargetStability(unittest.TestCase):
    def test_rotating_base_is_not_arm_motion(self):
        angle=np.linspace(0,.2,9);q=np.stack([np.cos(angle/2),angle*0,angle*0,np.sin(angle/2)],-1);base=np.stack([angle,angle*0,angle*0],-1);relative=np.repeat([[.4,0,.3]],9,axis=0);tcp=base+np.einsum('nij,nj->ni',rotation_wxyz(q),relative)
        result=motion(base,q,tcp);np.testing.assert_allclose(result['relative_tcp'],relative,atol=1e-14);self.assertLess(result['identity_residual_max_m'],1e-14)
        self.assertTrue(screen(np.arange(9)*.05,result['relative_tcp'])['qualified'])

    def test_unstable_short_or_nonunit_are_not_qualified(self):
        self.assertFalse(screen(np.arange(8)*.05,np.zeros((8,3)))['qualified'])
        x=np.zeros((9,3));x[-1,0]=.003
        self.assertFalse(screen(np.arange(9)*.05,x)['qualified'])
        with self.assertRaises(ValueError):rotation_wxyz([[.5,0,0,0]])


if __name__=='__main__':unittest.main()
