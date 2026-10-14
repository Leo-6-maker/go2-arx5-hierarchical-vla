"""Pure CPU frame and envelope checks; native lifecycle is separately qualified."""
import unittest
import numpy as np
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows,quat_to_matrix
from .base_intent import base_endpoints,native_document,measured_base,rotation


class BaseIntentTests(unittest.TestCase):
    def test_signed_exact_se2_and_tilted_origin(self):
        mount=np.eye(4)
        for axis in range(3):
            for sign in (-1,1):
                v=np.zeros((40,3));v[:,axis]=sign*.05;p=base_endpoints(mount,v)
                self.assertEqual(p.shape,(41,4,4))
                if axis<2:self.assertAlmostEqual(p[-1,axis,3],sign*.1)
                else:self.assertAlmostEqual(np.arctan2(p[-1,1,0],p[-1,0,0]),sign*.1)
        pitch=.2;roll=.3
        rx=np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
        ry=np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        mount[:3,:3]=rotation(np.pi-.01)@ry@rx
        offset=np.array([.1,0,.05]);omega=np.array([0,.1,.2])
        m,v=measured_base(mount,np.zeros(3),omega,offset)
        np.testing.assert_allclose(m[:3,3],mount[:3,:3]@offset)
        np.testing.assert_allclose(v[:2],(rotation(-np.pi+.01)@mount[:3,:3]@np.cross(omega,offset))[:2])
        self.assertNotAlmostEqual(v[2],omega[2])
        p=base_endpoints(m,np.tile([0,0,.1],(40,1)))
        np.testing.assert_allclose(p[-1,:3,:3],rotation(.2)@m[:3,:3],atol=1e-12)

    def test_world_eef_hold_zero_versus_absent_and_rejection(self):
        a=np.zeros((40,13));a[:,:3]=[.4,0,.6];a[:,3:9]=rotation_matrix_to_rot6d_rows(rotation(.3));a[:,9]=1
        mount=np.eye(4);a[:,10]=np.minimum(np.arange(40)*.01,.1)
        doc=native_document(a,mount,base_valid=True,previous_velocity=[0,0,0])
        self.assertEqual(len(doc['ee']['waypoints']),41);self.assertEqual(len(doc['base']['waypoints']),41)
        self.assertEqual(doc['ee']['waypoints'][0]['position'],doc['ee']['waypoints'][-1]['position'])
        np.testing.assert_allclose(quat_to_matrix(doc['ee']['waypoints'][0]['quaternion_xyzw']),rotation(.3),atol=1e-12)
        self.assertGreater(doc['base']['waypoints'][-1]['position'][0],0)
        zero=a.copy();zero[:,10:]=0
        self.assertIn('base',native_document(zero,mount,base_valid=True,previous_velocity=[0,0,0]))
        self.assertNotIn('base',native_document(zero,mount,base_valid=False,previous_velocity=[0,0,0]))
        for value in (np.nan,.16,.03):
            bad=a.copy();bad[0,10]=value
            with self.assertRaises(ValueError):native_document(bad,mount,base_valid=True,previous_velocity=[0,0,0])


if __name__=='__main__':unittest.main()
