"""Synthetic arithmetic fixtures, not new simulator/model evidence."""
import copy
import unittest
import numpy as np
from .saved_gripper_trace import analyze


class SavedGripperTraceTests(unittest.TestCase):
    def fixture(self,n=81):
        nq=2;nv=2;rows=np.zeros((n,1+nq+nv+8+7));rows[:,0]=np.arange(1,n+1)*.002
        rows[:,1:3]=.044
        mapping={name:dict(qpos=i,qvel=i,ctrl=i) for i,name in enumerate(('left_slide','right_slide'))}
        command=np.zeros((8,10));command[:,9]=.5
        target=.044
        for index in range(n):
            target+=np.clip(.022-target,-.00008,.00008)
            rows[index,5:7]=np.clip(400*(target-.044),-8,8)
        return rows,[command],mapping,np.full(nq,.044)

    def test_exact_slew_force_arithmetic_and_partial_prefix(self):
        value=analyze(*self.fixture(),nq=2,nv=2)
        self.assertEqual(value['executed_20Hz_points'],4)
        self.assertEqual(value['prefixes'][0]['points_executed_at_least_one_tick'],4)
        self.assertAlmostEqual(value['joints'][0]['final_slewed_target_m'],.044-81*.00008)
        self.assertLessEqual(value['motor_ctrl_max_abs_reconstruction_error_N'],1e-9)
        self.assertEqual(value['physics_integrations'],0)

    def test_out_of_range_command_never_clipped(self):
        args=self.fixture();args[1][0][0,9]=1.01
        with self.assertRaisesRegex(ValueError,'OUT_OF_CONTRACT'):analyze(*args,nq=2,nv=2)

    def test_bad_clock_ctrl_or_mapping_rejected(self):
        for kind in ('clock','ctrl','mapping'):
            rows,commands,mapping,initial=copy.deepcopy(self.fixture())
            if kind=='clock':rows[-1,0]+=.001
            elif kind=='ctrl':rows[-1,5]+=.1
            else:mapping['right_slide']['ctrl']=8
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                analyze(rows,commands,mapping,initial,nq=2,nv=2)

    def test_saved_limit_violation_without_new_integration(self):
        rows,commands,mapping,initial=self.fixture()
        rows[-1,2]=.045004271727453214
        value=analyze(rows,commands,mapping,initial,nq=2,nv=2)
        self.assertEqual(value['joints'][1]['first_limit_violation_tick'],81)
        self.assertEqual(value['model_loads'],0)


if __name__=='__main__':unittest.main()
