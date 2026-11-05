import unittest
import numpy as np
from go2arx5_vla.world_model.next1_contact import planned_path,chunk,GRADES
from scripts.pi0294.task_path_reference import rates

class GradedReference(unittest.TestCase):
    def test_exact_continuous_planned_suffix_and_small_first_grade(self):
        initial=np.eye(4);initial[:3,3]=[.274,-.02,.733];obj=np.eye(4);obj[:3,3]=[.458,-.08,.46]
        path=planned_path(initial,obj,'FREE_X10')
        np.testing.assert_array_equal(path[0],initial)
        np.testing.assert_allclose(path[-1,:3,3]-initial[:3,3],[.01,0,0])
        np.testing.assert_array_equal(chunk(path,7),path[np.minimum(np.arange(7,47),len(path)-1)])
        for grade,cfg in GRADES.items():
            candidate=planned_path(initial,obj,grade);speed,spin=rates(candidate,np.arange(len(candidate))*.05)
            self.assertLess(speed,.06);self.assertLess(spin,.3)
            self.assertLessEqual((40 if grade in('DESCEND_20_OPEN','ABOVE_CONTACT40')else 48)+len(candidate)-1,cfg['task_atoms'])
