import unittest
from .task_contacts import allowed,original_allowed


class TaskContactTests(unittest.TestCase):
    def test_all_native_pairs_unchanged(self):
        names=(None,'floor','FL','FR','RL','RR','base_link','x5_link1','unnamed_robot_geom')
        for a in names:
            for b in names:
                self.assertEqual(allowed(a,b),original_allowed(a,b))

    def test_only_three_explicit_task_pairs_added(self):
        for a,b in (('task_table','task_cube_geom'),('task_left_pad','task_cube_geom'),('task_right_pad','task_cube_geom')):
            self.assertTrue(allowed(a,b));self.assertTrue(allowed(b,a))
        for a,b in (('task_table','task_left_pad'),('floor','task_cube_geom'),('task_table','FL'),
                    ('task_cube_geom',None),('task_left_pad','task_right_pad')):
            self.assertFalse(allowed(a,b))


if __name__=='__main__':unittest.main()
