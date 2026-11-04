import unittest
from go2arx5_vla.world_model.target_replay import validate_windows,validate_partitions

class TargetReplayContract(unittest.TestCase):
    def test_no_cross_parent_split_leakage(self):
        validate_partitions([dict(parent='p',split='TRAIN'),dict(parent='p',split='TRAIN')])
        with self.assertRaises(ValueError):validate_partitions([dict(parent='p',split='TRAIN'),dict(parent='p',split='DEV')])
    def test_exact_relative_grid_not_absolute_50ms_grid(self):
        rows=[dict(native_start_step=2503+25*i,native_end_step=2528+25*i,actual_ticks=25,mode='WBC',solver_policies_with_exact_installed_target=[{}])for i in range(24)]
        validate_windows(rows,2503,24)
        for field,value in(('actual_ticks',24),('mode','HOLD'),('native_end_step',2553),('solver_policies_with_exact_installed_target',[])):
            altered=[dict(x)for x in rows];altered[0][field]=value
            with self.assertRaises(ValueError):validate_windows(altered,2503,24)
        with self.assertRaises(ValueError):validate_windows(rows[:-1],2503,24)
if __name__=='__main__':unittest.main()
