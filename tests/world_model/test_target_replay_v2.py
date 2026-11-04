import unittest
import numpy as np
from go2arx5_vla.world_model.target_replay_v2 import window_indices,materialize,select_samples

class VariableReplay(unittest.TestCase):
    def test_windows_stop_at_episode_and_split_boundaries(self):
        idx=window_indices([24,48,96]);self.assertEqual(len(idx),1+25+73);self.assertEqual(idx[0].tolist(),[0,0]);self.assertEqual(idx[-1].tolist(),[2,72])
        for n in(23,161):
            with self.assertRaises(ValueError):window_indices([n])
        meta=dict(episodes=[dict(parent='p',split='TRAIN'),dict(parent='d',split='DEV')]);d=dict(episode_index=np.array([0,0,1,1]));self.assertEqual(select_samples(d,meta,'TRAIN').tolist(),[0,1]);self.assertEqual(select_samples(d,meta,'DEV').tolist(),[2,3])
    def test_window_time_is_actual_episode_relative_and_overlap_is_not_new_atoms(self):
        e=dict(parent='p',split='TRAIN',task_atoms=48,state_start=0,atom_start=0);meta=dict(episodes=[e],state_dimension=2);raw={}
        for k in('state','native_step','sim_time','contact_time','wall_monotonic','tcp','object','events'):raw['flat_'+k]=np.zeros((49,2))if k=='state'else np.arange(49,dtype=float)*.05+5.006
        raw['flat_native_step']=2503+np.arange(49)*25;raw['flat_contact_time']=raw['flat_sim_time']-.002
        raw.update(flat_source_age=np.zeros(48),episode_goal=np.zeros((1,17)),window_index=window_indices([48]));d,_=materialize(raw,meta);self.assertEqual(d['state'].shape,(25,25,2));self.assertAlmostEqual(d['window_start_s'][-1],1.2);self.assertEqual(d['window_start_atom'][-1],24)
        raw['window_index'][-1]=[0,25]
        with self.assertRaises(ValueError):materialize(raw,meta)
if __name__=='__main__':unittest.main()
