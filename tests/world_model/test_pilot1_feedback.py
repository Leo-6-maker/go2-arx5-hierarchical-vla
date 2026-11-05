import unittest,tempfile,json
from pathlib import Path
import numpy as np
from scripts.wm02.build_pilot1_feedback import merge
from go2arx5_vla.world_model.target_replay import load


def saved(path,splits,prefix,feedback=False):
    episodes=[dict(episode=prefix+str(i),parent=prefix+str(i),split=s,task_atoms=24,state_start=25*i,atom_start=24*i,window_count=1)for i,s in enumerate(splits)]
    if feedback:
        for e in episodes:e['pilot1_feedback']=dict(action=0,goal_offset_xy_m=[0,0],local_XY_success=False)
    n=len(episodes);steps=np.tile(3503+np.arange(25)*25,n)
    meta=dict(schema='WM02_TARGET_LATE_DECISION_REPLAY_V3',state_fields=[dict(name='qpos',size=106)],state_dimension=106,
              joints=[],robot_fingerprint='unit-test',executor_sha256={'native_binary':'unit-test'},collection_protocol='FIXED_HOLD40_BURN8_DECIDE_RAMP8_V1',
              action='ACTUAL_HOLD8_PAST_GOALS_THEN_CHOSEN_RAMP8_FUTURE_GOAL_V3',scene_assets_sha256=['unit-test'],episodes=episodes,task_atoms=24*n,windows=n)
    raw=dict(flat_state=np.full((25*n,106),1.if prefix=='old'else 2.),flat_native_step=steps,flat_sim_time=steps*.002,
             flat_contact_time=steps*.002-.002,flat_wall_monotonic=np.arange(25*n,dtype=float),flat_source_age=np.zeros(24*n),
             flat_tcp=np.zeros((25*n,4,4)),flat_object=np.zeros((25*n,4,4)),flat_events=np.zeros((25*n,7),dtype=bool),
             flat_goal=np.zeros((24*n,17)),episode_goal=np.zeros((n,17)),window_index=np.asarray([[i,0]for i in range(n)]),
             metadata_utf8=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8))
    np.savez_compressed(path,**raw)


class FeedbackReplay(unittest.TestCase):
    def test_append_preserves_old_dev_and_offsets_each_new_physical_episode(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);a=root/'old.npz';b=root/'new.npz';out=root/'all.npz'
            saved(a,['TRAIN','DEV'],'old');saved(b,['TRAIN'],'new',True)
            result=merge(a,b,out);data,meta=load(out)
            self.assertEqual(result['new_TRAIN_windows'],1);self.assertEqual(meta['episodes'][1]['split'],'DEV')
            self.assertEqual(meta['episodes'][2]['state_start'],50);self.assertEqual(meta['episodes'][2]['atom_start'],48)
            np.testing.assert_array_equal(data['state'][1],1.);np.testing.assert_array_equal(data['state'][2],2.)
    def test_dev_cannot_be_appended_as_train_feedback(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);saved(root/'old.npz',['TRAIN','DEV'],'old');saved(root/'new.npz',['DEV'],'new',True)
            with self.assertRaises(ValueError):merge(root/'old.npz',root/'new.npz',root/'all.npz')


if __name__=='__main__':unittest.main()
