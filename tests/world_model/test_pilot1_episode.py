import unittest, tempfile, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from go2arx5_vla.world_model.pilot1_episode import choose
from go2arx5_vla.world_model.storage import digest


class PredictionBeforeNativeChoice(unittest.TestCase):
    def test_each_method_is_saved_at_real_prefix_before_any_action_can_be_installed(self):
        from tests.world_model.test_pilot1_actor import fixture
        meta,model,_=fixture()
        sample=dict(qpos=np.zeros(34),qvel=np.zeros(32),ctrl=np.zeros(20),tcp=np.eye(4),object_pose=np.eye(4),finger_q=np.zeros(2))
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);path=root/'ARX.npz';np.savez_compressed(path,**model.arrays())
            metadata=root/'META.json';metadata.write_text(json.dumps(meta))
            task=root/'TASK.json';task.write_text(json.dumps(dict(schema='LOCAL_EEF_REACH_PILOT_V1',nonhold_action_cost_m=.00002,mean_response_xy_m=[[0,0],[.001,0],[-.001,0],[0,.001]])))
            # Native inference has the same robot-only input ABI as the learner.
            from go2arx5_vla.world_model.pilot1_numpy import actor_features
            data=dict(state=np.zeros((1,9,106)),history_goal=np.zeros((1,8,17)),source_age=np.zeros((1,24)),future_prefix_remaining=np.asarray([7.]))
            dim=actor_features(data,meta,np.asarray([0]),np.zeros((1,2))).shape[1]
            actor=root/'ACTOR.npz';np.savez_compressed(actor,mean=np.zeros(dim,dtype=np.float32),scale=np.ones(dim,dtype=np.float32),w0=np.zeros((2,dim),dtype=np.float32),b0=np.zeros(2,dtype=np.float32),w2=np.zeros((2,2),dtype=np.float32),b2=np.zeros(2,dtype=np.float32),w4=np.zeros((4,2),dtype=np.float32),b4=np.asarray([0,1,0,0],dtype=np.float32))
            spec=dict(task=str(task),teacher_metadata=str(metadata),teacher_metadata_sha256=digest(metadata),ARX=str(path),ARX_sha256=digest(path),actor_numpy=str(actor),actor_numpy_sha256=digest(actor),source_sha='test-only',task_sha256=digest(task))
            events=[]
            def event(value):
                self.assertTrue((root/'PILOT1_DECISION.private.json').exists());events.append(value)
            adapter=SimpleNamespace(spec=spec,output=root,wait_source=lambda:None,observe=lambda:dict(step=3703,time_s=7.406),latest=dict(ee=SimpleNamespace(time=7.406)),next1_samples=[sample]*9,active_reference=dict(action_EEF10=np.zeros((40,10))),base_pose_hold=[0,0,0,0,0,0,1],measured_task_state=lambda:sample,command_event=event)
            episode=dict(task_goal_offset_xy_m=[.0005,0],purpose='EVALUATION',method='TRAINED_ACTOR',split='TRAIN',scene_id='fixture')
            np.testing.assert_array_equal(choose(adapter,episode),[.001,0,0])
            value=json.loads((root/'PILOT1_DECISION.private.json').read_text())
            self.assertFalse(value['future_response_observed']);self.assertEqual(value['selected_action'],1)
            self.assertEqual(events[0]['phase'],'pilot1-choice-before-install')


if __name__=='__main__':unittest.main()
