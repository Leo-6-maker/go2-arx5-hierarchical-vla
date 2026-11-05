import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from go2arx5_vla.world_model.next2_episode import predict_before_install
from go2arx5_vla.world_model.storage import digest

class PredictionFirst(unittest.TestCase):
    def test_all_candidates_are_durable_while_native_is_paused(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);fields=[('qpos',34),('qvel',32),('ctrl',20),('tcp_position_m',3),('tcp_rotation6d',6),('object_position_m',3),('object_rotation6d',6),('finger_position_m',2)];meta=dict(state_fields=[dict(name=k,size=n)for k,n in fields]);f=dict(metadata=meta)
            for label in ['ARX','NO_FUTURE_CHOICE_ARX','HOLD_RESPONSE']:
                path=root/(label+'.npz')
                if label=='HOLD_RESPONSE':arrays=dict(x_mean=np.zeros(1),x_scale=np.ones(1),baseline_weight=np.zeros((1,144)),response_weight=np.zeros((2,144)))
                else:arrays=dict(mean=np.zeros(106),scale=np.ones(106),feature_mean=np.zeros(233),feature_scale=np.ones(233),weight=np.zeros((234,106)))
                np.savez_compressed(path,**arrays);f[label]=dict(path=str(path),sha256=digest(path),baseline='constant',ridge=.1)
            frozen=root/'FROZEN.json';frozen.write_text(json.dumps(f));qpos=np.zeros(34);qpos[3]=1.;sample=dict(qpos=qpos,qvel=np.zeros(32),ctrl=np.zeros(20),tcp=np.eye(4),object_pose=np.eye(4),finger_q=np.zeros(2));events=[];adapter=SimpleNamespace(spec=dict(source_sha='fixture',frozen_predictors=str(frozen),frozen_predictors_sha256=digest(frozen),frozen_thresholds_sha256='fixture'),wait_source=lambda:None,observe=lambda:dict(step=3703,time_s=7.406),latest=dict(ee=SimpleNamespace(time=7.406)),next1_samples=[sample]*9,active_reference=dict(action_EEF10=np.zeros((40,10))),base_pose_hold=[0,0,0,0,0,0,1],output=root)
            def event(value):
                self.assertTrue((root/'PREDICTIONS_BEFORE_ACTION.private.json').exists());events.append(value)
            adapter.command_event=event;r=predict_before_install(adapter,dict(offset_xyz_m=[.001,0,0]));self.assertEqual(r['model_input_clock']['window_start_s'],0.);self.assertEqual(len(r['candidates']),4);self.assertFalse(r['future_response_observed']);self.assertEqual(events[0]['phase'],'predictions-sealed-before-action');self.assertEqual(np.asarray(r['observed_history']).shape,(9,106))
            adapter.observe=lambda:dict(step=3728,time_s=7.456)
            with self.assertRaises(ValueError):predict_before_install(adapter,dict(offset_xyz_m=[0,0,0]))
if __name__=='__main__':unittest.main()
