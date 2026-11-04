import unittest,copy,tempfile
from pathlib import Path
import numpy as np,torch
from go2arx5_vla.world_model.target_late_decision import goals,SCHEMA
from go2arx5_vla.world_model.target_training import TargetWM,update,save,restore,tree_error
from go2arx5_vla.world_model.target_training_v2 import compatible

class LateDecisionContract(unittest.TestCase):
    def fixture(self):
        initial=np.array([.3,0,.7,1,0,0,0,1,0,.042/.044]);future=initial.copy();future[0]+=.001;base=[0,0,.34,0,0,0,1];cmd=[]
        for k in(0,7,8,15,22):
            a=np.repeat((initial if k<8 else future)[None],40,axis=0)
            if k==8:a[:,:3]=initial[:3]+np.minimum((np.arange(40)+1)/8,1.)[:,None]*(future[:3]-initial[:3])
            cmd.append(dict(phase='proposed',native_step=2503+25*k,wire_sha256=str(k),action_EEF10=a.tolist(),measured_BASE_pose_hold=base))
        cmd.append(dict(phase='decision-boundary',native_step=2703,future_response_observed=False));windows=[dict(reference_sha256=str(max(k for k in(0,7,8,15,22)if k<=i)))for i in range(24)];return cmd,windows
    def test_only_past_hold_and_newly_installed_future_goal_are_valid(self):
        cmd,windows=self.fixture();future,past=goals(cmd,windows,2503,[.001,0,0]);np.testing.assert_allclose(past[:8,:3],np.tile([.3,0,.7],(8,1)));np.testing.assert_allclose(past[8:,:3],np.tile(future[:3],(16,1)))
        leaked=copy.deepcopy(cmd);leaked[0]['action_EEF10'][-1][0]+=.001
        with self.assertRaises(ValueError):goals(leaked,windows,2503,[.001,0,0])
        changed=copy.deepcopy(windows);changed[7]['reference_sha256']='8'
        with self.assertRaises(ValueError):goals(cmd,changed,2503,[.001,0,0])
        with self.assertRaises(ValueError):goals(cmd,windows,2503,[0,.001,0])
    def test_v3_causal_features_and_full_checkpoint_recovery(self):
        torch.set_num_threads(1);torch.manual_seed(10);fields=[('qpos',7),('qvel',6),('ctrl',1),('tcp_position_m',3),('tcp_rotation6d',6),('object_position_m',3),('object_rotation6d',6),('finger_position_m',2)]
        meta=dict(schema=SCHEMA,action='late-decision',state_fields=[dict(name=n,size=s)for n,s in fields],state_dimension=sum(s for n,s in fields),joints=[],actuator_gear=[],actuator_gain=[],robot_fingerprint='fixture',executor_sha256={},action_encoding='SCALED_WORLD_RELATIVE_1CM_V2');m=TargetWM(meta,8)
        data=dict(state=torch.randn(2,25,meta['state_dimension'])*.01,goal=torch.randn(2,17)*.01,history_goal=torch.randn(2,8,17)*.01,source_age=torch.zeros(2,24),window_start_s=torch.zeros(2),window_start_atom=torch.zeros(2),future_prefix_remaining=torch.full((2,),7.));m.fit_normalizer(data['state']);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5);update(m,opt,data,0);m.eval();before=m.predict(**data)
        other={k:v.clone()for k,v in data.items()};other['state'][:,9:]+=10;other['source_age'][:,9:]+=20;self.assertEqual(tree_error(before,m.predict(**other)),0.)
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'model.pt';save(p,m,opt,1,8,dict(recipe='V3_component'));update(m,opt,data,8);clone,co,c=restore(p,'cpu',dict(recipe='V3_component'));update(clone,co,data,c['replay_cursor']);self.assertEqual(c['schema'],'WM02_TARGET_WM_CHECKPOINT_V3');self.assertEqual(tree_error(m.state_dict(),clone.state_dict()),0.);self.assertEqual(tree_error(opt.state_dict(),co.state_dict()),0.)
        with self.assertRaises(ValueError):compatible(dict(meta,schema='WM02_TARGET_NATIVE_REPLAY_V2'),meta)
if __name__=='__main__':unittest.main()
