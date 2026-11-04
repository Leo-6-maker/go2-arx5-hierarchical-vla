import unittest,tempfile
from pathlib import Path
import numpy as np,torch
from go2arx5_vla.world_model.target_training import TargetWM,update,save,restore,tree_error,tensors,rotation6d_angle
from go2arx5_vla.world_model.target_training_v2 import compatible,MIGRATION
from go2arx5_vla.world_model.target_neural import validate_device

class WindowTraining(unittest.TestCase):
    def test_decoded_SO3_reports_radians_and_masks_invalid_rotation(self):
        identity=torch.tensor([[1.,0,0,0,1,0]]);quarter=torch.tensor([[0.,1,0,-1,0,0]])
        angle,valid=rotation6d_angle(identity,identity);self.assertEqual(float(angle[0]),0.);self.assertTrue(bool(valid[0]))
        angle,valid=rotation6d_angle(quarter,identity);self.assertAlmostEqual(float(angle[0]),np.pi/2,places=6)
        _,valid=rotation6d_angle(torch.zeros_like(identity),identity);self.assertFalse(bool(valid[0]))
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(8)
        fields=[('qpos',7),('qvel',6),('ctrl',1),('tcp_position_m',3),('tcp_rotation6d',6),('object_position_m',3),('object_rotation6d',6),('finger_position_m',2)]
        self.meta=dict(schema='WM02_TARGET_NATIVE_REPLAY_V2',state_fields=[dict(name=n,size=s)for n,s in fields],state_dimension=sum(s for n,s in fields),joints=[],actuator_gear=[],actuator_gain=[],robot_fingerprint='fixture',original_model_sha256='original')
        self.data=dict(state=torch.randn(2,25,self.meta['state_dimension'])*.01,goal=torch.randn(2,17)*.01,source_age=torch.zeros(2,24),window_start_s=torch.tensor([.3,2.1]),window_start_atom=torch.tensor([6.,42.]));self.m=TargetWM(self.meta,8);self.m.fit_normalizer(self.data['state']);self.opt=torch.optim.AdamW(self.m.parameters(),lr=1e-3,weight_decay=1e-5)
    def test_window_elapsed_and_future_source_mask_are_causal(self):
        self.m.eval();a=self.m.predict(**self.data);other={k:v.clone()for k,v in self.data.items()};other['state'][:,9:]+=20.;other['source_age'][:,9:]+=20.;b=self.m.predict(**other);self.assertEqual(tree_error(a,b),0.)
    def test_versioned_action_units_ablation_and_checkpoint_recovery(self):
        state=self.data['state'][:,0];goal=self.data['goal'].clone();clock=torch.zeros(2);known=torch.ones(2)
        raw=self.m.encode(state,goal,clock,clock,known)
        scaled=TargetWM(dict(self.meta,action_encoding='SCALED_WORLD_RELATIVE_1CM_V2'),8);scaled.fit_normalizer(self.data['state']);features=scaled.encode(state,goal,clock,clock,known)
        torch.testing.assert_close(features[:,:3],raw[:,:3]/.01);torch.testing.assert_close(features[:,10:13],raw[:,10:13]/.01);torch.testing.assert_close(goal,self.data['goal'])
        none=TargetWM(dict(self.meta,action_encoding='NO_ACTION_V2'),8).encode(state,goal,clock,clock,known);self.assertEqual(float(none[:,:17].abs().max()),0.);torch.testing.assert_close(none[:,17:],raw[:,17:])
        opt=torch.optim.AdamW(scaled.parameters(),lr=1e-3,weight_decay=1e-5)
        with tempfile.TemporaryDirectory()as tmp:
            update(scaled,opt,self.data,0);p=Path(tmp)/'model.pt';save(p,scaled,opt,1,8,dict(recipe='scaled'))
            update(scaled,opt,self.data,8);clone,co,c=restore(p,'cpu',dict(recipe='scaled'));self.assertEqual(clone.action_encoding,'SCALED_WORLD_RELATIVE_1CM_V2');update(clone,co,self.data,c['replay_cursor']);self.assertEqual(tree_error(scaled.state_dict(),clone.state_dict()),0.)
    def test_v2_optimizer_RNG_cursor_and_schema_roundtrip(self):
        with tempfile.TemporaryDirectory()as tmp:
            update(self.m,self.opt,self.data,0);p=Path(tmp)/'model.pt';save(p,self.m,self.opt,1,8,dict(recipe='V2_component'))
            update(self.m,self.opt,self.data,8);clone,opt,c=restore(p,'cpu',dict(recipe='V2_component'));update(clone,opt,self.data,c['replay_cursor']);self.assertEqual(c['schema'],'WM02_TARGET_WM_CHECKPOINT_V2');self.assertEqual(tree_error(self.m.state_dict(),clone.state_dict()),0.);self.assertEqual(tree_error(self.opt.state_dict(),opt.state_dict()),0.)
    def test_legacy_evaluation_needs_named_migration_and_native_fingerprint(self):
        old=dict(self.meta,schema='WM02_TARGET_NATIVE_STATE_V1',model_sha256='original');old.pop('robot_fingerprint')
        with self.assertRaises(ValueError):compatible(old,self.meta)
        compatible(old,self.meta,migration=MIGRATION)
        with self.assertRaises(ValueError):compatible(dict(self.meta,robot_fingerprint='changed'),self.meta)
    def test_explicit_finite_endpoint_has_no_cumulative_engineering_cap(self):
        uuid='GPU-845348f1-1511-dc19-0877-cc97658e3edd';s=dict(night1_binding='actual-binding-verified-separately',GPU_uuid=uuid,profile='wm02-auto1-target-neural',physics=False,data_domain='WM02_TARGET_NATIVE_REPLAY_V2',endpoint=1000,Actor_updates=0,allocator_limit_bytes=4*1024**3);validate_device(s,dict(uuid=uuid,free_bytes=6*1024**3))
        with self.assertRaises(PermissionError):validate_device(dict(s,endpoint=True),dict(uuid=uuid,free_bytes=6*1024**3))
if __name__=='__main__':unittest.main()
