import unittest,tempfile
from pathlib import Path
from go2arx5_vla.world_model.target_neural import validate_device

try:
    import torch,numpy as np
    from go2arx5_vla.world_model.target_training import TargetWM,update,save,restore,tree_error
except ImportError:torch=None


class TargetDevice(unittest.TestCase):
    def test_only_authorized_UUID_and_real_headroom_not_empty_gpu_claim(self):
        uuid='GPU-845348f1-1511-dc19-0877-cc97658e3edd';spec=dict(GPU_uuid=uuid,profile='wm02-auto1-target-neural',physics=False,data_domain='WM02_TARGET_NATIVE_STATE_V1',endpoint=200,Actor_updates=0,allocator_limit_bytes=4*1024**3)
        validate_device(spec,dict(uuid=uuid,free_bytes=6*1024**3))
        for changed in(dict(endpoint=5000),dict(physics=True),dict(Actor_updates=1),dict(GPU_uuid='different')):
            with self.assertRaises(PermissionError):validate_device(dict(spec,**changed),dict(uuid=uuid,free_bytes=6*1024**3))
        with self.assertRaises(PermissionError):validate_device(spec,dict(uuid=uuid,free_bytes=4*1024**3))


@unittest.skipIf(torch is None,'torch required for neural component tests')
class TargetNeuralContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        fields=[('qpos',7),('qvel',6),('ctrl',1),('tcp_position_m',3),('tcp_rotation6d',6),('object_position_m',3),('object_rotation6d',6),('finger_position_m',2)]
        self.meta=dict(state_fields=[dict(name=n,size=s)for n,s in fields],state_dimension=sum(s for n,s in fields))
        self.data=dict(state=torch.randn(2,25,self.meta['state_dimension'])*.01,goal=torch.randn(2,17)*.01,source_age=torch.tensor([[.002+(i%7)*.05 for i in range(24)]]*2))
        self.m=TargetWM(self.meta,hidden=8);self.m.fit_normalizer(self.data['state']);self.opt=torch.optim.AdamW(self.m.parameters(),lr=1e-3,weight_decay=1e-5)
    def test_future_actual_states_and_post_renewal_source_times_are_not_inputs(self):
        self.m.eval()
        a=self.m.predict(**self.data)
        altered={k:v.clone()for k,v in self.data.items()};altered['state'][:,9:]+=100.;altered['source_age'][:,9:]+=100.
        b=self.m.predict(**altered);self.assertEqual(tree_error(a,b),0.)
        # Privileged teacher-forcing is a distinct diagnostic, never FREE input.
        teacher=self.m.predict(**self.data,teacher_forced=True)
        changed=self.m.predict(**altered,teacher_forced=True)
        self.assertGreater(tree_error(teacher,changed),0.)
    def test_stochastic_optimizer_RNG_and_cursor_restore_exactly(self):
        provenance=dict(purpose='SYNTHETIC_COMPONENT_TEST_NO_NATIVE_QUALIFICATION')
        with tempfile.TemporaryDirectory()as tmp:
            update(self.m,self.opt,self.data,0);path=Path(tmp)/'state.pt';save(path,self.m,self.opt,1,8,provenance)
            update(self.m,self.opt,self.data,8)
            other,opt,c=restore(path,'cpu',provenance);update(other,opt,self.data,c['replay_cursor'])
            self.assertEqual(tree_error(self.m.state_dict(),other.state_dict()),0.);self.assertEqual(tree_error(self.opt.state_dict(),opt.state_dict()),0.)
            with self.assertRaises(ValueError):restore(path,'cpu',dict(purpose='different'))
if __name__=='__main__':unittest.main()
