import unittest
import numpy as np
import torch
from go2arx5_vla.world_model.pilot1_residual import TCPResidual, loss, inputs


class LocalResidual(unittest.TestCase):
    def test_local_offline_admission_needs_actual_host_grant_and_forbids_physics(self):
        import tempfile,json,time,os
        from pathlib import Path
        from unittest.mock import patch
        from go2arx5_vla.world_model.runtime import admitted_run
        from go2arx5_vla.world_model.pilot1_scope import PLAN_SHA
        from go2arx5_vla.world_model.storage import digest
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);path=root/'SPEC.json';registry=root/'REGISTRY.json';lease=root/'LEASE.json'
            spec=dict(source_root=str(Path(__file__).resolve().parents[2]),source_sha='1'*40,source_files={},registered_root=str(root),
                profile='wm02-pilot1-offline-cpu',host=os.uname().nodename,uid=os.getuid(),deadline_epoch=time.time()+60,
                execution_site='LOCAL_PC_PILOT1_CPU',physics=False,scope='WM02_PILOT1_PC_MODEL_ONLY',plan_sha=PLAN_SHA,
                kind='ARX_RESIDUAL',device='cpu',endpoint=4,role='UNIT_TEST_ONLY',registry_snapshot=str(registry),lease=str(lease),cpus=[min(os.sched_getaffinity(0))])
            path.write_text(json.dumps(spec));lease.write_text(json.dumps(dict(status='ACTIVE',run_uuid=root.name,spec_sha256=digest(path))))
            entry=dict(role=spec['role'],path=str(root),source_sha=spec['source_sha'],executor_node=spec['host'],scope=spec['scope'],local_lease=str(lease),plan_sha=PLAN_SHA,valid_until_epoch=spec['deadline_epoch'])
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                registry.write_text(json.dumps(dict(entries=[dict(entry,scope='WRONG')])) )
                with self.assertRaises(PermissionError):
                    with admitted_run(path,'fit','wm02-pilot1-offline-cpu',root/'fit'):pass
                registry.write_text(json.dumps(dict(entries=[entry])))
                with self.assertRaises(PermissionError):
                    with admitted_run(path,'fit','wm02-pilot1-offline-cpu',root/'fit',physics=True):pass
                with admitted_run(path,'fit','wm02-pilot1-offline-cpu',root/'fit')as accepted:self.assertEqual(accepted['device'],'cpu')

    def test_inputs_exclude_future_observations_and_blind_only_new_choice(self):
        from go2arx5_vla.world_model.target_arx import CausalARX
        fields=[('qpos',34),('qvel',32),('ctrl',20),('tcp_position_m',3),('tcp_rotation6d',6),('object_position_m',3),('object_rotation6d',6),('finger_position_m',2)]
        meta=dict(state_fields=[dict(name=k,size=n)for k,n in fields],joints=[dict(name='task_cube_free',qpos_address=27,qvel_address=26)])
        teacher=CausalARX(meta);teacher.mean=np.zeros(106);teacher.scale=np.ones(106);teacher.feature_mean=np.zeros(233);teacher.feature_scale=np.ones(233);teacher.weight=np.zeros((234,106))
        rng=np.random.default_rng(36002);data=dict(state=rng.normal(size=(2,25,106)),history_goal=np.zeros((2,8,17)),goal=np.zeros((2,17)),source_age=np.zeros((2,24)),window_start_s=np.zeros(2),future_prefix_remaining=np.full(2,7.))
        data['goal'][:,0]=.001
        actual=inputs(teacher,data,meta);changed=dict(data,state=data['state'].copy());changed['state'][:,9:]=1e6
        for k in ('x','base','observed'):np.testing.assert_array_equal(actual[k],inputs(teacher,changed,meta)[k])
        blind=inputs(teacher,data,meta,blind_choice=True);np.testing.assert_array_equal(blind['x'][:,:-4],actual['x'][:,:-4]);np.testing.assert_array_equal(blind['x'][:,-2:],actual['x'][:,-2:])
        np.testing.assert_array_equal(blind['x'][:,-4:-2],0.)

    def test_zero_output_is_exact_ARX_and_future_labels_never_enter_forward(self):
        torch.manual_seed(36002)
        model=TCPResidual(12)
        x=torch.randn(4,12);base=torch.randn(4,16,3)*.001;observed=torch.randn(4,3)
        model.fit_normalizer(x)
        for horizon in (1,4,8,16):
            torch.testing.assert_close(model(x,base,observed,horizon),base[:,:horizon],atol=0,rtol=0)
        self.assertEqual(tuple(model(x,base,observed).shape),(4,16,3))

    def test_nonzero_update_and_full_optimizer_recovery_on_recurrent_loss(self):
        torch.manual_seed(36002);m=TCPResidual(12);x=torch.randn(4,12);base=torch.zeros(4,16,3);observed=torch.zeros(4,3)
        m.fit_normalizer(x);opt=torch.optim.AdamW(m.parameters(),lr=.001);target=torch.ones_like(base)*.0002
        def update(model,optimizer):
            optimizer.zero_grad();value=loss(model(x,base,observed),target,base);value.backward();optimizer.step();return float(value.detach())
        before=update(m,opt);saved_m={k:v.detach().clone()for k,v in m.state_dict().items()}
        import copy
        saved_o=copy.deepcopy(opt.state_dict());clone=TCPResidual(12);co=torch.optim.AdamW(clone.parameters(),lr=.001)
        clone.load_state_dict(saved_m);co.load_state_dict(saved_o);update(m,opt);update(clone,co)
        for a,b in zip(m.state_dict().values(),clone.state_dict().values()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertGreater(before,0);self.assertGreater(float(m(x,base,observed).detach().abs().sum()),0)


if __name__=='__main__':unittest.main()
