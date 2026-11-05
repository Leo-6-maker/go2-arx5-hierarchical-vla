import unittest
import numpy as np
import torch
from go2arx5_vla.world_model.pilot1_actor import TorchARX, Actor, features, export_arrays, numpy_logits
from go2arx5_vla.world_model.target_arx import CausalARX


def fixture():
    fields = [('qpos',34),('qvel',32),('ctrl',20),('tcp_position_m',3),('tcp_rotation6d',6),
              ('object_position_m',3),('object_rotation6d',6),('finger_position_m',2)]
    meta = dict(state_fields=[dict(name=k,size=n)for k,n in fields],
                joints=[dict(name='task_cube_free',qpos_address=27,qvel_address=26)])
    model = CausalARX(meta); n=106; f=233
    rng = np.random.default_rng(36)
    model.mean=rng.normal(size=n);model.scale=np.ones(n)
    model.feature_mean=np.zeros(f);model.feature_scale=np.ones(f);model.weight=rng.normal(size=(f+1,n))*.00001
    data=dict(state=rng.normal(size=(3,25,n)),goal=rng.normal(size=(3,17)),history_goal=rng.normal(size=(3,8,17)),
              source_age=np.full((3,24),.05),window_start_s=np.zeros(3),future_prefix_remaining=np.full(3,7.))
    return meta,model,data


class PilotActor(unittest.TestCase):
    def test_port_preserves_numpy_recurrence_and_model_input_gradient(self):
        meta,model,data=fixture();ids=np.arange(3);port=TorchARX(model)
        args=dict(prefix=torch.tensor(data['state'][:,:9]),goal=torch.tensor(data['goal'],requires_grad=True),
                  origin=torch.zeros(3,dtype=torch.float64),age=torch.full((3,),.05,dtype=torch.float64),remaining=torch.full((3,),7.,dtype=torch.float64))
        actual=port(**args);np.testing.assert_allclose(actual.detach(),model.predict(data,ids),atol=2e-14,rtol=0)
        gradient=torch.autograd.grad(actual[0,-1,86],args['goal'])[0][0,0].item();values=[]
        for sign in (-1,1):
            varied=dict(args);varied['goal']=args['goal'].detach().clone();varied['goal'][0,0]+=sign*1e-6
            values.append(port(**varied)[0,-1,86].item())
        self.assertAlmostEqual(gradient,(values[1]-values[0])/2e-6,places=8)

    def test_student_inputs_exclude_future_object_and_parent_information(self):
        meta,model,data=fixture();ids=np.arange(3);goals=np.zeros((3,2));expected=features(data,meta,ids,goals)
        data['state'][:,9:]=1e9;data['state'][:,:9,27:34]=1e9;data['state'][:,:9,60:66]=1e9
        data['state'][:,:9,95:104]=1e9;data['goal'][:]=1e9;data['episode_index']=np.array([99,98,97])
        np.testing.assert_array_equal(features(data,meta,ids,goals),expected)
        self.assertFalse(np.array_equal(features(data,meta,ids,goals+.001),expected))

    def test_trained_model_export_uses_same_float32_input_semantics(self):
        torch.manual_seed(36);actor=Actor(12);x=torch.randn(10,12);actor.fit_normalizer(x)
        opt=torch.optim.AdamW(actor.parameters(),lr=.001);loss=actor(x).square().mean();loss.backward();opt.step()
        actual=numpy_logits(export_arrays(actor),x.numpy().astype(np.float64))
        np.testing.assert_allclose(actual,actor(x).detach().numpy(),atol=1e-6,rtol=0)


if __name__=='__main__':unittest.main()
