import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from go2arx5_vla.world_model.features import STATE_DIM, ROTATIONS, SLICES
from go2arx5_vla.world_model.model import Ensemble, exp_rotation, multistep_loss
from go2arx5_vla.world_model.checkpoint import save, restore
from go2arx5_vla.world_model.reward import potential, reward

torch.set_num_threads(1)


def sample():
    s = torch.zeros(4, 4, STATE_DIM)
    for sl in ROTATIONS: s[..., sl] = torch.eye(3).ravel()
    a = torch.rand(4, 4, 56)*.001
    g = torch.tensor([[.45, .08, .46]]).repeat(4, 1)
    y = s.clone(); y[..., :6] += .001
    e = torch.zeros(4, 4, 4)
    return s, a, g, y, e


class ModelTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(1103)

    def model(self): return Ensemble(np.zeros(STATE_DIM), np.ones(STATE_DIM), width=32)

    def test_so3_zero_and_nonzero_gradients_are_finite(self):
        x = torch.tensor([[0., 0., 0.], [.02, -.01, .03]], requires_grad=True)
        r = exp_rotation(x)
        torch.testing.assert_close(r@r.transpose(-1, -2), torch.eye(3).expand(2, 3, 3), atol=1e-6, rtol=0)
        r[:, 0, 1].sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_frozen_world_model_keeps_action_gradient(self):
        model = self.model()
        original = copy.deepcopy(model.state_dict())
        for parameter in model.parameters(): parameter.requires_grad_(False)
        states, actions, goals, _, _ = sample()
        actions = actions[:, 0].clone().requires_grad_()
        prediction, _ = model(states[:, 0], actions, goals)[0]
        prediction[:, :6].square().sum().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(actions.grad.abs().sum().item(), 0)
        for key, value in model.state_dict().items(): torch.testing.assert_close(original[key], value, rtol=0, atol=0)

    def test_multistep_is_not_teacher_forced(self):
        model = self.model(); values = list(sample()); values[0].requires_grad_()
        loss, _ = multistep_loss(model, model.members[0], *values)
        loss.backward()
        self.assertEqual(values[0].grad[:, 1:].abs().sum().item(), 0.)
        self.assertGreater(values[0].grad[:, 0].abs().sum().item(), 0.)

    def test_rejected_command_event_does_not_regress_a_future_state(self):
        model=self.model(); states,actions,goals,_,_=sample()
        member=model.members[0]
        _,logits=member(states[:,0],actions[:,0],goals,model.mean,model.scale)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,-1],torch.ones(4))
        loss.backward()
        self.assertIsNone(member.delta.weight.grad)
        self.assertIsNone(member.rotation.weight.grad)
        self.assertGreater(member.events.weight.grad.abs().sum().item(),0.)

    def test_full_checkpoint_next_update_exact(self):
        model = self.model(); opts = [torch.optim.AdamW(m.parameters(), lr=3e-4) for m in model.members]
        rngs = [np.random.default_rng(1103+i) for i in range(3)]
        values = sample()
        def update(m, optimizers, generators):
            ids = []
            for member, optimizer, rng in zip(m.members, optimizers, generators):
                idx = rng.integers(0, 4, 4); ids.append(idx.copy())
                loss, _ = multistep_loss(m, member, *[v[idx] for v in values]); optimizer.zero_grad(); loss.backward(); optimizer.step()
            return ids
        update(model, opts, rngs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            save(path, model, opts, rngs, updates=1, replay_identity='TRAIN_ONLY', source_sha='a'*40, recipe={'effective_batch':256})
            expected_ids = update(model, opts, rngs); expected = copy.deepcopy(model.state_dict())
            replica = self.model(); new_opts = [torch.optim.AdamW(m.parameters(), lr=3e-4) for m in replica.members]
            new_rngs = [np.random.default_rng(i) for i in range(3)]
            recovered = restore(path, replica, new_opts, new_rngs, replay_identity='TRAIN_ONLY', device='cpu')
            self.assertEqual(recovered['replay_cursor'], 256)
            actual_ids = update(replica, new_opts, new_rngs)
            for a, b in zip(expected_ids, actual_ids): np.testing.assert_array_equal(a, b)
            for key, value in replica.state_dict().items(): torch.testing.assert_close(expected[key], value, rtol=0, atol=0)
            with self.assertRaises(ValueError): restore(path, replica, new_opts, new_rngs, replay_identity='TEST', device='cpu')

    def test_reward_order_and_real_model_call_equality(self):
        state = sample()[0][:, 0].clone(); goal = torch.tensor([[.45, .08, .46]]).repeat(4, 1)
        state[:, SLICES['object_position_m']] = goal
        state[:, SLICES['tcp_position_m']] = goal+torch.tensor([0., 0., .13])
        state[:, SLICES['finger_q_m']] = .044
        events = torch.zeros(4, 4)
        good = potential(state, goal, events)
        held = events.clone(); held[:, 2] = 1.
        self.assertTrue((good > potential(state, goal, held)).all())
        command = torch.zeros(4, 56)
        ordinary = reward(state, state, command, goal, events, events, torch.zeros(4))
        collided = reward(state, state, command, goal, events, events, torch.ones(4))
        torch.testing.assert_close(ordinary-collided, torch.ones(4))
        differentiable = state.clone().requires_grad_()
        torch.testing.assert_close(ordinary, reward(state, differentiable, command, goal, events, events, torch.zeros(4)))


if __name__ == '__main__': unittest.main()
