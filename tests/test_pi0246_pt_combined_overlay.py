import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PINNED_HELPERS = os.environ.get("PI0246_PT_TEST_RUNTIME")
if PINNED_HELPERS:
    sys.path.insert(0, PINNED_HELPERS)
from pi0246_pt_combined_overlay import install


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = nn.Identity()
        self.action_decoder = nn.Identity()
        self.predicted_velocity = nn.Parameter(torch.full((1, 40, 132), 0.1))
        self.forward_calls = 0

    def sample_time(self, batch):
        return torch.full((batch,), 0.25)

    def forward(self, _backbone, action_input):
        self.forward_calls += 1
        action = action_input.action
        noise = torch.randn(action.shape, device=action.device, dtype=action.dtype)
        tau = self.sample_time(action.shape[0])
        noisy = (1 - tau[:, None, None]) * noise + tau[:, None, None] * action
        self.action_encoder(noisy)
        pred = self.action_decoder(self.predicted_velocity.expand_as(action))
        mask = action_input.action_mask
        return {"loss": pred.square().mean(), "action_mask": mask}


class CombinedOverlayTests(unittest.TestCase):
    def test_combines_both_losses_from_one_native_forward_and_noise_draw(self):
        action = torch.zeros(1, 40, 132)
        mask = torch.zeros_like(action)
        mask[..., :10] = 1
        state = torch.zeros(1, 1, 132)
        state[0, 0, :9] = torch.tensor([.2, -.1, .3, 1., 0., 0., 0., 1., 0.])
        batch = SimpleNamespace(action=action, action_mask=mask, state=state)
        scale = torch.full((40, 3), .02)
        offset = torch.zeros(40, 3)
        state_min = torch.full((9,), -1.)
        state_max = torch.ones(9)
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "pt-forward.jsonl"
            torch.manual_seed(1)
            baseline = Head()
            baseline_state = baseline.state_dict()
            torch.manual_seed(22301)
            baseline_output = baseline(None, batch)
            baseline_next_rng = torch.randn(1)
            audit = install(Head, scale=scale, offset=offset, state_min=state_min,
                            state_max=state_max, audit_path=log)
            head = Head()
            head.load_state_dict(baseline_state)
            torch.manual_seed(22301)
            result = head(None, batch)
            next_rng = torch.randn(1)
            self.assertEqual(head.forward_calls, 1)
            self.assertEqual(audit["forward_calls"], 1)
            row = json.loads(log.read_text().strip())
            self.assertEqual(row["valid_points"], 40)
            self.assertEqual(row["valid_transitions"], 39)
            self.assertEqual(row["native_loss"], baseline_output["loss"].item())
            self.assertAlmostEqual(row["total_loss"], row["native_loss"] + .1 * row["physical_position_loss"] + .1 * row["physical_delta_loss"], places=6)
            self.assertTrue(torch.equal(next_rng, baseline_next_rng))
            result["loss"].backward()
            self.assertTrue(torch.isfinite(head.predicted_velocity.grad).all())


if __name__ == "__main__":
    unittest.main()
