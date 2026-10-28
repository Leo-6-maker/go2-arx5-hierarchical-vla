import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DELTA_SPEC = importlib.util.spec_from_file_location(
    "pi0245_physical_delta_test", ROOT / "scripts" / "pi0245_physical_delta.py"
)
DELTA = importlib.util.module_from_spec(DELTA_SPEC)
DELTA_SPEC.loader.exec_module(DELTA)
POSITION_SPEC = importlib.util.spec_from_file_location(
    "pi0246_physical_position_test", ROOT / "scripts" / "pi0246_physical_position.py"
)
POSITION = importlib.util.module_from_spec(POSITION_SPEC)
POSITION_SPEC.loader.exec_module(POSITION)


def tensors():
    clean = torch.zeros(1, 40, 132)
    noisy = torch.zeros_like(clean)
    pred = torch.zeros_like(clean, requires_grad=True)
    mask = torch.ones_like(clean)
    state = torch.tensor([[0.2, -0.1, 0.3, 1., 0., 0., 0., 1., 0.]])
    scale = torch.stack((torch.linspace(0.01, 0.04, 40),) * 3, dim=-1)
    offset = torch.stack((torch.linspace(-0.002, 0.002, 40),) * 3, dim=-1)
    tau = torch.tensor([0.25])
    return clean, noisy, pred, mask, state, scale, offset, tau


def test_oracle_zero_bias_and_masked_points():
    clean, noisy, pred, mask, state, scale, offset, tau = tensors()
    clean.uniform_(-0.1, 0.1)
    noisy.uniform_(-0.1, 0.1)
    with torch.no_grad():
        pred[..., :3] = (clean[..., :3] - noisy[..., :3]) / (1 - tau[:, None, None])
    loss, details = POSITION.physical_position_loss(
        predicted_velocity=pred, clean_action=clean, noisy_trajectory=noisy,
        tau=tau, action_mask=mask, state_eef=state, scale=scale, offset=offset,
    )
    assert loss.item() < 1e-10
    assert details["valid_points"] == 40
    mask[:, 0, :3] = 0
    masked, details = POSITION.physical_position_loss(
        predicted_velocity=pred, clean_action=clean, noisy_trajectory=noisy,
        tau=tau, action_mask=mask, state_eef=state, scale=scale, offset=offset,
    )
    assert torch.isfinite(masked) and details["valid_points"] == 39


def test_constant_physical_bias_uses_time_varying_scale_and_relative_frame():
    clean, noisy, pred, mask, state, scale, offset, tau = tensors()
    with torch.no_grad():
        pred[:, :, 0] = 0.2
    loss, _ = POSITION.physical_position_loss(
        predicted_velocity=pred, clean_action=clean, noisy_trajectory=noisy,
        tau=torch.zeros_like(tau), action_mask=mask, state_eef=state,
        scale=scale, offset=offset,
    )
    expected = ((0.2 * scale[:, 0] / POSITION.P_REF_M) ** 2).mean()
    assert torch.allclose(loss, expected, atol=1e-7, rtol=1e-6)
    # Pinned decoder keeps its original clipping and zero-range behavior.
    sample = torch.tensor([[[2.0, 0.0, -2.0]]])
    decoded = DELTA.decode_xyz(
        sample, state, torch.zeros(1, 3), torch.tensor([[0.01, 0.0, -0.01]])
    )
    assert torch.allclose(decoded[0, 0], torch.tensor([0.21, -0.1, 0.29]), atol=1e-6)


def test_lambda_zero_preserves_native_value_and_gradient_exactly():
    p = torch.tensor([1.5, -0.25], requires_grad=True)
    native = (p.square()).sum()
    position = (p * 7).sum()
    native_grad = torch.autograd.grad(native, p, retain_graph=True)[0]
    combined = POSITION.combine_loss(native, position, coefficient=0.0)
    combined_grad = torch.autograd.grad(combined, p)[0]
    assert combined.item() == native.item()
    assert torch.equal(combined_grad, native_grad)


def test_overlay_adds_position_loss_to_one_native_forward():
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_encoder = nn.Identity()
            self.action_decoder = nn.Identity()
            self.predicted_velocity = nn.Parameter(torch.full((1, 40, 132), 0.1))

        def sample_time(self, batch):
            return torch.full((batch,), 0.25)

        def forward(self, _backbone, action_input):
            action = action_input.action
            noise = torch.randn(action.shape)
            tau = self.sample_time(action.shape[0])
            noisy = (1 - tau[:, None, None]) * noise + tau[:, None, None] * action
            self.action_encoder(noisy)
            prediction = self.action_decoder(self.predicted_velocity.expand_as(action))
            return {"loss": prediction.square().mean(), "action_mask": torch.ones_like(action)}

    clean, _, _, _, state, scale, offset, _ = tensors()
    clean.zero_()
    state_min, state_max = torch.full((9,), -1.0), torch.ones(9)
    head = Head()
    with tempfile.TemporaryDirectory() as temp:
        audit_path = Path(temp) / "forward.jsonl"
        audit = POSITION.install(
            Head, scale=scale, offset=offset, state_min=state_min, state_max=state_max,
            audit_path=audit_path,
        )
        result = head(None, SimpleNamespace(action=clean, state=state[:, None]))
        assert audit["forward_calls"] == 1
        row = json.loads(audit_path.read_text().splitlines()[0])
        assert row["forward_index"] == 1 and row["valid_points"] == 40
        assert torch.isfinite(result["loss"])
        assert result["loss"].item() > row["native_loss"]
        result["loss"].backward()
        assert torch.isfinite(head.predicted_velocity.grad).all()


if __name__ == "__main__":
    test_oracle_zero_bias_and_masked_points()
    test_constant_physical_bias_uses_time_varying_scale_and_relative_frame()
    test_lambda_zero_preserves_native_value_and_gradient_exactly()
    test_overlay_adds_position_loss_to_one_native_forward()
