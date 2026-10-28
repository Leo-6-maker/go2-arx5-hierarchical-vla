#!/usr/bin/env python3
"""CPU observer check: same native random calls, lambda-zero gradients, one pass."""

from __future__ import annotations

import json
import argparse
import inspect
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ("scripts" if (root / "scripts").is_dir() else "code")))
from pi0245_native_overlay import install


class FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_encoder = nn.Linear(132, 132, bias=False)
        self.action_decoder = nn.Linear(132, 132, bias=False)

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.rand(batch_size, device=device, dtype=dtype)

    def forward(self, backbone_output: object, action_input: object) -> dict:
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        tau = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        noisy = (1 - tau[:, None, None]) * noise + tau[:, None, None] * actions
        pred = self.action_decoder(self.action_encoder(noisy))
        mask = action_input.action_mask
        action_loss = F.mse_loss(pred, actions - noise, reduction="none") * mask
        return {"loss": action_loss.sum() / (mask.sum() + 1e-6), "action_loss": action_loss, "action_mask": mask}


class Batch:
    pass


def one(head: FakeHead, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(22301)
    output = head(None, batch)
    gradient = torch.autograd.grad(output["loss"], head.action_decoder.weight)[0]
    next_random = torch.randn(3)
    return output["loss"], gradient, next_random


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groot-source", type=Path, required=True)
    args = parser.parse_args()
    batch = Batch()
    batch.action = torch.arange(40 * 132, dtype=torch.float32).reshape(1, 40, 132) / 20000
    batch.action_mask = torch.zeros_like(batch.action)
    batch.action_mask[..., :10] = 1
    batch.state = torch.zeros((1, 1, 132), dtype=torch.float32)
    batch.state[0, 0, 3:9] = torch.tensor([1, 0, 0, 0, 1, 0])
    torch.manual_seed(1)
    original = FakeHead()
    base = original.state_dict()
    direct, direct_grad, direct_next = one(original, batch)
    control = FakeHead()
    control.load_state_dict(base)
    scale = torch.full((40, 3), 0.1)
    offset = torch.zeros((40, 3))
    state_min = torch.full((9,), -1.0)
    state_max = torch.full((9,), 1.0)
    audit = install(FakeHead, scale=scale, offset=offset, state_min=state_min, state_max=state_max, treatment=False)
    observed, observed_grad, observed_next = one(control, batch)
    assert torch.equal(direct, observed)
    assert torch.equal(direct_grad, observed_grad)
    assert torch.equal(direct_next, observed_next)
    assert audit["forward_calls"] == 1 and audit["first_forward"]["valid_transitions"] == 39
    assert torch.randn.__name__ == "randn"
    class TreatmentHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_encoder = nn.Linear(132, 132, bias=False)
            self.action_decoder = nn.Linear(132, 132, bias=False)

        sample_time = FakeHead.sample_time
        forward = FakeHead.forward

    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "forward.jsonl"
        treatment = install(TreatmentHead, scale=scale, offset=offset, state_min=state_min,
                           state_max=state_max, treatment=True, audit_path=log_path)
        logged_head = TreatmentHead()
        torch.manual_seed(22301)
        treatment_output = logged_head(None, batch)
        treatment_output["loss"].backward()
        logged_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert len(logged_rows) == 1 and logged_rows[0]["forward_index"] == 1
        assert logged_rows[0]["valid_transitions"] == 39 and treatment["forward_calls"] == 1
        assert logged_head.action_decoder.weight.grad is not None
    sys.path.insert(0, str(args.groot_source))
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    source = inspect.getsource(Gr00tN1d7ActionHead.forward)
    assert "noise = torch.randn(actions.shape" in source and "t = self.sample_time" in source
    assert "action_loss = F.mse_loss(pred_actions, velocity" in source
    original = Gr00tN1d7ActionHead.forward
    with tempfile.TemporaryDirectory() as temp_dir:
        native_audit = install(Gr00tN1d7ActionHead, scale=scale, offset=offset, state_min=state_min,
                               state_max=state_max, treatment=True,
                               audit_path=Path(temp_dir) / "native-forward.jsonl")
    assert native_audit["forward_calls"] == 0 and Gr00tN1d7ActionHead.forward is not original
    Gr00tN1d7ActionHead.forward = original
    del Gr00tN1d7ActionHead._pi0245_overlay
    print(json.dumps({"status": "PASS_CPU_NATIVE_OBSERVER_PARITY", "same_loss": True, "same_gradient": True, "same_rng_suffix": True, "forward_calls": audit["forward_calls"], "native_import_and_patch": True, "model_weight_loads": 0}))


if __name__ == "__main__":
    main()
