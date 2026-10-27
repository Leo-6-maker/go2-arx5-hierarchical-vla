"""Optional actual upstream leaves on tiny CPU dimensions, never pretrained GR00T.

Set ARX5_PINNED_LEAF_SOURCE to the independently fetched fixed-blob file.
No downloads, model loading, CUDA or simulator imports are performed by tests.
"""
from copy import deepcopy
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest
torch = pytest.importorskip('torch')
from torch import nn

from scripts.pi0294.generalization.minimal_adapter import (
    ARMS, ParameterContract, export_official_state, frozen_fingerprint,
)


@pytest.fixture
def official():
    path = os.environ.get('ARX5_PINNED_LEAF_SOURCE')
    if not path:
        pytest.skip('Pinned upstream leaf cache not provided; no native claim')
    data = Path(path).read_bytes()
    blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    assert blob == '504785d57cc33a87613dd775cc415cc88574c2ee'
    spec = importlib.util.spec_from_file_location('arx5_fixed_n17_category_leaf_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def miniature_model(official):
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            # Retain padded132 and H40. Hidden width8 is a TEST reduction only.
            self.state_encoder = official.CategorySpecificMLP(32, 132, 8, 8)
            self.action_encoder = official.MultiEmbodimentActionEncoder(132, 8, 32)
            self.action_decoder = official.CategorySpecificMLP(32, 8, 8, 132)
            self.model, self.vlln, self.vl_self_attention = nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 8)
            self.position_embedding = nn.Embedding(40, 8)
            self.state_dropout_prob = 0.

        def forward(self, state, actions, time, ids):
            h = self.action_encoder(actions, time, ids) + self.state_encoder(state, ids)
            h = self.model(h) + self.position_embedding.weight
            return self.action_decoder(self.vl_self_attention(self.vlln(h)), ids)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, self.action_head = nn.Identity(), Head()

        def forward(self, state, actions, time, ids):
            return self.action_head(state, actions, time, ids)
    return Model().eval()


@pytest.mark.parametrize('arm', ARMS)
def test_actual_pinned_leaf_zero_step_132d_h40_backward_export(official, arm):
    torch.manual_seed(20261019)
    original = miniature_model(official)
    model = deepcopy(original)
    contract = ParameterContract(model, arm, 10)
    args = (torch.randn(2, 1, 132), torch.randn(2, 40, 132), torch.tensor([23, 76]), torch.tensor([10, 1]))
    with torch.no_grad():
        assert torch.equal(original(*args), model(*args))
    frozen = frozen_fingerprint(model)
    optimizer = torch.optim.AdamW(contract.parameters(), lr=.001, weight_decay=.1)
    model.train()
    value = model(*args)
    assert value.shape == (2, 40, 132)
    value.square().mean().backward()
    assert model.action_head.state_encoder.layer1.target_W.grad.abs().sum() > 0
    targets = [(n, p) for n, p in model.named_parameters() if '.target_' in n]
    assert len(targets) == 14
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for _, p in targets)
    optimizer.step()
    assert frozen_fingerprint(model) == frozen
    original.load_state_dict(export_official_state(model), strict=True)
    assert torch.equal(original(*args), model(*args))
