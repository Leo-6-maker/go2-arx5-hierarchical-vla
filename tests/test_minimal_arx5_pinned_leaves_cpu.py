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
    ARMS, LEAVES, ParameterContract, export_official_state, frozen_fingerprint,
    consume_for_official_inference,
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


@pytest.mark.parametrize('arm',ARMS)
def test_pinned_leaf_one_way_official_export_exact_and_no_second_full_table(official,arm):
    # Actual pinned upstream leaves, tiny hidden widths: not full GR00T proof.
    torch.manual_seed(20261019);original=miniature_model(official)
    model=deepcopy(original);contract=ParameterContract(model,arm,10)
    with torch.no_grad():
        for path in LEAVES:model.get_submodule(path).target_b.add_(.125)
    original.load_state_dict(export_official_state(model),strict=True)
    args=(torch.randn(2,1,132),torch.randn(2,40,132),torch.tensor([23,76]),torch.tensor([10,1]))
    with torch.no_grad():expected=model(*args).clone()
    pointers={path:model.get_submodule(path).frozen_W.data_ptr()for path in LEAVES}
    contract._training_session_used=True
    rng=torch.get_rng_state().clone();result=consume_for_official_inference(contract)
    assert result is model and torch.equal(torch.get_rng_state(),rng)
    assert set(model.state_dict())==set(original.state_dict())
    assert all(type(model.get_submodule(path))is official.CategorySpecificLinear for path in LEAVES)
    assert all(model.get_submodule(path).W.data_ptr()==pointers[path]for path in LEAVES)
    with torch.no_grad():assert torch.equal(model(*args),expected)
    for name,value in model.state_dict().items():assert torch.equal(value,original.state_dict()[name])
    assert all(not p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError):contract.validate()
    with pytest.raises(ValueError):consume_for_official_inference(contract)
