"""CPU engineering tests, not GR00T/CUDA/model or task qualification."""
from copy import deepcopy
import io

import pytest
torch = pytest.importorskip('torch')
from torch import nn

from scripts.pi0294.generalization.minimal_adapter import (
    ARMS, LEAVES, ParameterContract, TargetCategoryLinear,
    export_official_state, frozen_fingerprint, matched_flow_rng,
    restore_parameter_state,
)


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class CategoryLinear(nn.Module):
    """Shape/forward fixture; real pinned-source leaf is separately checked."""
    def __init__(self, cats=12, din=4, dout=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(cats, din, dout) * .2)
        self.b = nn.Parameter(torch.randn(cats, dout) * .1)

    def forward(self, x, ids):
        return torch.bmm(x, self.W[ids]) + self.b[ids].unsqueeze(1)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1, self.layer2 = CategoryLinear(), CategoryLinear()

    def forward(self, x, ids):
        return self.layer2(torch.relu(self.layer1(x, ids)), ids)


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1, self.W2, self.W3 = CategoryLinear(), CategoryLinear(), CategoryLinear()

    def forward(self, x, ids):
        return self.W3(torch.tanh(self.W2(self.W1(x, ids), ids)), ids)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder, self.action_encoder, self.action_decoder = MLP(), ActionEncoder(), MLP()
        self.model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.7), nn.BatchNorm1d(4))
        self.vlln, self.vl_self_attention = nn.LayerNorm(4), nn.Linear(4, 4)
        self.position_embedding = nn.Embedding(40, 4)
        self.state_dropout_prob = 0.

    def forward(self, x, ids):
        h = self.state_encoder(x, ids) + self.action_encoder(x, ids)
        h = self.vl_self_attention(self.vlln(h))
        h = self.model(h[:, 0])[:, None] + self.position_embedding.weight[0]
        return self.action_decoder(h, ids)


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.5))
        self.action_head = Head()

    def forward(self, x, ids):
        return self.action_head(self.backbone(x), ids)


def fixture_pair():
    torch.manual_seed(79)
    original = FixtureModel().eval()
    a, b = deepcopy(original), deepcopy(original)
    return original, a, b, ParameterContract(a, ARMS[0], 10), ParameterContract(b, ARMS[1], 10)


@pytest.mark.parametrize('ids', [[10, 10], [0, 11], [10, 2], [11, 10]])
def test_wrapped_zero_step_exact_matches_original_and_both_arms(ids):
    original, a, b, _, _ = fixture_pair()
    x, ids = torch.randn(2, 1, 4), torch.tensor(ids)
    with torch.no_grad():
        reference = original(x, ids)
        assert torch.equal(a(x, ids), reference)
        assert torch.equal(b(x, ids), reference)


@pytest.mark.parametrize('arm', ARMS)
def test_adamw_target_only_other_categories_and_shared_freeze(arm):
    _, a, b, ca, cb = fixture_pair()
    model, contract = (a, ca) if arm == ARMS[0] else (b, cb)
    frozen = frozen_fingerprint(model)
    original = export_official_state(model)
    optimizer = torch.optim.AdamW(contract.parameters(), lr=.01, weight_decay=.2)
    model.train()  # The Trainer does this too: hooks must reassert equal modes.
    x, ids = torch.randn(2, 1, 4), torch.tensor([10, 10])
    loss = model(x, ids).square().sum()
    loss.backward()
    for path in LEAVES:
        grad = model.get_submodule(path).target_W.grad
        assert grad is not None and torch.isfinite(grad).all()
    assert any(model.get_submodule(p).target_W.grad.abs().sum() > 0 for p in LEAVES[:2])
    optimizer.step()
    assert frozen_fingerprint(model) == frozen
    exported = export_official_state(model)
    for path in LEAVES:
        assert torch.equal(exported[path + '.W'][:10], original[path + '.W'][:10])
        assert torch.equal(exported[path + '.W'][11:], original[path + '.W'][11:])
    assert any(not torch.equal(exported[p + '.W'][10], original[p + '.W'][10]) for p in LEAVES)
    shared_changed = not torch.equal(exported['action_head.model.0.weight'],
                                     original['action_head.model.0.weight'])
    assert shared_changed is (arm == ARMS[1])
    assert not model.action_head.training and not model.backbone.training
    assert not model.action_head.model[1].training


def test_masked_full_table_adamw_is_a_real_negative_control():
    layer = CategoryLinear()
    before = layer.W.detach().clone()
    layer.W.register_hook(lambda g: g * torch.nn.functional.one_hot(torch.tensor(10), 12)[:, None, None])
    optimizer = torch.optim.AdamW(layer.parameters(), lr=.01, weight_decay=.2)
    layer(torch.ones(1, 1, 4), torch.tensor([10])).sum().backward()
    optimizer.step()
    assert not torch.equal(before[0], layer.W[0])  # Zero gradient did not avoid decay.


def test_shared_frozen_forward_still_has_input_gradient_path():
    _, model, _, contract, _ = fixture_pair()
    x = torch.randn(2, 1, 4, requires_grad=True)
    model(x, torch.tensor([10, 10])).square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert model.action_head.state_encoder.layer1.target_W.grad.abs().sum() > 0
    assert model.action_head.model[0].weight.grad is None
    assert all(p.requires_grad for p in contract.parameters())


def test_export_reconstructs_ordinary_model_with_strict_original_keys():
    original, model, _, contract, _ = fixture_pair()
    with torch.no_grad():
        model.action_head.action_decoder.layer2.target_b.add_(.3)
    original.load_state_dict(export_official_state(model), strict=True)
    x, ids = torch.randn(3, 1, 4), torch.tensor([10, 2, 11])
    assert torch.equal(model(x, ids), original(x, ids))
    assert set(export_official_state(model)) == set(original.state_dict())


@pytest.mark.parametrize('arm', ARMS)
def test_full_cpu_optimizer_scheduler_rng_cursor_resume_matches_uninterrupted(arm):
    _, a, b, ca, cb = fixture_pair()
    model, contract = (a, ca) if arm == ARMS[0] else (b, cb)
    x, ids = torch.randn(2, 1, 4), torch.tensor([10, 10])
    opt = torch.optim.AdamW(contract.parameters(), lr=.01, weight_decay=.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1 - step / 1024)

    def step(m, optimizer, schedule, index):
        with matched_flow_rng(20261019, step=index, microbatch=0):
            optimizer.zero_grad(set_to_none=True)
            (m(x + torch.randn_like(x), ids).square().sum()).backward()
            optimizer.step()
            schedule.step()

    step(model, opt, scheduler, 0)
    stream = io.BytesIO()
    torch.save(dict(model=model.state_dict(), contract=contract.describe(), optimizer=opt.state_dict(),
                    scheduler=scheduler.state_dict(), RNG=torch.get_rng_state(), step=1, cursor=32), stream)
    step(model, opt, scheduler, 1)
    _, fresh_a, fresh_b, fresh_ca, fresh_cb = fixture_pair()
    restored, fresh_contract = (fresh_a, fresh_ca) if arm == ARMS[0] else (fresh_b, fresh_cb)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restore_parameter_state(fresh_contract, saved['model'], saved['contract'])
    fresh_opt = torch.optim.AdamW(fresh_contract.parameters(), lr=.01, weight_decay=.2)
    fresh_scheduler = torch.optim.lr_scheduler.LambdaLR(fresh_opt, lambda step: 1 - step / 1024)
    fresh_opt.load_state_dict(saved['optimizer'])
    fresh_scheduler.load_state_dict(saved['scheduler'])
    torch.set_rng_state(saved['RNG'])
    assert saved['cursor'] == 32 * saved['step']
    step(restored, fresh_opt, fresh_scheduler, saved['step'])
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert scheduler.state_dict() == fresh_scheduler.state_dict()
    for i, state in opt.state_dict()['state'].items():
        for k, v in state.items():
            assert torch.equal(v, fresh_opt.state_dict()['state'][i][k])


def test_wrong_arm_or_category_restore_rejected_even_when_shapes_match():
    _, a, b, ca, cb = fixture_pair()
    with pytest.raises(ValueError, match='EXACT_ARM_SLOT'):
        restore_parameter_state(cb, a.state_dict(), ca.describe())
    wrong = FixtureModel()
    wrong_contract = ParameterContract(wrong, ARMS[0], 9)
    with pytest.raises(ValueError, match='EXACT_ARM_SLOT'):
        restore_parameter_state(wrong_contract, a.state_dict(), ca.describe())


def test_modes_match_after_train_and_extra_random_draws_do_not_shift_flow_stream():
    _, a, b, _, _ = fixture_pair()
    a.train(); b.train()
    x, ids = torch.randn(2, 1, 4), torch.tensor([10, 10])
    saved_rng = torch.get_rng_state()
    with matched_flow_rng(11, step=4, microbatch=3):
        noise_a = torch.randn_like(x)
        time_a = torch.distributions.Beta(1.5, 1.).sample((2,))
        out_a = a(x + noise_a, ids)
    assert torch.equal(saved_rng, torch.get_rng_state())
    torch.rand(17)
    with matched_flow_rng(11, step=4, microbatch=3):
        noise_b = torch.randn_like(x)
        time_b = torch.distributions.Beta(1.5, 1.).sample((2,))
        out_b = b(x + noise_b, ids)
    assert torch.equal(noise_a, noise_b) and torch.equal(time_a, time_b)
    assert torch.equal(out_a, out_b)
    assert [(n, m.training) for n, m in a.named_modules()] == [(n, m.training) for n, m in b.named_modules()]
    with matched_flow_rng(11, step=4, microbatch=4):
        assert not torch.equal(noise_a, torch.randn_like(x))


@pytest.mark.parametrize('ids', [torch.tensor([-1]), torch.tensor([12]), torch.tensor([10.]), torch.tensor([[10]])])
def test_invalid_category_dispatch_rejected(ids):
    layer = TargetCategoryLinear(CategoryLinear(), 10)
    with pytest.raises(ValueError):
        layer(torch.zeros(1, 1, 4), ids)


def test_reapplication_and_whitelist_mutation_rejected():
    _, a, _, contract, _ = fixture_pair()
    with pytest.raises(ValueError, match='ALREADY_INSTALLED'):
        ParameterContract(a, ARMS[0], 10)
    a.action_head.position_embedding.requires_grad_(True)
    with pytest.raises(ValueError, match='WHITELIST_CHANGED'):
        contract.parameters()


def test_failed_prevalidation_does_not_mutate_input_model():
    a = FixtureModel()
    before = {n: p.requires_grad for n, p in a.named_parameters()}
    a.action_head.state_dropout_prob = .1
    with pytest.raises(ValueError, match='STATE_DROPOUT'):
        ParameterContract(a, ARMS[0], 10)
    assert {n: p.requires_grad for n, p in a.named_parameters()} == before
