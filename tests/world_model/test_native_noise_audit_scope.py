"""Real CPU hook/RNG checks for the observed post-training inference rejection."""
from types import SimpleNamespace
import pytest
import torch
from scripts.pi0294.native_train import NativeNoiseAudit,rng_identity


class Encoder(torch.nn.Module):
    def forward(self,actions,times):return actions+times[:,None,None]


class Head(torch.nn.Module):
    state_dropout_prob=0
    num_timestep_buckets=1000

    def __init__(self,corrupt=False):
        super().__init__();self.state_encoder=torch.nn.Identity()
        self.action_encoder=Encoder();self.corrupt=corrupt;self.fail=False

    def sample_time(self,batch,device,dtype):return torch.rand(batch,device=device,dtype=dtype)

    def forward(self,backbone,inputs):
        actions=inputs['action'];self.state_encoder(actions[:,0])
        if self.fail:raise RuntimeError('ORIGINAL_FORWARD_FAILURE')
        noise=torch.randn_like(actions)
        times=self.sample_time(actions.shape[0],actions.device,actions.dtype)
        noisy=(1-times[:,None,None])*noise+times[:,None,None]*actions
        return self.action_encoder(noisy+(1 if self.corrupt else 0),(times*self.num_timestep_buckets).long())

    def generate(self):
        self.state_encoder(torch.ones(1,10))
        return self.action_encoder(torch.randn(1,40,10),torch.zeros(1,dtype=torch.long))


@pytest.fixture
def cpu_fork(monkeypatch):
    original=torch.random.fork_rng
    monkeypatch.setattr(torch.random,'fork_rng',lambda devices:original(devices=[]))


def test_native_audit_keeps_training_noise_and_rng_exact_and_ignores_later_generation(cpu_fork):
    head=Head();actions=torch.arange(400).reshape(1,40,10).float()
    torch.manual_seed(17);expected=head(None,{'action':actions});after=rng_identity()
    audit=NativeNoiseAudit(SimpleNamespace(action_head=head))
    torch.manual_seed(17);actual=head(None,{'action':actions})
    assert torch.equal(actual,expected)and rng_identity()==after
    assert audit.record['exact']and not audit.active and audit.pending is None
    training_record=audit.record
    torch.manual_seed(91);head.generate();after_generation=rng_identity()
    audit.close();torch.manual_seed(91);head.generate()
    assert rng_identity()==after_generation and audit.record==training_record


def test_native_training_noise_corruption_is_still_rejected(cpu_fork):
    head=Head(corrupt=True);audit=NativeNoiseAudit(SimpleNamespace(action_head=head))
    with pytest.raises(ValueError,match='OBSERVED_NATIVE_NOISE_OR_TIME_DIFFERS'):
        head(None,{'action':torch.zeros(1,40,10)})
    assert not audit.active
    head.generate();audit.close()


def test_original_forward_error_cleans_audit_without_masking_failure(cpu_fork):
    head=Head();audit=NativeNoiseAudit(SimpleNamespace(action_head=head));head.fail=True
    with pytest.raises(RuntimeError,match='ORIGINAL_FORWARD_FAILURE'):
        head(None,{'action':torch.zeros(1,40,10)})
    assert not audit.active and audit.pending is None
    head.generate();audit.close()
