"""Actual small CPU autograd/AdamW/LambdaLR; not a GR00T qualification."""
import pytest
torch=pytest.importorskip('torch')

from test_minimal_arx5_adapter_contract import fixture_pair
from test_minimal_arx5_training import resume_packet
from scripts.pi0294.generalization import minimal_qualification as q, minimal_training as t
from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint, parameter_state_fingerprint
from scripts.pi0294.e1.trainer import learning_rate_multiplier


@pytest.mark.parametrize('arm',[0,1])
def test_original_LambdaLR_first_call_does_not_update_second_call_does(arm):
    _,a,b,ca,cb=fixture_pair();model,contract=(a,ca)if arm==0 else(b,cb)
    before=parameter_state_fingerprint(model);frozen=frozen_fingerprint(model)
    optimizer=torch.optim.AdamW(contract.parameters(),lr=1e-5,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda i:learning_rate_multiplier(i,1024,64))
    assert optimizer.param_groups[0]['lr']==0
    grads=None
    for i in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(2,1,4),torch.tensor([10,10])).square().mean().backward()
        grads=q.gradients(contract);optimizer.step();scheduler.step()
        if i==0:
            assert parameter_state_fingerprint(model)==before
            with pytest.raises(ValueError,match='NO_REAL_TARGET'):
                q.after_optimizer(contract,initial=before,frozen=frozen,gradient_rows=grads,step=2,cursor=64)
    out=q.after_optimizer(contract,initial=before,frozen=frozen,gradient_rows=grads,step=2,cursor=64)
    assert out['optimizer_calls']==2 and out['cursor']==64
    assert out['formal_training_start_requires_fresh_raw_parent']
    assert out['native_model_qualification_complete'] is False
    assert frozen_fingerprint(model)==frozen


def test_qualification_stops_at_two_original_calls_without_third_step_or512_save():
    for step in (0,1):
        x=t.step_boundary(step,step*32,pending=[],unknown=[],session_stop=2,qualification_only=True)
        assert not x['should_training_stop'] and not x['should_save']
    x=t.step_boundary(2,64,pending=[],unknown=[],session_stop=2,qualification_only=True)
    assert x['should_training_stop'] and x['should_save']
    for kwargs in [dict(session_stop=2),dict(session_stop=512,qualification_only=True)]:
        with pytest.raises(ValueError):t.step_boundary(2,64,pending=[],unknown=[],**kwargs)
    with pytest.raises(ValueError):t.step_boundary(3,96,pending=[],unknown=[],session_stop=2,qualification_only=True)


def test_qualification_full_state_cannot_be_resumed_as_formal512():
    packet=resume_packet(step=2);packet[1]['qualification_only']=True
    with pytest.raises(ValueError,match='OWN_COMPLETE'):
        t.resume_state(*packet,session_stop=512)


def test_target_gradient_path_must_exist_and_frozen_update_is_refused():
    _,model,_,contract,_=fixture_pair()
    with pytest.raises(ValueError,match='NO_AUTOGRAD_PATH'):q.gradients(contract)
    frozen=frozen_fingerprint(model);before=parameter_state_fingerprint(model)
    model(torch.randn(2,1,4),torch.tensor([10,10])).square().mean().backward()
    grads=q.gradients(contract)
    with torch.no_grad():model.action_head.position_embedding.weight.add_(1)
    with pytest.raises(ValueError,match='CHANGED_FROZEN'):
        q.after_optimizer(contract,initial=before,frozen=frozen,gradient_rows=grads,step=2,cursor=64)


def test_nonfinite_weight_cannot_be_misreported_as_a_successful_parameter_update():
    _,model,_,contract,_=fixture_pair();frozen=frozen_fingerprint(model);before=parameter_state_fingerprint(model)
    model(torch.randn(2,1,4),torch.tensor([10,10])).square().mean().backward();grads=q.gradients(contract)
    with torch.no_grad():next(p for p in contract.parameters()).reshape(-1)[0]=float('nan')
    with pytest.raises(ValueError,match='NONFINITE_UPDATED_PARAMETER'):
        q.after_optimizer(contract,initial=before,frozen=frozen,gradient_rows=grads,step=2,cursor=64)
