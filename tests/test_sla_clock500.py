import numpy as np
import pytest

from go2arx5_vla.sla_adapter import adapt5k_contract as adapt
from go2arx5_vla.sla_adapter.clock500 import event_weighted_width_loss, width_weights


def test_clock_width_weights_preserve_phase_precedence_and_boundary_support():
    label = np.ones((40,10))
    consumed = np.arange(40)<10
    weights = width_weights(label,consumed,.04,'HOLD')
    assert weights.tolist()==[4.]*10+[1.]*30
    for phase in ('DESCEND','OPEN'):
        np.testing.assert_array_equal(width_weights(label,consumed,.04,phase),np.ones(40))
    label[:,9] = 0.
    assert adapt.phase_bucket(label,consumed,.04,'DESCEND')=='P10_DUE_CLOSE'
    assert width_weights(label,consumed,.04,'DESCEND').tolist()==[4.]*3+[1.]*37
    label[:9,9] = 1.
    assert np.flatnonzero(width_weights(label,consumed,.04,'DESCEND')==4).tolist()==[7,8,9,10,11]
    label[9,9] = 1.
    assert adapt.phase_bucket(label,consumed,.04,'DESCEND')=='APPROACH'
    np.testing.assert_array_equal(width_weights(label,consumed,.04,'DESCEND'),np.ones(40))
    with pytest.raises(ValueError,match='ORIGINAL_FINITE_FULL_H40'):
        width_weights(label[:10],consumed,.04,'DESCEND')


def test_clock_all_ones_matches_original_width_gradient_and_suffix_is_supervised():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter.grasp_supervision_d5 import width_losses

    expected = jnp.full((1,40),.02)
    predict = jnp.linspace(.006,.035,40)[None,:]
    original = lambda x:width_losses(x,expected,jnp.array([.024]),xp=jnp)[0]
    future = lambda x,w:event_weighted_width_loss(x,expected,w,xp=jnp)
    a,da = jax.value_and_grad(original)(predict)
    b,db = jax.value_and_grad(future)(predict,jnp.ones_like(predict))
    np.testing.assert_allclose(a,b,rtol=2e-6)
    np.testing.assert_allclose(da,db,rtol=2e-6)
    weight = jnp.ones_like(predict).at[:,:10].set(4.)
    weighted_grad = jax.grad(future)(predict,weight)
    assert weighted_grad[0,-1]!=0
    np.testing.assert_allclose(weighted_grad/da,(40/70)*weight,rtol=2e-6)


def test_clock_decision_distinguishes_nonzero_semantic_failure_and_execution_readiness(monkeypatch,tmp_path):
    from scripts.sla import clock500_trigger as trigger

    row = dict(timely_and_gates=True)
    monkeypatch.setattr(trigger,'exact_panel',lambda _: (dict(checkpoint='2k'),{},[row],[row],{}))
    monkeypatch.setattr(trigger.audit,'distribution',lambda rows:{})
    def failure(*_):
        raise ValueError('bad label identity')
    monkeypatch.setattr(trigger,'label_proof',failure)
    assert trigger.decision({}, {})=='NOT_TRIGGERED'
    row['timely_and_gates'] = False
    receipt = {}
    assert trigger.decision({},receipt)=='BLOCKED_DIAGNOSIS_OR_SEMANTICS'
    assert 'bad label identity' in receipt['first_failure']
    monkeypatch.setattr(trigger,'label_proof',lambda *_: {'labels_verified':88})
    assert trigger.decision(dict(curriculum_decision=str(tmp_path/'unset')),receipt)=='BLOCKED_ARTIFACT'
    assert receipt['scientific_trigger'] and receipt['joint_numerator']==0
