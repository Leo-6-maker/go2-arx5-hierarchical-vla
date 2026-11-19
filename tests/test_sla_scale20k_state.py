"""Regression checks for exact matched exposure and long-run optimizer state."""
import numpy as np
import pytest

from go2arx5_vla.sla_adapter import scale20k_checkpoint as checkpoint
from go2arx5_vla.sla_adapter.scale20k_contract import (
    DOMAINS, PLAN, RECIPE, bank_sha256, cursors, learning_rates, make_adamw, visits,
)


def test_shared_learning_rate_table_and_only_admitted_training_across_cursors():
    rates = learning_rates()
    assert len(rates) == 20000 and rates[499] == 2e-5 and rates[-1] == 2e-6
    assert np.all(np.diff(rates[:500]) > 0) and np.all(np.diff(rates[499:]) < 0)
    assert cursors('SCALE-MIX20K',5000) == dict(target=160000,robodojo=240000,libero=240000)
    assert cursors('SCALE-TGT5K',5000) == dict(target=160000,robodojo=0,libero=0)
    rows = [dict(episode='a',task='short',split='TRAIN',admitted=True,window_count=1),
            dict(episode='b',task='long',split='TRAIN',admitted=True,window_count=30)]
    bank = visits(rows,'target')
    assert len(bank) == 640000 and (bank==0).sum() == 320000 and bank.max() == 30
    np.testing.assert_array_equal(bank,visits(rows,'target'))
    with pytest.raises(ValueError,match='UNIQUE_ADMITTED_TRAIN'):
        visits([rows[0],dict(rows[1],split='SOURCE_DEV')],'target')


def test_private_moments_and_decay_and_next_update_survive_checkpoint(tmp_path):
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    pytest.importorskip('optax')
    init,apply = make_adamw(beta1=.9,beta2=.95,eps=1e-8,weight_decay=.01,clip_norm=1.)
    params = {name:(jnp.asarray([.4,1.2],dtype=jnp.float32),) for name in ('shared',*DOMAINS)}
    optimizer = init(params)
    rng = jax.random.PRNGKey(4)
    gradient = {name:(jnp.asarray([.5,1.1],dtype=jnp.float32),) for name in params}
    target,states,_ = apply(params,optimizer,{key:gradient[key] for key in ('shared','target')},2e-5)
    for source in DOMAINS[1:]:
        assert target[source] is params[source] and states[source] is optimizer[source]
    active,state,_ = apply(params,optimizer,gradient,2e-5)
    arm,digest = 'SCALE-MIX20K','4'*64
    binding = dict(recipe=RECIPE,plan_sha=PLAN,arm=arm,learning_rates_sha256=bank_sha256(learning_rates()),
        parent_manifest_sha256=digest,partition_sha256=digest,initial_state_sha256=digest,optimizer_sha256=digest,
        data_manifest_sha256=dict.fromkeys(DOMAINS,digest),visit_bank_sha256=dict.fromkeys(DOMAINS,digest))
    pin = checkpoint.save(tmp_path/'ready',arm=arm,step=1,parameters=active,optimizer=state,rng=rng,binding=binding)
    restored,step,offsets = checkpoint.restore(pin,arm=arm,parameters=params,optimizer=optimizer,rng=rng,binding=binding)
    assert step == 1 and offsets == dict(target=32,robodojo=48,libero=48)
    np.testing.assert_array_equal(restored['rng'],rng)
    left = apply(active,state,gradient,2e-6)[:2]
    right = apply(restored['parameters'],restored['optimizer'],gradient,2e-6)[:2]
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right),strict=True):
        np.testing.assert_array_equal(a,b)
    member = tmp_path/'ready/000000.npy'
    member.write_bytes(member.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='ARRAY_IDENTITY_MISMATCH'):
        checkpoint.restore(pin,arm=arm,parameters=params,optimizer=optimizer,rng=rng,binding=binding)
