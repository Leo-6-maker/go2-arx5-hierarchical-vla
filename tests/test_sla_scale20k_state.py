"""Regression checks for exact matched exposure and long-run optimizer state."""
import numpy as np
import pytest

from go2arx5_vla.sla_adapter import scale20k_checkpoint as checkpoint
from go2arx5_vla.sla_adapter.scale20k_contract import (
    DOMAINS, PLAN, PRIVATE_ROOTS, RECIPE, bank_sha256, cursors, learning_rates, make_adamw, parameter_partition, visits,
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


def test_upstream_mlp_lora_members_are_active_alongside_attention_lora():
    llm = ('PaliGemma','llm','layers')
    mlp = [(*llm,'mlp',name) for name in ('gating_einsum_lora_a','gating_einsum_lora_b','linear_lora_a','linear_lora_b')]
    attention = (*llm,'attn','q_einsum','lora_b')
    full_expert = (*llm,'mlp_1','linear')
    frozen = [(*llm,'mlp','linear'),('PaliGemma','img','kernel')]
    private = [(root,'kernel') for root in PRIVATE_ROOTS]
    groups = parameter_partition([*frozen,*mlp,attention,full_expert,*private])
    assert set(groups['vlm_lora']) == {*mlp,attention}
    assert groups['action_expert'] == (full_expert,)
    assert set(groups['frozen']) == set(frozen)


def test_core_successor_binds_only_a_closed_hash_verified_training_segment(tmp_path, monkeypatch):
    from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
    from scripts.pi0292.durable import write
    from scripts.sla import scale20k_core_prefix as core

    arm = 'SCALE-MIX20K'
    previous = tmp_path/'old'
    checkpoint_root = previous/'train/step00200'
    checkpoint_root.mkdir(parents=True)
    manifest = checkpoint_root/'COMPLETE.json'
    write(manifest, dict(step=200))
    result_file = previous/'train/RESULT.private.json'
    write(result_file,dict(scope=RECIPE,source_sha='f'*40,arm=arm,mode='train',status='FIXED_SEGMENT_SAVED',
        exit_code=0,global_step=200,global_cursors=cursors(arm,200),new_native_requests=0,new_native_ticks=0,
        checkpoint=dict(path=str(checkpoint_root),sha256=sha(manifest))))
    prior = dict(path=str(result_file),sha256=sha(result_file))
    (previous/'control').mkdir()
    lease = previous/'control/LEASE.json'
    write(lease,dict(status='CLOSED',unit='wm02-old.service'))
    source = previous/'source'
    (source/'scripts/sla').mkdir(parents=True)
    (source/'scripts/sla/train_scale20k.py').touch()
    write(previous/'RESULT.private.json',dict(stages=dict(train=prior)))
    config = dict(arm=arm,source_sha='f'*40,previous_train_result=prior)
    monkeypatch.setattr(core.subprocess,'check_output',lambda *args,**kwargs:'0\n')
    assert core.closed_predecessor(config,tmp_path/'new') == (prior,source,200)
    monkeypatch.setattr(core.subprocess,'check_output',lambda *args,**kwargs:'975\n')
    with pytest.raises(PermissionError,match='STILL_OWNED'):
        core.closed_predecessor(config,tmp_path/'new')
