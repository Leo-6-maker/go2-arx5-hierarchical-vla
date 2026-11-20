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


def test_clock_queued_arm_admission_preserves_negative_and_semantic_block():
    from scripts.sla import clock500_activation as activation
    from scripts.sla.clock500_trigger import SCOPE

    registered = dict(job_id='1598095',source_sha='source')
    actual = dict(registered,scope=SCOPE,scientific_trigger=False,status='NOT_TRIGGERED',
                  optimizer_updates=0,model_generations=0,native_requests=0,native_ticks=0)
    assert activation.trigger_status(actual,registered)=='NOT_TRIGGERED'
    assert activation.trigger_status(dict(actual,scientific_trigger=None,status='BLOCKED_DIAGNOSIS_OR_SEMANTICS'),
                                     registered)=='BLOCKED_DIAGNOSIS_OR_SEMANTICS'
    semantic = dict(labels_verified=88,due_experts_verified=24,original_H40_and_seam_verified=88)
    ready = dict(actual,scientific_trigger=True,status='TRIGGERED_AWAITING_EXECUTION_READINESS',joint_numerator=0,
                 eligible_windows=24,semantic_audit=semantic)
    assert activation.trigger_status(ready,registered)==activation.ACTIVE
    missing = dict(ready,status='BLOCKED_ARTIFACT',first_failure='MISSING_COLD_D2_CURRICULUM_DECISION')
    assert activation.trigger_status(missing,registered)==activation.ACTIVE
    invalid = dict(missing,first_failure='COLD_D2_CURRICULUM_INELIGIBLE_OR_MISMATCHED')
    assert activation.trigger_status(invalid,registered)=='BLOCKED_ARTIFACT'
    with pytest.raises(ValueError,match='ZERO_OF_24'):
        activation.trigger_status(dict(ready,joint_numerator=1),registered)


def test_clock_branch_saves_all_state_and_rejects_cross_arm_or_formal_resume(tmp_path):
    jax = pytest.importorskip('jax')
    from go2arx5_vla.sla_adapter import adapt5k_checkpoint, clock500_checkpoint
    from go2arx5_vla.sla_adapter.clock500 import ARMS, PLAN
    from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_STATISTICS
    from go2arx5_vla.sla_adapter.scale20k_contract import bank_sha256

    marker = 'a'*64
    contract = dict.fromkeys(('parent_checkpoint_sha256','parent_training_spec_sha256','target_state_sha256',
        'target_D1_sha256','optimizer_sha256','tokenizer_sha256','partition_sha256','parent_manifest_sha256',
        'R200_membership_sha256','LIBERO_membership_sha256','stage_target_data_sha256'),marker)
    contract.update(plan=adapt.PLAN,recipe=adapt5k_checkpoint.RECIPE,parent_step=500,source_revision='b'*40,
        learning_rates_sha256=bank_sha256(adapt.phase_rates()),stage_start=2000,stage_stop=5000,
        stage_bank_sha256=dict.fromkeys(adapt.DOMAINS,marker),frozen_C20_action_statistics_sha256=C20_STATISTICS)
    binding = dict(plan=PLAN,arm=ARMS[1],parent_phase=2000,original_adapt_contract=contract,
                   **{key:dict(path=key,sha256=marker) for key in ('parent_state','D2','comparison')})
    keys = ('shared',*adapt.DOMAINS)
    parameters = {d:{str(i):np.array([i],np.float32) for i in range(12-(d=='libero'))} for d in keys}
    optimizer = {d:{str(i):np.array([i+1],np.float32) for i in range(23)} for d in keys}
    named = dict(parameters=parameters,optimizer=optimizer,rng=jax.random.PRNGKey(9))
    cp = clock500_checkpoint.save(tmp_path/'phase02250',phase=2250,binding=binding,**named)
    _,sealed = clock500_checkpoint.read_complete(cp,binding)
    assert sealed['branch_updates']==250 and sealed['branch_training_H40']==1000
    assert sealed['formal_ADAPT_parent_eligible'] is False and len(sealed['arrays'])==140
    restored,phase = clock500_checkpoint.restore(cp,binding,**named)
    assert phase==2250
    for observed,expected in zip(jax.tree.leaves(restored),jax.tree.leaves(named),strict=True):
        np.testing.assert_array_equal(observed,expected)
    with pytest.raises(ValueError,match='IDENTITY_OR_140'):
        clock500_checkpoint.read_complete(cp,dict(binding,arm=ARMS[0]))
    with pytest.raises(ValueError,match='COMPLETE_OPTIMIZER_RNG_AND_PHASE_CLOCK'):
        adapt5k_checkpoint.read_complete(cp,contract)
    with pytest.raises(ValueError,match='PINNED_PARENT_D2_AND_ARM'):
        clock500_checkpoint.boundary(binding,2501)


def test_clock_prefix_checks_original_lr_slots_and_actual_d2_bank_bytes(tmp_path):
    import hashlib
    import json
    from scripts.sla.train_clock500 import validate_ledger
    from scripts.wm02.multisource_d4_vanda import pin

    phase = 2001
    banks = {d:np.arange(n,dtype='<i8') for d,n in adapt.BATCH.items()}
    row = adapt.progress(500,phase,2000)
    rows = {d:dict(visit_slice_sha256=hashlib.sha256(a.tobytes()).hexdigest()) for d,a in banks.items()}
    rows['target'].update(auxiliary_slots=adapt.auxiliary_slots(phase-1),training_full_generations=4)
    row.update(domains=rows,grad_norm=.5,learning_rate=adapt.phase_rates()[phase-1])
    ledger = tmp_path/'UPDATES.jsonl'
    ledger.write_text(json.dumps(row)+'\n')
    validate_ledger(pin(ledger),2000,2001,banks=banks,parent_step=500)
    changed = dict(banks,target=banks['target'][::-1])
    with pytest.raises(ValueError,match='UNMATCHED_BANK_SLICE:2001:target'):
        validate_ledger(pin(ledger),2000,2001,banks=changed,parent_step=500)
    row['learning_rate'] = adapt.phase_rates()[0]
    ledger.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='UNMATCHED_SAMPLE_CLOCK'):
        validate_ledger(pin(ledger),2000,2001,banks=banks,parent_step=500)
