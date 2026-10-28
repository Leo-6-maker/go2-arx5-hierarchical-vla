import copy

import numpy as np
import pytest

from go2arx5_vla.flowdagger.contracts import Candidate, SafetyReceipt, arbitrate, label_record, summarize


def candidates():
    a = np.tile([.3, .1, .5, 1, 0, 0, 0, 1, 0, .8], (40, 1))
    return Candidate('obs-1', 'STUDENT', a, 40), Candidate('obs-1', 'EXPERT', a+.0, 40)


def proof(candidate, passed=True):
    return SafetyReceipt(candidate.observation_id, candidate.sha256, 'PASS', 'PASS' if passed else 'REJECT', 'a'*64)


def fixture():
    student, expert = candidates()
    decision = arbitrate(student, proof(student), expert, proof(expert), expert_valid=True)
    receipt = dict(observation_id='obs-1', candidate_sha256=student.sha256,
        installed_prefix=decision['prefix_to_install'], integrations=200, native_safe=True, ack_sha256='d'*64)
    return dict(observation_id='obs-1', source_domain='FIXED_BASE_ARX5_TASK_SIM', split='TRAIN',
        lineage_splits=['TRAIN'], student=student, expert=expert, expert_valid=True, expert_queried=True,
        decision=decision, execution_receipt=receipt, actual_result='TASK_FAIL', student_visited_state=True,
        observation_tick=200, state_visit_receipt=dict(observation_id='obs-1',previous_executed_by='STUDENT',
            tick_after=200,integrations=200,ack_sha256='e'*64),
        evidence_kind='SAVED_ACTUAL_STATE_WITH_VERIFIED_ORACLE',
        scene_group='fixture-original-group',
        model_sha256='b'*64, statistics_sha256='c'*64)


def test_counterfactual_expert_label_not_success_demo():
    row = label_record(**fixture())
    assert row['expert_valid'] and row['optimizer_eligible'] and row['executed_by'] == 'STUDENT'
    assert row['actual_result'] == 'TASK_FAIL' and not row['successful_demo_admitted']
    assert summarize([row])['student_execution_fraction'] == 1


def test_arbitration_is_not_proof_of_execution():
    values = fixture();values.update(execution_receipt=None, actual_result='PENDING')
    row = label_record(**values)
    assert row['selected_executor'] == 'STUDENT' and row['executed_by'] == 'NOT_EXECUTED'
    assert row['installed_prefix'] is None and row['actual_integrations'] == 0
    assert summarize([row])['student_execution_fraction'] is None
    values['actual_result'] = 'TASK_SUCCESS'
    with pytest.raises(ValueError):label_record(**values)


def test_native_fail_cannot_be_rescued_by_expert():
    s, e = candidates()
    result = arbitrate(s, proof(s), e, proof(e), expert_valid=True, native_safe=False)
    assert result['selected_executor'] == 'STOP' and result['prefix_to_install'] is None


def test_unsafe_student_only_valid_same_state_expert_may_take_over():
    s, e = candidates()
    assert arbitrate(s, proof(s, False), e, proof(e), expert_valid=True)['selected_executor'] == 'EXPERT'
    assert arbitrate(s, proof(s, False), e, proof(e, False), expert_valid=True)['selected_executor'] == 'STOP'
    assert arbitrate(s, proof(s), expert_valid=False, request_intervention=True)['selected_executor'] == 'STUDENT'


def test_explicit_intervention_and_expert_full_horizon():
    s, e = candidates()
    assert arbitrate(s, proof(s), e, proof(e), expert_valid=True, request_intervention=True)['selected_executor'] == 'EXPERT'
    short = Candidate('obs-1', 'EXPERT', e.action[:8], 8)
    with pytest.raises(ValueError):short.checked_action()


def test_stale_state_or_modified_action_cannot_reuse_safety():
    s, e = candidates();saved = proof(s)
    s.action[0, 0] += .001
    with pytest.raises(ValueError):saved.permits(s)
    stale = Candidate('obs-2', 'EXPERT', e.action, 40)
    with pytest.raises(ValueError):arbitrate(s, proof(s), stale, proof(stale), expert_valid=True)


@pytest.mark.parametrize('change', [dict(split='TRAIN', lineage_splits=['DEV']),
    dict(split='TRAIN', lineage_splits=['TRAIN', 'DEV']), dict(source_domain='GO2_ARX5_FULLBODY')])
def test_no_dev_or_target_domain_relabel(change):
    values=fixture();values.update(change)
    with pytest.raises(ValueError):label_record(**values)


def test_dev_labels_are_evaluation_only():
    values=fixture();values.update(split='DEV', lineage_splits=['DEV'])
    row=label_record(**values)
    assert not row['optimizer_eligible'] and row['expert_valid']


def test_unknown_expert_and_actual_safety_not_hidden_in_denominator():
    values=fixture();values.update(expert=None, expert_valid=False, expert_failure_reason='NO_VALID_CORRECTION')
    row=label_record(**values)
    assert not row['optimizer_eligible'] and row['expert_action'] is None
    assert summarize([row])['unavailable_expert_labels'] == 1
    values['execution_receipt']['native_safe'] = False
    with pytest.raises(ValueError):label_record(**values)
    values['actual_result'] = 'SAFETY_FAIL'
    assert label_record(**values)['actual_result'] == 'SAFETY_FAIL'


def test_installed_prefix_cannot_be_arbitrarily_changed():
    values=fixture();values['execution_receipt']=copy.deepcopy(values['execution_receipt'])
    values['execution_receipt']['installed_prefix'][0][0] += .01
    with pytest.raises(ValueError):label_record(**values)


@pytest.mark.parametrize('mutation',[dict(state_visit_receipt=None),dict(observation_tick=0)])
def test_student_state_claim_requires_actual_prior_ack(mutation):
    values=fixture();values.update(mutation)
    with pytest.raises(ValueError):label_record(**values)


def test_safety_rejected_expert_is_not_a_valid_training_label():
    values=fixture();s,e=values['student'],values['expert']
    values['decision']=arbitrate(s,proof(s),e,proof(e,False),expert_valid=True)
    with pytest.raises(ValueError):label_record(**values)


def test_reference_reject_does_not_require_new_shadow_simulation():
    s,_=candidates()
    receipt=SafetyReceipt(s.observation_id,s.sha256,'REJECT','NOT_RUN','a'*64)
    assert not receipt.permits(s)
