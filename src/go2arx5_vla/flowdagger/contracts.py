"""Pure decision/label contracts. A label is not an executed success demo."""
from dataclasses import dataclass
import hashlib
import json

from .codec import eef10


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    observation_id: str
    source: str
    action: object
    valid_steps: int

    def checked_action(self):
        if not self.observation_id or self.source not in ('STUDENT', 'EXPERT') or self.valid_steps != 40:
            raise ValueError('SAME_OBSERVATION_FULL_VALID_H40_REQUIRED')
        return eef10(self.action)

    @property
    def sha256(self):
        return digest(dict(observation_id=self.observation_id, source=self.source,
                           action=self.checked_action().tolist(), valid_steps=self.valid_steps))


@dataclass(frozen=True)
class SafetyReceipt:
    observation_id: str
    candidate_sha256: str
    reference_gate: str
    S1: str
    proof_sha256: str

    def permits(self, candidate):
        if (self.observation_id != candidate.observation_id or self.candidate_sha256 != candidate.sha256
                or len(self.proof_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.proof_sha256)):
            raise ValueError('SAFETY_PROOF_MUST_BIND_EXACT_STATE_AND_UNMODIFIED_ACTION')
        if self.reference_gate == 'REJECT' and self.S1 == 'NOT_RUN':
            return False
        if self.reference_gate not in ('PASS', 'REJECT') or self.S1 not in ('PASS', 'REJECT'):
            raise ValueError('UNKNOWN_SAFETY_OUTCOME_FAIL_CLOSED')
        return self.reference_gate == 'PASS' and self.S1 == 'PASS'


def arbitrate(student, student_safety, expert=None, expert_safety=None, *,
              request_intervention=False, expert_valid=False, native_safe=True):
    """No sampling, repair, physics, S1 implementation or automatic safety override."""
    if type(native_safe) is not bool or type(expert_valid) is not bool or type(request_intervention) is not bool:
        raise ValueError('EXPLICIT_BOOLEAN_DECISIONS_REQUIRED')
    if student.source != 'STUDENT':
        raise ValueError('STUDENT_PROPOSAL_REQUIRED')
    student.checked_action()
    if not native_safe:
        return dict(selected_executor='STOP', reason='ACTUAL_NATIVE_SAFETY_FAIL', prefix_to_install=None, expert_label_qualified=False)
    student_ok = student_safety.permits(student)
    expert_ok = False
    if expert_valid:
        if expert is None or expert_safety is None or expert.source != 'EXPERT' or expert.observation_id != student.observation_id:
            raise ValueError('VALID_EXPERT_AND_SAFETY_REQUIRED_ON_SAME_OBSERVATION')
        expert_ok = expert_safety.permits(expert)
    evidence = dict(expert_label_qualified=expert_ok, student_safety_proof_sha256=student_safety.proof_sha256,
                    expert_safety_proof_sha256=expert_safety.proof_sha256 if expert_valid else None)
    if expert_ok and (request_intervention or not student_ok):
        choice, reason = expert, 'REQUESTED_EXPERT' if request_intervention else 'UNSAFE_STUDENT_VALID_EXPERT'
    elif student_ok:
        choice, reason = student, 'STUDENT_SAFE_NO_VALID_EXPERT_TAKEOVER'
    else:
        return dict(selected_executor='STOP', reason='NEITHER_CANDIDATE_SAFE', prefix_to_install=None, **evidence)
    return dict(selected_executor=choice.source, reason=reason, candidate_sha256=choice.sha256,
                prefix_to_install=choice.checked_action()[:8].tolist(), **evidence)


def label_record(*, observation_id, source_domain, split, lineage_splits, student,
                 expert, expert_valid, expert_queried, decision, execution_receipt,
                 actual_result, student_visited_state, observation_tick, state_visit_receipt,
                 model_sha256, statistics_sha256, expert_failure_reason=None):
    if source_domain != 'FIXED_BASE_ARX5_TASK_SIM' or split not in ('TRAIN', 'DEV'):
        raise ValueError('EXPLICIT_SOURCE_DOMAIN_AND_SPLIT_REQUIRED')
    if not lineage_splits or any(s not in ('TRAIN', 'DEV') for s in lineage_splits):
        raise ValueError('ORIGINAL_SPLIT_LINEAGE_REQUIRED')
    if split == 'TRAIN' and any(s != 'TRAIN' for s in lineage_splits):
        raise ValueError('DEV_STATE_MUST_NOT_ENTER_TRAIN_OR_OPTIMIZER')
    if any(type(v) is not bool for v in (expert_valid, expert_queried, student_visited_state)) or expert_valid and not expert_queried:
        raise ValueError('EXPLICIT_LABEL_AND_VISIT_IDENTITY_REQUIRED')
    if expert_queried and not expert_valid:
        if not isinstance(expert_failure_reason, str) or not 0 < len(expert_failure_reason) <= 4096:
            raise ValueError('UNAVAILABLE_EXPERT_REASON_MUST_BE_RECORDED')
    elif expert_failure_reason is not None:
        raise ValueError('EXPERT_REASON_MUST_MATCH_QUERY_OUTCOME')
    if type(observation_tick) is not int or observation_tick < 0 or observation_tick % 200:
        raise ValueError('ACTUAL_PREFIX_BOUNDARY_OBSERVATION_TICK_REQUIRED')
    if student_visited_state:
        visit = state_visit_receipt
        if (not isinstance(visit, dict) or set(visit) != {'observation_id', 'previous_executed_by', 'tick_after', 'integrations', 'ack_sha256'}
                or visit['observation_id'] != observation_id or visit['previous_executed_by'] != 'STUDENT'
                or type(visit['tick_after']) is not int or visit['tick_after'] != observation_tick or observation_tick == 0
                or type(visit['integrations']) is not int or visit['integrations'] != 200
                or len(visit['ack_sha256']) != 64 or any(c not in '0123456789abcdef' for c in visit['ack_sha256'])):
            raise ValueError('STUDENT_VISIT_REQUIRES_PREVIOUS_ACTUAL_STUDENT_PREFIX_ACK')
    if student.observation_id != observation_id or student.source != 'STUDENT':
        raise ValueError('STUDENT_STATE_BINDING_REQUIRED')
    action = student.checked_action()
    correction = None
    if expert_valid:
        if (expert is None or expert.observation_id != observation_id or expert.source != 'EXPERT'
                or decision.get('expert_label_qualified') is not True):
            raise ValueError('EXPERT_LABEL_MUST_BIND_STUDENT_CURRENT_STATE')
        correction = expert.checked_action().tolist()
    executor = decision['selected_executor']
    if executor not in ('STUDENT', 'EXPERT', 'STOP') or executor == 'EXPERT' and not expert_valid:
        raise ValueError('EXPLICIT_VALID_EXECUTOR_REQUIRED')
    expected = None if executor == 'STOP' else (action if executor == 'STUDENT' else expert.checked_action())[:8].tolist()
    chosen = None if executor == 'STOP' else (student if executor == 'STUDENT' else expert).sha256
    if decision['prefix_to_install'] != expected or executor != 'STOP' and decision.get('candidate_sha256') != chosen:
        raise ValueError('PROPOSED_PREFIX_MUST_MATCH_SELECTED_UNCHANGED_ACTION')
    if actual_result not in ('PENDING', 'TASK_SUCCESS', 'TASK_FAIL', 'SAFETY_FAIL', 'ENGINEERING_INCOMPLETE'):
        raise ValueError('EXPLICIT_RESULT_NOT_INFERRED_FROM_LABEL_REQUIRED')
    installed, executed_by, ticks = None, 'NOT_EXECUTED', 0
    if execution_receipt is not None:
        required = {'observation_id', 'candidate_sha256', 'installed_prefix', 'integrations', 'native_safe', 'ack_sha256'}
        if (set(execution_receipt) != required or executor == 'STOP'
                or execution_receipt['observation_id'] != observation_id
                or execution_receipt['candidate_sha256'] != chosen
                or execution_receipt['installed_prefix'] != expected
                or type(execution_receipt['integrations']) is not int or not 1 <= execution_receipt['integrations'] <= 200
                or type(execution_receipt['native_safe']) is not bool
                or len(execution_receipt['ack_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in execution_receipt['ack_sha256'])):
            raise ValueError('ACTUAL_EXECUTION_PROOF_REQUIRED_NOT_JUST_ARBITRATION')
        if not execution_receipt['native_safe'] and actual_result != 'SAFETY_FAIL':
            raise ValueError('ACTUAL_NATIVE_FAILURE_MUST_REMAIN_SAFETY_FAIL')
        installed, executed_by, ticks = expected, executor, execution_receipt['integrations']
    elif actual_result not in ('PENDING', 'ENGINEERING_INCOMPLETE'):
        raise ValueError('NO_TASK_RESULT_WITHOUT_ACTUAL_EXECUTION_PROOF')
    for value in (model_sha256, statistics_sha256):
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('PINNED_MODEL_AND_STATISTICS_REQUIRED')
    row = dict(schema='DAGGER_CORRECTION_LABEL_V1', observation_id=observation_id,
        source_domain=source_domain, split=split, lineage_splits=list(lineage_splits),
        student_action=action.tolist(), expert_action=correction, expert_valid=expert_valid, expert_queried=expert_queried,
        expert_failure_reason=expert_failure_reason,
        selected_executor=executor, executed_by=executed_by, installed_prefix=installed,
        actual_integrations=ticks, execution_receipt=execution_receipt, actual_result=actual_result,
        student_visited_state=student_visited_state, observation_tick=observation_tick,
        state_visit_receipt=state_visit_receipt, safety_proofs={k: decision.get(k) for k in
            ('student_safety_proof_sha256', 'expert_safety_proof_sha256')},
        optimizer_eligible=expert_valid and split == 'TRAIN',
        successful_demo_admitted=False, success_demo_requires_separate_full_episode_audit=True,
        model_sha256=model_sha256, statistics_sha256=statistics_sha256,
        selected='NONE', deployment='NONE')
    row['record_sha256'] = digest(row)
    return row


def summarize(records):
    rows = list(records)
    executed = [r for r in rows if r['executed_by'] in ('STUDENT', 'EXPERT')]
    return dict(decision_blocks=len(rows), executed_blocks=len(executed),
        expert_queries=sum(r['expert_queried'] for r in rows), valid_expert_labels=sum(r['expert_valid'] for r in rows),
        unavailable_expert_labels=sum(r['expert_queried'] and not r['expert_valid'] for r in rows),
        student_executed_blocks=sum(r['executed_by'] == 'STUDENT' for r in rows),
        student_execution_fraction=(sum(r['executed_by'] == 'STUDENT' for r in executed)/len(executed)) if executed else None,
        student_visited_correction_labels=sum(r['expert_valid'] and r['student_visited_state'] for r in rows),
        training_labels=sum(r['optimizer_eligible'] for r in rows),
        all_expert_execution_is_student_state_DAgger=False,
        independent_episode_success_rate='REQUIRES_EPISODE_RESULTS_NOT_LABEL_COUNT')
