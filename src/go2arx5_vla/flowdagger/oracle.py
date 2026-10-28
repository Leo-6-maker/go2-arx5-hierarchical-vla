"""Offline source oracle adapter; no engine, S1 simulation or recovery claim."""
import copy
from dataclasses import dataclass

import numpy as np

from .abi import validate_source_abi
from .codec import eef10
from .contracts import Candidate, digest
from ..task_data.codec import pose10


@dataclass(frozen=True)
class OracleRequest:
    observation_id: str
    tick: int
    task: str
    measured_state10: object
    evidence_kind: str
    binding: object

    def validate(self):
        validate_source_abi(self.binding)
        if (not self.observation_id or type(self.tick) is not int or self.tick < 0 or self.tick % 200
                or self.task not in ('T1', 'T2') or self.evidence_kind not in ('SYNTHETIC_CPU_FIXTURE', 'SAVED_ACTUAL_STATE')):
            raise ValueError('BOUND_OFFLINE_SOURCE_STATE_REQUIRED_NO_LIVE_RUN_ENTRY')
        eef10(self.measured_state10, horizon=None)


class SourceOracle:
    def __init__(self, provider, *, version, safety_check=None):
        if not version:raise ValueError('PINNED_ORACLE_VERSION_REQUIRED')
        self.provider, self.version, self.safety_check = provider, version, safety_check

    def query(self, request, privileged_truth, executed_history):
        request.validate()
        raw = self.provider(copy.deepcopy(request), copy.deepcopy(privileged_truth), copy.deepcopy(executed_history))
        result = dict(observation_id=request.observation_id, tick=request.tick, task=request.task,
            evidence_kind=request.evidence_kind, oracle_version=self.version,
            valid=False, expert_H40=None, active_dim_mask=None, phase=raw.get('phase'),
            reason=raw.get('reason'), proposal_H40_audit_only=None, certificate=None,
            scientific_recovery_qualified=False, physically_executed=False)
        if type(raw.get('valid')) is not bool or not isinstance(raw.get('reason'), str) or not raw['reason']:
            raise ValueError('ORACLE_VALIDITY_PHASE_AND_FAILURE_REASON_REQUIRED')
        if not raw['valid']:
            if raw.get('expert_H40') is not None:raise ValueError('INVALID_ORACLE_MUST_NOT_SUPPLY_TRAIN_LABEL')
            return result
        if not isinstance(raw.get('phase'), str) or not raw['phase']:
            raise ValueError('EXPLICIT_EXPERT_PHASE_REQUIRED')
        try:
            action = eef10(raw['expert_H40'])
        except ValueError:
            result['reason'] = 'INCOMPLETE_OR_INVALID_FULL_H40_NO_TAIL_PADDING'
            return result
        if raw.get('observation_id') != request.observation_id or raw.get('tick') != request.tick:
            raise ValueError('EXPERT_PROPOSAL_BOUND_TO_DIFFERENT_STATE')
        result['proposal_H40_audit_only'] = action.tolist()
        candidate = Candidate(request.observation_id, 'EXPERT', action, 40)
        if self.safety_check is None:
            result['reason'] = 'ORIGINAL_GATE_AND_S1_CERTIFICATE_NOT_AVAILABLE_NO_LABEL'
            return result
        proof = self.safety_check(candidate, request)
        result['certificate'] = dict(proof_sha256=proof.proof_sha256, candidate_sha256=candidate.sha256,
                                   reference_gate=proof.reference_gate, S1=proof.S1)
        if not proof.permits(candidate):
            result['reason'] = 'ORIGINAL_SAFETY_REJECT_NO_VALID_CORRECTION'
            return result
        result.update(valid=True, expert_H40=action.tolist(), active_dim_mask=np.ones((40, 10), dtype=bool).tolist(),
            validity_ticks=[request.tick, request.tick+975], reason='VALID_OFFLINE_PROPOSAL_NOT_PHYSICAL_RECOVERY')
        result['reply_sha256'] = digest(result)
        return result


def legacy_saved_teacher_provider(teacher, saved_state):
    """Reuse a copied qualified Teacher.plan, without inferring a new phase."""
    original_teacher, original_state = copy.deepcopy(teacher), copy.deepcopy(saved_state)

    def provider(request, privileged_truth, history):
        state = copy.deepcopy(original_state)
        measured = eef10(request.measured_state10, horizon=None)
        actual = pose10(state['tcp'], state['gripper_fraction'])
        if not np.allclose(actual, measured, atol=1e-12, rtol=0) or abs(state['time_s']-request.tick*.002) > 1e-10:
            raise ValueError('SAVED_TEACHER_STATE_MUST_MATCH_EXACT_REQUEST')
        local = copy.deepcopy(original_teacher)
        plan = local.plan(state)
        if (np.asarray(plan['poses']).shape != (40, 4, 4) or np.asarray(plan['opening']).shape != (40,)
                or not np.allclose(plan['offset_s'], np.arange(40)/20, atol=1e-12, rtol=0)):
            raise ValueError('ORIGINAL_TEACHER_FULL_H40_CLOCK_REQUIRED')
        action = np.stack([pose10(p, g) for p, g in zip(plan['poses'], plan['opening'])])
        return dict(valid=True, expert_H40=action, phase=plan['stage'], reason='COPIED_LEGACY_PLAN_ONLY',
                    observation_id=request.observation_id, tick=request.tick)

    return provider
