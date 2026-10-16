"""In-memory CPU chunk transactions, not a durable or live robot collector."""
import copy

from .abi import validate_source_abi
from .codec import eef10
from .contracts import arbitrate, digest


class ChunkTransaction:
    def __init__(self, episode_id, *, context, max_chunks, evidence_kind='SYNTHETIC_CPU_FIXTURE'):
        if (not episode_id or type(max_chunks) is not int or not 0 < max_chunks <= 80
                or evidence_kind not in ('SYNTHETIC_CPU_FIXTURE', 'SAVED_ACTUAL_RECEIPT')):
            raise ValueError('BOUNDED_OFFLINE_TRANSACTION_REQUIRED')
        required = {'source_domain', 'task', 'split', 'scene_group', 'language', 'model_sha256', 'statistics_sha256'}
        if (set(context) != required or context['source_domain'] != 'FIXED_BASE_ARX5_TASK_SIM'
                or context['task'] not in ('T1', 'T2') or context['split'] not in ('TRAIN', 'DEV')
                or any(not isinstance(v, str) or not v for v in context.values())):
            raise ValueError('EXPLICIT_FROZEN_SOURCE_EPISODE_CONTEXT_REQUIRED')
        self.context = copy.deepcopy(context)
        self.episode_id, self.max_chunks, self.evidence_kind = episode_id, max_chunks, evidence_kind
        self.phase, self.tick, self.sequence = 'WAIT_OBSERVATION', 0, 0
        self.snapshots, self.records = set(), []
        self.observation, self.decision = None, None

    def observe(self, *, observation_id, tick, timestamp_ns, measured_state10, rgb_sha256, binding):
        validate_source_abi(binding)
        if (self.phase != 'WAIT_OBSERVATION' or not observation_id or observation_id in self.snapshots
                or type(tick) is not int or tick != self.tick
                or type(timestamp_ns) is not int or timestamp_ns != tick*2_000_000
                or len(rgb_sha256) != 2 or any(len(s) != 64 or any(c not in '0123456789abcdef' for c in s) for s in rgb_sha256)):
            raise ValueError('FRESH_NEXT_OBSERVATION_AFTER_ACTUAL_PREFIX_ACK_REQUIRED')
        state = eef10(measured_state10, horizon=None)
        self.snapshots.add(observation_id)
        self.observation = dict(observation_id=observation_id, tick=tick, timestamp_ns=timestamp_ns,
            measured_state10=state.tolist(), measured_BASE=None, base_observation_present=False,
            rgb_sha256=list(rgb_sha256), binding=copy.deepcopy(binding), context=copy.deepcopy(self.context), episode_id=self.episode_id)
        self.phase = 'COMPLETE' if self.sequence == self.max_chunks else 'OBSERVED'
        return copy.deepcopy(self.observation)

    def choose(self, student, student_safety, expert=None, expert_safety=None, *, student_noise_id, **kwargs):
        if self.phase != 'OBSERVED' or student.observation_id != self.observation['observation_id']:
            raise ValueError('ONE_SELECTION_ON_CURRENT_STATE_REQUIRED')
        self.decision = arbitrate(student, student_safety, expert, expert_safety, **kwargs)
        if not isinstance(student_noise_id,str) or not student_noise_id:raise ValueError('PINNED_STUDENT_DRAW_ID_REQUIRED')
        self.proposals = dict(student_H40=student.checked_action().tolist(), student_noise_id=student_noise_id,
            expert_H40=expert.checked_action().tolist() if expert is not None and self.decision['expert_label_qualified'] else None,
            expert_valid=self.decision['expert_label_qualified'],
            active_dim_mask=[[True]*10 for _ in range(40)] if self.decision['expert_label_qualified'] else None)
        self.phase = 'STOPPED' if self.decision['selected_executor'] == 'STOP' else 'SELECTED'
        if self.phase == 'STOPPED':
            self.records.append(dict(observation=copy.deepcopy(self.observation), proposals=copy.deepcopy(self.proposals), decision=copy.deepcopy(self.decision),
                execution=None, bank='FAILURE_DIAGNOSTIC', evidence_kind=self.evidence_kind))
        return copy.deepcopy(self.decision)

    def dispatch_intent(self):
        if self.phase != 'SELECTED':raise ValueError('NO_REPEATED_OR_UNSELECTED_DISPATCH')
        self.phase = 'WAIT_ACK'
        self.intent_sha256 = digest(dict(episode_id=self.episode_id, sequence=self.sequence,
            observation=self.observation, decision=self.decision))
        return dict(intent_sha256=self.intent_sha256, reference_id=self.intent_sha256, prefix=copy.deepcopy(self.decision['prefix_to_install']),
                    tick_before=self.tick, requested_integrations=200)

    def acknowledge(self, *, intent_sha256, installed_prefix, tick_before, tick_after,
                    native_safe, task_success=False, early_end=False, object_contact=False):
        if (self.phase != 'WAIT_ACK' or intent_sha256 != self.intent_sha256 or tick_before != self.tick
                or type(tick_before) is not int or type(tick_after) is not int
                or not self.tick < tick_after <= self.tick+200
                or installed_prefix != self.decision['prefix_to_install']
                or any(type(v) is not bool for v in (native_safe, task_success, early_end, object_contact))):
            raise ValueError('ACTUAL_ONE_SHOT_PREFIX_ACK_REQUIRED')
        if tick_after != tick_before+200 and not early_end and native_safe:
            raise ValueError('PARTIAL_ADVANCE_MUST_BE_EXPLICIT_EARLY_END')
        record = dict(observation=copy.deepcopy(self.observation), proposals=copy.deepcopy(self.proposals), decision=copy.deepcopy(self.decision),
            execution=dict(intent_sha256=intent_sha256, installed_prefix=copy.deepcopy(installed_prefix),
                installed_ref_id=intent_sha256,
                tick_before=tick_before, tick_after=tick_after, integrations=tick_after-tick_before,
                native_safe=native_safe, early_end=early_end, object_contact=object_contact),
            evidence_kind=self.evidence_kind, task_success_claim=task_success,
            bank='INTERVENTION' if native_safe else 'FAILURE_DIAGNOSTIC', successful_BC_demo_admitted=False)
        self.records.append(record);self.tick=tick_after;self.sequence+=1
        self.phase = 'STOPPED' if not native_safe or early_end else 'WAIT_OBSERVATION'
        return copy.deepcopy(record)

    def unknown_dispatch(self):
        if self.phase != 'WAIT_ACK':raise ValueError('NO_PENDING_DISPATCH')
        self.phase = 'UNKNOWN_STOPPED_NO_RETRY'

    def early_end(self, reason):
        if self.phase not in ('WAIT_OBSERVATION', 'OBSERVED') or not reason:
            raise ValueError('UNKNOWN_OR_DISPATCHED_ACTION_CANNOT_BE_DECLARED_UNEXECUTED')
        self.phase = 'STOPPED'
        self.records.append(dict(reason=reason, execution=None, bank='FAILURE_DIAGNOSTIC', evidence_kind=self.evidence_kind))
