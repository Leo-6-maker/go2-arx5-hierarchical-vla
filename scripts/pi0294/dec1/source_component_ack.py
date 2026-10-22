"""Cold original-feedback Teacher ACK; CPU source components only.

No neural identity, RelayTeacher.pending, relay packet or model ACK contract.
The owner must already admit the fixed component plan/condition and original
source runtime. This module neither admits nor runs physics or debits a ledger.
The actual owner record callback must durably preserve its source events.
read_events() must independently read persisted bytes and verify the entire
original sequence/hash chain before returning event dictionaries. This is a
required cold-owner seam, not an assertion of native qualification.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading

import numpy as np
from scripts.pi0292.durable import read, write
from scripts.pi0292.eval_contract import FRAME, TCP
from scripts.pi0292.relay_protocol import exact_keys, uuid_string
from scripts.pi0294.scene_cards import digest
from scripts.pi0294.teacher import PickPlaceTeacher, PushTeacher
from scripts.pi0294.dec1 import source_commissioning_plan
from scripts.pi0294.dec1.mapped_ack import (
    _hash, _int, _finite, _identifier, _mapped_hash, raw_h40_hash,
    REFERENCE_KEYS, SEMANTICS, S1_VERSION,
)
from scripts.pi0294.dec1.source_episode_wire import NativeMappedPrefixAck
from go2arx5_vla.task_data.codec import pose10

POLICY = 'ORIGINAL_FEEDBACK_TEACHER_NO_GR00T'
PURPOSE = 'SOURCE_INTERFACE_COMPONENT_QUALIFICATION_ONLY'
SCHEMA = 'DEC1_SOURCE_COMPONENT_MAPPED_ACK_V1'
PLAN_SCHEMA = 'DEC1_SOURCE_COMPONENT_RAW_PLAN_PREBIND_V1'
PROOF_KEYS = {'nonce', 'raw_chunk_id', 'episode_id', 'executor_id',
    'raw_h40_sha256', 'action_sha256', 'proposal_digests', 'phases', 'actual_ticks',
    'tick_before', 'tick_after', 'initial_adapter_ack_id', 's1',
    'actual_reference_acks', 'maximum_live_shadow_difference', 'ABI'}


def _bytes(value):
    # Exact original durable.write encoding; no relay sealing/resealing.
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class SourceComponentAcknowledger:
    """One new owner, engine and original Teacher; no resume or replay.

    Construct only in an admitted cold component factory. Wrap the ORIGINAL
    feedback Teacher with wrap_teacher(), then pass that wrapped teacher to
    SourceInterfaceTeacher and self.acknowledge as acknowledge_source. Never
    nest E1ProtectedTeacher or the old S1 path around this executor.
    SourceInterfaceTeacher remains the sole caller of executor.prepare/begin.

    Each entry into Teacher.plan or acknowledge is terminal on failure,
    including interrupted/unknown persistence. Files remain preserved. The
    existing owner handles all lease/CPU/integration/clone accounting.
    """
    def __init__(self, engine, directory, *, plan_bytes, expected_plan_sha256,
                 population_bytes, owner_release, condition_id, episode_id,
                 executor_id, adapter, runtime_release, read_events, guard, writer=write, reader=read):
        if (type(plan_bytes) is not bytes
                or hashlib.sha256(plan_bytes).hexdigest() != _hash(expected_plan_sha256)
                or not isinstance(runtime_release, str)
                or re.fullmatch('[0-9a-f]{40}', runtime_release) is None
                or not all(callable(f) for f in (guard, writer, reader, read_events))):
            raise ValueError('FIXED_COMPONENT_PLAN_BYTES_AND_OWNER_CALLBACKS_REQUIRED')
        plan = source_commissioning_plan.validate(json.loads(plan_bytes), population_bytes,
            owner_release=owner_release)
        found = [c for c in plan['conditions'] if c['condition_id'] == condition_id]
        if (len(found) != 1 or found[0]['executor'] != 'E_SOURCE'
                or found[0]['policy'] != POLICY or found[0]['purpose'] != PURPOSE
                or found[0]['interface'] not in ('I_TIME', 'I_PROGRESS')
                or digest(engine.card) != found[0]['card_sha256']
                or type(engine.steps) is not int or engine.steps != 0
                or float(engine.data.time) != 0. or engine.safety):
            raise ValueError('FRESH_ENGINE_AND_EXACT_FOUR_SOURCE_COMPONENT_CONDITION_REQUIRED')
        _identifier(episode_id); _identifier(executor_id)
        if (adapter.episode_id != episode_id or adapter.mode != found[0]['interface']
                or adapter.state()['pending_command_id'] is not None
                or adapter.state()['predecessor_ack_id'] is not None):
            raise ValueError('EXACT_IDLE_COMPONENT_INTERFACE_ADAPTER_REQUIRED')
        directory = Path(directory)
        if not directory.is_dir() or directory.is_symlink() or any(directory.iterdir()):
            raise ValueError('NEW_EMPTY_OWNER_COMPONENT_ACK_DIRECTORY_REQUIRED')
        self.engine = engine
        self.adapter = adapter
        self.directory = directory
        self.read_events = read_events
        self.condition = deepcopy(found[0])
        self.plan_sha = expected_plan_sha256
        self.owner_release = owner_release
        self.runtime_release = runtime_release
        self.episode_id, self.executor_id = episode_id, executor_id
        self.guard, self.writer, self.reader = guard, writer, reader
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._busy = False
        self._unknown = True
        self._sequence = 0
        self._prior_ack = None
        self._pending = None
        self._teacher = None
        for name in ('plans', 'physics', 'acks'):
            (directory / name).mkdir()
        _sync_directory(directory)
        binding = dict(schema='DEC1_SOURCE_COMPONENT_OWNER_BINDING_V1',
            component_plan_sha256=self.plan_sha, plan_owner_release=owner_release,
            runtime_release=runtime_release,
            condition_id=condition_id, card_sha256=self.condition['card_sha256'],
            episode_id=episode_id, executor_id=executor_id,
            interface_mode=self.condition['interface'], execution_semantics=SEMANTICS,
            policy=POLICY, purpose=PURPOSE, frame=FRAME, tcp=TCP,
            student_neural_ACK_qualification=False, GR00T_queries=0)
        self._write_once(directory / 'OWNER_COMPONENT_BINDING.json', binding)
        self._unknown = False

    def _write_once(self, path, document):
        if path.exists() or path.is_symlink():
            raise FileExistsError('PRESERVE_FIRST_COMPONENT_EFFECT_NO_REPLAY')
        self.writer(path, document)
        raw = path.read_bytes()
        if self.reader(path) != document or raw != _bytes(document):
            raise ValueError('ORIGINAL_COMPONENT_DURABLE_BYTES_READBACK_REQUIRED')
        return hashlib.sha256(raw).hexdigest()

    def _enter(self):
        if self._busy or self._unknown:
            raise ValueError('COMPONENT_OUTCOME_UNKNOWN_NO_RETRY')
        self._busy = True
        self._unknown = True
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            raise ValueError('COMPONENT_SINGLE_ACTUAL_OWNER_REQUIRED')
        if self.adapter.episode_id != self.episode_id or self.adapter.mode != self.condition['interface']:
            raise ValueError('FIXED_COMPONENT_INTERFACE_ADAPTER_CHANGED')
        self.guard()

    def _leave(self):
        self._busy = False
        self._unknown = False

    def wrap_teacher(self, teacher):
        expected = PickPlaceTeacher if self.condition['task'] == 'T1' else PushTeacher
        if self._teacher is not None or type(teacher) is not expected or self._unknown or self._busy:
            raise ValueError('ONE_ORIGINAL_UNWRAPPED_FEEDBACK_TEACHER_REQUIRED')
        self._teacher = teacher
        owner = self

        class PreboundFeedbackTeacher:
            def __getattr__(self, name):
                return getattr(teacher, name)

            def plan(self, state):
                # Owner entry precedes the sole original Teacher call. A failed
                # Teacher call cannot be retried to produce a replacement plan.
                return owner._plan(teacher, state)

        return PreboundFeedbackTeacher()

    def _plan(self, teacher, state):
        self._enter()
        tick = self._sequence * 200
        if (teacher is not self._teacher or self._pending is not None or self._sequence >= 80
                or type(self.engine.steps) is not int or self.engine.steps != tick
                or not math.isfinite(float(state['time_s']))
                or abs(float(state['time_s']) - tick * .002) > 1e-8
                or float(state['time_s']) != float(self.engine.data.time)
                or self.engine.safety or self.adapter.state()['pending_command_id'] is not None
                or self.adapter.state()['predecessor_ack_id'] != self._prior_ack):
            raise ValueError('FRESH_TEACHER_PLAN_ONLY_AFTER_ACTUAL_PREVIOUS_ACK')
        original = teacher.plan(state)  # Exactly one call; no independent Teacher replay.
        poses = np.asarray(original['poses'], dtype=float)
        grips = np.asarray(original['opening'], dtype=float)
        offsets = np.asarray(original['offset_s'], dtype=float)
        if (poses.shape != (40, 4, 4) or grips.shape != (40,)
                or offsets.shape != (40,) or not np.isfinite(poses).all()
                or not np.isfinite(grips).all() or not np.array_equal(offsets, np.arange(40) / 20)):
            raise ValueError('ORIGINAL_FINITE_TEACHER_FULL_H40_AND_20HZ_OFFSETS_REQUIRED')
        rows = np.stack([pose10(p, g) for p, g in zip(poses, grips)]).tolist()
        chunk = original.get('reference_uuid', self.episode_id + ':ref' + str(self._sequence))
        _identifier(chunk)
        binding = dict(schema=PLAN_SCHEMA, component_plan_sha256=self.plan_sha,
            condition_id=self.condition['condition_id'], interface_mode=self.condition['interface'],
            policy=POLICY, purpose=PURPOSE, episode_id=self.episode_id, executor_id=self.executor_id,
            sequence=self._sequence, tick_before=tick, raw_chunk_id=chunk,
            raw_h40_sha256=raw_h40_hash(rows), raw_H40_eef10=rows,
            offsets_s=offsets.tolist(), observation_timestamp=float(state['time_s']),
            actual_feedback_eef10=pose10(state['tcp'], state['gripper_fraction']).tolist(),
            initial_adapter_ack_id=self._prior_ack, frame=FRAME, tcp=TCP)
        path = self.directory / 'plans' / f'{self._sequence:03d}.json'
        binding_sha = self._write_once(path, binding)
        # Preserved raw plan is bound BEFORE caller executor.prepare/begin/live.
        self._pending = (deepcopy(binding), binding_sha, path)
        self._leave()
        return original

    def _proof(self, proof, binding):
        exact_keys(proof, PROOF_KEYS)
        tick = binding['tick_before']
        if (proof['episode_id'] != self.episode_id or proof['executor_id'] != self.executor_id
                or proof['raw_chunk_id'] != binding['raw_chunk_id']
                or proof['raw_h40_sha256'] != binding['raw_h40_sha256']
                or _int(proof['actual_ticks']) != 200 or _int(proof['tick_before']) != tick
                or _int(proof['tick_after']) != tick + 200
                or proof['initial_adapter_ack_id'] != self._prior_ack
                or proof['ABI'] != 'SINGLE_4x4_POSE_PLUS_GRIPPER_NOT_PREVIEW9'):
            raise ValueError('PREBOUND_COMPONENT_RAW_PLAN_AND_ACTUAL_PREFIX200_REQUIRED')
        uuid_string(proof['nonce']); _hash(proof['raw_h40_sha256']); _hash(proof['action_sha256'])
        s1 = proof['s1']
        exact_keys(s1, ('version', 'decision', 'nonce', 'state_sha256', 'action_sha256', 'shadow_integrations'))
        if (s1['version'] != S1_VERSION or s1['decision'] != 'PASS'
                or s1['nonce'] != proof['nonce'] or s1['action_sha256'] != proof['action_sha256']
                or _int(s1['shadow_integrations']) != 200):
            raise ValueError('ORIGINAL_ONE_CLONE200_S1_NONCE_STATE_ACTION_PASS_REQUIRED')
        _hash(s1['state_sha256']); _hash(s1['action_sha256'])
        maximum = _finite(proof['maximum_live_shadow_difference'])
        if not 0 <= maximum <= 1e-10:
            raise ValueError('ORIGINAL_COMPONENT_S1_LIVE_SHADOW_PARITY_REQUIRED')
        refs = proof['actual_reference_acks']
        digests, phases = proof['proposal_digests'], proof['phases']
        if any(not isinstance(v, list) or len(v) != 8 for v in (refs, digests, phases)):
            raise ValueError('EIGHT_ACTUAL_COMPONENT_REFERENCE_ACKS_REQUIRED')
        prior = self._prior_ack
        seen = set()
        for i, row in enumerate(refs):
            exact_keys(row, REFERENCE_KEYS)
            name = _identifier(row['command_id']); _hash(row['proposal_sha256']); _hash(digests[i])
            if (type(row['reference']) is not int or row['reference'] != i or name in seen or name == prior
                    or row['ack_id'] != name or row['predecessor_ack_id'] != prior
                    or row['proposal_sha256'] != digests[i] or _finite(row['phase']) != _finite(phases[i])
                    or _int(row['tick_before']) != tick + 25 * i
                    or _int(row['tick_after']) != tick + 25 * (i + 1)
                    or abs(_finite(row['execution_end_timestamp']) - row['tick_after'] * .002) > 1e-8):
                raise ValueError('ACTUAL_COMPONENT_REFERENCE_PHASE_ACK_CLOCK_CHAIN_CHANGED')
            seen.add(name); prior = name
        if _mapped_hash(refs) != proof['action_sha256']:
            raise ValueError('ACTUAL_COMPONENT_MAPPED_EIGHT_ACTION_HASH_CHANGED')
        return prior

    def _events(self, proof):
        # This reads real owner-recorded events; no fallback to the supplied
        # proof, copied shadow predictions or reconstructed no-op ACK records.
        events = self.read_events()
        if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
            raise ValueError('ACTUAL_DURABLE_SOURCE_EVENT_LIST_REQUIRED')
        owner = dict(episode_id=self.episode_id, executor_id=self.executor_id)
        nonce = proof['nonce']
        def one(kind, expected):
            selected = [e for e in events if e.get('kind') == kind and e.get('nonce') == nonce]
            if selected != [{**owner, **expected, 'kind':kind}]:
                raise ValueError('ACTUAL_OWNER_SOURCE_EVENT_BINDING_REQUIRED:' + kind)
        one('SOURCE_PREFIX200_ACK', proof)
        one('SOURCE_PREFIX_SHADOW_COMPLETE', dict(nonce=nonce, shadow_ticks=200, clone_attempts=1,
            proposal_digests=proof['proposal_digests'], phases=proof['phases']))
        one('SOURCE_MAPPED_PREFIX_BIND_CONFIRMED', dict(nonce=nonce))
        one('SOURCE_MAPPED_PREFIX_INSTALL_INTENT', dict(nonce=nonce,
            raw_h40_sha256=proof['raw_h40_sha256'], action_sha256=proof['action_sha256'],
            proposal_digests=proof['proposal_digests'], phases=proof['phases'],
            mapped_poses=[r['target'] for r in proof['actual_reference_acks']],
            mapped_grippers=[r['opening'] for r in proof['actual_reference_acks']],
            native_step_before=proof['tick_before']))
        actual = [e for e in events if e.get('kind') == 'SOURCE_ACTUAL_REFERENCE_ACK' and e.get('nonce') == nonce]
        if actual != [dict(kind='SOURCE_ACTUAL_REFERENCE_ACK', **owner, nonce=nonce, **r)
                      for r in proof['actual_reference_acks']]:
            raise ValueError('EIGHT_ACTUAL_OWNER_ACK_EVENTS_REQUIRED')
        ordered = [e['kind'] for e in events if e.get('nonce') == nonce and e.get('kind') in (
            'SOURCE_PREFIX_SHADOW_COMPLETE', 'SOURCE_MAPPED_PREFIX_INSTALL_INTENT',
            'SOURCE_MAPPED_PREFIX_BIND_CONFIRMED', 'SOURCE_ACTUAL_REFERENCE_ACK', 'SOURCE_PREFIX200_ACK')]
        if ordered != ['SOURCE_PREFIX_SHADOW_COMPLETE', 'SOURCE_MAPPED_PREFIX_INSTALL_INTENT',
                'SOURCE_MAPPED_PREFIX_BIND_CONFIRMED'] + ['SOURCE_ACTUAL_REFERENCE_ACK'] * 8 + ['SOURCE_PREFIX200_ACK']:
            raise ValueError('ACTUAL_SOURCE_BIND_SHADOW_ACK_EVENT_ORDER_CHANGED')
        # Owner callable reads/verifies persisted bytes/sequence/hash chain;
        # this is the digest of that complete verified EVENT LIST, not a claim
        # about the physical JSONL encoding or file-byte SHA.
        return hashlib.sha256(_bytes(events)).hexdigest()

    def acknowledge(self, engine, rows, proof):
        self._enter()  # Failure from here is non-retryable, even before IO.
        if self._pending is None or engine is not self.engine:
            raise ValueError('ACTUAL_PREBOUND_COMPONENT_ENGINE_AND_PLAN_REQUIRED')
        binding, binding_sha, binding_path = self._pending
        if (self.reader(binding_path) != binding or binding_path.read_bytes() != _bytes(binding)
                or _sha(binding_path) != binding_sha):
            raise ValueError('ORIGINAL_PREEXECUTION_TEACHER_PLAN_BYTES_CHANGED')
        final_ack = self._proof(proof, binding)
        tick, end = proof['tick_before'], proof['tick_after']
        if (type(engine.steps) is not int or engine.steps != end or engine.safety
                or not math.isfinite(float(engine.data.time))
                or abs(float(engine.data.time) - end * .002) > 1e-8
                or self.adapter.state()['pending_command_id'] is not None
                or self.adapter.state()['predecessor_ack_id'] != final_ack):
            raise ValueError('ACTUAL_SAFE_NATIVE_ENGINE_PREFIX200_CLOCK_REQUIRED')
        actual = np.asarray(rows, dtype=float).copy()
        columns = 1 + engine.model.nq + engine.model.nv + engine.model.nu + 6 + 1
        if (actual.shape != (200, columns) or not np.isfinite(actual).all()
                or np.any(actual[:, -1] != 0)
                or np.max(np.abs(actual[:, 0] - np.arange(tick + 1, end + 1) * .002)) > 1e-8):
            raise ValueError('ACTUAL_ORIGINAL_200ROW_SAFE_PHYSICS_AND_CLOCK_REQUIRED')
        event_sha = self._events(proof)
        physics = self.directory / 'physics' / f'{self._sequence:03d}.npz'
        ack_path = self.directory / 'acks' / f'{self._sequence:03d}.json'
        if physics.exists() or physics.is_symlink() or ack_path.exists() or ack_path.is_symlink():
            raise FileExistsError('PRESERVE_FIRST_COMPONENT_ACK_EFFECTS_NO_REPLAY')
        # Exclusive native artifact path; no overwrite or save retry.
        with physics.open('xb') as handle:
            np.savez_compressed(handle, rows=actual, tick_before=tick, tick_after=end)
            handle.flush(); os.fsync(handle.fileno())
        _sync_directory(physics.parent)
        physics_sha = _sha(physics)
        with np.load(physics, allow_pickle=False) as stored:
            if (set(stored.files) != {'rows', 'tick_before', 'tick_after'}
                    or stored['tick_before'].shape != () or stored['tick_after'].shape != ()
                    or stored['tick_before'].dtype.kind not in 'iu' or stored['tick_after'].dtype.kind not in 'iu'
                    or int(stored['tick_before']) != tick or int(stored['tick_after']) != end
                    or not np.array_equal(stored['rows'], actual)):
                raise ValueError('COMPLETE_ACTUAL_PHYSICS_NPZ_READBACK_REQUIRED')
        document = dict(schema=SCHEMA, component_plan_sha256=self.plan_sha,
            plan_owner_release=self.owner_release, runtime_release=self.runtime_release,
            condition_id=self.condition['condition_id'],
            card_sha256=self.condition['card_sha256'], interface_mode=self.condition['interface'],
            episode_id=self.episode_id, executor_id=self.executor_id, sequence=self._sequence,
            policy=POLICY, purpose=PURPOSE, execution_semantics=SEMANTICS, frame=FRAME, tcp=TCP,
            teacher_raw_plan_sha256=binding_sha, verified_source_event_list_sha256=event_sha,
            physics_receipt_sha256=physics_sha, proof=deepcopy(proof),
            student_neural_ACK_qualification=False, GR00T_queries=0)
        ack_sha = self._write_once(ack_path, document)
        if (_sha(physics) != physics_sha or _sha(binding_path) != binding_sha
                or self._events(proof) != event_sha or engine.steps != end
                or not math.isfinite(float(engine.data.time))
                or abs(float(engine.data.time) - end * .002) > 1e-8 or engine.safety
                or self.adapter.mode != self.condition['interface']
                or self.adapter.episode_id != self.episode_id
                or self.adapter.state()['pending_command_id'] is not None
                or self.adapter.state()['predecessor_ack_id'] != final_ack):
            raise ValueError('COMPLETE_COMPONENT_ACK_FINAL_READBACK_CHANGED')
        receipt = NativeMappedPrefixAck(self.episode_id, self.executor_id, proof['nonce'],
            proof['raw_h40_sha256'], proof['action_sha256'], tick, end,
            ack_path.relative_to(self.directory).as_posix(), physics_sha, ack_sha)
        self._sequence += 1
        self._prior_ack = final_ack
        self._pending = None
        self._leave()
        return receipt

    def state(self):
        return dict(sequence=self._sequence, unknown=self._unknown, entered=self._busy,
            pending_raw_plan=self._pending is not None, last_adapter_ack_id=self._prior_ack,
            policy=POLICY, purpose=PURPOSE, student_neural_ACK_qualification=False,
            native_qualification='NATIVE_COMMISSIONING_NOT_RUN')
