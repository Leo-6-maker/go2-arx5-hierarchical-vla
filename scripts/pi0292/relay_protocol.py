"""Durable CPU-only protocol for one paused-observation relay episode.

This module neither advances physics nor invokes a model or network endpoint.
An accepted command is handed out once, after its reservation is durable. A lost
ACK must be diagnosed, never handled by resending the physical command.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import uuid

from .durable import OwnerLock, read, write
from .eval_contract import FRAME, TCP, gate_action

PROTOCOL = 'PI0292_PAUSED_RELAY_V1'
CAMERAS = ('external_rgb', 'wrist_rgb')
RGB_SIZE = (256, 256)  # Frozen PI0291 renderer pixels, before official processing.
IDENTITY_KEYS = {'profile', 'model_revision', 'source_sha', 'config_sha256',
                 'processor_sha256', 'statistics_sha256'}


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def sealed(value):
    body = copy.deepcopy(value)
    return dict(body=body, sha256=digest(body))


def exact_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError('UNEXPECTED_PACKET_FIELDS')


def unseal(packet):
    exact_keys(packet, ('body', 'sha256'))
    if packet['sha256'] != digest(packet['body']):
        raise ValueError('PACKET_HASH_MISMATCH')
    return copy.deepcopy(packet['body'])


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('NONNEGATIVE_INTEGER_REQUIRED')
    return value


def finite_vector(value, count):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError('VECTOR_SHAPE')
    if not all(type(x) in (int, float) and math.isfinite(x) for x in value):
        raise ValueError('NONFINITE_VECTOR')


def uuid_string(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError('CANONICAL_UUID_REQUIRED')


def validate_identity(value):
    exact_keys(value, IDENTITY_KEYS)
    if value['profile'] not in ('B0', 'B1', 'TEACHER_REFERENCE'):
        raise ValueError('UNREGISTERED_PROFILE')
    for key in IDENTITY_KEYS - {'profile'}:
        item = value[key]
        if key == 'model_revision' and value['profile'] == 'TEACHER_REFERENCE' and item == 'NOT_APPLICABLE':
            continue
        length = 40 if key.endswith(('revision', 'source_sha')) else 64
        if not isinstance(item, str) or len(item) != length or any(c not in '0123456789abcdef' for c in item):
            raise ValueError('IMMUTABLE_IDENTITY_REQUIRED')


def validate_observation(payload):
    exact_keys(payload, ('snapshot_uuid', 'language', 'state_eef10', 'images', 'frame', 'tcp'))
    uuid_string(payload['snapshot_uuid'])
    if not isinstance(payload['language'], str) or not payload['language'] or len(payload['language']) > 4096:
        raise ValueError('ORIGINAL_LANGUAGE_REQUIRED')
    if payload['frame'] != FRAME or payload['tcp'] != TCP:
        raise ValueError('FRAME_OR_TCP_CHANGED')
    finite_vector(payload['state_eef10'], 10)
    if not isinstance(payload['images'], list) or len(payload['images']) != 2:
        raise ValueError('TWO_ORIGINAL_RGB_IMAGES_REQUIRED')
    from PIL import Image
    for expected, image in zip(CAMERAS, payload['images']):
        exact_keys(image, ('camera', 'png_base64', 'sha256'))
        if image['camera'] != expected or not isinstance(image['png_base64'], str):
            raise ValueError('CAMERA_ORDER_CHANGED')
        if len(image['png_base64']) > 6 * 1024**2:
            raise ValueError('RGB_PACKET_SIZE_LIMIT')
        raw = base64.b64decode(image['png_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != image['sha256']:
            raise ValueError('RGB_HASH_MISMATCH')
        with Image.open(io.BytesIO(raw)) as picture:
            if picture.format != 'PNG' or picture.mode != 'RGB' or picture.size != RGB_SIZE:
                raise ValueError('ORIGINAL_RGB_FORMAT_REQUIRED')
            picture.verify()


class RelayExchange:
    """One durable, single-owner episode; all protocol errors stop the attempt."""

    def __init__(self, directory, *, run_uuid=None, episode_uuid=None, identity=None, now=None):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = OwnerLock(self.root / 'OWNER.lock')
        self.path = self.root / 'STATE.json'
        try:
            if run_uuid is None:
                self.state = read(self.path)
                if self.state['protocol'] != PROTOCOL:
                    raise ValueError('PROTOCOL_CHANGED')
            else:
                if self.path.exists():
                    raise FileExistsError('NEVER_RESTART_EXISTING_EPISODE')
                if not isinstance(run_uuid, str) or not run_uuid:
                    raise ValueError('RUN_UUID_REQUIRED')
                uuid_string(episode_uuid)
                validate_identity(identity)
                if not isinstance(now, (int, float)) or not math.isfinite(now):
                    raise ValueError('WALL_CLOCK_REQUIRED')
                self.state = dict(protocol=PROTOCOL, run_uuid=run_uuid, episode_uuid=episode_uuid,
                    identity=copy.deepcopy(identity), started=now, last_wall=now, deadline=now + 900,
                    phase='WAIT_OBSERVATION', sequence=0, tick=0, history=[], used_snapshots=[], pending=None)
                write(self.path, self.state)
        except BaseException:
            self.lock.close()
            raise

    def close(self):
        self.lock.close()

    def _clock(self, now):
        if not isinstance(now, (int, float)) or not math.isfinite(now) or now < self.state['last_wall']:
            raise ValueError('WALL_CLOCK_REVERSED_OR_INVALID')
        if now >= self.state['deadline']:
            raise ValueError('EPISODE_TIMEOUT')
        pending = self.state['pending']
        if pending and now >= pending['expires']:
            raise ValueError('QUERY_TIMEOUT')
        self.state['last_wall'] = now

    def _fail(self, error):
        self.state['phase'] = 'STOPPED'
        self.state.setdefault('failure', str(error))
        if self.state['pending'] and 'command_sha256' in self.state['pending']:
            self.state['unacknowledged_physics'] = 'UNKNOWN_DO_NOT_RESEND_COMMAND'
        write(self.path, self.state)

    def _envelope(self, body, kind):
        exact_keys(body, ('protocol', 'kind', 'run_uuid', 'episode_uuid', 'sequence', 'tick', 'payload'))
        if (body['protocol'], body['kind'], body['run_uuid'], body['episode_uuid']) != (
                PROTOCOL, kind, self.state['run_uuid'], self.state['episode_uuid']):
            raise ValueError('PACKET_IDENTITY_MISMATCH')
        if integer(body['sequence']) != self.state['sequence'] or integer(body['tick']) != self.state['tick']:
            raise ValueError('STALE_DUPLICATE_OR_REORDERED_PACKET')

    def observe(self, packet, now):
        try:
            self._clock(now)
            if self.state['phase'] != 'WAIT_OBSERVATION' or self.state['sequence'] >= 80:
                raise ValueError('OBSERVATION_NOT_EXPECTED')
            body = unseal(packet)
            self._envelope(body, 'observation')
            payload = body['payload']
            validate_observation(payload)
            if payload['snapshot_uuid'] in self.state['used_snapshots']:
                raise ValueError('SNAPSHOT_REUSE')
            query_uuid, reference_uuid = str(uuid.uuid4()), str(uuid.uuid4())
            pending = dict(query_uuid=query_uuid, reference_uuid=reference_uuid,
                observation_sha256=packet['sha256'], anchor_sha256=digest(payload['state_eef10']),
                expires=min(now + 60, self.state['deadline']), state_eef10=payload['state_eef10'])
            request = sealed(dict(protocol=PROTOCOL, kind='student_request', run_uuid=self.state['run_uuid'],
                episode_uuid=self.state['episode_uuid'], sequence=self.state['sequence'], tick=self.state['tick'],
                query_uuid=query_uuid, reference_uuid=reference_uuid, expires=pending['expires'],
                observation_sha256=packet['sha256'], anchor_sha256=pending['anchor_sha256'],
                identity=self.state['identity'], payload=payload))
            self.state['used_snapshots'].append(payload['snapshot_uuid'])
            self.state.update(phase='WAIT_ACTION', pending=pending)
            write(self.root / ('observation-%03d.json' % self.state['sequence']), packet)
            write(self.root / ('request-%03d.json' % self.state['sequence']), request)
            write(self.path, self.state)
            return request
        except (ValueError, TypeError, KeyError, OSError) as error:
            self._fail(error)
            raise

    def action(self, packet, now):
        try:
            self._clock(now)
            if self.state['phase'] != 'WAIT_ACTION':
                raise ValueError('ACTION_NOT_EXPECTED_OR_ALREADY_RESERVED')
            body = unseal(packet)
            self._envelope(body, 'action')
            value = body['payload']
            exact_keys(value, ('query_uuid', 'reference_uuid', 'observation_sha256', 'anchor_sha256',
                               'identity', 'trajectory_eef10', 'offsets_ns'))
            for key in ('query_uuid', 'reference_uuid', 'observation_sha256', 'anchor_sha256'):
                if value[key] != self.state['pending'][key]:
                    raise ValueError('ACTION_OBSERVATION_BINDING_MISMATCH')
            if value['identity'] != self.state['identity']:
                raise ValueError('MODEL_OR_PROCESSOR_IDENTITY_CHANGED')
            if (not isinstance(value['offsets_ns'], list)
                    or any(type(x) is not int for x in value['offsets_ns'])
                    or value['offsets_ns'] != [i * 50_000_000 for i in range(40)]):
                raise ValueError('NATIVE_H40_CLOCK_CHANGED')
            trajectory = value['trajectory_eef10']
            if not isinstance(trajectory, list) or len(trajectory) != 40:
                raise ValueError('NATIVE_H40_REQUIRED')
            for row in trajectory:
                finite_vector(row, 10)
            gate = gate_action(trajectory, self.state['pending']['state_eef10'],
                               self.state['tick'] * .002, value['reference_uuid'])
            write(self.root / ('action-%03d.json' % self.state['sequence']), dict(packet=packet, gate=gate))
            if gate['decision'] not in ('ACCEPT', 'WARN'):
                raise ValueError('FROZEN_SAFETY_GATE_REJECT')
            prefix = trajectory[:8]
            command = sealed(dict(protocol=PROTOCOL, kind='install_prefix8', run_uuid=self.state['run_uuid'],
                episode_uuid=self.state['episode_uuid'], sequence=self.state['sequence'], tick=self.state['tick'],
                query_uuid=value['query_uuid'], reference_uuid=value['reference_uuid'],
                observation_sha256=value['observation_sha256'], anchor_sha256=value['anchor_sha256'],
                prefix_eef10=prefix, prefix_sha256=digest(prefix), advance_integrations=200,
                frame=FRAME, tcp=TCP, expires=self.state['pending']['expires']))
            self.state['pending'].update(command_sha256=command['sha256'], prefix_sha256=digest(prefix))
            self.state.update(phase='WAIT_ACK')
            write(self.root / ('command-%03d.json' % self.state['sequence']), command)
            write(self.path, self.state)
            return command
        except (ValueError, TypeError, KeyError, OSError) as error:
            self._fail(error)
            raise

    def acknowledge(self, packet, now):
        try:
            self._clock(now)
            if self.state['phase'] != 'WAIT_ACK':
                raise ValueError('ACK_NOT_EXPECTED')
            body = unseal(packet)
            self._envelope(body, 'ack')
            value = body['payload']
            exact_keys(value, ('query_uuid', 'reference_uuid', 'command_sha256', 'prefix_sha256',
                               'tick_after', 'integrations', 'physics_receipt_sha256'))
            for key in ('query_uuid', 'reference_uuid', 'command_sha256', 'prefix_sha256'):
                if value[key] != self.state['pending'][key]:
                    raise ValueError('ACK_COMMAND_BINDING_MISMATCH')
            if integer(value['integrations']) != 200 or integer(value['tick_after']) != self.state['tick'] + 200:
                raise ValueError('PHYSICS_PROGRESS_MISMATCH')
            receipt = value['physics_receipt_sha256']
            if not isinstance(receipt, str) or len(receipt) != 64 or any(c not in '0123456789abcdef' for c in receipt):
                raise ValueError('PHYSICS_RECEIPT_HASH_REQUIRED')
            write(self.root / ('ack-%03d.json' % self.state['sequence']), packet)
            self.state['history'].append(dict(sequence=self.state['sequence'], tick_before=self.state['tick'],
                tick_after=value['tick_after'], query_uuid=value['query_uuid'], ack_sha256=packet['sha256']))
            self.state.update(sequence=self.state['sequence'] + 1, tick=value['tick_after'], pending=None,
                              phase='DONE' if value['tick_after'] == 16000 else 'WAIT_OBSERVATION')
            write(self.path, self.state)
        except (ValueError, TypeError, KeyError, OSError) as error:
            self._fail(error)
            raise

    def check_timeout(self, now):
        try:
            self._clock(now)
            write(self.path, self.state)
        except ValueError as error:
            self._fail(error)
            raise
