"""One-shot bounded file endpoints; SSH transport and accounting stay with owner."""
from pathlib import Path
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0292.physics_relay import accept_prefix
from scripts.pi0292.relay_protocol import PROTOCOL, digest, exact_keys, sealed, unseal, uuid_string, validate_identity
from .native_dataset import sha
from .relay_transport import teacher_action
from .run_admission import validate


def inside(root, relative):
    path = Path(relative)
    result = root / path
    if (path.is_absolute() or '..' in path.parts or result.is_symlink()
            or not result.resolve().is_relative_to(root.resolve())):
        raise PermissionError('REGISTERED_ENDPOINT_PATH_REQUIRED')
    return result


class Endpoint:
    def __init__(self, root, registry_relative, registry_sha256, role, *, now=time.time):
        self.root, self.role, self.now = Path(root), role, now
        if role not in ('core', 'hopper'):
            raise PermissionError('EXACT_ENDPOINT_ROLE_REQUIRED')
        manifest = validate(self.root, profile='4080-simulation' if role == 'core' else 'hopper-neural')
        path = inside(self.root, registry_relative)
        if sha(path) != registry_sha256:
            raise PermissionError('SEALED_ENDPOINT_REGISTRY_REQUIRED')
        self.registry = read(path)
        if (self.registry['run_uuid'] != self.root.name
                or self.registry[role+'_source_sha'] != manifest['code']['vla_commit']
                or self.registry[role+'_manifest_sha256'] != sha(self.root/'RUN_MANIFEST.json')
                or now() >= self.registry['deadline_epoch']):
            raise PermissionError('ENDPOINT_RUN_SOURCE_AND_ORIGINAL_DEADLINE_REQUIRED')

    def case(self, episode):
        uuid_string(episode)
        case = self.registry['cases'].get(episode)
        if case is None or case['episode_uuid'] != episode:
            raise PermissionError('EXACT_REGISTERED_EPISODE_REQUIRED')
        validate_identity(case['identity'])
        return case

    def core(self, operation, episode, sequence, bundle=None):
        if self.role != 'core' or type(sequence) is not int or not 0 <= sequence < 80:
            raise PermissionError('CORE_SEQUENCE_REQUIRED')
        case = self.case(episode)
        folder = inside(self.root, 'episodes/'+case['attempt_id'])
        request_path = inside(self.root, case['core_request_relative'])
        if sha(request_path) != case['core_request_sha256']:
            raise PermissionError('REGISTERED_CORE_REQUEST_CHANGED')
        request = read(request_path)
        if (request['episode_uuid'] != episode or request['identity'] != case['identity']
                or request['attempt_id'] != case['attempt_id']
                or request['source_sha'] != self.registry['core_source_sha']
                or request['run_manifest_sha256'] != self.registry['core_manifest_sha256']):
            raise PermissionError('EXACT_PHYSICAL_REQUEST_REQUIRED')
        relay = inside(self.root, str(folder.relative_to(self.root))+'/relay')
        names = {'observation': 'observations/'+f'{sequence:03d}.json',
                 'teacher': f'teacher-{sequence:03d}.json', 'ack': 'acks/'+f'{sequence:03d}.json'}
        if operation in names:
            if operation == 'teacher' and case['identity']['profile'] != 'TEACHER_REFERENCE':
                raise PermissionError('MODEL_CANNOT_READ_TEACHER_ANSWER')
            path = inside(relay, names[operation])
            return read(path) if path.is_file() else None
        if operation != 'install' or not isinstance(bundle, dict) or set(bundle) != {'command','action','request'}:
            raise PermissionError('EXACT_ONE_SHOT_INSTALL_REQUIRED')
        observation = read(inside(relay, names['observation']))
        command, _ = accept_prefix(bundle['command'], bundle['action'], observation,
            request=bundle['request'], identity=case['identity'], now=self.now())
        if command['episode_uuid'] != episode or command['sequence'] != sequence:
            raise ValueError('INSTALL_EPISODE_AND_SEQUENCE_CHANGED')
        previous = read(inside(relay, 'acks/'+f'{sequence-1:03d}.json')) if sequence else None
        if unseal(bundle['request'])['previous_ack'] != previous:
            raise ValueError('PREVIOUS_ACTUAL_ACK_REQUIRED')
        path = inside(relay, 'commands/'+f'{sequence:03d}.json')
        lock = OwnerLock(relay/'INSTALL_OWNER.lock')
        try:
            if path.exists() or inside(relay, 'intents/'+f'{sequence:03d}.json').exists():
                raise FileExistsError('NEVER_REPLAY_AN_INSTALL')
            write(path, bundle)
            return dict(status='BUNDLE_INSTALLED_NOT_PHYSICAL_ACK', command_sha256=bundle['command']['sha256'])
        finally:
            lock.close()

    def hopper(self, operation, episode, *, query=None, teacher=None, query_uuid=None):
        if self.role != 'hopper':
            raise PermissionError('HOPPER_ENDPOINT_REQUIRED')
        case = self.case(episode)
        directory = inside(self.root, case['hopper_session_relative'])
        if operation == 'response':
            uuid_string(query_uuid)
            path = inside(directory, 'outbox/'+query_uuid+'.json')
            return read(path) if path.is_file() else None
        if operation != 'query':
            raise PermissionError('REGISTERED_HOPPER_QUERY_REQUIRED')
        body = unseal(query)
        from scripts.pi0292.neural_relay import REQUEST_KEYS
        exact_keys(body, REQUEST_KEYS)
        for key in ('episode_uuid', 'query_uuid', 'reference_uuid'):
            uuid_string(body[key])
        if (body['run_uuid'] != self.root.name or body['episode_uuid'] != episode
                or body['identity'] != case['identity'] or body['protocol'] != PROTOCOL
                or body['kind'] != 'student_request' or not self.now() < body['expires'] <= self.now()+60):
            raise ValueError('EXACT_FRESH_REGISTERED_QUERY_REQUIRED')
        path = inside(directory, 'inbox/'+body['query_uuid']+'.json')
        lock = OwnerLock(directory/'ENDPOINT_OWNER.lock')
        try:
            if path.exists():
                raise FileExistsError('GENERATION_INTENT_ALREADY_RESERVED_NO_REQUERY')
            if case['identity']['profile'] != 'TEACHER_REFERENCE':
                if teacher is not None:
                    raise PermissionError('NO_TEACHER_INPUT_TO_SHARED_MODEL')
                # Only the already admitted resident can generate; this endpoint
                # never imports a model or opens a listener on the login host.
                write(path, query)
                return None
            from .resident import QueryReservations
            state_path = directory/'QUERY_STATE.json'
            if state_path.exists():
                reservations = QueryReservations.__new__(QueryReservations)
                reservations.directory, reservations.run, reservations.path = directory, self.root.name, state_path
                reservations.max_queries = 80
                reservations.state = read(state_path)
            else:
                reservations = QueryReservations(directory, self.root.name, max_queries=80)
            from scripts.pi0292.relay_protocol import validate_observation
            validate_observation(body['payload'])
            observation = sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid=self.root.name,
                episode_uuid=episode,sequence=body['sequence'],tick=body['tick'],payload=body['payload']))
            if (type(body['sequence']) is not int or not 0 <= body['sequence'] < 80
                    or type(body['tick']) is not int or body['tick'] != body['sequence']*200
                    or body['observation_sha256'] != observation['sha256']
                    or body['anchor_sha256'] != digest(body['payload']['state_eef10'])
                    or (body['sequence'] == 0 and body['previous_ack'] is not None)):
                raise ValueError('ACTUAL_TEACHER_QUERY_OBSERVATION_BINDING_REQUIRED')
            reservations.begin(body, self.now())
            write(path, query)
            action = teacher_action(query, teacher)
            prefix = unseal(action)['payload']['trajectory_eef10'][:8]
            command = sealed(dict(protocol=PROTOCOL,kind='install_prefix8',run_uuid=self.root.name,
                episode_uuid=episode,sequence=body['sequence'],tick=body['tick'],
                **{k:body[k] for k in ('query_uuid','reference_uuid','observation_sha256','anchor_sha256')},
                prefix_eef10=prefix,prefix_sha256=digest(prefix),advance_integrations=200,
                frame=body['payload']['frame'],tcp=body['payload']['tcp'],expires=body['expires']))
            accept_prefix(command, action, observation, request=query, identity=case['identity'], now=self.now())
            reservations.complete(body,action['sha256'],prefix_sha256=digest(prefix),command_sha256=command['sha256'])
            write(inside(directory, 'outbox/'+body['query_uuid']+'.json'), action)
            return action
        finally:
            lock.close()
