"""V5 paused-query lifecycle candidate, exclusively nonphysical.

Reuse the original full-H40 compiler, TTL/renewal and mock receiver. There is
deliberately no simulator, ROS, actuator, hook installer or native resume API.
An independent READMISSION plus a separately bound native adapter is still
required. A Python STOP poll is not a measured native stopping latency.
"""
import copy
import math
import time

from .hybrid_nonphysical_v3 import Receiver, check_live, identity, read, native_s
from .storage import digest

PROFILE = {
    'schema': 'MFREE_SIM_PAUSED_NATIVE_CONTRACT_V5_CANDIDATE',
    'profile': 'SIM_PAUSED_NATIVE',
    'receiver': 'MOCK_NO_INTEGRATION',
    'architecture': 'SHIFT7_SINGLE',
    'mode': 'M_FREE',
    'physics_dt_s': .002,
    'atom_dt_s': .05,
    'H40_points': 40,
    'P10_points': 10,
    'query_period_native_s': .5,
    'reference_ttl_native_s': .4,
    'renew_before_atom': 7,
    'external_STOP_clock': 'MONOTONIC_WALL',
    'inference_clock': 'MONOTONIC_WALL_FINITE_SERVICE_TIMEOUT',
    'native_clock_during_query': 'FROZEN_AT_OBSERVATION_QUERY_BOUNDARY',
    'native_execution_admitted': False,
    'strict_realtime_qualified': False,
}
PROFILE_SHA256 = identity(PROFILE)
SELECTED_SHIFT7_MANIFEST_SHA256 = 'ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb'


class PausedMockSession:
    """One nonintegrating owner, one episode, no recovery by replay.

    Observed native ticks are supplied by the mock caller. Their consistency
    can be tested here; this cannot prove a real simulator is paused. The
    recorded initial reference is not relabeled as a new native emission.
    """
    def __init__(self, root, episode, native_step, seed, seed_evidence, bindings,
                 *, service_timeout_s, operator_stop=None, now=None, sleep=None):
        recorded_seed = (
            seed_evidence.get('kind') == 'RECORDED_NATIVE_COMMAND'
            and len(seed_evidence.get('sha256', '')) == 64
        ) or (
            seed_evidence.get('kind') == 'RECORDED_NATIVE_COMMAND_NOT_SLA_FEEDBACK'
            and len(seed_evidence.get('bank', {}).get('sha256', '')) == 64
            and type(seed_evidence.get('index')) is int
        )
        if (not math.isfinite(service_timeout_s) or service_timeout_s <= 0
                or not episode or not recorded_seed
                or bindings.get('architecture') != 'SHIFT7_SINGLE'
                or bindings.get('SHIFT7_package_manifest_sha256') != SELECTED_SHIFT7_MANIFEST_SHA256):
            raise ValueError('EXPLICIT_FINITE_SERVICE_AND_RECORDED_SINGLE_MODEL_BINDINGS_REQUIRED')
        self.now = time.monotonic if now is None else now
        self.sleep = time.sleep if sleep is None else sleep
        self.service_timeout_s = float(service_timeout_s)
        self.receiver = Receiver(root, episode, native_step, copy.deepcopy(seed),
            copy.deepcopy(seed_evidence), dict(copy.deepcopy(bindings),
                paused_contract_sha256=PROFILE_SHA256,
                paused_service_timeout_s=self.service_timeout_s), operator_stop=operator_stop)
        self.request = self.response = self.ack = self.token = None
        self._deadline = None
        self._phase = 'QUERY_BOUNDARY'
        self._event('CREATED')

    def _event(self, event):
        record = self.receiver.state.setdefault('sim_paused_v5', {})
        record.update(profile_sha256=PROFILE_SHA256, profile=PROFILE['profile'],
                      phase=self._phase, last_event=event, wall_monotonic=self.now(),
                      native_execution_admitted=False, real_integrations=0)
        self.receiver.save()

    def _abort(self, reason):
        # Keep immutable install/emission receipts and the real partial count.
        # Invalidate executable history; an old seed/response cannot be reused.
        self.receiver.cancel(reason)
        self.receiver.state.update(previous=None, previous_evidence=None)
        self.receiver._compiled = None
        self.token = None
        self._phase = 'TERMINAL'
        self._event(reason)

    def _guard(self, *, native_step, paused):
        try:
            self.receiver.require_running()
            wall = self.now()
            if (type(native_step) is not int
                    or native_step != self.receiver.state['native_step']
                    or not math.isfinite(wall)):
                raise RuntimeError('OBSERVED_NATIVE_CLOCK_OR_WALL_CLOCK_CHANGED')
            if paused:
                if (self.request is None or native_step != self.request['native_step']
                        or wall < self._available_wall or wall >= self._deadline):
                    raise RuntimeError('PAUSE_CLOCK_OR_FINITE_WALL_SERVICE_TIMEOUT')
                check_live(self.request, self.receiver.state, now=self.now)
            return wall
        except (ValueError, RuntimeError, TimeoutError):
            self._abort('STOP_CANCEL_CLOCK_OR_SERVICE_FAILURE')
            raise

    def begin_query(self, input_pin, language, noise_id, *, native_step,
                    observation_native_step, observation_available_monotonic):
        if self._phase != 'QUERY_BOUNDARY':
            raise RuntimeError('NEW_QUERY_REQUIRES_FINISHED_P10_OR_NEW_EPISODE')
        wall = self._guard(native_step=native_step, paused=False)
        available = float(observation_available_monotonic)
        if (type(observation_native_step) is not int
                or observation_native_step != native_step
                or not math.isfinite(available) or available > wall
                or wall >= available + self.service_timeout_s):
            self._abort('INVALID_OR_EXPIRED_OBSERVATION')
            raise ValueError('QUERY_BOUNDARY_OBSERVATION_AND_ORIGINAL_AVAILABLE_TIME_REQUIRED')
        self._available_wall = available
        self._deadline = available + self.service_timeout_s
        # Keep original observation availability separately; never move it to
        # request creation or install ACK. Reference TTL remains native-clock.
        try:
            self.request = self.receiver.request(copy.deepcopy(input_pin), language, noise_id,
                timeout_s=self._deadline-wall, clock_mode='SIM_PAUSED_ONLY')
        except Exception:
            self._abort('OBSERVATION_OR_REQUEST_VALIDATION_FAILED')
            raise
        self.response = self.ack = self.token = None
        self._phase = 'PAUSED_WAIT'
        self.receiver.state['sim_paused_v5'].update(
            observation_available_monotonic=available,
            observation_native_step=observation_native_step,
            request_created_monotonic=self.request['created_monotonic'],
            finite_service_deadline_monotonic=self._deadline)
        self._event('QUERY_NATIVE_CLOCK_FROZEN')
        self._guard(native_step=native_step, paused=True)
        return copy.deepcopy(self.request)

    def wait_response(self, response_path, *, observed_native_step, alive):
        """External wall STOP is polled even when no model reply is present."""
        if self._phase != 'PAUSED_WAIT':
            raise RuntimeError('NOT_WAITING_FOR_RESPONSE')
        while True:
            self._guard(native_step=observed_native_step(), paused=True)
            try:
                response = read(response_path)
            except FileNotFoundError:
                if not alive():
                    self._abort('WRITER_TERMINAL_NO_RESPONSE')
                    raise RuntimeError('WRITER_TERMINAL_NO_RESPONSE')
                self.sleep(.005)
                continue
            except (ValueError, OSError):
                self._abort('INVALID_RESPONSE_NO_REPLAY')
                raise
            self._guard(native_step=observed_native_step(), paused=True)
            return response

    def install(self, response, *, native_step):
        if self._phase != 'PAUSED_WAIT':
            raise RuntimeError('NOT_WAITING_FOR_INSTALL')
        self._guard(native_step=native_step, paused=True)
        try:
            ack = self.receiver.install(self.request, response)
            self._guard(native_step=native_step, paused=True)
            self.response = copy.deepcopy(response)
            self.ack = copy.deepcopy(ack)
            self._phase = 'PAUSED_ACKED'
            self._event('ORIGINAL_FULL_H40_MOCK_INSTALL_ACK')
            return copy.deepcopy(ack)
        except Exception:
            self._abort('INSTALL_FAILED_NO_RESUME')
            raise

    def _verify_ack_and_hashes(self, ack):
        installed = self.receiver.state['installed']
        if (not installed or ack != self.ack or ack != installed['ack']
                or ack['request_id'] != self.request['request_id']
                or not ack['full_H40_compiled']
                or ack['receiver_kind'] != 'MOCK_NO_INTEGRATION'
                or ack['native_execution_admitted']
                or digest(self.response['action']['path']) != ack['action_sha256']
                or digest(installed['compiled']['path']) != installed['compiled']['sha256']):
            raise RuntimeError('ACK_OR_ACTION_OR_COMPILED_HASH_CHANGED')
        document = read(self.receiver.root / ('install-%04d.json' % self.request['sequence']))
        bundle = document['bundle']
        if (document['ack'] != ack or bundle['request_id'] != ack['request_id']
                or identity({k:v for k,v in bundle.items() if k != 'bundle_sha256'})
                    != ack['bundle_sha256']
                or 'base' in bundle['eef_wire']
                or bundle['mechanical_H40_pin'] != installed['compiled']):
            raise RuntimeError('ORIGINAL_INSTALL_BUNDLE_CHANGED_OR_BASE_ORACLE')

    def prepare_mock_resume(self, ack, *, native_step):
        """Return only a mock token, never a native actuator/resume permit."""
        if self._phase != 'PAUSED_ACKED':
            raise RuntimeError('RESUME_REQUIRES_FULL_H40_AND_ACK')
        self._guard(native_step=native_step, paused=True)
        try:
            self._verify_ack_and_hashes(ack)
            self._guard(native_step=native_step, paused=True)
            token = dict(request_id=self.request['request_id'],
                bundle_sha256=ack['bundle_sha256'], native_step=native_step,
                generation=self.receiver.state['generation'],
                profile_sha256=PROFILE_SHA256, receiver_kind='MOCK_NO_INTEGRATION',
                native_resume_authorized=False)
            token['token_id'] = identity(token)
            self.token = token
            self._phase = 'MOCK_RESUME_PREPARED'
            self._event('PRE_RESUME_STOP_CLOCK_AND_HASH_RECHECKED')
            return copy.deepcopy(token)
        except Exception:
            self._abort('PRE_RESUME_FAILURE')
            raise

    def resume_mock(self, token, *, native_step):
        if self._phase != 'MOCK_RESUME_PREPARED':
            raise RuntimeError('MOCK_RESUME_TOKEN_ALREADY_USED_OR_MISSING')
        self._guard(native_step=native_step, paused=True)
        try:
            if token != self.token:
                raise RuntimeError('MOCK_RESUME_IDENTITY_CHANGED')
            self._verify_ack_and_hashes(self.ack)
            self._guard(native_step=native_step, paused=True)
            self.token = None
            self._phase = 'MOCK_CONSUMING'
            self._event('MOCK_ONLY_NO_PHYSICS_RESUME')
        except Exception:
            self._abort('RESUME_FAILURE')
            raise

    def consume_mock(self, count, *, native_step):
        if self._phase != 'MOCK_CONSUMING' or type(count) is not int or count <= 0:
            raise RuntimeError('POSITIVE_MOCK_ATOMS_AFTER_RESUME_ONLY')
        self._guard(native_step=native_step, paused=False)
        if self.receiver.state['emitted_atoms'] + count > 10:
            raise ValueError('NO_MORE_THAN_P10_NO_UNEXECUTED_SUFFIX_HISTORY')
        try:
            for _ in range(count):
                self._verify_ack_and_hashes(self.ack)
                self._guard(native_step=self.receiver.state['native_step'], paused=False)
                result = self.receiver.consume(1)
            self._event('PARTIAL_OR_FULL_MOCK_ACK_ONLY')
            return result
        except Exception:
            self._abort('MOCK_CONSUMPTION_FAILED')
            raise

    def finish(self):
        if self._phase != 'MOCK_CONSUMING':
            raise RuntimeError('NO_MOCK_P10_TO_FINISH')
        self._guard(native_step=self.receiver.state['native_step'], paused=False)
        self.receiver.finish()  # Partial P10 explicitly cannot finish.
        self._phase = 'QUERY_BOUNDARY'
        self._event('FULL_MOCK_P10_FINISHED')

    def cancel(self):
        self._abort('OPERATOR_CANCEL_HISTORY_INVALIDATED')

    def restart(self):
        self.receiver.restart()
        self._abort('COLD_RESTART_OLD_HISTORY_AND_RESPONSES_INVALIDATED')

    def stop(self):
        self.receiver.stop()
        self._abort('WALL_STOP_LATCHED_HISTORY_INVALIDATED')
