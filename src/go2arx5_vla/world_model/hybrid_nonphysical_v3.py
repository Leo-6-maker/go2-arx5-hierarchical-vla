"""V3 request/receiver seam. File IPC and mock emission; no physics/actuator API.

The receiver is the sole state writer. The GPU process only reads live.json.
SIM_PAUSED_ONLY freezes native time, not measured wall latency. It grants no
continuous-time, native WBC, collision, or safety readmission qualification.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.gripper_time_v1 import reference_width
from .storage import digest
from .target_reference import require_install_clock, require_live_window

SCHEMA = 'MFREE_HYBRID_NONPHYSICAL_REQUEST_V3_1'


def native_s(step):
    # Match the existing ROS integer-nanosecond source domain.
    return round(step * .002 * 1e9) / 1e9


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def wait_receipt(path, *, alive, stopped, timeout_s):
    """Read the atomic publication directly; exists() is not a read receipt.

    Only absence is retried. Invalid bytes/identities and terminal writers fail.
    This performs no write, model call, installation, or effect retry.
    """
    until = time.monotonic()+timeout_s
    while True:
        try:
            return read(path)
        except FileNotFoundError:
            if not alive():
                raise RuntimeError('WRITER_TERMINAL_WITHOUT_READABLE_RECEIPT')
            if stopped() or time.monotonic() >= until:
                raise TimeoutError('FINITE_RECEIPT_WAIT_STOP_OR_TIMEOUT')
            time.sleep(.005)


def load_input(request):
    pin = request['input']
    if digest(pin['path']) != pin['sha256']:
        raise ValueError('UNIFIED_INPUT_BYTES_CHANGED')
    with np.load(pin['path'], allow_pickle=False) as z:
        values = {k: z[k].copy() for k in ('state_eef10', 'body_rgb', 'wrist_rgb',
                  'T_policy_from_mount', 'policy_from_world', 'accepted_reference_TCP')}
    for key in ('body_rgb', 'wrist_rgb'):
        a = values[key]
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError('ORDINARY_UINT8_BODY_AND_WRIST_RGB_REQUIRED')
    if values['state_eef10'].shape != (10,) or not all(np.isfinite(a).all() for a in values.values()):
        raise ValueError('FINITE_CAPTURED_QUERY_REQUIRED')
    return values


def check_live(request, state, *, now=None):
    wall = (time.monotonic if now is None else now)()
    body = {k: v for k, v in request.items() if k != 'request_id'}
    if (identity(body) != request['request_id'] or request['schema'] != SCHEMA
            or (state.get('operator_stop') and Path(state['operator_stop']).exists())
            or state['stop'] or state['cancelled'] or state['pending'] != request['request_id']
            or state['episode'] != request['episode'] or state['generation'] != request['generation']
            or state['native_step'] != request['native_step']
            or state['sequence'] != request['sequence'] or state['bindings'] != request['bindings']
            or state['previous'] != request['previous_reference']
            or not math.isfinite(wall) or wall >= request['service_deadline_monotonic']):
        raise RuntimeError('STOP_CANCEL_CLOCK_IDENTITY_OR_SERVICE_TIMEOUT')
    if request['clock_mode'] == 'WALL_CLOCK_STRICT':
        require_wall_window(request['created_monotonic'], wall, 0.)
    reference_width(request['previous_reference'])
    require_install_clock(request['source_native_s'], native_s(state['native_step']))


def require_wall_window(source, wall, duration):
    if (not all(math.isfinite(v) for v in (source, wall, duration))
            or wall < source or duration < 0 or wall+duration >= source+.4-1e-10):
        raise TimeoutError('WALL_CLOCK_TTL_EXPIRED_NO_INSTALL_OR_EMISSION')


class Receiver:
    """Single-writer, nonintegrating native-format receiver substitute.

    seed is a proven recorded command, not a newly emitted native command.
    Subsequent previous references advance only after the atomic mock ACK.
    """
    def __init__(self, root, episode, native_step, seed, seed_evidence, bindings, operator_stop=None):
        self.root = Path(root)
        self.root.mkdir()
        if type(native_step) is not int or abs(native_s(native_step) - seed['query_native_s']) > 1e-8:
            raise ValueError('CAPTURED_NATIVE_STEP_AND_SEED_CLOCK')
        reference_width(seed)
        self.state = dict(episode=episode, generation=0, sequence=0, native_step=native_step,
            previous=seed, previous_evidence=seed_evidence, bindings=bindings,
            operator_stop=None if operator_stop is None else str(operator_stop),
            stop=False, cancelled=False, pending=None, installed=None, emitted_atoms=0)
        self.save()

    def save(self):
        write(self.root / 'live.json', self.state)

    def require_running(self):
        flag = self.state.get('operator_stop')
        if flag and Path(flag).exists() and not self.state['stop']:
            self.stop()  # Removing the external flag later cannot un-latch STOP.
        if self.state['stop'] or self.state['cancelled']:
            raise RuntimeError('STOP_OR_CANCEL_LATCHED_NO_LIVE_COMMAND')

    def request(self, input_pin, language, noise_id, *, timeout_s=60., clock_mode='SIM_PAUSED_ONLY'):
        self.require_running()
        s = self.state
        if s['stop'] or s['cancelled'] or s['pending'] is not None or s['installed'] is not None:
            raise RuntimeError('RESET_OR_FINISH_PREVIOUS_REQUEST_FIRST')
        if clock_mode not in ('SIM_PAUSED_ONLY', 'WALL_CLOCK_STRICT'):
            raise ValueError('EXPLICIT_CLOCK_MODE_REQUIRED')
        reference_width(s['previous'])
        s['sequence'] += 1
        s['emitted_atoms'] = 0
        start = time.monotonic()
        request = dict(schema=SCHEMA, episode=s['episode'], generation=s['generation'],
            sequence=s['sequence'], native_step=s['native_step'], source_native_s=native_s(s['native_step']),
            input=input_pin, language=language, noise_id=noise_id, previous_reference=s['previous'],
            previous_evidence=s['previous_evidence'], bindings=s['bindings'],
            created_monotonic=start, service_deadline_monotonic=start+timeout_s,
            clock_mode=clock_mode, native_ttl_s=.4, renewal_target_s=.35,
            mode='M_FREE', independent_gripper_write=False)
        request['request_id'] = identity(request)
        s['pending'] = request['request_id']; self.save()
        load_input(request)
        return request

    def install(self, request, response):
        self.require_running()
        s = self.state
        check_live(request, s)
        if response['request_id'] != request['request_id'] or response['bindings'] != request['bindings']:
            raise ValueError('ATOMIC_RESPONSE_IDENTITY_MISMATCH')
        if s['installed'] is not None:
            if s['emitted_atoms'] == 0 and s['installed']['action'] == response['action']:
                return s['installed']['ack']  # idempotent ACK, never a second installation
            raise RuntimeError('LATE_RESPONSE_AFTER_CONSUMPTION')
        data = load_input(request)
        from scripts.pi0292.eval_contract import matrix10
        from .sla_dense_policy_v5 import compile_proposal
        frame = data['policy_from_world']
        actual = np.linalg.inv(frame) @ matrix10(data['state_eef10'])
        program, evidence = compile_proposal(response['action']['path'], response['action']['sha256'],
            source_from_target=frame, query_state10=data['state_eef10'], actual_TCP=actual,
            consumed_reference_TCP=data['accepted_reference_TCP'], base_hold=np.r_[np.zeros(6), 1.])
        check_live(request, read(self.root/'live.json'))  # compile/installation race boundary
        from scripts.pi0294.base_intent import eef_document
        path = self.root / ('compiled-%04d.npz' % s['sequence'])
        with path.open('xb') as f:
            np.savez(f, reference_H40=program.reference_path, nominal_H40=program.desired_path)
        wire = eef_document(program.preview(0))
        wire['_pi0294'] = dict(episode=s['episode'], sequence=s['sequence'],
            source_mpc_s=request['source_native_s'], ttl_s=.4, operation='install')
        if 'base' in wire:
            raise ValueError('M_FREE_MUST_HAVE_NO_BASE_REFERENCE')
        bundle = dict(request_id=request['request_id'], eef_wire=wire,
            mechanical_H40_pin=dict(path=str(path), sha256=digest(path)),
            unified_EEF_and_gripper=True, original_compiler=evidence,
            gripper_mapping_count=1, nominal_reference_mapping_count=1)
        bundle['bundle_sha256'] = identity(bundle)
        ack = dict(schema='NONPHYSICAL_ACCEPT_AND_INSTALL_ACK_V3_1', request_id=request['request_id'],
            bundle_sha256=bundle['bundle_sha256'], action_sha256=response['action']['sha256'],
            receiver_kind='MOCK_NO_INTEGRATION', full_H40_compiled=True, EEF_dimension=7,
            native_WBC_certified=False, native_execution_admitted=False)
        # One atomic publication contains the full path and its installation ACK.
        self.require_running()
        check_live(request, read(self.root/'live.json'))
        write(self.root / ('install-%04d.json' % s['sequence']), dict(bundle=bundle, ack=ack))
        s['installed'] = dict(action=response['action'], compiled=bundle['mechanical_H40_pin'],
            ack=ack, plan_source_native_s=request['source_native_s'], start_step=s['native_step'],
            clock_mode=request['clock_mode'], source_wall_monotonic=request['created_monotonic'])
        s['emitted_atoms'] = 0; self.save()
        return ack

    def consume(self, count, *, renew_at_seven=True):
        if type(count) is not int or count <= 0:
            raise ValueError('POSITIVE_ATOM_COUNT')
        for _ in range(count):
            self.require_running()
            s = self.state; installed = s['installed']; k = s['emitted_atoms']
            if s['stop'] or s['cancelled'] or installed is None or k >= 10:
                raise RuntimeError('NO_LIVE_ACKED_P10')
            if installed['clock_mode'] == 'WALL_CLOCK_STRICT':
                require_wall_window(installed['source_wall_monotonic'], time.monotonic(), 0.)
            if s['native_step'] != installed['start_step']+25*k:
                raise ValueError('NO_PARTIAL_OR_STALE_ATOM_ACK')
            pin = installed['compiled']
            if digest(pin['path']) != pin['sha256']:
                raise ValueError('COMPILED_REFERENCE_CHANGED')
            with np.load(pin['path'], allow_pickle=False) as z:
                path = z['reference_H40'].copy()
            if k == 7 and renew_at_seven:
                # Exact existing native consumer behavior: same H40 suffix,
                # fresh measured source. This substitutes a clock, not WBC ACK.
                renewal_native_source = native_s(s['native_step'])
                renewal_wall_source = time.monotonic()
                from scripts.pi0294.base_intent import eef_document
                suffix = path[np.minimum(np.arange(k, k+40), 39)]
                doc = eef_document(suffix)
                doc['_pi0294'] = dict(episode=s['episode'], sequence=s['sequence'],
                    source_mpc_s=renewal_native_source, ttl_s=.4, operation='install')
                self.require_running()
                write(self.root/('renew-%04d.json' % s['sequence']), dict(request_id=s['pending'],
                    exact_same_H40_suffix=True, eef_wire=doc, mock_ack=True))
                self.require_running()
                if installed['clock_mode'] == 'WALL_CLOCK_STRICT':
                    # A late acknowledgement cannot resurrect the old plan or
                    # move the source timestamp from acquisition to ACK time.
                    require_wall_window(installed['source_wall_monotonic'], time.monotonic(), 0.)
                installed['plan_source_native_s'] = renewal_native_source
                installed['source_wall_monotonic'] = renewal_wall_source
            require_live_window(installed['plan_source_native_s'], native_s(s['native_step']), 25)
            source = native_s(s['native_step'])
            emitted = dict(request_id=s['pending'], atom=k, source_native_s=source,
                end_native_step=s['native_step']+25, EEF10_mechanical=path[k].tolist(),
                mock_receiver_ACK=True, native_emission=False, real_integrations=0)
            self.require_running()
            if installed['clock_mode'] == 'WALL_CLOCK_STRICT':
                require_wall_window(installed['source_wall_monotonic'], time.monotonic(), .05)
            write(self.root/('emitted-%04d-%02d.json' % (s['sequence'], k)), emitted)
            # Only a receiver-confirmed emission advances command history.
            s['native_step'] += 25; s['emitted_atoms'] += 1
            s['previous'] = dict(width_m=.044*float(path[k,9]), source_native_s=source,
                                 query_native_s=native_s(s['native_step']))
            s['previous_evidence'] = dict(kind='MOCK_EMISSION_ACK', request_id=s['pending'], atom=k)
            reference_width(s['previous']); self.save()
        return dict(consumed_atoms=self.state['emitted_atoms'],
                    consumed_s=self.state['emitted_atoms']*.05, previous=self.state['previous'],
                    full_P10=self.state['emitted_atoms']==10, native_integrations=0)

    def finish(self):
        if self.state['emitted_atoms'] != 10:
            raise RuntimeError('PARTIAL_P10_MUST_NOT_BE_MARKED_COMPLETE')
        self.state.update(pending=None, installed=None); self.save()

    def cancel(self, reason='CANCEL'):
        self.state.update(cancelled=True, pending=None, installed=None)
        write(self.root/'CANCEL_ACK.json', dict(reason=reason, consumed_atoms=self.state['emitted_atoms'],
            full_P10=self.state['emitted_atoms']==10, receiver='MOCK_NO_PHYSICS'))
        self.save()

    def stop(self):
        self.state['stop'] = True; self.cancel('STOP_LATCHED')

    def restart(self):
        # Cold service generation invalidates every old request and response.
        self.state.update(generation=self.state['generation']+1, pending=None, installed=None,
                          cancelled=True)
        self.save()
