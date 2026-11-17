import copy
import json

import numpy as np
import pytest

from go2arx5_vla.world_model.sim_paused_contract_v5 import (
    PausedMockSession, PROFILE, PROFILE_SHA256, SELECTED_SHIFT7_MANIFEST_SHA256,
)
from go2arx5_vla.world_model.storage import digest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    state = np.array([.4, 0., .3, 1., 0., 0., 0., 1., 0., 1.])
    inp = tmp_path / 'observation.npz'
    np.savez(inp, state_eef10=state, body_rgb=np.zeros((16,16,3), np.uint8),
             wrist_rgb=np.ones((16,16,3), np.uint8), T_policy_from_mount=np.eye(4),
             policy_from_world=np.eye(4), accepted_reference_TCP=np.array(
                 [[1,0,0,.4],[0,1,0,0],[0,0,1,.3],[0,0,0,1]]))
    stop = tmp_path/'EXTERNAL_STOP'
    session = PausedMockSession(tmp_path/'receiver', 'episode-1', 525,
        dict(width_m=.042, source_native_s=1., query_native_s=1.05),
        dict(kind='RECORDED_NATIVE_COMMAND', sha256='a'*64),
        dict(architecture='SHIFT7_SINGLE', SHIFT7_package_manifest_sha256=SELECTED_SHIFT7_MANIFEST_SHA256),
        service_timeout_s=10., operator_stop=stop)
    q = session.begin_query(dict(path=str(inp), sha256=digest(inp)), 'pick', 0,
        native_step=525, observation_native_step=525, observation_available_monotonic=99.98)
    action = tmp_path/'action.npz'
    path = np.repeat(state[None], 40, axis=0)
    # Small monotonic gripper movement distinguishes emitted and future points.
    path[:,9] = 1. - .0005*np.arange(40)
    np.savez(action, physical_H40=path, normalized=np.zeros((1,40,32)))
    response = dict(request_id=q['request_id'], bindings=q['bindings'],
                    action=dict(path=str(action), sha256=digest(action)))
    return session, q, response, clock, stop, path


def resumed(f):
    session, q, response, clock, stop, path = f
    ack = session.install(response, native_step=525)
    token = session.prepare_mock_resume(ack, native_step=525)
    assert token['native_resume_authorized'] is False
    session.resume_mock(token, native_step=525)
    return session


def test_pause_is_native_clock_not_a_retimed_wall_TTL(setup):
    s, q, response, clock, _, _ = setup
    clock[0] += 2.  # >400ms of wall inference; explicitly distinct profile.
    ack = s.install(response, native_step=525)
    token = s.prepare_mock_resume(ack, native_step=525)
    assert q['source_native_s'] == 1.05
    assert q['native_ttl_s'] == .4
    assert q['created_monotonic'] == 100.
    assert s.receiver.state['sim_paused_v5']['observation_available_monotonic'] == 99.98
    s.resume_mock(token, native_step=525)
    result = s.consume_mock(10, native_step=525)
    assert result['full_P10'] and result['native_integrations'] == 0
    renewal = json.loads((s.receiver.root/'renew-0001.json').read_text())
    assert renewal['exact_same_H40_suffix']
    assert renewal['eef_wire']['_pi0294']['source_mpc_s'] == pytest.approx(1.4)
    assert PROFILE['native_execution_admitted'] is False
    assert PROFILE_SHA256 == q['bindings']['paused_contract_sha256']


def test_partial_history_uses_mock_emissions_and_cannot_finish(setup):
    s = resumed(setup)
    initial = copy.deepcopy(s.receiver.state['previous'])
    result = s.consume_mock(3, native_step=525)
    assert result['consumed_atoms'] == 3 and not result['full_P10']
    assert result['previous']['width_m'] != initial['width_m']
    assert result['previous']['source_native_s'] == pytest.approx(1.15)
    assert result['previous']['query_native_s'] == pytest.approx(1.20)
    assert s.receiver.state['previous_evidence']['atom'] == 2
    with pytest.raises(RuntimeError, match='PARTIAL'): s.finish()
    assert len(list(s.receiver.root.glob('emitted-*'))) == 3
    s.cancel()
    assert s.receiver.state['previous'] is None
    assert s.receiver.state['previous_evidence'] is None
    assert s.receiver.state['emitted_atoms'] == 3  # Never changed into ten.


def test_next_query_uses_last_emitted_not_future_suffix(setup):
    s, q, _, _, _, path = setup
    resumed(setup).consume_mock(10, native_step=525)
    s.finish()
    q2 = s.begin_query(q['input'], 'pick', 0, native_step=775,
        observation_native_step=775, observation_available_monotonic=100.)
    assert q2['previous_evidence']['atom'] == 9
    # Use the compiler's actual emitted mechanical command, not a second
    # independent implementation of its nominal-to-mechanical mapping.
    emitted = json.loads((s.receiver.root/'emitted-0001-09.json').read_text())
    assert q2['previous_reference']['width_m'] == pytest.approx(.044*emitted['EEF10_mechanical'][9])
    assert q2['previous_reference']['width_m'] != pytest.approx(.002+.040*path[39,9])
    assert q2['source_native_s'] == 1.55


@pytest.mark.parametrize('fault', ['stop', 'cancel', 'restart', 'clock', 'timeout', 'response_id', 'binding'])
def test_fault_during_inference_never_installs_or_resumes(setup, fault):
    s, q, response, clock, stop, _ = setup
    step = 525
    if fault == 'stop': stop.write_text('STOP')
    elif fault == 'cancel': s.cancel()
    elif fault == 'restart': s.restart()
    elif fault == 'clock': step += 1
    elif fault == 'timeout': clock[0] = 109.98
    elif fault == 'response_id': response['request_id'] = 'wrong'
    else: response['bindings'] = {'wrong': True}
    with pytest.raises((ValueError, RuntimeError)):
        s.install(response, native_step=step)
    assert not list(s.receiver.root.glob('install-*'))
    assert not list(s.receiver.root.glob('emitted-*'))
    assert s.receiver.state['emitted_atoms'] == 0


def test_STOP_polled_during_model_wait_without_a_reply(setup):
    s, _, _, clock, stop, _ = setup
    def sleep(dt):
        clock[0] += dt
        stop.write_text('STOP')
    s.sleep = sleep
    with pytest.raises(RuntimeError, match='STOP'):
        s.wait_response(s.receiver.root/'NO_RESPONSE', observed_native_step=lambda:525, alive=lambda:True)
    stop.unlink()
    assert s.receiver.state['stop']  # Removing flag cannot resurrect it.
    assert s.receiver.state['previous'] is None
    assert s.receiver.state['emitted_atoms'] == 0


def test_wait_timeout_includes_age_before_request_creation(setup):
    s, _, _, clock, _, _ = setup
    clock[0] = 109.979
    s.sleep = lambda dt: clock.__setitem__(0, clock[0] + dt)
    with pytest.raises(RuntimeError, match='TIMEOUT'):
        s.wait_response(s.receiver.root/'NO_RESPONSE', observed_native_step=lambda:525, alive=lambda:True)
    assert s.receiver.state['previous'] is None


@pytest.mark.parametrize('fault', ['stop', 'clock', 'timeout', 'action', 'compiled', 'ack', 'wire'])
def test_pre_resume_rechecks_all_live_evidence(setup, fault):
    s, q, response, clock, stop, _ = setup
    ack = s.install(response, native_step=525)
    token = s.prepare_mock_resume(ack, native_step=525)
    step = 525
    if fault == 'stop': stop.write_text('STOP')
    elif fault == 'clock': step += 25
    elif fault == 'timeout': clock[0] = 110.
    elif fault == 'action':
        with open(response['action']['path'], 'ab') as f: f.write(b'changed')
    elif fault == 'compiled':
        with open(s.receiver.state['installed']['compiled']['path'], 'ab') as f: f.write(b'changed')
    elif fault == 'ack': s.receiver.state['installed']['ack']['action_sha256'] = 'x'*64
    else:
        p = s.receiver.root/'install-0001.json'
        data = json.loads(p.read_text()); data['bundle']['eef_wire']['base'] = {'oracle': True}
        p.write_text(json.dumps(data))
    with pytest.raises((ValueError, RuntimeError)):
        s.resume_mock(token, native_step=step)
    assert s.receiver.state['emitted_atoms'] == 0
    assert s.receiver.state['previous'] is None


def test_illegal_unconsumed_H40_suffix_still_rejects(setup):
    s, q, response, _, _, path = setup
    path[30,0] += .1
    np.savez(response['action']['path'], physical_H40=path, normalized=np.zeros((1,40,32)))
    response['action']['sha256'] = digest(response['action']['path'])
    with pytest.raises(ValueError, match='FULL_PROPOSAL_REJECTED'):
        s.install(response, native_step=525)
    assert s.receiver.state['previous'] is None
    assert not list(s.receiver.root.glob('install-*'))


def test_STOP_between_partial_atoms_preserves_true_count(setup):
    s, _, _, _, stop, _ = setup
    resumed(setup).consume_mock(4, native_step=525)
    stop.write_text('STOP')
    with pytest.raises(RuntimeError): s.consume_mock(6, native_step=625)
    assert s.receiver.state['emitted_atoms'] == 4
    assert s.receiver.state['previous'] is None
    assert len(list(s.receiver.root.glob('emitted-*'))) == 4


def test_resume_token_one_use_and_episode_bound(setup):
    s, _, response, _, _, _ = setup
    ack = s.install(response, native_step=525)
    token = s.prepare_mock_resume(ack, native_step=525)
    s.resume_mock(token, native_step=525)
    with pytest.raises(RuntimeError): s.resume_mock(token, native_step=525)
    s.restart()
    assert s.receiver.state['generation'] == 1
    assert s.receiver.state['previous'] is None
    with pytest.raises(RuntimeError): s.install(response, native_step=525)


def test_compilation_time_is_inside_wall_service_deadline(setup, monkeypatch):
    from go2arx5_vla.world_model import sla_dense_policy_v5 as compiler
    s, _, response, clock, _, _ = setup
    original = compiler.compile_proposal
    def delayed(*args, **kwargs):
        result = original(*args, **kwargs); clock[0] = 110.; return result
    monkeypatch.setattr(compiler, 'compile_proposal', delayed)
    with pytest.raises(RuntimeError, match='TIMEOUT'):
        s.install(response, native_step=525)
    assert s.receiver.state['previous'] is None
    assert s.receiver.state['installed'] is None
    assert s.receiver.state['emitted_atoms'] == 0
