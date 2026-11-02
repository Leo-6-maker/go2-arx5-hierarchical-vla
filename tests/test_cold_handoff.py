import signal

import pytest

from scripts.pi0294.night10 import cold_handoff as c


def fake_process(monkeypatch, *, children=(), start='same'):
    signals=[];events=[]
    monkeypatch.setattr(c, 'process_identity', lambda _:dict(proc_start=start,
        state='T' if signal.SIGSTOP in signals else 'S', children=list(children)))
    monkeypatch.setattr(c.os, 'pidfd_open', lambda _:42)
    monkeypatch.setattr(c.os, 'close', lambda _:None)
    monkeypatch.setattr(c.signal, 'pidfd_send_signal', lambda fd,s:signals.append(s))
    monkeypatch.setattr(c.select, 'select', lambda *a:([42],[],[]))
    return signals,events


def test_only_exact_frozen_childless_cold_manager_can_end(monkeypatch):
    signals,events=fake_process(monkeypatch)
    assert c.stop_idle_manager(dict(PID=10,proc_start='same'),lambda:True,events.append)
    assert signals==[signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT]
    assert events[0]['verified_cold_wait'] and events[0]['child_processes']==0


@pytest.mark.parametrize('children,verified', [(['11'],True),([],False)])
def test_active_helper_or_lost_boundary_resumes_without_termination(monkeypatch,children,verified):
    signals,events=fake_process(monkeypatch,children=children)
    assert not c.stop_idle_manager(dict(PID=10,proc_start='same'),lambda:verified,events.append)
    assert signals==[signal.SIGSTOP,signal.SIGCONT] and not events


def test_unknown_cold_evidence_resumes_and_propagates(monkeypatch):
    signals,events=fake_process(monkeypatch)
    def fail():raise PermissionError('Unknown native effect')
    with pytest.raises(PermissionError):c.stop_idle_manager(dict(PID=10,proc_start='same'),fail,events.append)
    assert signals==[signal.SIGSTOP,signal.SIGCONT]


def test_reused_pid_cannot_receive_signal(monkeypatch):
    signals,events=fake_process(monkeypatch,start='different')
    with pytest.raises(PermissionError):c.stop_idle_manager(dict(PID=10,proc_start='same'),lambda:True,events.append)
    assert not signals and not events


def test_gpu1_unavailable_does_not_block_ready_gpu0():
    assert c.inference_choice(True,False)==0
    assert c.inference_choice(True,True)==1
    assert c.inference_choice(False,True)==1
    assert c.inference_choice(False,False)is None
