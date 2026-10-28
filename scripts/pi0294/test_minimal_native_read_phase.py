"""Cold qstat-read failure contracts, not native GPU/model qualification."""
import json,subprocess
from types import SimpleNamespace
import pytest
from scripts.pi0294.generalization import minimal_session_gate as g

JOB='CPU_NATIVE_JOB_FIXTURE'
COMMAND=['/opt/pbs/bin/qstat','-f','-F','json',JOB]


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def sleep(self,seconds):self.now+=seconds


def run(tmp_path,function,clock,*,guard=lambda:None,until=180.,failures=None):
    return g._within_phase_reads(function,guard=guard,until=until,clock=clock,sleep=clock.sleep,
        job_id=JOB,observations=tmp_path/'observations',failures=[]if failures is None else failures)


@pytest.mark.parametrize('failure',[subprocess.CalledProcessError(255,COMMAND,output='',stderr='native read unavailable'),
                                  subprocess.TimeoutExpired(COMMAND,25,output=b'',stderr=b'read timeout')])
def test_real_error_types_do_not_grant_permission_and_later_read_must_succeed(tmp_path,failure):
    clock=Clock();calls=[];guard=[]
    def observe():
        calls.append(clock())
        if len(calls)==1:raise failure
        return {'actual_fresh_read':True}
    assert run(tmp_path,observe,clock,guard=lambda:guard.append(clock()))=={'actual_fresh_read':True}
    assert calls==[0.,2.]and len(guard)==2
    saved=json.loads((tmp_path/'observations/READ_FAILURE_01.json').read_text())
    assert saved['model_or_update_permission']is False and saved['error_type']==type(failure).__name__
    assert saved['returncode']==(255 if isinstance(failure,subprocess.CalledProcessError)else None)


def test_all_reads_share_original_deadline_and_late_success_is_refused(tmp_path):
    clock=Clock();clock.now=150.;failures=[]
    def first():
        clock.now+=25.;raise subprocess.TimeoutExpired(COMMAND,25)
    with pytest.raises(TimeoutError,match='ORIGINAL180S'):
        run(tmp_path,first,clock,failures=failures)
    assert clock.now>=180 and len(failures)<=3
    clock.now=179.
    def late():clock.now=181.;return 'TOO_LATE'
    with pytest.raises(TimeoutError,match='ORIGINAL180S'):run(tmp_path,late,clock)


def test_third_failed_read_stops_without_resetting_phase_or_replaying_effects(tmp_path):
    clock=Clock();calls=[];failures=[]
    def failed():
        calls.append(clock());raise subprocess.CalledProcessError(255,COMMAND)
    with pytest.raises(subprocess.CalledProcessError):run(tmp_path,failed,clock,failures=failures)
    assert calls==[0.,2.,4.]and len(failures)==3
    assert len(list((tmp_path/'observations').glob('*.json')))==3


@pytest.mark.parametrize('error',[PermissionError('STOP_OR_WRONG_IDENTITY'),ValueError('BAD_NATIVE_JSON'),
    subprocess.CalledProcessError(1,['NOT_QSTAT']),subprocess.TimeoutExpired(COMMAND[:-1]+['OTHER_JOB'],25)])
def test_nonread_or_other_identity_errors_propagate_without_retry(tmp_path,error):
    clock=Clock();calls=[]
    def invalid():calls.append(1);raise error
    with pytest.raises(type(error)):run(tmp_path,invalid,clock)
    assert calls==[1]and not(tmp_path/'observations').exists()


def test_guard_checks_stop_before_every_read_and_ready_and_verify_share_failures(tmp_path):
    clock=Clock();calls=[];failures=[]
    def read():calls.append(1);raise subprocess.CalledProcessError(255,COMMAND)
    def guard():
        if calls:raise PermissionError('STOP')
    with pytest.raises(PermissionError,match='STOP'):run(tmp_path,read,clock,guard=guard,failures=failures)
    assert len(calls)==len(failures)==1
    # Simulate a later verification after two earlier native-read failures.
    failures.append(dict(previous='saved'))
    with pytest.raises(subprocess.CalledProcessError):run(tmp_path,read,clock,failures=failures)
    assert len(calls)==2 and len(failures)==3


def test_full_wait_refuses_release_which_becomes_stale_during_native_read(tmp_path,monkeypatch):
    from scripts.pi0294.generalization import minimal_native_context as context
    clock=Clock();slot=tmp_path/'slot';slot.mkdir();(slot/'READY.json').write_text('{}');(slot/'RELEASE.json').write_text('{"observed_epoch": 100.0}')
    effects=tmp_path/'effects.json';effects.write_text('{}')
    ledger=SimpleNamespace(request={},request_path=tmp_path/'request',path=effects,guard=lambda:{})
    checks=[]
    def fresh(value,body,request,*,job_id,now_epoch):
        checks.append(now_epoch)
        if now_epoch-value['observed_epoch']>120:raise PermissionError('STALE_RELEASE_AFTER_NATIVE_READ')
        return value
    def native(*args,**kwargs):clock.now+=130.;return True
    monkeypatch.setattr(g,'publish_ready',lambda *args:slot)
    monkeypatch.setattr(g,'check_ready',lambda *args,**kwargs:None)
    monkeypatch.setattr(g,'check_release',fresh)
    monkeypatch.setattr(g.time,'time',lambda:100.+clock())
    monkeypatch.setattr(context,'verify_release',native)
    with pytest.raises(PermissionError,match='STALE_RELEASE'):
        g.wait_release(tmp_path,tmp_path,{'job_id':JOB,'request_sha256':'CPU_SHA_FIXTURE'},ledger,
            'E1_ADAPTER_ONLY_QUALIFY2','FIRST_UPDATE',guard=lambda:None,clock=clock,sleep=clock.sleep)
    assert checks==[100.,230.]and not(slot/'RELEASE_CONSUMED.json').exists()
