"""Actual error shape replay, CPU only: return0 does not mean an empty queue."""
import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from . import vanda_campaign as q
from .owner_transport import ReadOnlyFailure

ERROR='qselect: cannot connect to server  (errno=0)\n'


def native(stdout='',stderr='',returncode=0):
    return SimpleNamespace(stdout=stdout,stderr=stderr,returncode=returncode)


@pytest.mark.parametrize('stderr',[ERROR,'qselect: unknown diagnostic\n'])
def test_zero_exit_with_native_stderr_cannot_be_published_as_empty(stderr):
    with pytest.raises(ReadOnlyFailure)as caught:
        q.collect_queue(runner=lambda *args,**kwargs:native(stderr=stderr))
    diagnostic=caught.value.diagnostic
    assert diagnostic['returncode']==0 and diagnostic['stdout']=='' and diagnostic['stderr']==stderr
    assert diagnostic['argv']==['/opt/pbs/bin/qselect','-u','e1192548']


def test_clean_genuinely_empty_queue_retains_original_behavior():
    calls=[]
    def runner(argv,**kwargs):calls.append(argv);return native()
    view=q.collect_queue(runner=runner,clock=lambda:1000.)
    assert q.queue_rows(view,now_epoch=1001.)==[] and view['qstat']is None and len(calls)==1


def test_successful_exit_with_qstat_diagnostic_is_not_complete_queue():
    def runner(argv,**kwargs):
        if 'qselect'in argv[0]:return native('123.stdct-mgmt-02\n')
        return native(json.dumps(dict(Jobs={})),stderr='qstat: cannot connect to server\n')
    with pytest.raises(ReadOnlyFailure)as caught:q.collect_queue(runner=runner)
    assert caught.value.diagnostic['argv'][0]=='/opt/pbs/bin/qstat'


def test_saved_bad_proof_is_independently_rejected_without_rerunning_a_command():
    view=dict(schema=q.QUEUE,user='e1192548',observed_epoch=1000.,
        qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout='',stderr=ERROR),qstat=None)
    with pytest.raises(PermissionError):q.queue_rows(view,now_epoch=1001.)


def test_exact_missing_job_read_preserves_native_failure_instead_of_terminal_inference(tmp_path,monkeypatch):
    from scripts.pi0294.pv1 import mix_owner as m
    from scripts.pi0294.test_v5_closed_training_handoff import fixture
    g,_,_=fixture();p=tmp_path/'GLOBAL.json';p.write_bytes(m.a.raw(g))
    owner=m.Owner.__new__(m.Owner);owner.global_path=p
    view=q.collect_queue(runner=lambda *args,**kwargs:native(),clock=__import__('time').time)
    monkeypatch.setattr(m.queue,'collect_queue',lambda:view)
    monkeypatch.setattr(m.subprocess,'run',lambda *args,**kwargs:native(
        '{"pbs_server":""}',stderr='qstat: illegally formed job identifier: 1492540.stdct-mgmt-02\n',returncode=1))
    with pytest.raises(m.ReadOnlyFailure)as caught:owner.fresh(job='1492540.stdct-mgmt-02')
    assert caught.value.diagnostic['returncode']==1 and 'illegally formed'in caught.value.diagnostic['stderr']
    assert caught.value.diagnostic['argv'][-1]=='1492540.stdct-mgmt-02'


def test_original_three_read_interruptions_retain_diagnostics_and_never_heartbeat(tmp_path):
    from . import cold_owner as watch_module
    from scripts.pi0292.durable import read,write
    from scripts.pi0294.test_v5_closed_training_handoff import fixture
    from scripts.pi0294.pv1.successor_cd import FIELD
    g,_,_=fixture();marker=g[FIELD]
    for name,value in [('SOURCE_REQUEST.private.json',dict(request=dict(submit_count=26),directory=str(tmp_path))),
        ('SOURCE_JOB_BOUND.private.json',dict(job_id='1492540.stdct-mgmt-02')),
        ('TRAIN_REGISTERED.private.json',dict(root=str(tmp_path)))]:write(tmp_path/name,value)
    diagnostic=dict(returncode=0,stdout='',stderr=ERROR);records=[];sleeps=[]
    def fresh(*args,**kwargs):
        error=watch_module.ReadOnlyFailure('NATIVE_DIAGNOSTIC_NOT_EMPTY');error.diagnostic=diagnostic;raise error
    def forbidden(*args,**kwargs):raise AssertionError('NO_HEARTBEAT_OR_MODEL_RELEASE_ON_UNKNOWN_QUEUE')
    owner=SimpleNamespace(BASE=tmp_path,native=SimpleNamespace(T0=1),cold_rpc_bound=True,successor_scope=marker,read=read,
        fresh=fresh,record=lambda name,value:records.append((name,value)),
        previous=SimpleNamespace(rpc=forbidden),release=forbidden,auth=forbidden)
    with pytest.raises(watch_module.ReadOnlyFailure):
        watch_module.watch(owner,None,clock=lambda:1001.,sleep=sleeps.append,scope_binding=marker)
    interrupted=[v for name,v in records if name.startswith('READ_ONLY_OBSERVATION_INTERRUPTION-')]
    assert len(interrupted)==3 and all(v['diagnostic']==diagnostic for v in interrupted)
    assert sleeps==[30,30]
