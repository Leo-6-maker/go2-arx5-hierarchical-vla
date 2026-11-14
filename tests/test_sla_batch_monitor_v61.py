import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.wm02.sla_batch_monitor_v61 import training_state,evaluation_state,panel_snapshot
from scripts.wm02.sla_monitor_wake_bridge_v61 import queue_once
from scripts.pi0294.e1.journal import Journal,MutationOutcomeUnknown


def closed():
    row=dict(spec_sha256='s'*64,result_schema='ACTUAL',updates=3000,target_visits=96000,source_visits=0)
    state=dict(LoadState='loaded',MainPID='0')
    lease=dict(status='CLOSED',worker_exit_code=0,spec_sha256='s'*64)
    result=dict(schema='ACTUAL',step=3000,target_visits=96000,source_visits=0,actual=dict(optimizer_updates=3000,completed_training_visits=96000),exit_code=0,fresh_RGB16_bit_exact=True)
    return row,state,lease,result,dict(exit_code=0)


def test_terminal_files_do_not_override_a_live_or_missing_process_handle():
    r,s,l,v,w=closed();assert training_state(r,dict(s,MainPID='123'),l,v,w)['status']=='RUNNING'
    assert training_state(r,dict(s,LoadState='not-found'),l,v,w)['status']=='UNKNOWN_HANDLE'
    assert training_state(r,s,l,v,w)['complete']
    assert training_state(r,s,dict(l,spec_sha256='x'*64),v,w)['status']=='TERMINAL_INCONSISTENT'


@pytest.mark.parametrize('bad',['missing_worker','wrong_updates','wrong_visits','RGB_not_qualified'])
def test_training_barrier_requires_the_actual_qualified_endpoint(bad):
    r,s,l,v,w=closed()
    if bad=='missing_worker':w=None
    elif bad=='wrong_updates':v['actual']['optimizer_updates']=2999
    elif bad=='wrong_visits':v['target_visits']=95968
    else:v['fresh_RGB16_bit_exact']=False
    assert not training_state(r,s,l,v,w)['complete']


def test_failed_evaluation_is_retained_as_closed_failure_not_success():
    r,s,l,v,w=closed();l['worker_exit_code']=v['exit_code']=w['exit_code']=1
    assert evaluation_state(r,s,l,v,w)=='CLOSED_FAILURE_RETAINED'
    assert evaluation_state(r,dict(s,MainPID='456'),l,v,w)=='RUNNING'


def test_cumulative_development_groups_are_not_added_twice(tmp_path):
    (tmp_path/'PANEL.private.json').write_text(json.dumps(dict(assignments=[{}]*8)))
    for n,count in enumerate((4,8)):
        g=tmp_path/('panel-group-'+str(n));(g/'runs').mkdir(parents=True)
        rows=[dict(native_run='run'+str(i),task_success=False,technical_valid=True,hard_safety_stop=False,actual_SLA_queries=2,actual_native_integrations=5953)for i in range(count)]
        (g/'OUTCOMES.private.json').write_text(json.dumps(rows))
        for i in range(count):
            r=g/'runs'/('run'+str(i));r.mkdir();(r/'SUPERVISOR.private.json').write_text('{}')
            gpu=r.with_name(r.name+'-gpu');gpu.mkdir();(gpu/'SUPERVISOR.private.json').write_text('{}')
    v=panel_snapshot(tmp_path);assert(v['terminal'],v['started'],v['SLA_queries_closed'])==(8,8,16)


def test_wake_is_once_only_and_queue_ack_does_not_claim_actual_resume(tmp_path):
    calls=[]
    def runner(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=0,stdout='Queued message mid for thread tid.\n',stderr='Warning: failed to resume daemon')
    spec=dict(thread_id='tid',codex='/codex');event=dict(result_sha256='e'*64)
    a=queue_once(tmp_path,spec,event,runner=runner);b=queue_once(tmp_path,spec,event,runner=runner)
    assert a==b and len(calls)==1 and a['queue_ack']
    assert a['automatic_resume']=='UNAVAILABLE_OR_WARNING_RETAINED'
    assert '--resume'in calls[0]


def test_unknown_queue_intent_is_never_resent(tmp_path):
    j=Journal(tmp_path/'journal',tmp_path.name);p=j.path('QUEUE-TERMINAL-EVENT');p.write_text(json.dumps(dict(run_uuid=tmp_path.name,status='INTENT',payload_sha256='unknown')))
    with pytest.raises((MutationOutcomeUnknown,ValueError)):
        queue_once(tmp_path,dict(thread_id='tid',codex='/codex'),dict(result_sha256='e'*64),runner=lambda *_a,**_k:pytest.fail('must not send again'))
