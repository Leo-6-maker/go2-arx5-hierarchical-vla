"""Runner fakes only; no scheduler, network, model or native physics."""
from pathlib import Path
from types import SimpleNamespace as R
import json
import pytest
from . import queue_read_guard as q

@pytest.mark.parametrize('ret',[1,162])
def test_failure_preserves_scheduler_stderr_never_empty_queue(capsys,ret):
 calls=[]
 def run(args,**kw):calls.append(kw);return R(returncode=ret,stdout='',stderr='actual scheduler diagnostic')
 with pytest.raises(SystemExit) as e:q.queue_run(['/cm/shared/apps/pbspro/current/bin/qselect','-u','fixture'],capture_output=True,text=True,timeout=25,check=True,runner=run)
 assert e.value.code==75 and len(calls)==1 and calls[0]['check'] is False
 v=json.loads(capsys.readouterr().err);assert v['returncode']==ret and v['queue_known'] is False and v['stderr']=='actual scheduler diagnostic'
 r=R(returncode=75,stdout='',stderr=json.dumps(v))
 with pytest.raises(q.QueueReadFailure):q.decode_read_result(r,effect='READ_ONLY')
 with pytest.raises(RuntimeError):q.decode_read_result(r,effect='MUTATION')

@pytest.mark.parametrize('args,timeout',[(['/bin/qsub'],25),(['/opt/pbs/bin/qstat'],26),(['/opt/pbs/bin/qstat'],True)])
def test_bad_command_or_deadline_before_runner(args,timeout):
 with pytest.raises(ValueError):q.queue_run(args,capture_output=True,text=True,timeout=timeout,check=True,runner=lambda *a,**k:pytest.fail('NO_RUN'))

def test_no_job_success_is_only_actual_zero_status():
 r=R(returncode=0,stdout='',stderr='');assert q.queue_run(['/opt/pbs/bin/qselect','-u','fixture'],capture_output=True,text=True,timeout=25,check=True,runner=lambda *a,**kw:r) is r
 assert q.decode_read_result(R(returncode=0,stdout='{"jobs":{}}',stderr=''),effect='READ_ONLY')=={'jobs':{}}

@pytest.mark.parametrize('effect',['MUTATION','HEARTBEAT','READ_ONLY'])
def test_untyped_native_assertions_are_never_read_retry(effect):
 with pytest.raises(RuntimeError):q.decode_read_result(R(returncode=1,stdout='',stderr='AssertionError: WRONG_AUTHORITY'),effect=effect)

@pytest.mark.parametrize('bad',['extra','stdout','returncode','schema','queue_known'])
def test_forged_or_mixed_marker_refused(bad):
 v=dict(schema=q.SCHEMA,command=['/opt/pbs/bin/qselect'],returncode=162,stdout='',stderr='diagnostic',queue_known=False);out=''
 if bad=='extra':v['extra']=True
 if bad=='stdout':out='partial output'
 if bad=='returncode':v['returncode']=False
 if bad=='schema':v['schema']='OTHER'
 if bad=='queue_known':v['queue_known']=True
 with pytest.raises(RuntimeError):q.decode_read_result(R(returncode=75,stdout=out,stderr=json.dumps(v)),effect='READ_ONLY')

def test_only_exact_queue_program_three_calls_wrapped():
 source="import subprocess\nfrom pathlib import Path\nq='/opt/pbs/bin/qstat'\nselection=subprocess.run([str(Path(q).with_name('qselect'))],capture_output=True,text=True,timeout=25,check=True)\na=subprocess.run([q],capture_output=True,text=True,timeout=25,check=True)\nb=subprocess.run([q],capture_output=True,text=True,timeout=25,check=True)\nassert selection.stdout==a.stdout==b.stdout\n"
 wrapped=q.wrap_program(source,Path(q.__file__).read_text());compile(wrapped,'<cpu_only>','exec');assert 'assert selection.stdout==a.stdout==b.stdout' in wrapped
 with pytest.raises(ValueError):q.wrap_program('write_authority()',Path(q.__file__).read_text())
 with pytest.raises(ValueError):q.wrap_program(wrapped,Path(q.__file__).read_text())

@pytest.mark.parametrize('effect',['READ_ONLY','MUTATION'])
def test_original_journal_keeps_read_failures_distinct_from_unknown_writes(tmp_path,effect):
 from .owner_transport import Journal,ReadOnlyFailure,UnknownMutation
 j=Journal(tmp_path,dict(run_uuid='fixture',job_id='fixture-job',source_sha='a'*40,request_sha256='b'*64))
 j.call('SEED','MUTATION',lambda *a:{'confirmed':True})
 marker=dict(schema=q.SCHEMA,command=['/opt/pbs/bin/qselect'],returncode=162,stdout='',stderr='scheduler diagnostic',queue_known=False)
 def call(*a):return q.decode_read_result(R(returncode=75,stdout='',stderr=json.dumps(marker)),effect=effect)
 with pytest.raises(ReadOnlyFailure if effect=='READ_ONLY' else UnknownMutation):j.call('QUEUE',effect,call)
 state=json.loads((tmp_path/'RPC_EFFECT_STATE.json').read_bytes());assert state['status']==('CONFIRMED' if effect=='READ_ONLY' else 'UNKNOWN')
 if effect=='READ_ONLY':assert j.call('QUEUE','READ_ONLY',lambda *a:{'jobs':{}})=={'jobs':{}}
