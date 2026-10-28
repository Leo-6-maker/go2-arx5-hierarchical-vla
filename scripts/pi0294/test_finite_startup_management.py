"""Original CAS/startup seams on real temp files; server/process are fixtures.

These tests do not connect, register an actor, load GR00T or run physics.
The native platform assertion is explicitly replaced ONLY in the CPU fixture;
the published management programs keep it and every original runtime gate.
"""
import ast,base64,json,subprocess,sys,time
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import finite_startup_management as n
from scripts.pi0294.pv1 import mix_accounting as a,mix_core_accounting as c,mix_core_delegation as d
from scripts.pi0294.test_v5_core_full_admission import full,captured_global
from scripts.pi0294.test_mix_core_delegation import activated
from scripts.pi0294.dec1.owner_transport import Journal,UnknownMutation

GUARD="assert os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')and not os.environ.get('PBS_JOBID');sys.dont_write_bytecode=True"

@pytest.fixture(scope='module')
def bind_case():
 g,l,r,_=__import__('scripts.pi0294.test_v5_core_stage_accounting',fromlist=['registered']).registered()
 # Join the existing full captured budget/history fixture, as the original
 # admission test does; the future process remains an explicit CPU fixture.
 actual=captured_global.__wrapped__();inputs=full.__wrapped__(actual)
 g,l=map(json.loads,(inputs['global_bytes'],inputs['local_bytes']));p=json.loads(inputs['native_facts_bytes']);rid=p['request_id']
 for v in(g,l):
  row=v[c.FIELD][rid];row.update(status=c.STATES[0],native_handle=None)
  for k in('boot_id','pid','proc_start','process_started_epoch','native_bind_proof_sha256'):row.pop(k,None)
 g.setdefault('revision',0);before=a.raw(g)
 proposal=c.bind(g,l,request_id=rid,process=p,now_epoch=1205.);after=proposal['proposed_global'];after['revision']+=1
 c.checked(after)
 return before,a.raw(after),rid,after[c.FIELD][rid]

def invoke_program(tmp_path,case,*,wrong_after=False):
 before,after,rid,row=case;path=tmp_path/'GLOBAL.json';path.write_bytes(before)
 args=dict(code=str(Path(__file__).resolve().parents[2]),path=str(path),before=a.sha(before),after='0'*64 if wrong_after else a.sha(after),
  kind='bind',row=row,request_id=rid,observed_epoch=time.time(),key='CPU-ONLY-EXACT-BIND')
 # The path below is the repository, never a production capsule.
 args['code']=str(Path(__file__).resolve().parents[2])
 source=n.GLOBAL;assert source.count(GUARD)==1
 source=source.replace(GUARD,"assert CPU_FIXTURE_NOT_NATIVE;sys.dont_write_bytecode=True")
 body='CPU_FIXTURE_NOT_NATIVE=True\nP='+repr(args)+'\n'+source
 return path,lambda:subprocess.run([sys.executable,'-B','-c',body],capture_output=True,text=True,timeout=35)

def test_real_temp_original_CAS_changes_only_the_bound_row_and_revision(bind_case,tmp_path):
 path,call=invoke_program(tmp_path,bind_case);reply=call();assert reply.returncode==0,reply.stderr
 assert path.read_bytes()==bind_case[1]
 receipt=json.loads(path.with_name('CPU-ONLY-EXACT-BIND.RECEIPT.json').read_bytes())
 assert receipt['after_sha256']==a.sha(bind_case[1])and receipt['before_sha256']==a.sha(bind_case[0])
 before,after=map(json.loads,bind_case[:2]);assert after['used']==before['used']and after['PBS_submissions']==before['PBS_submissions']
 assert all('ssh'not in p for p in n.PROGRAMS.values())

def test_wrong_complete_proposal_hash_is_refused_before_any_native_intent(bind_case,tmp_path):
 path,call=invoke_program(tmp_path,bind_case,wrong_after=True);reply=call();assert reply.returncode!=0
 assert path.read_bytes()==bind_case[0]and not list(tmp_path.glob('*.INTENT.json'))

def test_lost_reply_preserves_original_Journal_UNKNOWN_and_real_CAS_without_replay(bind_case,tmp_path):
 path,call=invoke_program(tmp_path,bind_case);jroot=tmp_path/'journal';jroot.mkdir()
 j=Journal(jroot,dict(run_uuid='CPU_ONLY_STARTUP',job_id='CPU_ONLY_TEMP',source_sha='c'*40,request_sha256='d'*64));calls=[]
 def lost(intent,limit):
  calls.append(intent);assert min(35,limit)==35
  reply=call();assert reply.returncode==0,reply.stderr
  raise TimeoutError('CPU_FIXTURE_LOST_REPLY_AFTER_REAL_CAS')
 with pytest.raises(UnknownMutation):j.call('BIND','MUTATION',lost)
 assert path.read_bytes()==bind_case[1]and json.loads(j.state_path.read_bytes())['status']=='UNKNOWN'
 with pytest.raises(UnknownMutation):j.call('BIND','MUTATION',lost)
 assert len(calls)==1 and path.with_name('CPU-ONLY-EXACT-BIND.RECEIPT.json').exists()

def test_initial_stores_use_six_actual_temp_adoption_bytes_and_publish_READY_last(full,tmp_path,monkeypatch):
 v=activated(full,tmp_path);body=json.loads(v['grant_bytes'])['body'];N=tmp_path/'neural';(N/'evidence').mkdir(parents=True);BASE=tmp_path/'owner';BASE.mkdir();writes=[]
 ns=dict(os=__import__('os'),json=json,Path=Path,raw=a.raw)
 # Execute the actual original exclusive-file helper, not a test copy.
 fn=next(x for x in ast.parse(n.COMMON).body if isinstance(x,ast.FunctionDef)and x.name=='put')
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-exclusive-startup-put>','exec'),ns)
 original=ns['put']
 def put(path,value,mode=0o400):writes.append(path);return original(path,value,mode)
 ns.update(put=put,d=d,c=c,base64=base64,time=time,N=N,BASE=BASE,grant=v['grant_bytes'],gb=v['global_bytes'],args=v['activation_args'],
  owner=dict(pid=12345,proc_start=9999),p=body['admission']['allocation']['native_process'],r=body['request'],rb=a.raw(body['request']),sha=a.sha)
 start=n.ACTIVATE.index('now=time.time();state=')
 with monkeypatch.context()as patch:
  patch.setattr(time,'time',lambda:1206.)
  exec(compile(n.ACTIVATE[start:],'<actual-activation-store-publication-tail>','exec'),ns)
 assert writes[-1]==BASE/'CORE_FINITE_ACTIVATION_READY.json'
 ready=json.loads(writes[-1].read_bytes());assert ready['model_permission']is ready['physics_permission']is False
 assert len(writes)==6 and not list(tmp_path.glob('**/*MODEL_LOAD*'))
