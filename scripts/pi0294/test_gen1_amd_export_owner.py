"""Original Journal/temp GLOBAL and real bootstrap child; no native job."""
import base64,json,os,shlex,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.pi0294.generalization import amd_export_owner as w,amd_export_accounting as e
from scripts.pi0294.dec1.owner_transport import Journal,UnknownMutation

ROOT=Path(__file__).resolve().parents[4]
B=ROOT/'work/private/v5-execution-20261025'

def test_real_bootstrap_keeps_request_and_native_parent_identity(tmp_path):
 root=tmp_path/'campaign/children/role';(root/'control').mkdir(parents=True);req=dict(request_id='fixture',submission_nonce='nonce',source_sha='a'*40)
 (root/'REQUEST.json').write_bytes(w.raw(req));(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json').write_bytes(w.raw(dict(PBS_submissions=[dict(request_id='fixture',job_id='123.stdct-mgmt-02')])))
 (root/'control/CPU_RUNNING_RELEASE.json').write_bytes(w.raw(dict(job_id='123.stdct-mgmt-02',request_id='fixture',observed_epoch=__import__('time').time())))
 body=w.script(root,tmp_path/'code',req,sys.executable).decode();assert '#PBS -m n'in body and '#PBS -r n'in body and 'ngpus=0'in body and 'ncpus=1'in body
 words=shlex.split(body,comments=True);index=words.index(sys.executable);argv=words[index:index+3];assert argv[:2]==[sys.executable,'-c'];v=subprocess.run(argv,text=True,capture_output=True,timeout=10,env=dict(os.environ,PBS_JOBID='123.stdct-mgmt-02'));assert v.returncode==0,v.stderr
 record=w.read(root/'control/CPU_COMPUTE_PROCESS.json');assert record['pid']==os.getpid()and record['request_id']=='fixture'and record['procstart']==Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]

def native():
 return dict(id='123.stdct-mgmt-02',job_state='F',Exit_status=0,project='CFP04-CF-007',Job_Name='pi0294_gen1_export',Job_Owner='e1192548@fixture',Resource_List=dict(ngpus=0,ncpus=1),Variable_List=dict(GEN1_REQUEST_ID='fixture',GEN1_SUBMISSION_NONCE='nonce'))

@pytest.mark.parametrize('bad',['state','project','name','owner','GPU','CPU','request','nonce','exit_type'])
def test_other_job_unknown_or_wrong_effect_identity_is_not_terminal(bad):
 j=native()
 if bad=='state':j['job_state']='R'
 elif bad=='project':j['project']='other-project'
 elif bad=='name':j['Job_Name']='other-job'
 elif bad=='owner':j['Job_Owner']='other-user'
 elif bad=='GPU':j['Resource_List']['ngpus']=1
 elif bad=='CPU':j['Resource_List']['ncpus']=4
 elif bad=='request':j['Variable_List']['GEN1_REQUEST_ID']='other'
 elif bad=='nonce':j['Variable_List']['GEN1_SUBMISSION_NONCE']='other'
 else:j['Exit_status']='unknown'
 with pytest.raises(PermissionError):w.payload_terminal(j,dict(request_id='fixture',submission_nonce='nonce'),'a'*64)

@pytest.fixture
def owner(tmp_path):
 required=B/'GEN1_VANDA_OFFICIAL_EXPORT_PREPARED.private.json'
 if not required.exists():pytest.skip('private full captured budget inputs unavailable')
 p=w.read(required);captured=w.read(B/'GEN1_VANDA_OFFICIAL_EXPORT_ROLE_INDEPENDENT_READONLY.private.json');root=tmp_path/'campaign/children/gen1-vanda-official-export-01';(root/'pbs').mkdir(parents=True);target=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';target.write_bytes(base64.b64decode(captured['GLOBAL_base64']))
 out=w.Owner.__new__(w.Owner);out.root=root;out.global_path=target;out.code=tmp_path/'code';out.request=p['request'];out.m=dict(execution=dict(operation_deadline_epoch=1793422349.8133287,python=sys.executable));out.e=e;out.runtime=SimpleNamespace(validate=lambda *a,**k:None);out.BASE=root/'control/owner';out.BASE.mkdir(parents=True);rpc=out.BASE/'setup-rpc';rpc.mkdir();out.setup=Journal(rpc,dict(run_uuid=root.name,job_id='CPU_FIXTURE_ONLY',source_sha=p['source'],request_sha256=w.sha(w.raw(p['request']))));out.fresh=lambda secret:dict(own=[],fixture=True);(root/'REQUEST.json').write_bytes(w.raw(p['request']));out.binding=dict(run_manifest_sha256=p['binding']['run_manifest_sha256'],lease=str(tmp_path/'OWN_CPU_LEASE.json'));Path(out.binding['lease']).write_bytes(w.raw(dict(status='ACTIVE',manifest_sha256=out.binding['run_manifest_sha256'],run_uuid=root.name)));return out

def test_single_submit_and_original_once_global_bind(owner,monkeypatch):
 calls=[]
 def query(argv,**kwargs):calls.append(argv);return SimpleNamespace(returncode=0,stdout='123.stdct-mgmt-02\n',stderr='')
 monkeypatch.setattr(w.subprocess,'run',query);before=w.read(owner.global_path);job=owner.submit(None);after=w.read(owner.global_path)
 assert job=='123.stdct-mgmt-02'and e.rows(after)[0]['job_id']==job and after['used']==before['used']
 with pytest.raises(PermissionError):owner.submit(None)
 assert len(calls)==1

def test_qsub_unknown_latches_before_bind_without_resubmitting(owner,monkeypatch):
 calls=[]
 def query(argv,**kwargs):calls.append(argv);raise subprocess.TimeoutExpired(argv,30)
 monkeypatch.setattr(w.subprocess,'run',query);before=owner.global_path.read_bytes()
 with pytest.raises(UnknownMutation):owner.submit(None)
 assert owner.global_path.read_bytes()==before and w.read(owner.setup.state_path)['status']=='UNKNOWN'
 with pytest.raises(FileExistsError):owner.submit(None)
 assert len(calls)==1

def test_original_fresh_full_budget_accepts_only_exact_unbound_CPU_join(owner,monkeypatch):
    from scripts.pi0294.generalization import amd_export_queue
    from scripts.pi0294.dec1 import vanda_campaign
    import time
    p=w.read(B/'GEN1_VANDA_OFFICIAL_EXPORT_OWNER_COLD_PREPARED.private.json')
    owner.CONFIG=p['config'];owner.queue=amd_export_queue
    owner.auth=lambda secret:dict(balance_CPU_hours=100000,balance_GPU_hours=100000)
    owner.program=lambda name:dict(report_age_s=0,available_bytes_lower=10**15)
    proof=dict(schema=vanda_campaign.QUEUE,user='e1192548',observed_epoch=time.time(),qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout='',stderr=''),qstat=None)
    monkeypatch.setattr(vanda_campaign,'collect_queue',lambda:proof)
    value=w.Owner.fresh(owner,None)
    assert value['own']==[]and value['exact_CPU_queue_join']['own_CPU_unbound']
    assert not value['full_remaining_proof']['cap_changes_required']
    assert value['full_queue']==proof and value['full_remaining_proof']['LOCAL_is_historical']
    g=w.read(owner.global_path);g['PBS_submissions'].append(dict(request_id='unrelated-unknown',job_id=None,status=e.STATES[0]));owner.global_path.write_bytes(w.raw(g))
    with pytest.raises(PermissionError,match='COMPLETE_ACCOUNT_QUEUE_NO_UNKNOWN'):w.Owner.fresh(owner,None)

def test_complete_CPU_with_failed_R0_support_settles_but_never_claims_training_bank(owner):
 from scripts.pi0294.generalization import amd_export,amd_pilot_scope as scope
 p=w.read(B/'GEN1_VANDA_OFFICIAL_EXPORT_PREPARED.private.json');plan=json.loads(base64.b64decode(p['small']['plans/GEN1_SAVED_EXPORT_PLAN.private.json']));packet=base64.b64decode(p['packet_base64']);identities={x['attempt_id']:x for x in scope.pilot_plan(packet)['attempts']};records=[]
 for case in plan['cases']:
  row=dict(identities[case['id']],id=case['id'],source_raw_manifest_sha256=case['raw_manifest_sha256'],source_full_audit_sha256=case['audit_sha256'],source_packet_sha256=scope.PACKET_SHA,statistics_sha256=scope.a.STATS,admitted_for_training=False,codec=dict(status='BLOCKED_FROZEN_CODEC_PHYSICAL_ERROR'),fixture=True)
  records.append(row);folder=owner.root/'datasets'/case['id'];folder.mkdir(parents=True);(folder/'EXPORT_RECEIPT.json').write_bytes(w.raw(row))
 support=amd_export.support_after_export(packet,plan['capture_outcomes'],records);assert support['status']=='DATA_SUPPORT_INCOMPLETE_NO_TRAINING'
 (owner.root/'plans').mkdir();(owner.root/'plans/GEN1_SAVED_EXPORT_PLAN.private.json').write_bytes(w.raw(plan));(owner.root/'plans/GEN1_V5_PREVIEW_V2.private.json').write_bytes(packet)
 ledger=dict(status='KNOWN_GEN1_EXPORT_TERMINAL',unknown_effects=[],error=None,CPU_s=13.1,completed=len(records),admitted=0,constructors=0,physics_ticks=0,episode_starts=0,model_loads=0,queries=0,updates=0);(owner.root/'RESOURCE_LEDGER.json').write_bytes(w.raw(ledger))
 (owner.root/'evidence').mkdir();result=dict(status='KNOWN_GEN1_OFFICIAL_EXPORT_TERMINAL',export_finished=True,rows=records,support=support,bank_ready=None,constructors=0,physics_ticks=0,episode_starts=0,model_loads=0,queries=0,updates=0);(owner.root/'evidence/GEN1_EXPORT_RESULT.private.json').write_bytes(w.raw(result))
 (owner.root/'control/CPU_COMPUTE_PROCESS.json').write_bytes(w.raw(dict(job_id='123.stdct-mgmt-02',request_id=owner.request['request_id'],fixture=True)))
 before=w.read(owner.global_path);bound=e.bind(before,request_id=owner.request['request_id'],job_id='123.stdct-mgmt-02');owner.global_path.write_bytes(w.raw(bound));job=native();job['Variable_List']=dict(GEN1_REQUEST_ID=owner.request['request_id'],GEN1_SUBMISSION_NONCE=owner.request['submission_nonce']);job['resources_used']=dict(cput='00:00:16')
 assert owner.terminal('123.stdct-mgmt-02',job)is False
 after=w.read(owner.global_path);row=e.rows(after)[0];assert row['export_finished']and not row['training_bank_qualified']and after['used']==before['used']and w.read(owner.binding['lease'])['status']=='CLOSED'
