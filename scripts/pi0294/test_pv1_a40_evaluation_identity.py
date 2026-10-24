"""CPU fixtures only: no native allocation, model, physics or endurance PASS."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1 import evaluation_platform as e
from scripts.pi0294.pv1 import heartbeat_capture as heartbeat
from scripts.pi0294.pv1 import pre_model_qualification as qualification
from scripts.pi0294.dec1.vanda_campaign import QUEUE
from scripts.pi0294.dec1.owner_transport import ReadOnlyFailure


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_text(json.dumps(value));path.chmod(0o400)


@pytest.fixture
def case(tmp_path):
    root=tmp_path/'evaluation';directory=root/'pbs/alloc-24'
    started=e.T0+100000;now=started+100
    m=dict(schema='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1',scope=e.ROLE,run_uuid=root.name,
        platform_evaluation_version=e.VERSION,execution=dict(profile=e.PROFILE),
        authority=dict(protocol_ref=e.AUTHORITY),code=dict(vla_commit='a'*40),
        limits=dict(PV1_T0_epoch=e.T0,PV1_hard_close_epoch=e.HARD_CLOSE))
    put(root/'RUN_MANIFEST.json',m)
    r=dict(schema='PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1',run_uuid=root.name,
        campaign_run_uuid='20261018T182956Z-localpc-pv1',authority_release=e.AUTHORITY,
        platform_evaluation_version=e.VERSION,native_platform=e.PLATFORM,
        run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),source_sha='a'*40,
        request_id='12345678-1234-4234-8234-123456789abc',
        submission_nonce='12345678-1234-4234-8234-123456789abd',
        stage='P4_DEV',evaluation_candidate='A_CONTROL',checkpoint_manifest_sha256=e.ENDPOINTS['A_CONTROL'],
        parent_checkpoint_manifest_sha256=e.PARENT,statistics_sha256=e.R0,optimizer_updates=0,
        Teacher_takeover=False,normalizer_refit=False,evaluation_data_for_training=False,LOCKBOX_opened=False,
        numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',project='CFP04-CF-007',job_name='pi0294_pv1_a40_dev',
        submit_count=24,active_cpu_limit=4,requested_ncpus=4,requested_memory_bytes=32*1024**3,
        allowed_native_capacity_profiles=[dict(ncpus=4,memory_bytes=32*1024**3),
            dict(ncpus=36,memory_bytes=240*1024**3)],
        walltime_s=28800,created_epoch=started-100,original_T0_epoch=e.T0,hard_close_epoch=e.HARD_CLOSE)
    put(directory/'REQUEST.json',r)
    a=dict(status=e.STATUS,platform_evaluation_version=e.VERSION,run_uuid=root.name,
        job_id='123.stdct-mgmt-02',hostname='GN-A40-016',project=r['project'],native_platform=e.PLATFORM,
        source_sha=r['source_sha'],manifest_sha256=r['run_manifest_sha256'],
        request_sha256=sha(directory/'REQUEST.json'),request_id=r['request_id'],submission_nonce=r['submission_nonce'],
        stage=r['stage'],evaluation_candidate=r['evaluation_candidate'],submit_count=24,
        checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],statistics_sha256=e.R0,
        allocation_start_epoch=started,allocation_end_epoch=started+28800,
        native_select='1:ncpus=4:mem=32gb:ngpus=1',native_allocated_ncpus=4,
        native_allocated_memory_bytes=32*1024**3,model_loaded=False,new_optimizer_updates=0)
    put(directory/'ALLOCATION_ADMISSION.json',a)
    j=dict(job_state='R',Job_Owner='e1192548@stdct-login-01',project=r['project'],Job_Name=r['job_name'],
        queue='batch_gpu',Rerunable=False,run_count=1,stime=started,exec_host='GN-A40-016/0',
        Variable_List=dict(PV1_REQUEST_ID=r['request_id'],PV1_SUBMISSION_NONCE=r['submission_nonce']),
        Resource_List=dict(ncpus=4,ngpus=1,mem='32gb',select=a['native_select'],walltime='08:00:00'))
    return dict(root=root,directory=directory,m=m,r=r,a=a,j=j,now=now)


def proof(c):
    jid=c['a']['job_id']
    return dict(schema=QUEUE,user='e1192548',observed_epoch=c['now'],
        qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout=jid+'\n',stderr=''),
        qstat=dict(argv=['/opt/pbs/bin/qstat','-f','-F','json',jid],returncode=0,
            stdout=json.dumps(dict(Jobs={jid:c['j']})),stderr=''))


def sealed(c):
    return e.sealed_identity(c['m'],c['r'],c['a'],run_uuid=c['root'].name,
        manifest_sha256=sha(c['root']/'RUN_MANIFEST.json'),request_sha256=sha(c['directory']/'REQUEST.json'))


def native(c,p=None):
    return e.native_identity(c['r'],c['a'],now=c['now'],user='e1192548',
        hostname=c['a']['hostname'],job_id=c['a']['job_id'],queue_proof=proof(c) if p is None else p)


@pytest.mark.parametrize('candidate',list(e.ENDPOINTS))
def test_exact_frozen_endpoints_and_distinct_dev_identity(case,candidate):
    c=case;r,a=c['r'],c['a']
    for x in (r,a):
        x.update(evaluation_candidate=candidate,checkpoint_manifest_sha256=e.ENDPOINTS[candidate],
            stage='B0_DEV' if candidate=='B0_CORR' else 'P4_DEV')
    put(c['directory']/'REQUEST.json',r);a['request_sha256']=sha(c['directory']/'REQUEST.json')
    assert sealed(c) is None and native(c) is None
    assert a['model_loaded'] is False and a['new_optimizer_updates']==0


@pytest.mark.parametrize('field,value',[
    ('platform_training_version','old-train'),('evaluation_candidate','REPLICA_A_CONTROL'),
    ('stage','P2_PRIMARY'),('statistics_sha256','f'*64),('checkpoint_manifest_sha256','f'*64),
    ('optimizer_updates',1),('optimizer_updates',False),('training_block',{'arm':'A_CONTROL'}),
    ('Teacher_takeover',True),('normalizer_refit',True),('LOCKBOX_opened',True),
    ('source_sha','f'*40),('request_id','not-a-uuid'),('submission_nonce','12345678-1234-1234-8234-123456789abd'),
    ('original_T0_epoch',e.T0+1),('created_epoch',e.STOP_NEW),('active_cpu_limit',36),
    ('walltime_s',28801),('native_platform','HOPPER_H200')])
def test_no_training_relabel_replica_changed_math_or_clock(case,field,value):
    case['r'][field]=value
    with pytest.raises(PermissionError):sealed(case)


@pytest.mark.parametrize('target,field,value',[
    ('m','scope','PI0294_PV1_SHARED_POLICY_HOPPER_V1'),
    ('m','platform_evaluation_version','old'),
    ('a','status','PASS_DEC1_OWN_NATIVE_A40_SOURCE_TRAIN_ALLOCATION'),
    ('a','model_loaded',True),('a','new_optimizer_updates',False),
    ('a','manifest_sha256','f'*64)])
def test_profile_and_admission_cannot_be_caller_pass_strings(case,target,field,value):
    case[target][field]=value
    with pytest.raises(PermissionError):sealed(case)


@pytest.mark.parametrize('field,value',[
    ('job_state','Q'),('project','other'),('Job_Owner','other@host'),('Job_Name','pi0294_dec1_a40_train'),
    ('queue','other'),('Rerunable',True),('run_count',2),('exec_host','hopper-h200-01/0'),
    ('stime',e.T0+99999)])
def test_current_native_job_must_match_exact_admitted_identity(case,field,value):
    case['j'][field]=value
    with pytest.raises(PermissionError):native(case)


@pytest.mark.parametrize('cpus,mem',[(4,32),(36,240)])
def test_original_allowed_native_profiles_do_not_increase_active_cpu(case,cpus,mem):
    c=case;select=f'1:ncpus={cpus}:mem={mem}gb:ngpus=1'
    c['j']['Resource_List'].update(ncpus=cpus,mem=f'{mem}gb',select=select)
    c['a'].update(native_select=select,native_allocated_ncpus=cpus,native_allocated_memory_bytes=mem*1024**3)
    native(c);assert c['r']['active_cpu_limit']==4


@pytest.mark.parametrize('change',['nonce','wall','two_gpu','wrong_gpu','memory','profile','stale','dropped','failed','empty','end'])
def test_full_queue_capacity_nonce_and_clock_fail_closed(case,change):
    c=case;j=c['j'];p=None
    if change=='nonce':j['Variable_List']['PV1_SUBMISSION_NONCE']='other'
    if change=='wall':j['Resource_List']['walltime']='07:59:59'
    if change=='two_gpu':j['Resource_List']['ngpus']=2
    if change=='wrong_gpu':j['Resource_List']['select']+=':gpu_model=H200'
    if change=='memory':j['Resource_List']['mem']='240gb'
    if change=='profile':c['r']['allowed_native_capacity_profiles'].append(dict(ncpus=99,memory_bytes=1))
    if change=='stale':p=proof(c);p['observed_epoch']-=121
    if change=='dropped':p=proof(c);p['qstat']['stdout']='{"Jobs":{}}'
    if change=='failed':p=proof(c);p['qselect']['returncode']=162
    if change=='empty':p=proof(c);p['qselect']['stdout']='';p['qstat']=None
    if change=='end':c['a']['allocation_end_epoch']+=1
    with pytest.raises(PermissionError):native(c,p)


def test_mature_identity_and_current_directory_use_vanda_without_hopper_fallback(case):
    c=case;root,d=c['root'],c['directory']
    with patch.dict(heartbeat.os.environ,dict(PBS_JOBID=c['a']['job_id'],USER='e1192548')),\
         patch.object(heartbeat.socket,'gethostname',return_value=c['a']['hostname']),\
         patch.object(heartbeat.time,'time',return_value=c['now']),\
         patch.object(e,'collect_queue',return_value=proof(c)) as queue,\
         patch.object(heartbeat,'owned_jobs',side_effect=AssertionError('NO_HOPPER_READ')):
        assert heartbeat.identity(root,d,c['a'])['job_id']==c['a']['job_id']
        assert qualification.current_directory(root,c['a'])==d
        assert queue.call_count==1
        queue.side_effect=ReadOnlyFailure('PRESERVED_DIAGNOSTIC')
        with pytest.raises(ReadOnlyFailure):heartbeat.identity(root,d,c['a'])
        assert not (d/heartbeat.FILENAME).exists()
        put(d/'PBS_PROCESS_FINAL.json',dict(job_id=c['a']['job_id']))
        with pytest.raises(PermissionError,match='PRECEDE_MODEL'):heartbeat.identity(root,d,c['a'])


def test_wrong_typed_identity_fails_before_queue_transport(case):
    c=case;c['r']['platform_training_version']='old'
    with patch.object(e,'collect_queue') as queue,pytest.raises(PermissionError):
        e.current_identity(c['m'],c['r'],c['a'],run_uuid=c['root'].name,
            manifest_sha256=sha(c['root']/'RUN_MANIFEST.json'),request_sha256=sha(c['directory']/'REQUEST.json'),
            user='e1192548',hostname=c['a']['hostname'],job_id=c['a']['job_id'])
    queue.assert_not_called()


def test_typed_vanda_request_cannot_choose_legacy_h200_identity(case):
    c=case;c['a']['status']='PASS_PV1_OWN_NATIVE_H200_ALLOCATION'
    put(c['directory']/'ALLOCATION_ADMISSION.json',c['a'])
    with patch.dict(heartbeat.os.environ,dict(PBS_JOBID=c['a']['job_id'],USER='e1192548')),\
         patch.object(heartbeat.socket,'gethostname',return_value=c['a']['hostname']),\
         patch.object(heartbeat.time,'time',return_value=c['now']),\
         patch.object(heartbeat,'owned_jobs') as hopper,patch.object(e,'collect_queue') as vanda:
        with pytest.raises(PermissionError,match='FALL_BACK_TO_H200'):
            heartbeat.identity(c['root'],c['directory'],c['a'])
        with pytest.raises(PermissionError,match='FALL_BACK_TO_H200'):
            qualification.current_directory(c['root'],c['a'])
        hopper.assert_not_called();vanda.assert_not_called()
