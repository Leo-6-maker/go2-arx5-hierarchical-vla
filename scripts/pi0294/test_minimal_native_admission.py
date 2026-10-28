"""Captured full budgets; all future PBS/project/process values are fixtures.

These cases cannot issue a native permission or start a model.
"""
import base64,gzip,json,os,time
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.test_minimal_execution import actual,request,closed_CPU_fixture
from scripts.pi0294.generalization import minimal_execution as e,minimal_native_admission as n
from scripts.pi0294.dec1 import vanda_campaign as q


def queue_limits(observed_epoch):
    reads={}
    for name in ('auto','batch_gpu'):
        v=dict(enabled='True',started='True',resources_max=dict(walltime='168:00:00'))
        if name=='auto':v['route_destinations']=['batch_gpu']
        reads[name]=dict(argv=['/opt/pbs/bin/qstat','-Q','-f','-F','json',name],returncode=0,stderr='',
            stdout=json.dumps(dict(Queue={name:v})))
    return dict(source='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS',requested_queue='auto',
        destination='batch_gpu',maximum_wall_s=604800,observed_epoch=observed_epoch,raw_queue_readbacks=reads)


@pytest.fixture
def packet(actual):
    g,cap=actual;g=closed_CPU_fixture(g);r=request(g,cap);now=1793131105.;job='12345.stdct-mgmt-02'
    limits=e.raw(queue_limits(now-6))
    m=dict(scope=e.scope.SCOPE,platform_training_version=e.VERSION,run_uuid='CPU-fixture-first512',
        output_root='/CPU-fixture/children/CPU-fixture-first512',registered_runs_root='/CPU-fixture/children',
        execution=dict(profile=e.PROFILE,agent_liveness_required=False,sealed_inputs={'plans/MINIMAL_EFFECTIVE_QUEUE_LIMITS.private.json':e.sha(limits)}),
        code=dict(vla_commit=r['source_sha']))
    r.update(run_uuid=m['run_uuid'],created_epoch=now-5,submit_count=len(g['PBS_submissions'])+1,run_manifest_sha256=e.sha(e.raw(m)),effective_queue_proof_sha256=e.sha(limits))
    v=e.register(e.raw(g),request_bytes=e.raw(r),Core_measurement_capture_bytes=cap,now_epoch=now-4)['proposed_global']
    v=e.bind(v,r['request_id'],job,now_epoch=now-3)
    root=Path(os.environ['ARX5_FIRST512_MEASUREMENT_CAPTURE']).parent
    local=json.loads((root/'CORE_RETRY12_CONTINUATION_INDEPENDENT_DUAL_READBACK.private.json').read_bytes())['local']
    lb=gzip.decompress(base64.b64decode(local['LOCAL_gzip_base64']))
    native=dict(job_state='R',queue='batch_gpu',Rerunable=False,run_count=1,project=r['project'],
        Job_Name=r['job_name'],Job_Owner='e1192548@stdct-login-01',exec_host='GN-A40-022/0*36',
        Variable_List=dict(PV1_REQUEST_ID=r['request_id'],PV1_SUBMISSION_NONCE=r['submission_nonce']),
        Resource_List=dict(ncpus=36,ngpus=1,mem='240gb',walltime='16:30:00',select='1:ncpus=36:mem=240gb:ngpus=1:gpu_model=A40'),
        stime=time.strftime('%a %b %d %H:%M:%S %Y',time.localtime(now-2)))
    queue=dict(schema=q.QUEUE,user='e1192548',observed_epoch=now,
        qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout=job+'\n',stderr=''),
        qstat=dict(argv=['/opt/pbs/bin/qstat','-f','-F','json',job],returncode=0,stdout=json.dumps(dict(Jobs={job:native})),stderr=''))
    site=dict(schema=n.SITE,source='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE',
        run_uuid=r['run_uuid'],request_id=r['request_id'],submission_nonce=r['submission_nonce'],
        request_sha256=e.sha(e.raw(r)),source_sha=r['source_sha'],job_id=job,observed_epoch=now,
        project=dict(name=r['project'],active=True,users=['e1192548'],cluster=['stdct-mgmt-02'],start_date='2026-03-12',end_date='2027-03-12'),
        balance_GPU_hours=1000.,balance_CPU_hours=1000.,
        quota=dict(observed_epoch=now,user='e1192548',scope='OWN_PROJECT_AND_FILESYSTEM_MINIMUM',available_bytes=1024**5))
    return dict(global_bytes=e.raw(v),local_bytes=lb,request_bytes=e.raw(r),manifest_bytes=e.raw(m),job_id=job,
        hostname='GN-A40-022',user='e1192548',native_job_bytes=e.raw(native),queue_proof=queue,
        site_bytes=e.raw(site),effective_queue_bytes=limits,independent_job_bytes=None,now_epoch=now)


def test_actual_fourteen_dimension_budget_and_future_native_fixture_cannot_write(packet):
    before=deepcopy(packet);proof=n.validate(**packet)
    assert packet==before and proof['whole_remaining_package_shortfalls']=={}
    assert len(proof['full_remaining_proof']['dimensions'])==14
    assert proof['native_allocated_ncpus']==36 and proof['active_CPU_limit']==8
    assert proof['model_loaded'] is proof['permission_file_created'] is False
    assert proof['new_optimizer_updates']==proof['new_physics']==0


@pytest.mark.parametrize('change',['unknown','pending','project_balance','stale_site','stale_quota',
    'quota_shortfall','missing_queue_row','unconfirmed_foreign_GPU','old_scope','different_raw_job',
    'native_restart','wrong_nonce','wrong_node','original_clock','queue_limit'])
def test_missing_native_or_full_remaining_evidence_is_not_an_allocation(packet,change):
    p=deepcopy(packet)
    def update(key,fn):
        value=json.loads(p[key]);fn(value);p[key]=e.raw(value)
    if change=='unknown':update('global_bytes',lambda v:v.update(unknown_effects=['PRESERVED']))
    elif change=='pending':update('global_bytes',lambda v:v.update(pending_updates=[{'PRESERVED':True}]))
    elif change=='project_balance':update('site_bytes',lambda v:v.update(balance_CPU_hours=595.19))
    elif change=='stale_site':update('site_bytes',lambda v:v.update(observed_epoch=p['now_epoch']-121))
    elif change=='stale_quota':update('site_bytes',lambda v:v['quota'].update(observed_epoch=p['now_epoch']-901))
    elif change=='quota_shortfall':update('site_bytes',lambda v:v['quota'].update(available_bytes=0))
    elif change=='missing_queue_row':p['queue_proof']['qstat']['stdout']='{"Jobs":{}}'
    elif change=='unconfirmed_foreign_GPU':
        proof=p['queue_proof'];jid='9999999.stdct-mgmt-02';native=json.loads(p['native_job_bytes'])
        native.update(Job_Name='other',project='CFP04-CF-009');jobs=json.loads(proof['qstat']['stdout'])
        jobs['Jobs'][jid]=native;proof['qstat']['stdout']=json.dumps(jobs)
        proof['qstat']['argv'].append(jid);proof['qselect']['stdout']+=jid+'\n'
    elif change=='old_scope':update('request_bytes',lambda v:v.update(platform_training_version='OLD_CD'))
    elif change=='different_raw_job':update('native_job_bytes',lambda v:v.update(exec_host='GN-A40-999/0*36'))
    elif change=='wrong_node':p['hostname']='GN-A40-023'
    elif change=='original_clock':p['now_epoch']=1793451149.8133287
    elif change=='queue_limit':update('effective_queue_bytes',lambda v:v.update(maximum_wall_s=1))
    else:
        native=json.loads(p['native_job_bytes'])
        if change=='native_restart':native['run_count']=2
        else:native['Variable_List']['PV1_SUBMISSION_NONCE']='another-nonce'
        p['native_job_bytes']=e.raw(native)
        p['queue_proof']['qstat']['stdout']=json.dumps(dict(Jobs={p['job_id']:native}))
    with pytest.raises((PermissionError,KeyError,ValueError)):n.validate(**p)
    assert packet!=p


def test_independent_native_observations_allow_changing_usage_counters(packet):
    p=deepcopy(packet);native=json.loads(p['native_job_bytes'])
    native.update(resources_used=dict(cput='00:01:11',walltime='00:02:00'),mtime='later',comment='updated')
    p['native_job_bytes']=e.raw(native)
    assert n.validate(**p)['job_id']==p['job_id']


@pytest.mark.parametrize('key,value', [('exec_host','GN-A40-999/0*36'),('stime','another-start'),('job_state','F'),('run_count',2)])
def test_independent_native_observations_reject_changed_identity(packet,key,value):
    p=deepcopy(packet);native=json.loads(p['native_job_bytes']);native[key]=value;p['native_job_bytes']=e.raw(native)
    with pytest.raises((PermissionError,KeyError,ValueError)):n.validate(**p)
