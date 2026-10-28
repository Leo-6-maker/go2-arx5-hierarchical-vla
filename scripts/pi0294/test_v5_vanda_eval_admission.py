"""Captured real science/budget inputs with explicit native-process fixtures.

No test result is a native qstat, GPU, render, model or physics permission.
"""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import pytest
from scripts.pi0294.test_v5_vanda_cohost import captured
from scripts.pi0294.test_v5_vanda_eval_accounting import inputs
from scripts.pi0294.pv1 import mix_accounting as a,mix_vanda_eval_identity as identity
from scripts.pi0294.pv1 import mix_vanda_eval_accounting as accounting,mix_vanda_eval_admission as admission
from scripts.pi0294.pv1 import mix_evaluation,successor_cd as s
from scripts.pi0294.dec1 import vanda_campaign


@pytest.fixture
def packet(captured):
    g,args=inputs(captured);r=args['request'];record=args['record'];now=args['now_epoch']+30
    workspace=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
    path=workspace/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/CORE_PAIRED_PEER_TEMPLATES_7cc75180.private.json'
    if not path.exists():pytest.skip('private captured original population unavailable')
    template=json.loads(path.read_bytes())['rows']['A_CONTROL']['core']['files']
    population=base64.b64decode(template['plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'])
    groups=json.loads(population)['groups'];plan=a.raw(mix_evaluation.candidate_plan(groups,'C50'))
    effective=a.raw(dict(source='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS',requested_queue='auto',destination='batch_gpu',maximum_wall_s=604800,observed_epoch=args['now_epoch']-1))
    r.update(planned_panel_sha256=a.sha(plan),queue_limit_relative='plans/ACTUAL_QUEUE_LIMITS.json',queue_limit_sha256=a.sha(effective))
    sealed={'plans/V5_CD_CLOSED_TRAINING_CAPTURE.private.json':a.sha(a.raw(args['capture'])),
        'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json':a.sha(population),'plans/SOURCE_EVALUATION_PLAN.json':a.sha(plan),
        'plans/PRIOR_EVALUATION_OUTCOMES.json':a.sha(a.raw([])),'plans/ACTUAL_QUEUE_LIMITS.json':a.sha(effective)}
    m=dict(scope=identity.NEURAL_ROLE,platform_evaluation_version=identity.VERSION,run_uuid=r['run_uuid'],
        code=dict(vla_commit=r['source_sha']),execution=dict(profile=identity.NEURAL_PROFILE,sealed_inputs=sealed),
        authority=dict(protocol_ref=s.TASK_SOURCE,scope_grant_sha256=s.GRANT_SHA,T_CD=r['T_CD'],deadline_epoch=r['deadline_epoch']))
    r['run_manifest_sha256']=a.sha(a.raw(m))
    record.update({k:r[k] for k in record if k in r});record['request_sha256']=a.sha(a.raw(r))
    g=accounting.register(g,**args);jid='7654321.stdct-mgmt-02'
    g=accounting.bind_job(g,request_id=r['request_id'],job_id=jid,now_epoch=now-10)
    job=dict(job_state='R',queue='batch_gpu',Rerunable=False,run_count=1,project='CFP04-CF-007',
        Job_Name=identity.JOB_NAME,Job_Owner='e1192548@stdct-login-01',exec_host='GN-A40-022/0*36',
        Variable_List=dict(PV1_REQUEST_ID=r['request_id'],PV1_SUBMISSION_NONCE=r['submission_nonce']),
        Resource_List=dict(ncpus=36,ngpus=1,mem='240gb',walltime='08:00:00',select='1:ncpus=36:mem=240gb:ngpus=1:gpu_model=A40'),
        stime=time.strftime('%a %b %d %H:%M:%S %Y',time.localtime(now-9)))
    allocation=identity.native_allocation(job,r,job_id=jid,hostname='GN-A40-022',now_epoch=now)
    facts=dict(schema='V5_CD_VANDA_NEURAL_PROCESS_STARTED_V1',uid=22182,job_id=jid,native_handle=jid,
        request_id=r['request_id'],request_sha256=a.sha(a.raw(r)),submission_nonce=r['submission_nonce'],source_sha=r['source_sha'],
        run_manifest_sha256=r['run_manifest_sha256'],native_model_permission=False,physics_authorized=False,
        pid=987654,proc_start=123456,boot_id='c1968358-1c82-4c14-8156-ed02b885a806',cgroup='0::/pbs-fixture\n',
        GPU_uuid='GPU-12345678-1234-1234-1234-123456789abc',visible='0',started_epoch=now-8,observed_epoch=now,
        hostname='GN-A40-022',actual_PBS=job,actual_PBS_sha256=a.sha(a.raw(job)),raw_GPU_processes=[],
        allocation_start_epoch=allocation['allocation_start_epoch'],allocation_end_epoch=allocation['allocation_end_epoch'])
    g=accounting.bind_neural_process(g,request_id=r['request_id'],process=facts,now_epoch=now)
    queue=dict(schema=vanda_campaign.QUEUE,user='e1192548',observed_epoch=now,
        qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout=jid+'\n',stderr=''),
        qstat=dict(argv=['/opt/pbs/bin/qstat','-f','-F','json',jid],returncode=0,stdout=json.dumps(dict(Jobs={jid:job})),stderr=''))
    phase=now-1
    site=dict(schema=admission.SITE,source='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE',run_uuid=r['run_uuid'],job_id=jid,
        request_id=r['request_id'],request_sha256=a.sha(a.raw(r)),submission_nonce=r['submission_nonce'],source_sha=r['source_sha'],
        phase_started_epoch=phase,observed_epoch=now,balance_GPU_hours=1000.,balance_CPU_hours=1000.,
        project=dict(name='CFP04-CF-007',active=True,users=['e1192548'],cluster=['stdct-mgmt-02'],start_date='2026-03-12',end_date='2027-03-12'),
        quota=dict(user='e1192548',scope='OWN_PROJECT_AND_FILESYSTEM_MINIMUM',observed_epoch=now,available_bytes=1024**5))
    return dict(manifest_bytes=a.raw(m),request_bytes=a.raw(r),global_bytes=a.raw(g),historical_LOCAL_bytes=captured['historical_LOCAL_bytes'],
        training_capture_bytes=a.raw(args['capture']),population_bytes=population,plan_bytes=plan,prior_bytes=a.raw([]),
        native_facts_bytes=a.raw(facts),queue_bytes=a.raw(queue),site_bytes=a.raw(site),queue_limit_bytes=effective,
        independent_job_bytes=None,phase_started_epoch=phase,now_epoch=now)


def test_raw_join_retains_real_all_remaining_budget_and_grants_no_model_or_physics(packet):
    before=deepcopy(packet);p=admission.recompute(**packet)
    assert packet==before and p['shortfalls']=={}
    assert p['allocation']['status']==identity.STATUS
    assert p['model_permission'] is p['physics_authorized'] is False
    assert p['historical_LOCAL_is_native_authority'] is False and p['canonical_LOCAL_sync']=='PENDING'
    assert p['dimensions']['PBS_submissions']['projected']==34
    assert p['allocation']['allocation_number']==1 and p['allocation']['submit_count']==30


@pytest.mark.parametrize('change',['Core_host','CPU_only','new_project','other_nonce','wrong_PID','old_site','old_quota','empty_queue',
                                  'read_failed','other_active_job','missed_condition','new_checkpoint','wrong_LOCAL','shortfall','wrong_phase','not_active_project'])
def test_incomplete_native_identity_or_fresh_budget_cannot_release(packet,change):
    p=deepcopy(packet)
    def update(key,fn):
        value=json.loads(p[key]);fn(value);p[key]=a.raw(value)
    if change in ('Core_host','CPU_only','new_project','other_nonce','wrong_PID'):
        def bad(v):
            if change=='Core_host':v['hostname']='corelab-4080s'
            elif change=='CPU_only':v['actual_PBS']['Resource_List']['ngpus']=0
            elif change=='new_project':v['actual_PBS']['project']='CFP04-CF-009'
            elif change=='other_nonce':v['actual_PBS']['Variable_List']['PV1_SUBMISSION_NONCE']='other'
            else:v['pid']+=1
            v['actual_PBS_sha256']=a.sha(a.raw(v['actual_PBS']))
        update('native_facts_bytes',bad)
    elif change=='old_site':update('site_bytes',lambda v:v.update(observed_epoch=p['now_epoch']-121))
    elif change=='old_quota':update('site_bytes',lambda v:v['quota'].update(observed_epoch=p['now_epoch']-901))
    elif change=='empty_queue':update('queue_bytes',lambda v:v.update(qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout='',stderr=''),qstat=None))
    elif change=='read_failed':update('queue_bytes',lambda v:v['qselect'].update(returncode=162))
    elif change=='other_active_job':
        def extra(v):
            v['qselect']['stdout']+='9999999.stdct-mgmt-02\n';v['qstat']['argv'].append('9999999.stdct-mgmt-02')
            jobs=json.loads(v['qstat']['stdout']);j=deepcopy(next(iter(jobs['Jobs'].values())));j.update(Job_Name='new-independent',project='CFP04-CF-009')
            jobs['Jobs']['9999999.stdct-mgmt-02']=j;v['qstat']['stdout']=json.dumps(jobs)
        update('queue_bytes',extra)
    elif change=='missed_condition':update('plan_bytes',lambda v:v['panels']['C50-DEV-N294101']['cards'].pop())
    elif change=='new_checkpoint':update('request_bytes',lambda v:v.update(checkpoint_manifest_sha256='0'*64))
    elif change=='wrong_LOCAL':p['historical_LOCAL_bytes']+=b' '
    elif change=='shortfall':update('global_bytes',lambda v:v.update(CPU_s=766799))
    elif change=='wrong_phase':p['phase_started_epoch']+=1
    else:update('site_bytes',lambda v:v['project'].update(active=False))
    with pytest.raises((PermissionError,KeyError,ValueError)):admission.recompute(**p)


@pytest.mark.parametrize('visible,count,name,memory', [('0,1',2,'NVIDIA A40',48*1024**3),('',0,None,0),('0',1,'RTX 4080',32*1024**3),('0',1,'NVIDIA A40',16*1024**3)])
def test_visible_device_is_real_A40_and_never_the_old_shared_Core_exception(visible,count,name,memory):
    with pytest.raises(PermissionError):identity.require_device(dict(platform_evaluation_version=identity.VERSION),
        dict(platform_evaluation_version=identity.VERSION,status=identity.STATUS),visible=visible,count=count,name=name,memory_bytes=memory)
