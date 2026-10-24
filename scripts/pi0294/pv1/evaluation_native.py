"""Typed Vanda native DEV admission and fresh query release, no new runner.

Reuses full resource recomputation and the original 175s owner wait. This
cannot replace registered-source validation, same-job 600s/80cycle proof,
real model/processor loading or actual robot safety/ACK/STOP qualification.
"""
import base64
from datetime import date,datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from . import evaluation_admission as a,evaluation_platform as e,evaluation_scope as scope
from .evaluation_account_jobs import checked_outside,sealed_confirmation
from .future_budget import _read,_fresh,_number
from scripts.pi0294.dec1.vanda_campaign import queue_rows,classify,collect_queue

GRANT='plans/V4_S2_GLOBAL_RESOURCE_AUTHORIZATION.json'
B_CLOSURE='plans/V4_B4096_INDEPENDENT_DUAL_CLOSURE.private.json'
PHASE='BEFORE_FIRST_DEVELOPMENT_QUERY_IN_THIS_NATIVE_BLOCK'


def need(ok,why):
    if not ok:raise PermissionError('PV1_V4_NATIVE_DEV_'+why)


def validate(*,recompute_inputs,queue_proof,site_bytes,effective_queue_bytes,
             user,hostname,job_id,grant_bytes,B_closure_bytes=None,independent_job_bytes=None,
             phase_started_epoch=None):
    """Recompute from raw actual inputs, never accept an owner's PASS flag."""
    i=recompute_inputs;now=i['now_epoch'];r,rs=_read(i['request_bytes']);m,ms=_read(i['manifest_bytes'])
    g,gs=_read(i['global_bytes']);l,ls=_read(i['local_bytes']);scope.grant(grant_bytes);adopted=scope.validate_pair(g,l)
    need(adopted['adopted_epoch']<=now,'ACTUAL_SCOPE_ALREADY_ADOPTED_BEFORE_NATIVE_PHASE')
    facts=e.bootstrap_identity(m,r,g,run_uuid=r['run_uuid'],manifest_sha256=ms,request_sha256=rs,
        now=now,user=user,hostname=hostname,job_id=job_id,queue_proof=queue_proof)
    rows=queue_rows(queue_proof,now_epoch=now);_,outside,unknown=classify(g,rows)
    need(not unknown,'NO_UNKNOWN_ACCOUNT_OR_CAMPAIGN_JOB')
    independent=checked_outside(rows,outside,independent_job_bytes)
    proof=a.recompute(**i,purpose='NATIVE_PREMODEL',queue_proof=queue_proof)
    need(proof['all_remaining_costs_fit'] and proof['current_stage_clock_fits'],
         'ALL_CURRENT_STAGE_COSTS_CLOCK_AND_ORIGINAL_PROTECTION')
    site,ss=_read(site_bytes);q,qs=_read(effective_queue_bytes)
    phase=facts['allocation_start_epoch'] if phase_started_epoch is None else phase_started_epoch
    need(type(phase) in (int,float) and facts['allocation_start_epoch']<=phase<=now
         and site.get('schema')=='PI0294_V4_NATIVE_RUNNING_SITE_ATTESTATION_V1'
         and site.get('source')=='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE'
         and all(site.get(k)==v for k,v in dict(run_uuid=r['run_uuid'],campaign_run_uuid=r['campaign_run_uuid'],
             job_id=job_id,request_id=r['request_id'],submission_nonce=r['submission_nonce'],request_sha256=rs,
             source_sha=r['source_sha'],phase_started_epoch=phase).items())
         and site.get('observed_epoch',0)>=phase,'SAME_RUNNING_REQUEST_AND_CURRENT_PHASE_NATIVE_PROJECT_BYTES')
    _fresh(site['observed_epoch'],now,120)
    snapshot,_=_read(i['snapshot_bytes'])
    need(snapshot['observed_epoch']>=phase,'NO_PREQUEUE_OR_PREVIOUS_PHASE_RESOURCE_SNAPSHOT')
    project=site['project'];today=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).date()
    wall=r['walltime_s']+60
    need(project.get('name')==r['project'] and project.get('active') is True
         and user in project.get('users',[]) and 'stdct-mgmt-02' in project.get('cluster',[])
         and date.fromisoformat(project['start_date'])<=today<=date.fromisoformat(project['end_date'])
         and _number(site['balance_GPU_hours'])>=wall/3600
         and _number(site['balance_CPU_hours'])>=facts['native_allocated_ncpus']*wall/3600,
         'REAL_PROJECT_MEMBERSHIP_DATE_AND_ACTUAL_ALLOCATED_CAPACITY_BALANCE')
    need(q.get('source')=='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS'
         and q.get('requested_queue')=='auto' and q.get('destination')=='batch_gpu'
         and _number(q['maximum_wall_s'],integer=True)>=r['walltime_s']
         and m['execution']['sealed_inputs'].get(r['queue_limit_relative'])==qs
         and r['sealed_inputs'].get(r['queue_limit_relative'])==qs
         and r['queue_limit_sha256']==qs,'SEALED_ACTUAL_EFFECTIVE_QUEUE_LIMIT')
    _fresh(q['observed_epoch'],r['created_epoch'],120)
    quota=site['quota'];_fresh(quota['observed_epoch'],now,900)
    dim=proof['dimensions']['Hopper_storage_bytes']
    required=dim['reserved']+dim['remaining_S2']-dim['exact_current_credit']+dim['matrix']+dim['final']
    need(quota.get('user')==user and quota.get('scope')=='OWN_PROJECT_AND_FILESYSTEM_MINIMUM'
         and _number(quota['available_bytes'],integer=True)>=required,
         'ACTUAL_AVAILABLE_QUOTA_ALL_FUTURE_OUTPUT_AND_FINAL')
    reuse=None
    if r['evaluation_candidate']=='B_COVERAGE':
        reuse=scope.require_B_reuse(g,l,checkpoint_sha256=r['checkpoint_manifest_sha256'],closure_bytes=B_closure_bytes)
        need(r['checkpoint_reuse_proof_sha256']==scope.B_CLOSURE,'EXACT_SAVED_B4096_CLOSURE_REQUEST_BINDING')
    return dict(facts,status=e.STATUS,platform_evaluation_version=e.VERSION,run_uuid=r['run_uuid'],
        campaign_run_uuid=r['campaign_run_uuid'],request_sha256=rs,manifest_sha256=ms,
        source_sha=r['source_sha'],request_id=r['request_id'],submission_nonce=r['submission_nonce'],stage=r['stage'],
        evaluation_candidate=r['evaluation_candidate'],checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],
        statistics_sha256=r['statistics_sha256'],native_platform=e.PLATFORM,project=r['project'],submit_count=r['submit_count'],
        model_loaded=False,new_optimizer_updates=0,project_attestation_sha256=ss,resource_recomputation=proof,
        complete_queue_sha256=a.digest(queue_proof),independent_outside_jobs=independent,
        B4096_reuse_exception=reuse,native_prefix_safety_or_task_PASS=False,physics_authorized=False)


def inputs(root,request_path,*,now_epoch):
    """Existing registered capsule only; caller first runs neural_scope.validate."""
    from .future_admission import _file
    root=Path(root);i=a.file_inputs(root,request_path,now_epoch=now_epoch)
    m,_=_read(i['manifest_bytes']);r,_=_read(i['request_bytes'])
    def sealed(name):
        data=_file(root,name);digest=hashlib.sha256(data).hexdigest()
        need(m['execution']['sealed_inputs'].get(name)==r['sealed_inputs'].get(name)==digest,
             'SAME_PRESEALED_AUTHORITY_BYTES:'+name)
        return data
    return i,dict(grant_bytes=sealed(GRANT),effective_queue_bytes=sealed(r['queue_limit_relative']),
        B_closure_bytes=sealed(B_CLOSURE) if r['evaluation_candidate']=='B_COVERAGE' else None,
        independent_job_bytes=sealed_confirmation(root,m,r))


def admit(root,request_path):
    from .future_admission import _file
    q=collect_queue();i,extra=inputs(root,request_path,now_epoch=time.time())
    return validate(recompute_inputs=i,queue_proof=q,
        site_bytes=_file(root,str((Path(request_path).parent/'RUNNING_PROJECT_ATTESTATION.json').relative_to(root))),
        user=os.environ['USER'],hostname=socket.gethostname(),job_id=os.environ['PBS_JOBID'],**extra)



def verify_release(*,recompute_inputs,queue_proof,owner_bytes,allocation,phase_started_epoch,extra):
    """Actual current phase bytes, not a model-ready or prequeue PASS summary."""
    i=recompute_inputs;r,rs=_read(i['request_bytes']);owner,_=_read(owner_bytes)
    need(owner.get('schema')=='PI0294_V4_FRESH_DEV_QUERY_RELEASE_V1'
         and owner.get('owner')=='EXISTING_LOCAL_PC'
         and all(owner.get(k)==v for k,v in dict(job_id=allocation['job_id'],run_uuid=r['run_uuid'],
             request_sha256=rs,request_id=r['request_id'],submission_nonce=r['submission_nonce'],
             phase=PHASE,phase_started_epoch=phase_started_epoch).items()),'EXACT_CURRENT_PHASE_OWNER_RELEASE')
    _fresh(owner['observed_epoch'],i['now_epoch'],120)
    need(all(hashlib.sha256(i[key]).hexdigest()==owner[field] for key,field in
         (('global_bytes','global_ledger_sha256'),('local_bytes','LOCAL_ledger_sha256'),
          ('snapshot_bytes','snapshot_sha256'))),'OWNER_RAW_CURRENT_RESOURCE_BYTES')
    site=base64.b64decode(owner['running_site_base64'],validate=True)
    receipt=validate(recompute_inputs=i,queue_proof=queue_proof,site_bytes=site,
        user=os.environ['USER'],hostname=socket.gethostname(),job_id=os.environ['PBS_JOBID'],
        phase_started_epoch=phase_started_epoch,**extra)
    need(all(receipt[k]==allocation[k] for k in ('job_id','hostname','allocation_start_epoch','allocation_end_epoch')),
         'SAME_FIRST_ADMITTED_ALLOCATION_AFTER_QUALIFICATION_AND_LOAD')
    return receipt


def wait_native_release(root,directory,allocation,guard,*,now=time.time,sleep=time.sleep):
    """Original bounded phase wait, all endpoints including partial B0."""
    from scripts.pi0292.durable import write
    from .future_admission import _file
    from .visibility import read_visible
    root,directory=Path(root),Path(directory);request_path=directory/'REQUEST.json'
    rb=_file(root,str(request_path.relative_to(root)));r,rs=_read(rb);phase=now()
    ready=directory/'PV1_FUTURE_PHASE_READY.json'
    need(not ready.exists(),'NO_PHASE_REPLAY_OR_NATIVE_RESTART')
    write(ready,dict(schema='PI0294_V4_DEV_QUERY_PHASE_READY_V1',run_uuid=root.name,job_id=allocation['job_id'],
        request_id=r['request_id'],submission_nonce=r['submission_nonce'],request_sha256=rs,
        phase=PHASE,phase_started_epoch=phase,new_updates=0,new_queries=0));ready.chmod(0o400)
    target=directory/'PV1_FUTURE_OWNER_ATTESTATION.private.json'
    while now()<min(phase+175,allocation['allocation_end_epoch']-2700):
        guard()
        if read_visible(target) is not None:
            owner_raw=_file(root,str(target.relative_to(root)))
            i,extra=inputs(root,request_path,now_epoch=now())
            q=collect_queue();i['now_epoch']=now()
            receipt=verify_release(recompute_inputs=i,queue_proof=q,owner_bytes=owner_raw,
                allocation=allocation,phase_started_epoch=phase,extra=extra)
            out=directory/'PV1_FUTURE_NATIVE_RECOMPUTATION.json'
            need(not out.exists(),'NO_QUERY_RELEASE_REPLAY')
            write(out,receipt);out.chmod(0o400);return receipt
        sleep(1)
    raise TimeoutError('PV1_V4_FRESH_DEV_RELEASE_REQUIRED_NO_NEW_DEVELOPMENT_QUERY')
