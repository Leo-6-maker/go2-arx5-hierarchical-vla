"""Read-only Vanda DEV identity seam; not resource or model admission.

The future cold allocator must still prove the complete resource envelope,
registered source, project balance and quota. A typed identity cannot grant
those permissions or turn a training allocation into an evaluation job.
"""
import json
import math
import re
import uuid
from pathlib import Path

from scripts.pi0294.dec1.a40_allocation import select_profile
from scripts.pi0294.dec1.authority import PARENT, R0
from scripts.pi0294.dec1.vanda_campaign import collect_queue, queue_rows
from scripts.pi0294.e0_scope import pbs_epoch
from .allocation import native_request_binding
from .decision import AUTHORITY
from .dataset import POPULATION

VERSION='PV1_VANDA_A40_SOURCE_DEV_V1'
STATUS='PASS_PV1_OWN_NATIVE_A40_SOURCE_DEV_ALLOCATION'
ROLE='PI0294_PV1_SHARED_POLICY_VANDA_EVALUATION_V1'
PROFILE='vanda-neural-evaluation'
PLATFORM='VANDA_A40'
T0=1792348196
STOP_NEW=T0+144*3600
STOP_PHYSICS=T0+160*3600
HARD_CLOSE=T0+168*3600
ENDPOINTS=dict(B0_CORR=PARENT,
    A_CONTROL='e0b9d8cb67efc82ae8daa67d71f95df8fb5c607e0f75e26d58cf01412a3e807a',
    B_COVERAGE='be7ab71dee0fb68e17efae06a04ff8dde9cd776ff429c1d6c408ae01323a9c19')


def need(ok,reason):
    if not ok:raise PermissionError('PV1_A40_DEV_'+reason)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def cold_profile(manifest):
    """Recognize the explicit evaluation role without any runtime permission."""
    m=manifest
    typed=('platform_evaluation_version' in m or m.get('scope')==ROLE
        or m.get('execution',{}).get('profile')==PROFILE)
    if not typed:return False
    need(m.get('platform_evaluation_version')==VERSION and 'platform_training_version' not in m
         and m.get('scope')==ROLE and m.get('execution',{}).get('profile')==PROFILE,
         'NO_PARTIAL_COLD_EVALUATION_ROLE_OR_PLATFORM_RELABEL')
    return True


def cold_plan(root,manifest,stage):
    """Join actual sealed DEV metadata and the saved endpoint, without loading it.

    The original stage_work and complete checkpoint loader remain mandatory.
    This adds no B reuse exception, project proof, budget, or execution release.
    """
    from scripts.pi0292.durable import read
    from scripts.pi0294.native_dataset import sha
    from .neural_scope import inside
    from .evaluation_plan import PLAN,POPULATION_FILE,development
    root=Path(root);m=manifest
    need(cold_profile(m),'EXPLICIT_COLD_EVALUATION_PROFILE')
    def sealed(name):
        p=inside(root,name)
        need(p.is_file() and not p.stat().st_mode&0o222
             and m['execution']['sealed_inputs'].get(name)==sha(p),'PRESEALED_DEV_INPUT:'+name)
        return read(p)
    need(m['limits'].get('PV1_T0_epoch')==T0 and m['limits'].get('PV1_hard_close_epoch')==HARD_CLOSE,
         'FIXED_ORIGINAL_COLD_WINDOW')
    plan=sealed(PLAN);population=sealed(POPULATION_FILE);candidate=plan.get('candidate')
    need(sha(root/POPULATION_FILE)==POPULATION,'ORIGINAL_FROZEN_POPULATION_NOT_A_NEW_DEV_PANEL')
    need(candidate in ENDPOINTS,'ONLY_EXISTING_B0_A_B_NO_NEW_MODEL_OR_REPLICA')
    expected=development(population['groups'],route=stage['route'],candidate=candidate,
        new_bank_manifest_sha256=stage['new_bank_manifest_sha256'])
    need(plan==expected,'EXACT_ORIGINAL_DEV48_PLAN_NO_POST_OUTCOME_SUBSET')
    # Prior outcomes keep all valid successes and valid negatives and are also
    # checked against each full panel by the existing stage_work/serve path.
    sealed('plans/PRIOR_EVALUATION_OUTCOMES.json')
    if candidate!='B0_CORR':
        checkpoint=sealed('plans/EVALUATION_CHECKPOINT.private.json')
        sealed('plans/INITIAL_CORR_PARAMETER_INVENTORY.json')
        p=Path(checkpoint['checkpoint']);complete=p/'CHECKPOINT_COMPLETE.json'
        need(not p.is_symlink() and p.stat().st_uid==root.stat().st_uid
             and not complete.is_symlink() and complete.is_file()
             and checkpoint.get('candidate')==candidate and checkpoint.get('global_step')==4096
             and checkpoint.get('checkpoint_manifest_sha256')==sha(complete)==ENDPOINTS[candidate],
             'EXACT_OWN_SAVED4096_ENDPOINT_NOT_ANOTHER_A_OR_B')
        saved=read(complete)
        need(saved.get('global_step')==4096 and saved.get('full_resume') is True,
             'ACTUAL_COMPLETE_SAVED_STATE_METADATA')
    return dict(candidate=candidate,stage='B0_DEV' if candidate=='B0_CORR' else 'P4_DEV',
        checkpoint_manifest_sha256=ENDPOINTS[candidate],statistics_sha256=R0,
        resource_admission=False,model_loaded=False,physics_authorized=False)


def sealed_identity(manifest,request,allocation,*,run_uuid,manifest_sha256,request_sha256):
    """Validate cold semantic identity only; do not authorize a model load."""
    m,r,a=manifest,request,allocation
    need(cold_profile(m),'EXPLICIT_COLD_EVALUATION_PROFILE')
    need(a.get('status')==STATUS and a.get('platform_evaluation_version')==VERSION
         and r.get('platform_evaluation_version')==VERSION
         and m.get('platform_evaluation_version')==VERSION
         and all('platform_training_version' not in x for x in (m,r,a)),
         'DISTINCT_EVALUATION_NOT_TRAINING_OR_H200')
    need(m.get('schema')=='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1'
         and m.get('scope')==ROLE and m.get('execution',{}).get('profile')==PROFILE
         and m.get('authority',{}).get('protocol_ref')==AUTHORITY
         and r.get('schema')=='PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1'
         and r.get('authority_release')==AUTHORITY
         and r.get('campaign_run_uuid')=='20261018T182956Z-localpc-pv1'
         and run_uuid!=r['campaign_run_uuid']
         and all(x.get('run_uuid')==run_uuid for x in (m,r,a)),
         'EXACT_COLD_PROFILE_AUTHORITY_AND_CHILD')
    need(all(re.fullmatch('[0-9a-f]{'+str(n)+'}',str(v)) is not None for v,n in
         ((r.get('source_sha'),40),(manifest_sha256,64),(request_sha256,64)))
         and m.get('code',{}).get('vla_commit')==r['source_sha']
         and r.get('run_manifest_sha256')==a.get('manifest_sha256')==manifest_sha256
         and a.get('request_sha256')==request_sha256 and a.get('source_sha')==r['source_sha'],
         'SEALED_SOURCE_MANIFEST_REQUEST')
    for key in ('request_id','submission_nonce'):
        try:value=uuid.UUID(r[key])
        except (KeyError,TypeError,ValueError,AttributeError):need(False,'CANONICAL_UUID')
        need(value.version==4 and str(value)==r[key] and a.get(key)==r[key],'CANONICAL_UUID')
    candidate=r.get('evaluation_candidate')
    need(candidate in ENDPOINTS and r.get('stage')==('B0_DEV' if candidate=='B0_CORR' else 'P4_DEV')
         and a.get('stage')==r['stage'] and a.get('evaluation_candidate')==candidate
         and r.get('checkpoint_manifest_sha256')==a.get('checkpoint_manifest_sha256')==ENDPOINTS[candidate]
         and r.get('statistics_sha256')==a.get('statistics_sha256')==R0
         and r.get('parent_checkpoint_manifest_sha256')==PARENT
         and not r.get('training_block') and r.get('optimizer_updates')==0
         and type(r.get('optimizer_updates')) is int
         and r.get('Teacher_takeover') is False and r.get('normalizer_refit') is False
         and r.get('evaluation_data_for_training') is False and r.get('LOCKBOX_opened') is False
         and r.get('numerical_mode')=='DETERMINISTIC_CUDA_MATH_SDP_V1',
         'ORIGINAL_ENDPOINT_R0_DEV_ONLY_ZERO_OPTIMIZATION')
    need(r.get('native_platform')==a.get('native_platform')==PLATFORM
         and r.get('project')==a.get('project')=='CFP04-CF-007'
         and r.get('job_name')=='pi0294_pv1_a40_dev'
         and type(r.get('submit_count')) is int and r['submit_count']>0
         and a.get('submit_count')==r['submit_count']
         and r.get('active_cpu_limit')==4 and type(r.get('active_cpu_limit')) is int
         and r.get('requested_ncpus')==4 and r.get('requested_memory_bytes')==32*1024**3,
         'EXACT_PROJECT_RESOURCE_IDENTITY')
    wall=r.get('walltime_s');created=r.get('created_epoch')
    need(type(wall) is int and 2760<wall<=28800 and finite(created) and T0<=created<STOP_NEW
         and r.get('original_T0_epoch')==T0 and r.get('hard_close_epoch')==HARD_CLOSE
         and m.get('limits',{}).get('PV1_T0_epoch')==T0
         and m['limits'].get('PV1_hard_close_epoch')==HARD_CLOSE
         and a.get('model_loaded') is False and type(a.get('new_optimizer_updates')) is int
         and a['new_optimizer_updates']==0,'ORIGINAL_CLOCK_AND_PREMODEL_ONLY')


def native_identity(request,allocation,*,now,user,hostname,job_id,queue_proof):
    """Join the complete actual own-account queue, not a caller PASS flag."""
    r,a=request,allocation
    need(finite(now) and T0<=now<STOP_PHYSICS and user=='e1192548'
         and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',str(job_id)) is not None
         and re.fullmatch(r'gn-a40-[0-9]+(?:\.[A-Za-z0-9.-]+)?',hostname,re.IGNORECASE) is not None,
         'CURRENT_VANDA_COMPUTE_HOST_ACCOUNT_AND_CLOCK')
    queue_rows(queue_proof,now_epoch=now)
    jobs=json.loads(queue_proof['qstat']['stdout'])['Jobs'] if queue_proof['qstat'] is not None else {}
    j=jobs.get(job_id)
    need(j is not None and j.get('job_state')=='R' and j.get('Job_Owner','').split('@')[0]==user
         and j.get('project')==r['project'] and j.get('Job_Name')==r['job_name']
         and j.get('queue')=='batch_gpu' and native_request_binding(j,r),'FRESH_OWN_RUNNING_REQUEST_NOT_RERUN')
    limits=j['Resource_List'];cpus,mem=select_profile(limits,r)
    text=limits.get('walltime','')
    need(re.fullmatch(r'[0-9]+:[0-5][0-9]:[0-5][0-9]',text) is not None,'EXACT_NATIVE_WALLTIME')
    wall=sum(int(x)*scale for x,scale in zip(text.split(':'),(3600,60,1)))
    started=pbs_epoch(j['stime']);end=min(started+wall,STOP_PHYSICS)
    nodes={x.split('/')[0].split('.')[0].lower() for x in j.get('exec_host','').split('+')}
    need(nodes=={hostname.split('.')[0].lower()} and wall==r['walltime_s']
         and finite(started) and r['created_epoch']<=started<=now<end
         and started+wall<=HARD_CLOSE and a.get('job_id')==job_id and a.get('hostname')==hostname
         and a.get('allocation_start_epoch')==started and a.get('allocation_end_epoch')==end
         and a.get('native_select')==limits['select'] and a.get('native_allocated_ncpus')==cpus
         and a.get('native_allocated_memory_bytes')==mem,'SAME_NATIVE_HOST_START_CAPACITY_AND_END')


def current_identity(manifest,request,allocation,*,run_uuid,manifest_sha256,request_sha256,
                     user,hostname,job_id):
    sealed_identity(manifest,request,allocation,run_uuid=run_uuid,
        manifest_sha256=manifest_sha256,request_sha256=request_sha256)
    proof=collect_queue(user=user)
    # Sample after the bounded reads; a stale pre-read timestamp is not freshness.
    import time
    native_identity(request,allocation,now=time.time(),user=user,hostname=hostname,
        job_id=job_id,queue_proof=proof)
