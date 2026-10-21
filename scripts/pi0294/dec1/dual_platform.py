"""Explicit two-platform campaign guards; account jobs remain fully visible."""
from collections import Counter
import hashlib
import json
import re

from scripts.pi0294.pv1 import resources

GRANT_SHA='79f5c77f4a59c4f86c912a0663f13775ad2dcf9e01e27ea7bcb99c25987505ab'
RELEASE='4b6ef6e3e6e741e423d7561d67312008b9b68024'
PLATFORMS=('HOPPER_H200','VANDA_A40')
TERMINAL=('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB')


def grant(raw):
    value=json.loads(raw)
    if (hashlib.sha256(raw).hexdigest()!=GRANT_SHA
            or value.get('schema')!='VAP1_DUAL_PLATFORM_CONCURRENCY_ADOPTION_V1'
            or value.get('policy_release')!=RELEASE
            or value.get('maximum_running_scientific_GPUs')!=2
            or value.get('maximum_running_GPUs_by_platform')!=dict(HOPPER_H200=1,VANDA_A40=1)
            or value.get('owner')!='EXISTING_LOCAL_PC_ONLY'
            or value.get('original_T0_epoch')!=1792348196
            or value.get('original_hard_close_epoch')!=1792952996):
        raise PermissionError('DUAL_PLATFORM_EXACT_BOTH_SITE_USER_GRANT_BYTES')
    budget=value['budget']
    expected=dict(additional_GPU_hours=0,additional_PBS_submissions=0,global_GPU_hours=96,
        DEC1_GPU_hours=48,global_PBS_limit=24,DEC1_new_PBS_limit_including_old_attempts=8,
        global_final_GPU_protected_s=72000,original_T0_epoch=1792348196,original_hard_close_epoch=1792952996)
    if any(budget.get(k)!=v for k,v in expected.items()):
        raise PermissionError('DUAL_PLATFORM_NO_ADDITIVE_BUDGET_OR_CLOCK')
    return value


def observe(*,grant_bytes,site_jobs,ledger_items,user,now_epoch,observed_epochs):
    authority=grant(grant_bytes)
    if (set(site_jobs)!=set(PLATFORMS) or set(observed_epochs)!=set(PLATFORMS)
            or any(not 0<=now_epoch-observed_epochs[p]<=120 for p in PLATFORMS)):
        raise PermissionError('DUAL_PLATFORM_FRESH_COMPLETE_TWO_SITE_VIEWS')
    registered={item['job_id']:item for item in ledger_items if item.get('job_id')}
    if len(registered)!=sum(bool(i.get('job_id')) for i in ledger_items):
        raise PermissionError('DUAL_PLATFORM_UNIQUE_LEDGER_JOB_IDS')
    observed={};campaign=[];outside=[];confirmed=[];unknown=[]
    known=authority['known_user_confirmed_independent_job']
    for platform,jobs in site_jobs.items():
        suffix='.hopper-m-02' if platform=='HOPPER_H200' else '.stdct-mgmt-02'
        for job_id,job in jobs.items():
            if (job_id in observed or not re.fullmatch(r'[0-9]+'+re.escape(suffix),job_id)
                    or job.get('Job_Owner','').split('@')[0]!=user):
                raise PermissionError('DUAL_PLATFORM_EXACT_OWN_NATIVE_SERVER_JOB_ID')
            observed[job_id]=job
            row=dict(platform=platform,job_id=job_id,state=job.get('job_state'),
                project=job.get('project'),name=job.get('Job_Name'))
            item=registered.get(job_id)
            if item is not None:
                if (item.get('project')!=row['project'] or item.get('job_name')!=row['name']
                        or item.get('native_platform') not in (None,*PLATFORMS)
                        or (item.get('native_platform')=='VANDA_A40')!=(platform=='VANDA_A40')
                        or item['status'] in TERMINAL and row['state']!='F'):
                    raise PermissionError('DUAL_PLATFORM_LEDGER_NATIVE_IDENTITY_OR_TERMINAL_CONTRADICTION')
                if row['state']!='F':
                    if int(job.get('Resource_List',{}).get('ngpus',0))!=1:
                        raise PermissionError('DUAL_PLATFORM_ONE_GPU_PER_CAMPAIGN_JOB')
                    campaign.append(row)
                    if row['state'] not in ('Q','R'):unknown.append(row)
            elif (job_id==known['job_id'] and row['project']==known['project']
                    and row['name']==known['job_name']):confirmed.append(row)
            elif row['name'] and row['name'].startswith('pi0294_'):
                unknown.append(row)
            else:
                outside.append(dict(row,classification='OUTSIDE_REGISTERED_CAMPAIGN_NOT_ADOPTED',
                    user_confirmed_independent=False))
    for item in ledger_items:
        if item['status'] not in TERMINAL:
            if not item.get('job_id') or item['job_id'] not in observed:
                unknown.append(dict(job_id=item.get('job_id'),reason='ACTIVE_LEDGER_WITHOUT_NATIVE_JOB_PROOF'))
    by_platform=Counter(r['platform'] for r in campaign if r['state']=='R')
    pending=Counter(r['platform'] for r in campaign)
    if any(v>1 for v in pending.values()) or sum(by_platform.values())>2:
        raise PermissionError('DUAL_PLATFORM_ONE_ACTIVE_CHAIN_EACH_NOT_DUPLICATE_QUEUE_RACING')
    return dict(schema='VAP1_DUAL_PLATFORM_NATIVE_CAMPAIGN_VIEW_V1',observed_epoch=now_epoch,
        status='PASS_TWO_PLATFORM_CAMPAIGN_VIEW' if not unknown else 'STOP_UNKNOWN_CAMPAIGN_STATE',
        grant_sha256=GRANT_SHA,campaign_jobs=campaign,campaign_running_by_platform=dict(by_platform),
        user_confirmed_independent_jobs=confirmed,outside_registered_campaign_jobs=outside,
        account_owned_job_count=len(observed),unknown_campaign_jobs=unknown,
        account_view_filtered_or_forged_empty=False,other_jobs_modified=False)


def reserve_check(*,grant_bytes,view,used,reserved,extra,platform,job_wall_s,
                  now_epoch,site_wall_limit_s=8*3600,final=False):
    grant(grant_bytes)
    if (view.get('status')!='PASS_TWO_PLATFORM_CAMPAIGN_VIEW' or view.get('grant_sha256')!=GRANT_SHA
            or not 0<=now_epoch-view['observed_epoch']<=120):
        raise PermissionError('DUAL_PLATFORM_KNOWN_CAMPAIGN_BEFORE_NEW_RESERVATION')
    if (platform not in PLATFORMS or extra.get('PBS_submissions')!=1
            or type(job_wall_s) is not int or type(site_wall_limit_s) is not int
            or not 0<job_wall_s<=min(8*3600,site_wall_limit_s)):
        raise PermissionError('DUAL_PLATFORM_ONE_TYPED_BOUNDED_JOB_PER_ATOMIC_RESERVATION')
    jobs=view['campaign_jobs']
    if (len(jobs)>=2 or any(j['platform']==platform for j in jobs)
            or any(j['state'] not in ('Q','R') for j in jobs)):
        raise PermissionError('DUAL_PLATFORM_NO_DUPLICATE_QUEUED_OR_RUNNING_CHAIN')
    # The new explicit concurrency check replaces only the legacy single-GPU
    # predicate. All original monetary/resource caps and final reserves remain.
    if sum(view['campaign_running_by_platform'].values())>2:
        raise PermissionError('DUAL_PLATFORM_RUNNING_LIMIT')
    proof=resources.admit_resources(used,reserved,extra,final=final,active_GPUs=0,
        requested_GPUs=1,job_wall_s=job_wall_s,site_wall_limit_s=site_wall_limit_s)
    return dict(proof,status='PASS_DUAL_PLATFORM_GLOBAL_RESOURCE_CHECK_NOT_NATIVE_OR_DEC1_ADMISSION',
        concurrent_GPU_limit=2,per_platform_GPU_limit=1,requested_platform=platform,
        grant_sha256=GRANT_SHA,DEC1_subcaps_and_native_permission_checked=False)
