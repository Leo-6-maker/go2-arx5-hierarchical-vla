"""Raw-byte C/D full-future join on the actual allocated Vanda site.

The unavailable Core LOCAL is explicitly historical and conservative. This
function never creates a substitute LOCAL, submits PBS or grants physical
prefix safety. The original live source gates and 600s/80cycle proof remain.
"""
from datetime import date,datetime,timedelta,timezone
import json

from . import mix_accounting as a, mix_vanda_cohost as cohost, successor_cd as scope
from . import mix_vanda_eval_accounting as c, mix_vanda_eval_identity as identity
from . import mix_vanda_first as first, mix_evaluation as plan
from .dataset import POPULATION
from scripts.pi0294.dec1.vanda_campaign import queue_rows,classify
from .evaluation_account_jobs import checked_outside

SITE='V5_CD_VANDA_COLOCATED_RUNNING_SITE_ATTESTATION_V1'


def need(ok,why):
    if not ok:raise PermissionError('V5_VANDA_EVAL_ADMISSION_'+why)


def load(raw):
    need(type(raw) is bytes,'RAW_BYTES_NOT_PREPARED_PASS');return json.loads(raw)


def recompute(*,manifest_bytes,request_bytes,global_bytes,historical_LOCAL_bytes,
              training_capture_bytes,population_bytes,plan_bytes,prior_bytes,
              native_facts_bytes,queue_bytes,site_bytes,queue_limit_bytes,
              independent_job_bytes,phase_started_epoch,now_epoch):
    m,r,g,local=map(load,(manifest_bytes,request_bytes,global_bytes,historical_LOCAL_bytes))
    identity.request_identity(m,r,a.sha(manifest_bytes));c.checked(g)
    need(cohost.validated(g) is not None and a.sha(historical_LOCAL_bytes)==first.LOCAL_SHA,
         'ACTUAL_COLOCATED_AUTHORITY_HISTORICAL_LOCAL_NOT_FRESH')
    marker=scope.validated(g);now=a.number(now_epoch);phase=a.number(phase_started_epoch)
    need(r['T_CD']==marker['T_CD'] and r['deadline_epoch']==marker['deadline_epoch']
         and marker['T_CD']<=phase<=now<marker['deadline_epoch']-2700,'ACTUAL_ORIGINAL_SCOPE_AND_CURRENT_PHASE')
    row=c.item(g,r['request_id'])
    need(row['status'] in c.STATES[1:3] and row['request_sha256']==a.sha(request_bytes)
         and row['run_manifest_sha256']==a.sha(manifest_bytes) and row['child_run_uuid']==r['run_uuid']
         and all(row.get(k)==r.get(k) for k in ('source_sha','submission_nonce','evaluation_candidate',
            'checkpoint_manifest_sha256','statistics_sha256','training_handoff_sha256','planned_panel_sha256')),
         'ACTUAL_BOUND_SAME_CURRENT_REQUEST')
    sealed=m['execution']['sealed_inputs']
    for name,b in [('plans/V5_CD_CLOSED_TRAINING_CAPTURE.private.json',training_capture_bytes),
                   ('plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json',population_bytes),
                   ('plans/SOURCE_EVALUATION_PLAN.json',plan_bytes),('plans/PRIOR_EVALUATION_OUTCOMES.json',prior_bytes)]:
        need(sealed.get(name)==a.sha(b),'EXACT_SEALED_'+name)
    endpoint=identity.bind_closed_endpoint(global_bytes,r,load(training_capture_bytes))
    need(a.sha(population_bytes)==POPULATION,'FULL_ORIGINAL_POPULATION')
    expected=plan.candidate_plan(load(population_bytes)['groups'],r['evaluation_candidate'])
    need(load(plan_bytes)==expected and r['planned_panel_sha256']==a.sha(plan_bytes),'EXACT_ALL_ORIGINAL_CONDITIONS')
    from .evaluation_plan import remaining_panel
    need(len(remaining_panel(load(prior_bytes),expected['panels'][r['evaluation_candidate']+'-DEV-N294101']))==24,
         'VALID_FAILURES_NEVER_REPEATED')
    facts=load(native_facts_bytes);job=facts['actual_PBS']
    allocation=identity.native_allocation(job,r,job_id=row['job_id'],hostname=facts['hostname'],now_epoch=now)
    need(facts.get('schema')=='V5_CD_VANDA_NEURAL_PROCESS_STARTED_V1' and facts['uid']==22182
         and facts['job_id']==facts['native_handle']==row['job_id']
         and facts['actual_PBS_sha256']==a.sha(a.raw(job))
         and facts['request_id']==r['request_id'] and facts['request_sha256']==a.sha(request_bytes)
         and facts['run_manifest_sha256']==a.sha(manifest_bytes) and facts['source_sha']==r['source_sha']
         and 0<=now-facts['observed_epoch']<=120,'FRESH_REAL_NEURAL_KERNEL_AND_NATIVE_JOB')
    bound=row.get('neural_process')
    need(type(bound) is dict and all(facts.get(k)==bound.get(k) for k in
         ('job_id','pid','proc_start','boot_id','cgroup','started_epoch','GPU_uuid','hostname','uid')),
         'ACTUALLY_BOUND_KERNEL_PROCESS_AND_DEVICE_NOT_FIXTURE_PID')
    queue=load(queue_bytes);rows=queue_rows(queue,now_epoch=now);owned,outside,unknown=classify(g,rows)
    need(not unknown and len(owned)==1 and owned[0]['job_id']==row['job_id'] and owned[0]['state']=='R',
         'COMPLETE_REAL_ACCOUNT_QUEUE_ONE_TASK_JOB')
    outside_proof=checked_outside(rows,outside,independent_job_bytes)
    site=load(site_bytes)
    need(site.get('schema')==SITE and site.get('source')=='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE'
         and all(site.get(k)==v for k,v in dict(run_uuid=r['run_uuid'],job_id=row['job_id'],
             request_id=r['request_id'],request_sha256=a.sha(request_bytes),source_sha=r['source_sha'],
             submission_nonce=r['submission_nonce'],phase_started_epoch=phase).items())
         and phase<=site['observed_epoch']<=now and now-site['observed_epoch']<=120,
         'FRESH_AFTER_PHASE_ORIGINAL_NATIVE_PROJECT_ATTESTATION')
    project=site['project'];today=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).date()
    native_cpus=allocation['native_capacity']['ncpus']
    need(project.get('name')=='CFP04-CF-007' and project.get('active') is True
         and 'e1192548' in project.get('users',[]) and 'stdct-mgmt-02' in project.get('cluster',[])
         and date.fromisoformat(project['start_date'])<=today<=date.fromisoformat(project['end_date'])
         and a.number(site['balance_GPU_hours'])>=28860/3600
         and a.number(site['balance_CPU_hours'])>=native_cpus*28860/3600,
         'REAL_PROJECT_AND_NATIVE_CAPACITY_BALANCE_NOT_ONLY_ACTIVE_CPU_LIMIT')
    effective=load(queue_limit_bytes)
    need(effective.get('source')=='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS'
         and effective.get('requested_queue')=='auto' and effective.get('destination')=='batch_gpu'
         and a.number(effective['maximum_wall_s'])>=28800
         and m['execution']['sealed_inputs'].get(r['queue_limit_relative'])==a.sha(queue_limit_bytes)
         and r['queue_limit_sha256']==a.sha(queue_limit_bytes)
         and 0<=r['created_epoch']-effective['observed_epoch']<=120,'SEALED_ACTUAL_EFFECTIVE_QUEUE_LIMIT')
    full=scope.recompute(g,local,Core_storage_upper_bytes=first.CORE_STORAGE,local_storage_upper_bytes=first.LOCAL_STORAGE)
    need(full['shortfalls']=={},'ALL_ORIGINAL_REMAINING_DIMENSIONS_MATRIX_AND_FINAL')
    quota=site['quota'];required=full['dimensions']['Hopper_storage_bytes']['projected']-g['new_storage_current_bytes']
    need(quota.get('user')=='e1192548' and quota.get('scope')=='OWN_PROJECT_AND_FILESYSTEM_MINIMUM'
         and 0<=now-quota['observed_epoch']<=900 and a.number(quota['available_bytes'],integer=True)>=required,
         'ACTUAL_AVAILABLE_PROJECT_AND_FILESYSTEM_ALL_FUTURE_OUTPUT')
    remaining_panels=2 if r['evaluation_candidate']=='C50' else 1
    need(allocation['allocation_start_epoch']+remaining_panels*28860<marker['deadline_epoch'],
         'ALL_REMAINING_PANELS_AND_CLOSEOUT_FIT_UNRESET_WINDOW')
    allocation.update(status=identity.STATUS,platform_evaluation_version=identity.VERSION,
        run_uuid=r['run_uuid'],request_id=r['request_id'],submission_nonce=r['submission_nonce'],
        request_sha256=a.sha(request_bytes),manifest_sha256=a.sha(manifest_bytes),source_sha=r['source_sha'],
        project=r['project'],evaluation_candidate=r['evaluation_candidate'],checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],
        statistics_sha256=a.STATS,GPU_uuid=facts['GPU_uuid'],visible=facts['visible'],native_process_sha256=a.sha(native_facts_bytes),
        allocation_number=1,submit_count=r['submit_count'],native_model_permission=False,physics_authorized=False)
    return dict(schema='V5_CD_VANDA_COLOCATED_FULL_FUTURE_NATIVE_RESOURCE_PROOF_V1',
        GLOBAL_sha256=a.sha(global_bytes),historical_LOCAL_sha256=a.sha(historical_LOCAL_bytes),
        canonical_LOCAL_sync='PENDING',historical_LOCAL_is_native_authority=False,
        request_sha256=a.sha(request_bytes),training_handoff=endpoint,all_remaining_costs_fit=True,
        shortfalls={},dimensions=full['dimensions'],allocation=allocation,
        native_process_sha256=a.sha(native_facts_bytes),queue_sha256=a.sha(queue_bytes),
        site_sha256=a.sha(site_bytes),independent_outside_jobs=outside_proof,
        model_permission=False,physics_authorized=False)
