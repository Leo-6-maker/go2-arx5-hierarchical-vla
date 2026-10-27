"""Raw-input first512 admission through the original native queue and budget.

This pure validator is called by both the existing login owner and its own
compute entry. It creates no allocation/permission file, reservation or job.
Actual native collectors and registered-source context remain mandatory.
"""
from datetime import date,datetime,timedelta,timezone
import json,re
from . import minimal_execution as e,minimal_scope as scope,minimal_session_gate as tickets
from scripts.pi0294.dec1 import vanda_campaign as queue
from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
from scripts.pi0294.pv1.neural_ledger import check
from scripts.pi0294.e0_scope import pbs_epoch

STATUS='PASS_ARX5_MINIMAL_FIRST512_OWN_NATIVE_ALLOCATION_V1'
SITE='ARX5_MINIMAL_FIRST512_CURRENT_NATIVE_SITE_ATTESTATION_V1'
need=e.need


def fresh(value,now,upper,label):
    need(type(value)in(int,float)and 0<=now-value<=upper,'ADMISSION_FRESH_'+label)


def validate(*,global_bytes,local_bytes,request_bytes,manifest_bytes,job_id,hostname,user,
             native_job_bytes,queue_proof,site_bytes,effective_queue_bytes,
             independent_job_bytes,now_epoch,ready_bytes=None,effect_bytes=None):
    g,l,r,m=[json.loads(b)for b in(global_bytes,local_bytes,request_bytes,manifest_bytes)]
    check(g);marker=e.validate(g);e.request(r);rows=e.rows(g);need(len(rows)==1,'ADMISSION_ONE_ADOPTED_MAIN')
    row=rows[0];source=r['source_sha'];rs=e.sha(request_bytes)
    need(marker is not None and row['status']==e.STATES[1]and row['job_id']==job_id
        and row['request_sha256']==rs and row['source_sha']==source
        and row['run_manifest_sha256']==e.sha(manifest_bytes)==r['run_manifest_sha256']
        and m['scope']==scope.SCOPE and m['platform_training_version']==e.VERSION
        and m['execution']['profile']==e.PROFILE and m['code']['vla_commit']==source
        and m['run_uuid']==r['run_uuid']and user=='e1192548'
        and g['unknown_effects']==g['pending_updates']==[],
        'ADMISSION_ACTUAL_ADOPTED_REQUEST_MANIFEST_OWNER_AND_GLOBAL_FENCES')
    # The complete native queue remains intact. Human-confirmed independent
    # jobs are neither adopted nor stopped. Core concurrency has its separate
    # exact main/D25 binding in the original GLOBAL validator.
    native_rows=queue.queue_rows(queue_proof,now_epoch=now_epoch)
    own,outside,unknown=queue.classify(g,native_rows);need(not unknown,'ADMISSION_COMPLETE_UNFILTERED_QUEUE_NO_UNKNOWN')
    independent=checked_outside(native_rows,outside,independent_job_bytes)
    need(len(own)==1 and own[0]['job_id']==job_id and own[0]['state']=='R',
         'ADMISSION_ONLY_OWN_RUNNING_VANDA_MAIN')
    raw_job=json.loads(native_job_bytes);recorded=json.loads(queue_proof['qstat']['stdout'])['Jobs'][job_id]
    need(raw_job==recorded and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job_id)
        and hostname.split('.')[0].lower().startswith('gn-a40-')
        and raw_job['job_state']=='R'and raw_job['Job_Owner'].split('@')[0]==user
        and raw_job['project']==r['project']and raw_job['Job_Name']==r['job_name']
        and raw_job['queue']=='batch_gpu'and raw_job['Rerunable']in(False,'False')
        and int(raw_job.get('run_count',1))==1
        and raw_job['Variable_List']['PV1_REQUEST_ID']==r['request_id']
        and raw_job['Variable_List']['PV1_SUBMISSION_NONCE']==r['submission_nonce'],
        'ADMISSION_NATIVE_SAME_JOB_SOURCE_REQUEST_NONCE_AND_NO_RESTART')
    cpus,memory=e.select_profile(raw_job['Resource_List'],r)
    nodes={p.split('/')[0].split('.')[0]for p in raw_job['exec_host'].split('+')}
    started=pbs_epoch(raw_job['stime']);end=min(started+r['walltime_s'],r['deadline_epoch'])
    need(nodes=={hostname.split('.')[0]}and r['T_CD']<=started<=now_epoch<end-2700
        and started+r['reserved_H200_s']<r['stop_new_training_epoch'],
        'ADMISSION_REAL_NODE_START_AND_ORIGINAL_FULL_SAVE_WINDOW')
    q=json.loads(effective_queue_bytes);need(q.get('source')=='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS'
        and q.get('requested_queue')=='auto'and q.get('destination')=='batch_gpu'
        and type(q.get('maximum_wall_s'))is int and q['maximum_wall_s']>=r['walltime_s']
        and m['execution']['sealed_inputs'].get('plans/MINIMAL_EFFECTIVE_QUEUE_LIMITS.private.json')==e.sha(effective_queue_bytes),
        'ADMISSION_ACTUAL_SEALED_EFFECTIVE_QUEUE_LIMIT')
    fresh(q['observed_epoch'],r['created_epoch'],120,'PRE_SUBMISSION_QUEUE_LIMIT')
    site=json.loads(site_bytes);need(site.get('schema')==SITE
        and site.get('source')=='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE'
        and all(site.get(k)==v for k,v in dict(run_uuid=r['run_uuid'],request_id=r['request_id'],
            submission_nonce=r['submission_nonce'],request_sha256=rs,source_sha=source,job_id=job_id).items()),
        'ADMISSION_CURRENT_NATIVE_PROJECT_BYTES_NOT_CPU_FLAGS')
    fresh(site['observed_epoch'],now_epoch,120,'RUNNING_PROJECT')
    project=site['project'];today=datetime.fromtimestamp(now_epoch,timezone(timedelta(hours=8))).date()
    capacity=e.project_envelope(r)
    need(project.get('name')==r['project']and project.get('active')is True and user in project.get('users',[])
        and 'stdct-mgmt-02'in project.get('cluster',[])
        and date.fromisoformat(project['start_date'])<=today<=date.fromisoformat(project['end_date'])
        and e.original.number(site['balance_GPU_hours'])>=capacity['GPU_hours']
        and e.original.number(site['balance_CPU_hours'])>=capacity['CPU_hours'],
        'ADMISSION_PROJECT_DATE_MEMBERSHIP_AND_FULL_NATIVE_CAPACITY_BALANCE')
    # budget_LOCAL projects only the original already-paid control if its
    # actual canonical mirror has not yet caught up. It never writes a mirror
    # or claims historical Core data are fresh. The existing forecast performs
    # this projection exactly once and retains every original future job.
    full=scope.first512_forecast(global_bytes,local_bytes)
    need(full['cap_changes_required']=={},'ADMISSION_COMPLETE_FUTURE_MATRIX_AND_FINAL:'+str(full['cap_changes_required']))
    quota=site['quota'];fresh(quota['observed_epoch'],now_epoch,900,'PROJECT_AND_FILESYSTEM_QUOTA')
    dimension=full['dimensions']['Hopper_storage_bytes'];required=dimension['projected']-g['new_storage_current_bytes']
    need(quota.get('user')==user and quota.get('scope')=='OWN_PROJECT_AND_FILESYSTEM_MINIMUM'
        and type(quota.get('available_bytes'))is int and quota['available_bytes']>=required,
        'ADMISSION_ACTUAL_PROJECT_AND_FILESYSTEM_ALL_REMAINING_OUTPUT')
    phase=None
    if ready_bytes is not None:
        need(type(effect_bytes)is bytes,'ADMISSION_PHASE_ACTUAL_CURRENT_LOCAL_EFFECT_BYTES')
        ready=json.loads(ready_bytes);effect=json.loads(effect_bytes)
        phase=tickets.check_ready(ready,r,request_sha256=rs,job_id=job_id,effect_value=effect)
        need(started<=ready['observed_epoch']<=site['observed_epoch']<=now_epoch,
             'ADMISSION_NO_PRE_PHASE_SITE_ATTESTATION')
        fresh(ready['observed_epoch'],now_epoch,180,'PHASE_READY')
    else:need(effect_bytes is None,'ADMISSION_NO_ORPHAN_EFFECT_CAPTURE')
    return dict(status=STATUS,job_id=job_id,hostname=hostname,request_id=r['request_id'],
        submission_nonce=r['submission_nonce'],request_sha256=rs,source_sha=source,
        run_manifest_sha256=e.sha(manifest_bytes),allocation_start_epoch=started,allocation_end_epoch=end,
        observed_epoch=now_epoch,native_allocated_ncpus=cpus,native_allocated_memory_bytes=memory,
        active_CPU_limit=8,CPU_accounting_upper_s=r['CPU_upper_s'],native_select=raw_job['Resource_List']['select'],
        complete_queue_sha256=e.sha(e.raw(queue_proof)),site_attestation_sha256=e.sha(site_bytes),
        GLOBAL_sha256=e.sha(global_bytes),LOCAL_capture_sha256=e.sha(local_bytes),
        complete_queue_verified=True,independent_jobs=independent,project_eligible=True,
        actual_project_quota_covers_remaining_output=True,whole_remaining_package_shortfalls={},
        global_unknown_effects=[],global_pending_updates=[],native_own_job_id=job_id,
        phase=phase,full_remaining_proof=full,model_loaded=False,new_optimizer_updates=0,
        new_physics=0,permission_file_created=False)
