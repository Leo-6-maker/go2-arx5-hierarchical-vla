"""PV1 native admission: projects fund one shared, bounded H200 campaign."""
from datetime import datetime,timedelta,timezone
import math

from scripts.pi0294.e0_scope import pbs_epoch
from .decision import AUTHORITY
from .resources import admit_resources,project_eligible

PROJECT_SOURCE='HOPPER_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE'


def single_h200_select(resources,cpus):
    parts=resources.get('select','').split(':')
    if not parts or parts[0]!='1' or '+' in resources.get('select',''):return False
    fields={}
    for part in parts[1:]:
        pair=part.split('=',1)
        if len(pair)!=2 or pair[0] in fields:return False
        fields[pair[0]]=pair[1]
    return (type(cpus) is int and cpus>=4 and resources.get('ncpus')==cpus
        and fields.get('gpu_model')=='H200' and fields.get('ngpus')=='1' and fields.get('ncpus')==str(cpus))


def validate_native(*,now,user,hostname,job_id,jobs,quota,adoption,request,ledger,project_proof,queue_proof,
                    root_run_uuid=None):
    """Only native metadata; this function never submits or loads a model."""
    if (adoption['authority_release']!=AUTHORITY
            or adoption['stop_new_jobs_epoch']!=adoption['T0_epoch']+144*3600
            or adoption['stop_new_training_epoch']!=adoption['T0_epoch']+156*3600
            or adoption['stop_new_physics_epoch']!=adoption['T0_epoch']+160*3600
            or adoption['hard_closeout_epoch']!=adoption['T0_epoch']+168*3600):
        raise PermissionError('PV1_ORIGINAL_SEVEN_DAY_CLOCK_NOT_RESET')
    stage=request.get('stage')
    if stage not in ('B0_DEV','P2_PRIMARY','P2_TRAIN_PROBE','P3_PAIRED','P4_REPLICA','P4_DEV','P5_FINAL'):
        raise PermissionError('PV1_ONLY_PREDECLARED_NEURAL_PHASES')
    final=stage=='P5_FINAL'
    cutoff=adoption['stop_new_training_epoch'] if stage in ('P2_PRIMARY','P3_PAIRED','P4_REPLICA') else adoption['stop_new_physics_epoch']
    expected_run=adoption['run_uuid'] if root_run_uuid is None else root_run_uuid
    if (request['schema']!='PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1'
            or request['run_uuid']!=expected_run
            or request.get('campaign_run_uuid',request['run_uuid'])!=adoption['run_uuid']
            or request['authority_release']!=AUTHORITY
            or not adoption['T0_epoch']<=request['created_epoch']<adoption['stop_new_jobs_epoch']
            or not adoption['T0_epoch']<=now<cutoff
            or request.get('data_bank_status')!='SEALED_FULL_SUCCESS_OFFICIAL_CODEC_ADMITTED'
            or len(request.get('new_bank_manifest_sha256',''))!=64
            or request.get('numerical_mode')!='DETERMINISTIC_CUDA_MATH_SDP_V1'):
        raise PermissionError('PV1_AUTHORIZED_CLOCK_AND_QUALIFIED_FROZEN_DATA_BEFORE_NEURAL_ALLOCATION')
    wall=request['walltime_s']
    if (type(wall) is not int or not 2760<wall<=8*3600
            or queue_proof.get('source')!='NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS'
            or queue_proof['queue']!='auto' or queue_proof['destination']!='small'
            or not 0<=request['created_epoch']-queue_proof['observed_epoch']<=120
            or not 0<wall<=queue_proof['maximum_wall_s']):
        raise PermissionError('PV1_ORIGINAL_AUTO_AND_SMALLER_NATIVE_QUEUE_WALLTIME')
    submission=[s for s in ledger['PBS_submissions'] if s['request_id']==request['request_id']]
    if (len(submission)!=1 or submission[0]['job_id']!=job_id
            or submission[0]['project']!=request['project'] or submission[0]['reserved_H200_s']!=wall
            or ledger['reserved']['H200_s']<wall or not 1<=len(ledger['PBS_submissions'])<=24
            or ledger['used']['PBS_submissions']!=len(ledger['PBS_submissions'])):
        raise PermissionError('PV1_ONE_EXACT_NATIVE_INTENT_AND_FULL_ALLOCATION_RESERVATION')
    job=jobs.get(job_id)
    if (not job or job['job_state']!='R' or job['Job_Owner'].split('@')[0]!=user
            or job.get('project')!=request['project'] or int(job['Resource_List']['ngpus'])!=1
            or not single_h200_select(job['Resource_List'],request['ncpus'])
            or job.get('Job_Name')!=request['job_name'] or not request['job_name'].startswith('pi0294_pv1_')):
        raise PermissionError('PV1_OWN_RUNNING_SELECTED_PROJECT_SINGLE_GPU_PBS_ONLY')
    registered={s['job_id'] for s in ledger['PBS_submissions'] if s.get('job_id')}
    if any(k!=job_id and j['job_state']=='R'
           and (k in registered or j.get('Job_Name','').startswith('pi0294_pv1_')) for k,j in jobs.items()):
        raise PermissionError('PV1_ONE_CONCURRENT_H200_ACROSS_ALL_PROJECTS')
    nodes={n.split('/')[0].split('.')[0] for n in job['exec_host'].split('+')}
    native_wall=sum(int(x)*scale for x,scale in zip(job['Resource_List']['walltime'].split(':'),(3600,60,1)))
    started=pbs_epoch(job['stime'])
    if (hostname.split('.')[0] not in nodes or hostname.split('.')[0].startswith(('hopper-l','hopper-m'))
            or native_wall!=wall or not adoption['T0_epoch']<=started<=now
            or started+wall>adoption['hard_closeout_epoch']):
        raise PermissionError('PV1_ACTUAL_COMPUTE_HOST_START_AND_WALLTIME')
    p=project_proof
    if (p.get('schema')!='PI0294_PV1_RUNNING_JOB_PROJECT_ATTESTATION_V1'
            or p.get('source')!=PROJECT_SOURCE or p['run_uuid']!=expected_run
            or p['request_id']!=request['request_id'] or p['job_id']!=job_id
            or p['submission_nonce']!=request['submission_nonce']
            or not started<=p['observed_epoch']<=now or not 0<=now-p['observed_epoch']<=1800):
        raise PermissionError('PV1_FRESH_JOB_BOUND_OWNER_PROJECT_PROOF_AFTER_QUEUE_WAIT')
    project=p['project']
    today=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).date().isoformat()
    if (project['name']!=request['project'] or user not in project['users']
            or 'hopper-m-02' not in project['cluster']
            or not project_eligible(dict(project,member=user in project['users'],balance_su=p['balance_su']),today)):
        raise PermissionError('PV1_SELECTED_PROJECT_DATE_MEMBER_CLUSTER_AND_POSITIVE_BALANCE')
    extra=request['additional_output_upper_bytes']
    reserve=0 if final else 40*1024**3
    if (type(extra) is not int or extra<=0 or extra>250*1024**3
            or quota['user']!=user or not 0<=now-quota['report_epoch']<=900
            or any(type(quota[k]) is not int or quota[k]<0 for k in ('used_bytes','hard_bytes'))
            or quota['hard_bytes']-quota['used_bytes']<extra+reserve
            or not math.isfinite(ledger['new_storage_peak_bytes'])
            or ledger['new_storage_peak_bytes']+extra+reserve>250*1024**3):
        raise PermissionError('PV1_REAL_PERSONAL_QUOTA_AND_FINAL_STORAGE_RESERVE')
    admit_resources(ledger['used'],ledger['reserved'],{},final=final,active_GPUs=1)
    return dict(status='PASS_PV1_OWN_NATIVE_H200_ALLOCATION',job_id=job_id,project=request['project'],
        hostname=hostname,observed_epoch=now,allocation_start_epoch=started,
        allocation_end_epoch=min(started+wall,cutoff),resource_clock_reset=False,
        native_select=job['Resource_List']['select'],native_ncpus=request['ncpus'],
        unrelated_owned_jobs_cancelled=False,model_loaded=False)
