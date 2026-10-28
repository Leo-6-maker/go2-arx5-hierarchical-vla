"""Explicit C/D cohost PBS identity; old A/B and Core identities stay strict.

These joins consume actual allocation evidence. They cannot manufacture a job,
resource grant, same-job qualification, model release or physical permission.
"""
import json
import math
from pathlib import Path
import re
import uuid

from . import mix_accounting as a, mix_vanda_cohost as cohost, successor_cd as s

VERSION = 'V5_CD_VANDA_COLOCATED_MATCHED_DEV24_V1'
NEURAL_ROLE = 'V5_CD_VANDA_COLOCATED_NEURAL_EVALUATION_V1'
PHYSICAL_ROLE = 'V5_CD_VANDA_COLOCATED_PHYSICAL_EVALUATION_V1'
OWNER_ROLE = 'V5_CD_VANDA_COLOCATED_RESIDENT_OWNER_V1'
NEURAL_PROFILE = 'vanda-cd-neural-evaluation'
PHYSICAL_PROFILE = 'vanda-cd-source-evaluation'
STATUS = 'PASS_V5_CD_OWN_NATIVE_VANDA_COLOCATED_ALLOCATION'
JOB_NAME = 'pi0294_cd_eval'
STATES = ('QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY','SUBMITTED_PENDING_NATIVE',
          'NATIVE_RUNNING_QUALIFICATION','NATIVE_TERMINAL')


def need(ok, why):
    if not ok: raise PermissionError('V5_VANDA_EVAL_IDENTITY_'+why)


def typed(value):
    return (value.get('platform_evaluation_version') == VERSION
            or value.get('evaluation_peer_platform') == VERSION
            or value.get('scope') in (NEURAL_ROLE,PHYSICAL_ROLE,OWNER_ROLE)
            or value.get('execution',{}).get('profile') in (NEURAL_PROFILE,PHYSICAL_PROFILE))


def cold_profile(manifest):
    if not typed(manifest): return False
    need(manifest.get('scope') == NEURAL_ROLE
         and manifest.get('platform_evaluation_version') == VERSION
         and manifest.get('execution',{}).get('profile') == NEURAL_PROFILE
         and 'platform_training_version' not in manifest,
         'COMPLETE_DISTINCT_NEURAL_SCOPE_NO_LEGACY_ALIAS')
    return True


def request_identity(manifest, request, manifest_sha256):
    need(cold_profile(manifest),'NEW_TYPED_PROFILE')
    r=request; arm=r.get('evaluation_candidate')
    exact=dict(platform_evaluation_version=VERSION,native_platform='VANDA_A40',
        authority_release=s.TASK_SOURCE,scope_grant_sha256=s.GRANT_SHA,
        stage='V5_MATCHED_DEV24',project='CFP04-CF-007',job_name=JOB_NAME,
        owner_uid=22182,owner_user='e1192548',noise_bank=294101,
        remaining_conditions=24,checkpoint_step=1024,
        parent_checkpoint_manifest_sha256=a.PARENT,statistics_sha256=a.STATS,
        physical_GPU_count=1,physical_GPU_kind='NVIDIA_A40',walltime_s=28800,
        accounting_pad_s=60,reserved_H200_s=28800,
        neural_CPU_upper_s=80000,physical_CPU_upper_s=93900,
        joint_CPU_upper_s=173900,active_cpu_limit=8,
        query_limit=2097,optimizer_updates=0,optimizer_update_limit=0,reserved_optimizer_updates=0,
        numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',source_renderer='MESA_LLVMPIPE_CPU_EGL',
        source_gl_GPU_visible=False,model_payload_copies=0,
        additional_output_upper_bytes=cohost.storage_contract()['original_gross_upper_bytes'])
    need(arm in ('C50','D25') and all(r.get(k) == v and type(r[k]) is type(v) for k,v in exact.items()),
         'EXACT_ORIGINAL_PANEL_AND_JOINT_ENVELOPE')
    need(r.get('run_uuid') == manifest['run_uuid'] and r.get('source_sha') == manifest['code']['vla_commit']
         and r.get('run_manifest_sha256') == manifest_sha256
         and manifest['authority']['protocol_ref'] == s.TASK_SOURCE
         and manifest['authority']['scope_grant_sha256'] == s.GRANT_SHA
         and manifest['authority']['T_CD'] == r['T_CD']
         and manifest['authority']['deadline_epoch'] == r['deadline_epoch'] == s.scope(r['T_CD'])['deadline_epoch'],
         'ONE_ACTUALLY_ADOPTED_CLOCK_AND_MANIFEST')
    need(type(r.get('allocation_number')) is int and r['allocation_number'] == 1
         and type(r.get('submit_count')) is int and 1 <= r['submit_count'] <= 34,
         'LOCAL_REQUEST_NAMESPACE_DISTINCT_FROM_GLOBAL_PBS_COUNT')
    for k in ('request_id','submission_nonce'):
        try: valid = str(uuid.UUID(r[k])) == r[k]
        except (KeyError,ValueError,TypeError,AttributeError): valid=False
        need(valid,'CANONICAL_REQUEST_AND_NONCE')
    for k,n in (('source_sha',40),('run_manifest_sha256',64),('checkpoint_manifest_sha256',64),
                ('training_handoff_sha256',64),('planned_panel_sha256',64),('cold_input_audit_sha256',64)):
        need(type(r.get(k)) is str and re.fullmatch('[0-9a-f]{'+str(n)+'}',r[k]),'SEALED_'+k)
    need('training_block' not in r and 'shared_GPU_grant_sha256' not in r and 'native_handle' not in r,
         'NO_TRAINING_CORE_HANDLE_OR_B24_SHARING')
    need(type(r.get('owner_manifest_sha256')) is str and re.fullmatch('[0-9a-f]{64}',r['owner_manifest_sha256']),
         'SEALED_REGISTERED_RESIDENT_OWNER')
    return r


def native_allocation(job,request,*,job_id,hostname,now_epoch):
    need(type(now_epoch) in (int,float) and math.isfinite(now_epoch)
         and request['T_CD'] <= now_epoch < request['deadline_epoch']-2700,'REAL_CURRENT_SCOPE_CLOCK')
    need(type(job_id) is str and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job_id)
         and job['job_state'] == 'R' and job['project'] == 'CFP04-CF-007'
         and job['Job_Name'] == JOB_NAME and job['Job_Owner'].split('@')[0] == 'e1192548'
         and job.get('queue')=='batch_gpu' and job.get('Rerunable') in (False,'False')
         and int(job.get('run_count',1))==1,
         'OWN_CURRENT_RUNNING_PBS_NOT_A_SAVED_STATUS_FLAG')
    variables=job['Variable_List']
    need(variables.get('PV1_REQUEST_ID') == request['request_id']
         and variables.get('PV1_SUBMISSION_NONCE') == request['submission_nonce'], 'ACTUAL_PBS_REQUEST_NONCE')
    hosts={x.split('/')[0].split('.')[0].lower() for x in job['exec_host'].split('+')}
    need(len(hosts)==1 and hostname.split('.')[0].lower() in hosts and hostname.lower().startswith('gn-a40-'), 'OWN_ALLOCATED_A40_HOST_NOT_LOGIN')
    resources=job['Resource_List']; h,m,sec=map(int,resources['walltime'].split(':'))
    from scripts.pi0294.dec1.a40_allocation import memory_bytes
    native=(int(resources['ncpus']),memory_bytes(resources['mem']))
    need(int(resources.get('ngpus',0)) == 1 and h*3600+m*60+sec == 28800
         and native in ((8,64*1024**3),(36,240*1024**3)), 'EXACT_NATIVE_8CPU_OR_SITE_36CPU_BUNDLE')
    select=resources.get('select','').split(':')
    need(select[0]=='1' and '+' not in resources.get('select','')
         and all(x.count('=')==1 for x in select[1:]),'ONE_NATIVE_A40_CHUNK')
    fields=dict(x.split('=',1) for x in select[1:])
    need(len(fields)==len(select)-1 and int(fields.get('ncpus',0))==native[0]
         and memory_bytes(fields.get('mem',''))==native[1] and fields.get('ngpus')=='1'
         and fields.get('gpu_model','A40')=='A40','SELECT_MATCHES_ACTUAL_NATIVE_CAPACITY')
    from scripts.pi0294.e0_scope import pbs_epoch
    start=pbs_epoch(job['stime'])
    need(start <= now_epoch < min(start+28800,request['deadline_epoch'])-2700,
         'REAL_PBS_START_AND_ORIGINAL_CLOSEOUT_RESERVE')
    return dict(job_id=job_id,hostname=hostname,allocation_start_epoch=start,
                allocation_end_epoch=min(start+28800,request['deadline_epoch']),
                native_capacity=dict(ncpus=native[0],memory_bytes=native[1]),
                numerical_model_permission=False,physics_authorized=False)


def require_device(request,allocation,*,visible,count,name,memory_bytes):
    need(request.get('platform_evaluation_version')==allocation.get('platform_evaluation_version')==VERSION
         and allocation.get('status')==STATUS and request.get('stage')=='V5_MATCHED_DEV24'
         and request.get('evaluation_candidate') in ('C50','D25')
         and allocation.get('evaluation_candidate')==request['evaluation_candidate']
         and request.get('checkpoint_manifest_sha256')==allocation.get('checkpoint_manifest_sha256')
         and request.get('statistics_sha256')==allocation.get('statistics_sha256')==a.STATS
         and type(request.get('optimizer_updates')) is int and request['optimizer_updates']==0
         and not any('platform_training_version' in v or 'training_block' in v for v in (request,allocation))
         and type(visible) is str and bool(visible) and visible==allocation.get('visible')
         and ',' not in visible and type(count) is int and count==1 and type(name) is str and re.fullmatch(r'(?:NVIDIA )?A40',name)
         and type(memory_bytes) is int and memory_bytes>=40*1024**3,
         'ONE_ACTUAL_ALLOCATED_A40_NO_CORE_OR_H200_FALLBACK')


def bind_closed_endpoint(global_bytes,request,capture):
    g=json.loads(global_bytes); marker=cohost.validated(g)
    need(marker is not None and request['T_CD'] == marker['T_CD']
         and request['deadline_epoch'] == marker['deadline_epoch'],'ACTUAL_CURRENT_EXISTING_GLOBAL_SITE_AUTHORITY')
    from .mix_core_context import captured_endpoint
    sealed=captured_endpoint(capture)
    proof=cohost.endpoint(global_bytes,capture)
    # Registration/heartbeats/effect debits legitimately advance GLOBAL. The
    # immutable terminal evidence must remain exact, while the fresh current
    # ledger is joined separately; never demand an obsolete GLOBAL byte hash.
    need(proof['arm'] == request['evaluation_candidate']
         and proof['checkpoint_manifest_sha256'] == request['checkpoint_manifest_sha256']
         and a.sha(a.raw(sealed)) == request['training_handoff_sha256']
         and {k:v for k,v in proof.items() if k!='global_sha256'} ==
             {k:v for k,v in sealed.items() if k!='global_sha256'}, 'TRUE_CURRENT_CLOSED_1024_ENDPOINT')
    repair=cohost.repair.validated(g)
    if request['evaluation_candidate']=='C50':
        need(request['request_id'] == repair['replacement_request_id']
             and request['submission_nonce'] == repair['replacement_submission_nonce']
             and request['run_uuid'] == repair['replacement_run_uuid'], 'ONLY_PREDECLARED_C50_REPLACEMENT_IDENTITY')
    return proof


def require_preparation_at_current_global(global_bytes,request,capture):
    """Mandatory before any request registration, inside the original CAS.

    A model-phase join tolerates its own later counter revisions. A NEW cold
    request does not get that tolerance: its handoff must be rebound to the
    exact actual pre-registration GLOBAL bytes, not an earlier prepared plan.
    """
    import base64
    need(base64.b64decode(capture['global_base64'],validate=True) == global_bytes,
         'NEW_REQUEST_HANDOFF_REBOUND_TO_EXACT_CURRENT_GLOBAL')
    return bind_closed_endpoint(global_bytes,request,capture)
