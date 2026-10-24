"""Pure Vanda DEV accounting in the existing global ledger; no writes or grants.

These functions return proposals, never dispatch or persist them. Reservation
and fresh admission remain separate. Existing once IO and owner locks must
persist any proposal; an unknown write never permits retrying it here.
Full S1/final protection and the independent DEV work envelope must still pass
before any native intent. No training-source or DEC1 budget is created here.
"""
import re
import uuid
import math
from copy import deepcopy

from scripts.pi0294.dec1.alternate_accounting import finite,digest
from scripts.pi0294.dec1.a40_allocation import BOUNDED_CPU_PROFILE
from . import evaluation_platform as e
from .decision import AUTHORITY
from .resources import pbs_limit

TERMINAL='NATIVE_TERMINAL'
STATES=('QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY','SUBMITTED_PENDING_NATIVE',
        'NATIVE_RUNNING_QUALIFICATION',TERMINAL)
GIB=1024**3


def need(ok,reason):
    if not ok:raise PermissionError('PV1_DEV_ACCOUNTING_'+reason)


def output_upper(conditions):
    need(type(conditions) is int and 1<=conditions<=24,'ONE_ORIGINAL_REMAINING_NOISE_PANEL')
    raw=conditions*80*(2*6*1024**2+128*1024)+512*1024**2
    return ((raw+GIB-1)//GIB)*GIB


def progress(row):
    """Actual pre-effect debits, needed to avoid charging them again at F."""
    q=finite(row.get('charged_query_attempts',0),integer=True)
    loads=finite(row.get('charged_model_loads',0),integer=True)
    cpu=finite(row.get('charged_CPU_s',0))
    effects=row.get('charged_effects',{})
    need(type(effects) is dict and len(effects)<=row['query_limit']+10,'BOUNDED_ACTUAL_EFFECT_INDEX')
    totals=dict(generation_queries=0,model_loads=0,CPU_s=0)
    for key,value in effects.items():
        need(type(key) is str and re.fullmatch(r'[A-Za-z0-9._:/-]{1,200}',key) is not None
             and type(value) is dict and set(value)=={'kind','count'}
             and value['kind'] in totals,'EXACT_EFFECT_ID_AND_KIND')
        amount=finite(value['count'],integer=value['kind']!='CPU_s')
        need(value['kind']=='CPU_s' or amount==1,'ONE_ACTUAL_MODEL_OR_QUERY_INTENT')
        totals[value['kind']]+=amount
    need(totals['generation_queries']==q and totals['model_loads']==loads
         and math.isclose(totals['CPU_s'],cpu,rel_tol=1e-12,abs_tol=1e-8)
         and q<=row['query_limit'] and loads<=2 and cpu<=row['CPU_upper_s'] and (not q or loads),
         'EFFECT_INDEX_MATCHES_BOUNDED_CHARGES')
    return q,loads,cpu


def validate_item(row):
    """Only consistency of an already represented row, never a job permission."""
    r=row;candidate=r.get('evaluation_candidate')
    conditions=r.get('remaining_conditions');output=output_upper(conditions)
    query_limit=80*conditions+17
    need(candidate in e.ENDPOINTS,'EXISTING_FROZEN_ENDPOINT_ONLY')
    expected=dict(native_platform=e.PLATFORM,platform_evaluation_version=e.VERSION,
        scientific_admission_version=e.VERSION,authority_release=AUTHORITY,
        campaign_run_uuid='20261018T182956Z-localpc-pv1',owner_user='e1192548',
        stage='B0_DEV' if candidate=='B0_CORR' else 'P4_DEV',project='CFP04-CF-007',
        job_name='pi0294_pv1_a40_dev',checkpoint_manifest_sha256=e.ENDPOINTS[candidate],
        parent_checkpoint_manifest_sha256=e.PARENT,statistics_sha256=e.R0,
        physical_GPU_kind='NVIDIA_A40',physical_GPU_count=1,legacy_GPU_seconds_field_is_projection=True,
        reserved_H200_s=28800,accounting_pad_s=60,CPU_upper_s=80000,
        CPU_envelope_profile=BOUNDED_CPU_PROFILE,declared_process_CPU_limit_s=79760,
        query_limit=query_limit,optimizer_update_limit=0,reserved_optimizer_updates=0,
        original_T0_epoch=e.T0,hard_close_epoch=e.HARD_CLOSE,charges_DEC1_subcaps=False,
        new_robot_physics=0,additional_output_upper_bytes=output)
    need(all(r.get(k)==v and type(r[k]) is type(v) for k,v in expected.items())
         and 'platform_training_version' not in r and not r.get('training_block')
         and r.get('noise_bank') in (294101,294102),'DISTINCT_ZERO_TRAINING_GLOBAL_EVALUATION_ENVELOPE')
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',str(r.get('child_run_uuid',''))) is not None
         and r['child_run_uuid']!=r['campaign_run_uuid']
         and e.T0<=finite(r.get('created_epoch'))<e.STOP_NEW,'ORIGINAL_CHILD_AND_REQUEST_CLOCK')
    for key,size in (('request_sha256',64),('run_manifest_sha256',64),('source_sha',40),
                     ('cold_input_audit_sha256',64),('planned_panel_sha256',64),
                     ('prior_outcomes_sha256',64),('checkpoint_reuse_proof_sha256',64),
                     ('prequeue_future_admission_sha256',64)):
        need(re.fullmatch('[0-9a-f]{'+str(size)+'}',str(r.get(key,''))) is not None,
             'ACTUAL_PROVENANCE_HASH:'+key)
    for key in ('request_id','submission_nonce'):
        try:value=uuid.UUID(r[key])
        except (KeyError,TypeError,ValueError,AttributeError):need(False,'CANONICAL_UUID')
        need(value.version==4 and str(value)==r[key],'CANONICAL_UUID')
    need(r.get('status') in STATES,'KNOWN_LIFECYCLE_NOT_AN_UNKNOWN_REFUND')
    job=r.get('job_id')
    need(job is None and r['status']==STATES[0]
         or isinstance(job,str) and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job) is not None,
         'EXACT_VANDA_HANDLE_OR_PRESERVED_UNBOUND_UNKNOWN')
    charged,loads,cpu=progress(r)
    if r['status']==TERMINAL:
        for key in ('terminal_native_sha256','terminal_effect_summary_sha256'):
            need(re.fullmatch('[0-9a-f]{64}',str(r.get(key,''))) is not None,'ACTUAL_TERMINAL_EVIDENCE')
        unknown=r.get('unknown_query_intents',[])
        need(type(unknown) is list and len(set(unknown))==len(unknown)
             and all(type(i) is int and 0<=i<r.get('query_attempts',0) for i in unknown),
             'EXPLICIT_TERMINAL_UNKNOWN_QUERY_INTENTS')
        need(type(r.get('native_exit_status')) is int
             and finite(r.get('conservative_H200_s'))<=28860
             and cpu<=finite(r.get('conservative_CPU_s'))<=80000
             and charged<=finite(r.get('query_attempts'),integer=True)<=query_limit
             and loads<=finite(r.get('model_load_attempts'),integer=True)<=2
             and type(r.get('reserved_neural_queries')) is int and r['reserved_neural_queries']==len(unknown)
             and type(r.get('committed_optimizer_updates')) is int and r['committed_optimizer_updates']==0,
             'REAL_EFFECTS_KEEP_FAILED_COSTS_ZERO_OPTIMIZATION')
    else:
        need(type(r.get('reserved_neural_queries')) is int and r['reserved_neural_queries']==query_limit-charged,
             'EVERY_QUERY_ALREADY_CHARGED_OR_STILL_RESERVED')
    return row


def validate_global(value,rows):
    need(value.get('schema')=='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1'
         and value.get('authority_release')==AUTHORITY
         and value.get('campaign_run_uuid')=='20261018T182956Z-localpc-pv1'
         and len(value['PBS_submissions'])<=pbs_limit(value),
         'SAME_GLOBAL_LEDGER_AND_ALREADY_ADOPTED_CAP_NOT_PROPOSED_S2_CAP')
    for row in rows:validate_item(row)
    active=[r for r in rows if r['status']!=TERMINAL]
    need(sum(s['status'] not in (TERMINAL,'NATIVE_QSUB_REJECTED_NO_JOB') for s in value['PBS_submissions'])<=1,
         'ONE_REGISTERED_GPU_CHAIN_ACROSS_TRAINING_AND_EVALUATION')
    for key in ('request_id','submission_nonce','job_id'):
        handles=[s.get(key) for s in value['PBS_submissions'] if s.get(key) is not None]
        need(len(handles)==len(set(handles)),'NO_REUSED_NATIVE_INTENT_OR_HANDLE:'+key)
    need(sum(r['reserved_H200_s']+r['accounting_pad_s'] for r in active)<=value['reserved'].get('H200_s',0)
         and sum(r['CPU_upper_s']-progress(r)[2] for r in active)<=value.get('CPU_reserved_s',0)
         and sum(r['additional_output_upper_bytes'] for r in active)<=value.get('storage_reserved_bytes',0),
         'NO_DROPPED_GPU_CPU_OR_OUTPUT_ESCROW')
    need(sum(r['reserved_neural_queries'] for r in rows)<=value['reserved'].get('neural_queries',0)
         and sum(r['query_attempts'] if r['status']==TERMINAL else progress(r)[0] for r in rows)<=value.get('generation_queries',0)
         and sum(r['model_load_attempts'] if r['status']==TERMINAL else progress(r)[1] for r in rows)<=value.get('model_loads',0)
         and sum(r['conservative_CPU_s'] if r['status']==TERMINAL else progress(r)[2] for r in rows)<=value.get('CPU_s',0)
         and sum(r['conservative_H200_s'] for r in rows if r['status']==TERMINAL)<=value.get('used',{}).get('H200_s',0),
         'ACTUAL_QUERY_MODEL_CPU_DEBITS_EXIST_IN_THE_SAME_GLOBAL_TOTAL')
    return value


def checked(value):
    from .neural_ledger import check
    return check(value,allow_unknown=True)


def item(value,request_id):
    rows=[r for r in value['PBS_submissions'] if r.get('request_id')==request_id]
    need(len(rows)==1,'ONE_EXACT_EVALUATION_REQUEST')
    return validate_item(rows[0])


def bind_job(value,*,request_id,job_id,now_epoch):
    """Bind one actual successful qsub reply; never qsub or rebind here."""
    v=deepcopy(checked(value));r=item(v,request_id)
    need(r['status']==STATES[0] and r.get('job_id') is None
         and type(job_id) is str and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job_id) is not None
         and all(s.get('job_id')!=job_id for s in v['PBS_submissions'])
         and finite(now_epoch)>=r['created_epoch'],'ONE_NATIVE_REPLY_NOT_RETRY_OR_REBIND')
    r.update(status='SUBMITTED_PENDING_NATIVE',job_id=job_id,native_reply_epoch=now_epoch)
    return checked(v)


def charge(value,*,request_id,job_id,kind,count,effect_id):
    """Transfer a known pre-effect debit out of escrow, once per actor intent."""
    v=deepcopy(checked(value));r=item(v,request_id)
    need(r['status'] in STATES[1:3] and r['job_id']==job_id and bool(job_id),
         'BOUND_LIVE_REQUEST_BEFORE_EFFECT_DEBIT')
    need(kind in ('generation_queries','model_loads','CPU_s') and type(effect_id) is str
         and re.fullmatch(r'[A-Za-z0-9._:/-]{1,200}',effect_id) is not None
         and effect_id not in r.get('charged_effects',{}),'ONE_UNIQUE_EFFECT_ID_NO_REPLAY')
    amount=finite(count,integer=kind!='CPU_s')
    need(kind=='CPU_s' or amount==1,'ONE_PRE_EFFECT_QUERY_OR_MODEL_ATTEMPT')
    # Actual CPU closeout stays legal with unrelated UNKNOWNs; new neural
    # effects remain fenced. No UNKNOWN marker is removed or reissued here.
    if kind!='CPU_s':need(not v['unknown_effects'] and not v['pending_updates'],'NO_NEW_EFFECT_WHILE_UNKNOWN')
    q,loads,cpu=progress(r)
    if kind=='generation_queries':
        need(q+amount<=r['query_limit'] and loads>0,'BOUND_QUERIES_AFTER_A_RECORDED_MODEL_LOAD')
        r['charged_query_attempts']=q+amount;r['reserved_neural_queries']-=amount
        v['generation_queries']+=amount;v['used']['neural_queries']+=amount;v['reserved']['neural_queries']-=amount
    elif kind=='model_loads':
        need(loads+amount<=2,'BOUNDED_MODEL_LOAD_ATTEMPTS')
        r['charged_model_loads']=loads+amount;v['model_loads']=v.get('model_loads',0)+amount
    else:
        need(cpu+amount<=r['CPU_upper_s'],'BOUNDED_ACTUAL_CPU')
        r['charged_CPU_s']=cpu+amount;v['CPU_s']+=amount;v['CPU_reserved_s']-=amount
    r.setdefault('charged_effects',{})[effect_id]=dict(kind=kind,count=amount)
    r['status']='NATIVE_RUNNING_QUALIFICATION'
    return checked(v)


def settle(value,*,request_id,native_proof,effects,now_epoch):
    """Settle F from independent native finality plus the complete effect journal.

    Per-effect charges are credited exactly once. Unknown queries keep their
    explicit holds and marker. This is financial closure, never a task PASS.
    """
    v=deepcopy(checked(value));r=item(v,request_id)
    need(r['status'] in STATES[1:3] and r.get('job_id'),'KNOWN_BOUND_TERMINAL_REQUEST_NOT_SETTLED_TWICE')
    expected=dict(schema='PV1_NATIVE_A40_DEV_TERMINAL_V1',job_id=r['job_id'],job_state='F',
        project=r['project'],job_name=r['job_name'],owner='e1192548',ngpus=1,
        request_id=request_id,submission_nonce=r['submission_nonce'],recorded_process_terminal=True,
        terminal_inventory_stable=True)
    need(all(native_proof.get(k)==x and type(native_proof[k]) is type(x) for k,x in expected.items())
         and type(native_proof.get('Exit_status')) is int
         and finite(now_epoch)>=r['created_epoch']
         and 0<=now_epoch-finite(native_proof.get('observed_epoch'))<=120,
         'FRESH_EXACT_NATIVE_F_AND_PROCESS_FINALITY_NOT_OBSERVATION_TIMEOUT')
    gpu=finite(native_proof.get('conservative_GPU_s'));cpu=finite(native_proof.get('conservative_CPU_s'))
    output=finite(native_proof.get('new_resident_allocated_upper_bytes'),integer=True)
    need(gpu<=28860 and cpu<=80000 and output<=r['additional_output_upper_bytes'],
         'ACTUAL_COSTS_INSIDE_RESERVED_ENVELOPE_OTHERWISE_KEEP_HOLDS')
    identity=dict(schema='PV1_A40_NATIVE_DEV_EFFECT_SUMMARY_V1',job_id=r['job_id'],request_id=request_id,
        request_sha256=r['request_sha256'],source_sha=r['source_sha'],
        run_manifest_sha256=r['run_manifest_sha256'],evaluation_candidate=r['evaluation_candidate'],
        checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],statistics_sha256=e.R0,
        optimizer_updates_committed=0,new_physics=0,new_episodes=0,effect_journal_complete=True)
    need(all(effects.get(k)==x and type(effects[k]) is type(x) for k,x in identity.items()),
         'EXACT_COMPLETE_NATIVE_QUERY_LOAD_JOURNAL_NO_TRAINING_OR_PHYSICS')
    q=finite(effects.get('query_attempts'),integer=True);loads=finite(effects.get('model_load_attempts'),integer=True)
    charged,charged_loads,charged_cpu=progress(r)
    need(charged<=q<=r['query_limit'] and charged_loads<=loads<=2 and charged_cpu<=cpu
         and (not q or loads) and (not loads or gpu>0),
         'ACTUAL_TOTALS_COVER_ALREADY_COMMITTED_DEBITS')
    unknown=effects.get('unknown_query_intents')
    need(type(unknown) is list and len(set(unknown))==len(unknown)
         and all(type(i) is int and 0<=i<q for i in unknown),
         'PRESERVE_EXACT_UNRESOLVED_QUERY_INTENTS')
    v['used']['H200_s']+=gpu;v['CPU_s']+=cpu-charged_cpu
    v['generation_queries']+=q-charged;v['used']['neural_queries']+=q-charged
    v['model_loads']=v.get('model_loads',0)+loads-charged_loads
    v['reserved']['H200_s']-=28860
    v['reserved']['neural_queries']-=r['reserved_neural_queries']-len(unknown)
    v['CPU_reserved_s']-=80000-charged_cpu
    v['storage_reserved_bytes']-=r['additional_output_upper_bytes']
    v['new_storage_current_bytes']+=output
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    if unknown:
        v['unknown_effects'].append(dict(kind='A40_DEV_UNKNOWN_QUERY_INTENTS_NO_RETRY',
            request_id=request_id,job_id=r['job_id'],unknown_query_intents=deepcopy(unknown)))
    r.update(status=TERMINAL,conservative_H200_s=gpu,physical_A40_allocated_s=gpu,
        conservative_CPU_s=cpu,query_attempts=q,model_load_attempts=loads,committed_optimizer_updates=0,
        reserved_neural_queries=len(unknown),unknown_query_intents=deepcopy(unknown),
        native_exit_status=native_proof['Exit_status'],settled_epoch=now_epoch,
        terminal_native_sha256=digest(native_proof),terminal_effect_summary_sha256=digest(effects),
        settlement_financial_deltas=dict(GPU_s=gpu,CPU_s=cpu-charged_cpu,queries=q-charged,model_loads=loads-charged_loads))
    return checked(v)
