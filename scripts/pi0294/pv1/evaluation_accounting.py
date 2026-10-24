"""Validate Vanda DEV rows in the existing global ledger; no writes or grants.

This is a classification guard, not reserve/bind/settle or fresh admission.
Full S1/final protection and the independent DEV work envelope must still pass
before any native intent. No training-source or DEC1 budget is created here.
"""
import re
import uuid

from scripts.pi0294.dec1.alternate_accounting import finite
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
    if r['status']==TERMINAL:
        for key in ('terminal_native_sha256','terminal_effect_summary_sha256'):
            need(re.fullmatch('[0-9a-f]{64}',str(r.get(key,''))) is not None,'ACTUAL_TERMINAL_EVIDENCE')
        need(type(r.get('native_exit_status')) is int
             and finite(r.get('conservative_H200_s'))<=28860
             and finite(r.get('conservative_CPU_s'))<=80000
             and finite(r.get('query_attempts'),integer=True)<=query_limit
             and type(r.get('reserved_neural_queries')) is int and r['reserved_neural_queries']==0
             and type(r.get('committed_optimizer_updates')) is int and r['committed_optimizer_updates']==0,
             'REAL_EFFECTS_KEEP_FAILED_COSTS_ZERO_OPTIMIZATION')
    else:
        need(type(r.get('reserved_neural_queries')) is int and r['reserved_neural_queries']==query_limit,
             'FULL_QUERY_ESCROW_RETAINED_UNTIL_NATIVE_TERMINAL')
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
         and sum(r['CPU_upper_s'] for r in active)<=value.get('CPU_reserved_s',0)
         and sum(r['additional_output_upper_bytes'] for r in active)<=value.get('storage_reserved_bytes',0),
         'NO_DROPPED_GPU_CPU_OR_OUTPUT_ESCROW')
    return value
