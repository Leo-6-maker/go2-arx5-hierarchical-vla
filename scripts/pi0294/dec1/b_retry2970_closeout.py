"""Pure closeout checks for the paid B2970 failure, without any retry grant.

Saved-state audit and a fresh native terminal read are separate evidence.
Nothing here connects, writes a ledger, closes a lease or submits a job.
"""
from copy import deepcopy
import hashlib
import json
import math

from . import b_retry2970_failure as failed


def need(ok, reason):
    if not ok:
        raise PermissionError('B_RETRY2970_CLOSEOUT_'+reason)


def financial_evidence(saved, observation, expected, *, now_epoch):
    """Join real full-member reads to a later exact terminal read, never restamp."""
    failed.audit(saved, expected)
    need(type(now_epoch) in (int, float) and math.isfinite(now_epoch), 'FINITE_ACTUAL_CLOCK')
    epoch=observation.get('observed_epoch')
    need(type(epoch) in (int, float) and math.isfinite(epoch)
         and 0<=now_epoch-epoch<=30, 'FRESH_NATIVE_TERMINAL_OBSERVATION')
    job=observation['job'];failed.native(job, expected)
    # Include all native resource/cost fields, not just F/Exit1. A later
    # accounting correction requires explicit reconciliation of the difference.
    need(job==saved['raw_native'], 'UNCHANGED_COMPLETE_NATIVE_TERMINAL_ROW')
    proof=saved['native_proof']
    need(proof.get('fresh_native_scheduler_check') is False
         and saved.get('financial_settlement_ready') is False,
         'EXPLICIT_SAVED_STATE_ONLY_STAGE')
    costs=[saved.get(k) for k in ('prior_failed_audit_CPU_upper_s','CPU_audit_actual_s')]
    need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in costs)
         and sum(costs)<=480,
         'BOTH_READONLY_AUDIT_COSTS_PRESERVED')
    value=deepcopy(saved)
    value['historical_native_observed_epoch']=proof['observed_epoch']
    value['native_proof'].update(observed_epoch=epoch, fresh_native_scheduler_check=True)
    value['native_proof']['native_record_sha256']=hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    value['financial_settlement_ready']=True
    value['fresh_native_observation']=deepcopy(observation)
    return value


def settle_once(global_before, dec1_before, *, evidence, expected, now_epoch):
    """Use the existing paired-ledger arithmetic; callers own the original CAS."""
    failed.audit(evidence, expected)
    need(evidence.get('financial_settlement_ready') is True
         and evidence['native_proof'].get('fresh_native_scheduler_check') is True,
         'NO_SETTLEMENT_FROM_HISTORICAL_F_ONLY')
    failed.prior.retry.validate_adopted(global_before, dec1_before)
    from .a40_accounting import settle
    h,d=settle(global_before, dec1_before, request_id=failed.REQUEST,
        native_proof=evidence['native_proof'], effects=evidence['effects'], now_epoch=now_epoch)
    validate_after(h, d, expected)
    return h,d


def validate_after(h, d, expected):
    failed.expected(expected);failed.prior.retry.validate_adopted(h,d)
    need(h['optimizer_updates']==d['optimizer_updates']==failed.CUMULATIVE_UPDATES
         and h['generation_queries']==1214,
         'ACTUAL8744_PLUS922_AND1197_PLUS17')
    b=failed.prior.b
    need(h['training_runs'][b.LINEAGE]['last_native_step']==2970
         and h['training_runs']['B_COVERAGE']['last_native_step']==1829
         and h['training_runs'][b.zero.HR.LINEAGE]['last_native_step']==4096,
         'BOTH_PAID_FAILED_LINEAGES_AND_COMPLETED_A_PRESERVED')
    rows=[r for r in h['PBS_submissions'] if r.get('request_id')==failed.REQUEST]
    need(len(rows)==1, 'ONE_ORIGINAL_REQUEST_ROW')
    row=rows[0]
    need(row['job_id']==failed.JOB and row['source_sha']==failed.SOURCE
         and row['request_sha256']==failed.REQUEST_SHA and row['status']=='NATIVE_TERMINAL'
         and row['native_exit_status']==1 and row['committed_optimizer_updates']==922
         and row['query_attempts']==17
         and row['reserved_optimizer_updates']==row['reserved_neural_queries']==0,
         'FAILED_EFFECT_ROW_NEVER_B4096_SUCCESS')
    need(h['unknown_effects']==h['pending_updates']==d['unknown_effects']==[]
         and not any(h['reserved'].values()) and h['CPU_reserved_s']==h['storage_reserved_bytes']==0
         and all(d.get(k,0)==0 for k in ('GPU_reserved_s','CPU_reserved_s',
             'optimizer_updates_reserved','neural_queries_reserved','GPU_site_output_reserved_bytes')),
         'REAL_RELEASED_RESERVATIONS_NO_UNKNOWN_REFUND')
    return row
