"""Pure closeout checks for the paid B2970 failure, without any retry grant.

Saved-state audit and a fresh native terminal read are separate evidence.
Nothing here connects, writes a ledger, closes a lease or submits a job.
"""
from copy import deepcopy
import hashlib
import json
import math

from . import b_retry2970_failure as failed
from . import vanda_control_handoff as handoff

READONLY_CPU_SCOPE='FAILED2970_READONLY_AUDIT_FROM_EXISTING54000_CPU_RESERVATION_V1'
FAILED_AUDIT_SHA='681668e0726ed1943853b428cdb3cc340efb9d83532c8f7e3049dc7d548a6163'


def readonly_cpu_allocation(allocation, h, d, expected):
    """Partition already-held CPU; never extend a limit or reuse a failed actor."""
    failed.expected(expected)
    need(isinstance(allocation,dict) and allocation.get('schema')==READONLY_CPU_SCOPE
         and allocation.get('job_id')==failed.JOB and allocation.get('request_sha256')==failed.REQUEST_SHA
         and allocation.get('previous_failed_audit_receipt_sha256')==FAILED_AUDIT_SHA
         and allocation.get('previous_audit_actor_confirmed_absent') is True
         and allocation.get('prior_all_audit_and_admin_CPU_upper_s')==480
         and allocation.get('new_finite_audit_CPU_upper_s')==600
         and allocation.get('observer_CPU_upper_s')==1500
         and allocation.get('remaining_closeout_CPU_upper_s')==120
         and allocation.get('native_PBS_CPU_s')==8323
         and allocation.get('original_job_CPU_reserved_s')==54000
         and allocation.get('new_CPU_prepayment')==0,
         'EXACT_SEPARATE_FINITE600_AUDIT_WITH_PRIOR480_PRESERVED')
    need(all(type(allocation.get(k)) is int for k in ('prior_all_audit_and_admin_CPU_upper_s',
         'new_finite_audit_CPU_upper_s','observer_CPU_upper_s','remaining_closeout_CPU_upper_s',
         'native_PBS_CPU_s','original_job_CPU_reserved_s','new_CPU_prepayment')),
         'INTEGER_ORIGINAL_RESERVATION_PARTITIONS')
    need(isinstance(h,dict) and isinstance(d,dict),'PAIRED_CURRENT_LEDGER_VALUES')
    need(type(expected['request'].get('CPU_upper_s')) is int
         and expected['request'].get('CPU_upper_s')==54000
         and h.get('CPU_reserved_s')==d.get('CPU_reserved_s')==54000
         and h.get('optimizer_updates')==d.get('optimizer_updates')==8744
         and h.get('unknown_effects')==h.get('pending_updates')==d.get('unknown_effects')==[],
         'SAME_FAILED_REQUEST_STILL_OWNS_ORIGINAL54000_RESERVATION')
    marker=h.get(handoff.FIELD)
    need(isinstance(marker,dict) and marker==d.get(handoff.FIELD)
         and marker.get('grant_sha256')==handoff.GRANT_SHA
         and marker.get('old_parent_mutations_permitted') is False
         and marker.get('origin_is_fresh_native_read') is False and marker.get('one_mutation_owner') is True,
         'CURRENT_MIGRATED_VANDA_SINGLE_AUTHORITY')
    rows=[r for r in h['PBS_submissions'] if r.get('request_id')==failed.REQUEST]
    need(len(rows)==1 and rows[0].get('job_id')==failed.JOB
         and rows[0].get('status')=='SUBMITTED_PENDING_NATIVE',
         'ONE_UNSETTLED_ORIGINAL_FAILED_REQUEST_NO_NEW_SUBMISSION')
    total=8323+480+600+1500+120
    need(total==11023 and total<=54000,'WHOLE_JOB_WITH_ALL_READS_INSIDE_ORIGINAL_CPU_ENVELOPE')
    return total


def need(ok, reason):
    if not ok:
        raise PermissionError('B_RETRY2970_CLOSEOUT_'+reason)


def financial_evidence(saved, observation, expected, *, now_epoch,
                       readonly_allocation=None, budget_before=None):
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
    limit=480
    if readonly_allocation is not None:
        need(isinstance(budget_before,tuple) and len(budget_before)==2,'PAIRED_REAL_CPU_BUDGET_BEFORE_FINANCIAL_JOIN')
        total=readonly_cpu_allocation(readonly_allocation,*budget_before,expected)
        need(costs[0]==480 and type(costs[1]) in (int,float) and costs[1]<=600
             and proof.get('conservative_CPU_s')==total,'ALL_PRIOR_AND_NEW_AUDIT_CPU_CHARGED')
        limit=1080
    need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in costs)
         and sum(costs)<=limit,
         'BOTH_READONLY_AUDIT_COSTS_PRESERVED')
    value=deepcopy(saved)
    value['historical_native_observed_epoch']=proof['observed_epoch']
    value['native_proof'].update(observed_epoch=epoch, fresh_native_scheduler_check=True)
    value['native_proof']['native_record_sha256']=hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    value['financial_settlement_ready']=True
    value['fresh_native_observation']=deepcopy(observation)
    if readonly_allocation is not None:value['readonly_audit_allocation']=deepcopy(readonly_allocation)
    return value


def settle_once(global_before, dec1_before, *, evidence, expected, now_epoch):
    """Use the existing paired-ledger arithmetic; callers own the original CAS."""
    failed.audit(evidence, expected)
    need(evidence.get('financial_settlement_ready') is True
         and evidence['native_proof'].get('fresh_native_scheduler_check') is True,
         'NO_SETTLEMENT_FROM_HISTORICAL_F_ONLY')
    failed.prior.retry.validate_adopted(global_before, dec1_before)
    if 'readonly_audit_allocation' in evidence:
        total=readonly_cpu_allocation(evidence['readonly_audit_allocation'],global_before,dec1_before,expected)
        need(evidence['native_proof']['conservative_CPU_s']==total,'RESERVED_CPU_ALLOCATION_RECHECKED_AT_SETTLEMENT')
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
