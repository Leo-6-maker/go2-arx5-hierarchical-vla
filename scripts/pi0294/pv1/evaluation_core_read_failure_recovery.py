"""One additional cold B24 after the exact model-zero scheduler-read failure.

All original rows, grants, costs and queue remain. No generic retry pool, new
panel, cap or clock is introduced. The sole new request still needs original
fresh full resources, native registration and actual transport qualification.
"""
from copy import deepcopy
from . import evaluation_core_B24_1000 as late
import re,hashlib,json
FIELD='Core_B_known_zero_scheduler_read_recovery'
SCHEMA='PV1_CORE_B_KNOWN_ZERO_SCHEDULER_READ_COLD_RECOVERY_V1'
FAILED='b43a9ef7-1636-4e91-a3e0-e1edb33eeb38'
REQUEST_SHA='d7f0f25a9e6318d1672002eddd3ce20f642a429147f1435dbdcf49e7c17b84b4'
FINAL_SHA='d1e1b3505888db52328d672b12a9916d9b83399564dcdd3ef35b2fcf8c10acb3'
RUN='pv1-v4a-b4096-n294101-read-recovery-02'

def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_READ_RECOVERY_'+why)

def validate(g,first_marker):
    from . import evaluation_core_accounting as c
    m=g.get(FIELD);rows=g[c.FIELD]
    need(type(m) is dict and set(m)=={'schema','previous_boundary_sha256','failed_request_id',
        'failed_terminal_native','failed_effects','failed_dual_closure','known_zero_storage',
        'exact_read_failure_diagnostic_sha256','independent_closure_readback_sha256',
        'replacement_request_id','replacement_submission_nonce','replacement_run_uuid',
        'replacement_source_sha','control_id','created_epoch','automatic_retry','maximum_additional_replacements'},
        'EXACT_ONE_KNOWN_READ_FAILURE_NOT_A_POOL')
    need(m['schema']==SCHEMA and m['previous_boundary_sha256']==c.digest(first_marker)
         and first_marker['replacement_request_id']==m['failed_request_id']==FAILED
         and m['automatic_retry'] is False and type(m['maximum_additional_replacements']) is int
         and m['maximum_additional_replacements']==1 and c.old.T0<=m['created_epoch']<late.limit_for_marker(g,m),
         'PREVIOUS_ONCE_BOUNDARY_AND_UNRESET_CLOCK')
    r=rows[FAILED];p=m['failed_terminal_native'];e=m['failed_effects'];close=m['failed_dual_closure'];storage=m['known_zero_storage']
    need(r['request_sha256']==REQUEST_SHA and r['source_sha']=='a6991fca99f82e8dc413816e8bde6c8f0e1552ea'
         and r['evaluation_candidate']=='B_COVERAGE' and r['noise_bank']==294101 and r['child_run_uuid']=='pv1-v4a-b4096-n294101-zero-start-recovery-01'
         and r['status']==c.STATES[3] and r['native_exit_status']==1
         and c.progress(r)[:2]==(0,0) and 0<=c.progress(r)[2]<=3
         and r['query_attempts']==r['model_load_attempts']==0 and r['unknown_query_intents']==[]
         and r['conservative_GPU_s']==678 and r['conservative_CPU_s']==3,
         'EXACT_REAL_TERMINAL_COSTS_ZERO_MODEL_QUERY_NOT_UNKNOWN_OR_REFUND')
    need(c.digest(p)==r['terminal_native_sha256'] and c.digest(e)==r['terminal_effect_summary_sha256']
         and p['actual_NATIVE_PROCESS_FINAL_sha256']==FINAL_SHA
         and p['request_id']==e['request_id']==close['request_id']==storage['request_id']==FAILED
         and close['terminal_native_sha256']==r['terminal_native_sha256'] and close['native_exit_status']==1
         and close['GPU_lease_status']==close['physical_lease_status']=='CLOSED'
         and all(type(close[k]) is int and close[k]==0 for k in ('active_native_actors','remaining_holds',
              'actual_valid_conditions','actual_model_loads','actual_queries','actual_physics','actual_training_updates'))
         and e['effect_journal_complete'] is True and e['unknown_query_intents']==[]
         and all(type(e[k]) is int and e[k]==0 for k in ('optimizer_updates_committed','new_physics',
              'new_episodes','query_attempts','model_load_attempts')),
         'FULL_ACTUAL_KNOWN_EFFECTS_AND_BOTH_LEGAL_CLOSURES')
    need(storage['schema']=='PV1_KNOWN_ZERO_CLOSED_UNIQUE_ALLOCATED_OUTPUT_V1'
         and storage['no_historical_charge_refunded'] is True and storage['no_future_upper_deleted'] is True
         and p['new_resident_allocated_upper_bytes']==storage['neural']['conservative_new_allocated_upper_bytes']
         and g['Core_B_cold_read_failure_storage']['proof_sha256']==hashlib.sha256((json.dumps(storage,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
         and all(64*1024**2<=storage[role]['conservative_new_allocated_upper_bytes']<=256*1024**2
             for role in ('neural','physical')),'KNOWN_ZERO_NEW_OUTPUT_AND_PREPAID_SHARED_INPUT_INODES')
    need(m['exact_read_failure_diagnostic_sha256']=='48062b4b1175e0a582be055747b55e1596728fe116cb1df1302500b39c094eae'
         and re.fullmatch('[0-9a-f]{64}',str(m['independent_closure_readback_sha256'])) is not None,
         'EXACT_QSELECT_READ_TIMEOUT_DIAGNOSTIC_NOT_A_MODEL_OR_SAFETY_FAILURE')
    rid=m['replacement_request_id'];nonce=m['replacement_submission_nonce']
    need(c.canonical_uuid(rid) and c.canonical_uuid(nonce) and c.canonical_uuid(m['control_id'])
         and rid not in (FAILED,first_marker['failed_request_id'],first_marker['A_dual_closure']['request_id'])
         and m['replacement_run_uuid']==RUN and re.fullmatch('[0-9a-f]{40}',str(m['replacement_source_sha'])) is not None
         and not any(row['submission_nonce']==nonce for key,row in rows.items() if key!=rid),
         'DISTINCT_NEW_COLD_REQUEST_RUN_SOURCE_NONCE')
    if rid in rows:
        rr=rows[rid]
        need(rr['evaluation_candidate']=='B_COVERAGE' and rr['noise_bank']==294101
             and rr['child_run_uuid']==RUN and rr['source_sha']==m['replacement_source_sha']
             and rr['submission_nonce']==nonce and rr.get('startup_recovery_sha256')==c.digest(m)
             and not any(k.startswith('queue_') for k in rr),'ONE_ORIGINAL_B_PANEL_NEW_IDENTITY_NO_QUEUE_REPLAY')
    return m

def install(g,l,marker):
    from . import evaluation_core_accounting as c,evaluation_core_zero_start_recovery as z
    c.checked(g);c.scope.validate_pair(g,l);need(FIELD not in g and FIELD not in l,'ONCE_NEW_COLD_BOUNDARY')
    need(all(r['status']==c.STATES[3] for r in g[c.FIELD].values())
         and g['unknown_effects']==l['unknown_effects']==[]
         and all(r['status']=='CLOSED' for r in g['V4_DEV_control_reservations'].values()),'ALL_OLD_NATIVE_AND_CONTROL_CLOSED')
    g,l=deepcopy(g),deepcopy(l)
    for value in (g,l):value[FIELD]=deepcopy(marker)
    validate(g,g[z.FIELD]);c.checked(g);c.scope.validate_pair(g,l);return g,l
