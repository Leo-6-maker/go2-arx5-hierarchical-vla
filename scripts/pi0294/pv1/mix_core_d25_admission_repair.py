"""One cold repair of the actual closed05 pre-grant PBS arithmetic failure.

The known native attempt, all invoices and its closed roles remain history.
Only the original D25 panel is restored to the existing remaining escrow.
No transport, registration, grant activation or model permission is provided.
"""
from copy import deepcopy
import json,re
from . import mix_accounting as a

FIELD='V5_CD_D25_closed05_pregrant_admission_repair06'
SCHEMA='V5_CD_D25_EXACT_CLOSED05_PREGRANT_REPAIR06_V1'
FAILED='3e774da8-1f75-4930-b732-3f0409f3d5a4'
RUN='v5-cd-d25-n294101-06'
AUTHORITY='49506e3a135ad6ad931cd672cb9f0b9f6214e47fb072fd160b3f29b5932f007b'
GLOBAL='d58c94168ca4701d1542f36a599ab5a15e942045e4a2062e8edce0f4a7ef0c83'
LOCAL='e02a670577c6b54884fff683c1d300b8a5142afba644e467516b224efb13ac9c'
FIXED_ADMISSION='d9b1eae4c7fc0289842cb8bc6a56b3619e109358eb8f05901d3e4a0070c1e2c0'

def need(ok,why):
    if not ok:raise PermissionError('D25_CLOSED05_PREGRANT_REPAIR_'+why)
def digest(v):return a.sha(a.raw(v))


def validate(g,financial_marker=None):
    """Nonrecursive exact history/identity check inside existing validators."""
    m=g.get(FIELD)
    if m is None:return None
    from . import mix_core_accounting as c,mix_core_d25_local_successor as paid
    from . import mix_core_d25_cold_source as cold
    original=financial_marker or g[paid.FIELD];row=g[c.FIELD][FAILED]
    need(m['schema']==SCHEMA and m['authority_sha256']==AUTHORITY
         and m['failed_request_id']==FAILED and m['failed_row_sha256']==digest(row)
         and m['financial_marker_sha256']==digest(original)
         and m['closed05_marker_sha256']==digest(g[cold.LOCAL05_FIELD])
         and m['before_GLOBAL_sha256']==GLOBAL and m['before_LOCAL_sha256']==LOCAL,
         'EXACT_EXISTING_AUTHORITY_AND_UNCHANGED_TERMINAL_HISTORY')
    c.validate_row(row)
    need(row['status']==c.STATES[3]and row['native_exit_status']==1
         and row['child_run_uuid']==cold.LOCAL05_RUN
         and row['source_sha']=='371c5cc4fc9bebb8e823c14f6f27d1ddd41926fa'
         and row['conservative_GPU_s']==243 and row['conservative_CPU_s']==7
         and row['query_attempts']==row['model_load_attempts']==0
         and row['unknown_query_intents']==[]and not row.get('charged_effects')
         and FAILED not in g.get('V5_CD_Core_finite_delegations',{}),
         'ONLY_REAL_OS1_BEFORE_GRANT_AND_ZERO_MODEL_EFFECTS')
    need(c.canonical_uuid(m['replacement_request_id'])and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_request_id']!=FAILED
         and m['replacement_submission_nonce']!=row['submission_nonce']
         and m['replacement_run_uuid']==m['run_uuid']==RUN
         and re.fullmatch('[0-9a-f]{40}',m['source_sha'])
         and m['source_sha']!=row['source_sha']
         and re.fullmatch('[0-9a-f]{64}',m['source_seal_sha256'])
         and m['fixed_admission_file_sha256']==FIXED_ADMISSION,
         'NEW_COLD_SOURCE_REQUEST_NONCE_AND_EXACT_ADMISSION_FIX')
    need(m['maximum_new_native_requests']==1 and m['new_control_payment']==0
         and m['old_control_owner_CPU_s_carried']==5
         and m['native_remaining_allocation']==paid.allocation(g,'D25',_validated_marker=original)
         and m['old_stage_completed']is False and m['closed_roles_reopened']is False
         and m['T_CD']==row['T_CD']and m['deadline_epoch']==row['deadline_epoch']
         and row['settled_epoch']<=m['adopted_epoch']<min(row['deadline_epoch']-28860,1793404800),
         'ONE_SAME_PANEL_FINANCIAL_ENVELOPE_AND_ORIGINAL_CLOCK')
    proof=m['closed05_readback']
    need(proof['status']=='ACTUAL_CLOSED05_NATIVE_OS1_NO_GRANT_AND_NEW06_ABSENT_BOUNDARY'
         and proof['LOCAL_sha256']==LOCAL and proof['registry_sha256']=='3fd4cd9a098faa45746bf4ca14b6e74662654985ce2cf1a6fd8e60c9d4976fdb'
         and proof['all_units_terminal_OS1']is proof['grant_and_activation_absent']is proof['new06_roots_absent']is True
         and proof['new_mutations']==0 and set(proof['roles'])=={'owner','core','hopper'}
         and digest(proof)==m['closed05_readback_sha256'],
         'EXACT_THREE_REAL_CLOSED_ROLES_AND_PRE_GRANT_FAILURE_READBACK')
    for name,item in proof['roles'].items():
        need(item['root'].endswith('/'+('v5-cd-d25-eval-owner-05'if name=='owner'else cold.LOCAL05_RUN))
             and item['lease']==item['root']+'/control/OWN_LEASE.private.json'
             and re.fullmatch('[0-9a-f]{64}',item['closed_lease_sha256']), 'OLD_LOCAL_CLOSED_ROLE:'+name)
    new=g[c.FIELD].get(m['replacement_request_id'])
    if new is not None:
        need(new['evaluation_candidate']=='D25'and new['child_run_uuid']==RUN
             and new['source_sha']==m['source_sha']and new['submission_nonce']==m['replacement_submission_nonce']
             and new['checkpoint_manifest_sha256']==row['checkpoint_manifest_sha256']
             and new['planned_panel_sha256']==row['planned_panel_sha256']
             and new['training_handoff_sha256']==row['training_handoff_sha256']
             and new.get('D25_local_successor_sha256')==digest(original)
             and new.get(cold.DESCRIPTOR)==digest(m),'ONLY_SAME_MODEL_POPULATION_AND_NEW_BOUND_IDENTITY')
    return m


def propose(global_bytes,local_bytes,*,authority_bytes,closed_readback,source_seal_bytes,
            frozen,source_readback,new_request_id,new_nonce,now_epoch):
    from . import mix_core_accounting as c,mix_core_d25_local_successor as paid,mix_core_d25_cold_source as cold
    from .evaluation_core_scope import ORIGINAL_NUMERICAL_TABLE_SHA
    g,l=map(json.loads,(global_bytes,local_bytes));c.checked(g);c.paired(g,l)
    need(FIELD not in g and FIELD not in l and a.sha(global_bytes)==GLOBAL and a.sha(local_bytes)==LOCAL
         and a.sha(authority_bytes)==AUTHORITY and json.loads(authority_bytes)['direct_answer']=='正式批准第5节全文并执行'
         and all(r['status']==c.STATES[3]for r in g[c.FIELD].values())
         and all(r['status']=='NATIVE_TERMINAL'for r in g['PBS_submissions'])
         and g['pending_updates']==g['unknown_effects']==[], 'EXACT_ONCE_COLD_SETTLED_BOUNDARY')
    need(0<=now_epoch-closed_readback['observed_epoch']<=120,'FRESH_ACTUAL_CLOSED05_BOUNDARY')
    original=paid.validated(g);seal=json.loads(source_seal_bytes)
    need(len(frozen)==15 and a.sha(json.dumps(frozen,sort_keys=True).encode())==ORIGINAL_NUMERICAL_TABLE_SHA
         and all(seal['files'].get(n)==pin for n,pin in frozen.items())
         and seal['files']['scripts/pi0294/pv1/mix_core_admission.py']==FIXED_ADMISSION,
         'ALL_ORIGINAL_NUMERICAL_BYTES_AND_PUBLISHED_PRECISE_FIX')
    need(source_readback['status']=='ACTUAL_COMPLETE_COLD_SOURCE_SHA_READBACK'
         and source_readback['source_sha']==seal['source_sha']
         and source_readback['source_seal_sha256']==a.sha(source_seal_bytes)
         and source_readback['file_sha256']==seal['files']
         and source_readback['hostname']=='corelab-4080s'and source_readback['uid']==2009
         and source_readback['model_calls']==source_readback['physics_ticks']==0
         and 0<=now_epoch-source_readback['observed_epoch']<=120,'ACTUAL_COMPLETE_NEW_SOURCE_BYTES')
    identities=[*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]
    need(not any(r.get('request_id')==new_request_id or r.get('submission_nonce')==new_nonce for r in identities),'DISTINCT_UNUSED_IDENTITIES')
    old=g[c.FIELD][FAILED];receipt=l['V5_CD_Core_terminal_settlements'][FAILED]
    need(receipt['terminal_native_sha256']==old['terminal_native_sha256']
         and receipt['GPU_s']==243 and receipt['neural_CPU_s']==7
         and receipt['physical_CPU_s']==300 and all(receipt['physical'][k]==0 for k in('episodes','constructors','live','shadow'))
         and receipt['stage_completed']is False and l['V5_CD_Core_CPU_substages'][c.key('D25')]['status']=='SETTLED_KNOWN_NATIVE_TERMINAL',
         'REAL_DUAL_SETTLEMENT_AND_UNUSED_ORIGINAL_PHYSICAL_PANEL')
    active=paid.allocation(g,'D25')
    m=dict(schema=SCHEMA,authority_sha256=AUTHORITY,failed_request_id=FAILED,failed_row_sha256=digest(old),
        financial_marker_sha256=digest(original),closed05_marker_sha256=digest(g[cold.LOCAL05_FIELD]),
        before_GLOBAL_sha256=GLOBAL,before_LOCAL_sha256=LOCAL,source_sha=seal['source_sha'],
        source_seal_sha256=a.sha(source_seal_bytes),fixed_admission_file_sha256=FIXED_ADMISSION,
        original_numerical_table_sha256=ORIGINAL_NUMERICAL_TABLE_SHA,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,replacement_run_uuid=RUN,run_uuid=RUN,
        maximum_new_native_requests=1,new_control_payment=0,old_control_owner_CPU_s_carried=5,
        native_remaining_allocation=active,old_stage_completed=False,closed_roles_reopened=False,
        closed05_readback=deepcopy(closed_readback),closed05_readback_sha256=digest(closed_readback),
        full_source_readback_sha256=digest(source_readback),T_CD=old['T_CD'],deadline_epoch=old['deadline_epoch'],adopted_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l)
    for value in(v,w):
        value[FIELD]=deepcopy(m)
        for key,cost in active.items():value[a.PACKAGE]['remaining_resources'][key]+=cost
    for key in('H200_s','neural_queries','optimizer_updates'):v['reserved'][key]+=active[key]
    v['CPU_reserved_s']+=active['Hopper_CPU_s']
    w['GPU_reserved_s']+=active['H200_s']
    for key,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=active[key]
    v['revision']+=1;validate(v,original);c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));need(full['cap_changes_required']=={},'ORIGINAL_ALL14_WITH_OLD_FEES_AND_ONE_REMAINING_PANEL')
    need(v[c.FIELD]==g[c.FIELD]and v['used']==g['used']and v['CPU_s']==g['CPU_s']and w['CPU_s']==l['CPU_s'],
         'NO_NATIVE_HISTORY_COST_OR_PREPAID_CONTROL_RESET')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,native_admission=False,original_onceIO_required=True)
