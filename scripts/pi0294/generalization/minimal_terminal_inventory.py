"""Once Core LOCAL catch-up to three actually closed first512 allocations.

Keep the original mirror and every paid projection. This is an accounting
inventory repair at an all-terminal boundary, not a live cross-site permit.
No request, budget reservation, GPU, model, heartbeat or physics is started.
"""
from copy import deepcopy
import json
from . import minimal_execution as e,minimal_retry_execution as retry
from . import minimal_partial_completion_scope as completion,minimal_concurrency as concurrency

FIELD='ARX5_first512_LOCAL_closed_inventory_catchup_1504095'
SCHEMA='ARX5_FIRST512_LOCAL_ACTUAL_THREE_CLOSED_REQUESTS_CATCHUP_V1'
LOCAL='91ce7cc7621d1ddce5cb2329314b35a8cad67c0d33d240bbb43a63c88e193dae'
IDS=('b2773e0d-24e6-4d0b-93db-62cccf3e7ef3','619032fd-30a8-41ea-88de-95510a4fade9',completion.REQUEST)
LATEST_FIELD=FIELD+'_1507480'
LATEST_LOCAL='cad80ab3cc880c0bb60b40c2eca113c5d3f46e2ea70462fc78806a231b8f98d8'
LATEST_IDS=('80aeab00-91c9-455d-96bc-587f716f227f',
            '586b6601-daf5-4823-950b-605896e611c0','920f2ed2-fa6b-41d5-9ad5-f1683e4759e8')


def propose(global_bytes,local_bytes,*,fresh_readback,now_epoch):
    from scripts.pi0294.pv1 import mix_core_accounting as core
    from scripts.pi0294.pv1.neural_ledger import check
    from .minimal_scope import first512_forecast
    g,l=json.loads(global_bytes),json.loads(local_bytes);check(g)
    e.need(e.sha(local_bytes)==LOCAL and FIELD not in l
        and completion.validate(g)is not None
        and g['unknown_effects']==g['pending_updates']==[]
        and l['unknown_effects']==l['pending_updates']==l['pending_queries']==[]
        and all(r['status']in e.original.TERMINAL for r in g['PBS_submissions'])
        and all(r['status']=='NATIVE_TERMINAL'for r in g.get(core.FIELD,{}).values()),
        'TERMINAL_MIRROR_EXACT_LAST_CONFIRMED_LOCAL_ALL_NATIVE_CLOSED_NOT_ACTIVE_ALIAS')
    e.need(0<=now_epoch-fresh_readback['observed_epoch']<=120
        and fresh_readback['LOCAL_sha256']==LOCAL
        and fresh_readback['GLOBAL_sha256']==e.sha(global_bytes)
        and fresh_readback['new_mutations']==0,'TERMINAL_MIRROR_FRESH_BOTH_CANONICAL_BYTES')
    e.need(len(l['PBS_submissions'])==36 and len(g['PBS_submissions'])==38
        and tuple(r.get('request_id')for r in g['PBS_submissions'][-3:])==IDS,
        'TERMINAL_MIRROR_EXACT_TWO_MISSING_ROWS_NO_NEW_POPULATION')
    old_mirror=l[concurrency.RETRY_LOCAL_MIRROR]
    e.need(old_mirror['main_binding']['main_request_id']==IDS[0]
        and old_mirror['main_binding']['main_job_id']=='1502754.stdct-mgmt-02',
        'TERMINAL_MIRROR_ORIGINAL_OLD_BINDING_PRESERVED')
    allowed=retry.TERMINAL_KEYS|{'status'}
    for i,old in enumerate(l['PBS_submissions']):
        actual=g['PBS_submissions'][i]
        if i<35:e.need(actual==old,'TERMINAL_MIRROR_NO_OLDER_ROW_REWRITE')
        else:
            e.need(old['request_id']==actual['request_id']==IDS[0]
                and old['status']==e.STATES[1]and actual['status']==e.STATES[2]
                and all(old[k]==actual.get(k)for k in old if k not in allowed)
                and set(actual)-set(old)<=allowed,
                'TERMINAL_MIRROR_ONLY_REAL_TERMINAL_FIELDS_ADDED_TO_OLD_ROW')
    terminal_rows=[]
    for generation,rid in enumerate(IDS,1):
        marker=retry.validate(g,generation=generation);_,row=retry._selected(g,generation=generation)
        e.need(row['request_id']==rid and row['status']==e.STATES[2]and row['native_exit_status']==1
            and row['scientific_endpoint_complete']is False,
            'TERMINAL_MIRROR_TRUE_TYPED_NATIVE_F1_AND_PARTIAL_INDEPENDENT_AUDIT')
        effects=marker['terminal']['effects']
        e.need(effects['pending_updates']==effects['unknown_query_intents']==[],
            'TERMINAL_MIRROR_DO_NOT_CLEAR_ANY_UNKNOWN_OR_PENDING')
        terminal_rows.append(dict(request_id=rid,job_id=row['job_id'],row_sha256=e.sha(e.raw(row)),
            native_sha256=row['terminal_native_sha256'],audit_sha256=row['independent_audit_sha256'],
            updates=row['committed_optimizer_updates'],queries=row['query_attempts'],loads=row['model_load_attempts']))
    before=first512_forecast(global_bytes,local_bytes)
    # These same paid-control projections are already included by the complete
    # forecast. Apply once to actual LOCAL, not a new payment or resource grant.
    value=completion.budget_LOCAL(g,retry.budget_LOCAL(g,l))
    value['PBS_submissions']=deepcopy(g['PBS_submissions'])
    value[FIELD]=dict(schema=SCHEMA,GLOBAL_sha256=e.sha(global_bytes),before_LOCAL_sha256=LOCAL,
        old_retry_mirror_sha256=e.sha(e.raw(old_mirror)),closed_rows=terminal_rows,
        actual_previous_mirror_preserved=True,observed_epoch=now_epoch,
        actual_GLOBAL_already_paid_controls=True,new_control_or_budget_payment=False,
        new_model_queries_updates_or_physics=0,native_admission=False)
    core.paired(g,value);after=first512_forecast(global_bytes,e.raw(value))
    e.need(value[concurrency.RETRY_LOCAL_MIRROR]==old_mirror
        and before['cap_changes_required']==after['cap_changes_required']=={}
        and all(before['dimensions'][k]['projected']==after['dimensions'][k]['projected']for k in before['dimensions']),
        'TERMINAL_MIRROR_ALL14_TOTALS_UNCHANGED_NO_COST_REFUND_OR_DROPPED_ESCROW')
    return dict(proposed_LOCAL=value,full_future_proof=after,terminal_rows=terminal_rows,
        original_onceIO_required=True,new_budget=False,new_request_PBS_model_or_physics=0,native_admission=False)


def propose_latest(global_bytes,local_bytes,*,fresh_GLOBAL,closed_D25_readback,now_epoch):
    """Mirror only the three known closed suffix rows before cold D25-05.

    The older catch-up and all failures remain. A proved unstarted closed04
    preparation is not represented as a native terminal or reopened role.
    """
    from scripts.pi0294.pv1 import mix_core_accounting as core,mix_vanda_first as first
    from scripts.pi0294.pv1.neural_ledger import check
    from . import minimal_execution_adapter as adapters
    from .minimal_scope import first512_forecast
    g,l=json.loads(global_bytes),json.loads(local_bytes);check(g)
    e.need(e.sha(local_bytes)==LATEST_LOCAL and LATEST_FIELD not in l
        and len(l['PBS_submissions'])==38 and len(g['PBS_submissions'])==41
        and l['PBS_submissions']==g['PBS_submissions'][:38]
        and tuple(r['request_id']for r in g['PBS_submissions'][38:])==LATEST_IDS
        and g['pending_updates']==g['unknown_effects']==l['pending_queries']==l['pending_updates']==l['unknown_effects']==[]
        and all(r['status']in e.original.TERMINAL for r in g['PBS_submissions']),
        'LATEST_LOCAL_EXACT_UNMODIFIED_PREFIX_AND_THREE_CLOSED_SUFFIX_ROWS')
    e.need(fresh_GLOBAL['GLOBAL_sha256']==e.sha(global_bytes)and fresh_GLOBAL['new_mutations']==0
        and closed_D25_readback['LOCAL_sha256']==LATEST_LOCAL
        and closed_D25_readback['status']=='INDEPENDENT_CLOSED04_BYTES_LOCAL_AND_REGISTRY_HASHES_QUALIFIED_NEW05_ABSENT'
        and closed_D25_readback['actors']==[]and closed_D25_readback['new_mutations']==0
        and all(0<=now_epoch-r['observed_epoch']<=120 for r in(fresh_GLOBAL,closed_D25_readback)),
        'LATEST_LOCAL_FRESH_BOTH_CANONICAL_HASHES_AND_NO_OLD_OWNER')
    active=[r for r in g[core.FIELD].values()if r['status']!=core.STATES[3]]
    from scripts.pi0294.pv1.mix_core_d25_cold_source import REQUEST
    e.need(len(active)==1 and active[0]['request_id']==REQUEST and active[0]['status']==core.STATES[0]
        and active[0]['native_handle']is None and not active[0].get('charged_effects')
        and not active[0].get('query_attempts',0)and not active[0].get('model_load_attempts',0),
        'LATEST_LOCAL_ONLY_ORIGINAL_UNSTARTED_D25_ESCROW')
    for role,item in closed_D25_readback['roles'].items():
        record=item['record']
        e.need(record['status']=='ACTUAL_UNUSED_COLD_ROLE_CLOSED_WITH_NO_NATIVE_ACTOR'
            and record['request_id']==REQUEST and record['role']==role
            and record['models_queries_physics']==0 and record['live_actors']==[]
            and e.sha(e.raw(record))==item['record_sha256'],
            'LATEST_LOCAL_REAL_CLOSED04_RECORD:'+role)
    e.need(set(closed_D25_readback['roles'])=={'owner','physical','neural'}
        and closed_D25_readback['roles']['neural']['manifest_sha256']==active[0]['run_manifest_sha256'],
        'LATEST_LOCAL_ALL_THREE_CLOSURES_MATCH_PENDING_SCIENTIFIC_ROW')
    terminal=[]
    for rid in LATEST_IDS:
        adapter,row,marker=adapters.registered(g,rid)
        effects=marker['terminal']['effects']
        e.need(row['status']==e.STATES[2]and row['native_exit_status']==1
            and effects['pending_updates']==effects['unknown_query_intents']==[],
            'LATEST_LOCAL_ACTUAL_TYPED_FAILURE_AND_KNOWN_EFFECTS_NOT_A_NEW_COMPLETION')
        terminal.append(dict(request_id=rid,job_id=row['job_id'],row_sha256=e.sha(e.raw(row)),
            native_sha256=row['terminal_native_sha256'],audit_sha256=row['independent_audit_sha256']))
    before=first512_forecast(global_bytes,local_bytes)
    value=first.budget_LOCAL(g,l)
    value['PBS_submissions']=deepcopy(g['PBS_submissions'])
    value['GPU_allocated_s']=max(value['GPU_allocated_s'],g['used']['H200_s'])
    value['CPU_s']['Hopper']=max(value['CPU_s']['Hopper'],g['CPU_s'])
    for name,key in(('optimizer_updates','optimizer_updates'),('neural_queries','neural_queries')):
        value[name]=max(value[name],g['used'][key])
    value[LATEST_FIELD]=dict(schema='ARX5_LOCAL_REAL_CLOSED_SUFFIX_1505362_1506193_1507480_V1',
        GLOBAL_sha256=e.sha(global_bytes),before_LOCAL_sha256=LATEST_LOCAL,closed_rows=terminal,
        closed_D25_readback_sha256=e.sha(e.raw(closed_D25_readback)),
        old_history_and_all_paid_projections_retained=True,new_budget_or_control_payment=False,
        new_models_queries_updates_physics=0,native_admission=False,observed_epoch=now_epoch)
    core.paired(g,value);after=first512_forecast(global_bytes,e.raw(value))
    e.need(before['cap_changes_required']==after['cap_changes_required']=={}
        and all(before['dimensions'][k]['projected']==after['dimensions'][k]['projected']for k in before['dimensions']),
        'LATEST_LOCAL_ALL14_PROJECTED_TOTALS_IDENTICAL_NO_REFUND_OR_EXTRA_HOLD')
    return dict(proposed_LOCAL=value,full_future_proof=after,terminal_rows=terminal,
        original_onceIO_required=True,new_budget=False,new_request_PBS_model_or_physics=0,native_admission=False)
