"""Once-only original GLOBAL settlement of the three remaining real sessions."""
from copy import deepcopy
import math
from . import minimal_execution as e,minimal_e2only_execution as execution,minimal_terminal as original

SCHEMA='ARX5_E2ONLY_ORIGINAL_GLOBAL_ONCE_NATIVE_TERMINAL_V1'
need=e.need


def fields(native,effects,audit,amounts):
    return dict(native_exit_status=native['Exit_status'],conservative_H200_s=amounts['GPU_s'],
        conservative_CPU_s=amounts['CPU_s'],committed_optimizer_updates=amounts['updates'],query_attempts=amounts['queries'],
        model_load_attempts=amounts['loads'],terminal_native_sha256=e.sha(e.raw(native)),
        terminal_effect_summary_sha256=e.sha(e.raw(effects)),independent_audit_sha256=e.sha(e.raw(audit)),
        scientific_endpoint_complete=amounts['success'])


def validate_terminal(g,row,m):
    t=m.get('terminal',{});need(t.get('schema')==SCHEMA,'REMAINING_TERMINAL_COMPLETE_ORIGINAL_NATIVE_RECORDS')
    amounts=original.terminal_values(row,t['native'],t['effects'],t['audit'],e2only_global=g)
    expected=fields(t['native'],t['effects'],t['audit'],amounts)
    need(all(row.get(k)==v and type(row.get(k))is type(v)for k,v in expected.items())
        and type(row.get('settled_epoch'))in(int,float)and math.isfinite(row['settled_epoch'])
        and t['native']['observed_epoch']<=row['settled_epoch']
        and m['status']==('COMPLETE'if amounts['success']else'INCOMPLETE')and m['completed_sessions']==amounts['complete_sessions'],
        'REMAINING_TERMINAL_DERIVED_ROWS_NOT_PREPARED_OR_RUNTIME_FLAGS')
    from .minimal_e2only_physical import check_proof
    check_proof(t['physical'],m['physical_custody'],row,now_epoch=row['settled_epoch'],observed_epoch=t['physical_observed_epoch'])
    b=m['before']
    need(g['used']['H200_s']>=b['H200_s']+amounts['GPU_s']and g['CPU_s']>=b['CPU_s']+amounts['CPU_s']
        and g['optimizer_updates']>=b['optimizer_updates']+amounts['updates']
        and g['generation_queries']>=b['generation_queries']+amounts['queries']
        and g['model_loads']>=b['model_loads']+amounts['loads'],
        'REMAINING_TERMINAL_ALL_OLD_AND_NEW_COSTS_RETAINED')
    return amounts


def settle(g,request_id,*,native,effects,audit,physical_proof,physical_observed_epoch,now_epoch):
    execution.validate(g);m,row=execution.basic(g)
    need(row['request_id']==request_id and row['status']==e.STATES[1]
        and g['pending_updates']==g['unknown_effects']==[]and native['observed_epoch']<=now_epoch,
        'REMAINING_TERMINAL_EXACT_BOUND_REQUEST_ONCE_NO_UNKNOWN_CLEAR')
    amounts=original.terminal_values(row,native,effects,audit,e2only_global=g)
    v=deepcopy(g);index,r=execution.basic(v);held=execution.native_resources()
    for k in e.package.TRACKED:v['reserved'][k]-=held[k]
    v['CPU_reserved_s']-=held['Hopper_CPU_s'];v['storage_reserved_bytes']-=held['Hopper_storage_bytes']
    v['used']['H200_s']+=amounts['GPU_s'];v['CPU_s']+=amounts['CPU_s'];v['optimizer_updates']+=amounts['updates']
    v['used']['optimizer_updates']=v['optimizer_updates'];v['generation_queries']+=amounts['queries']
    v['used']['neural_queries']=v['generation_queries'];v['model_loads']+=amounts['loads']
    index.update(status='COMPLETE'if amounts['success']else'INCOMPLETE',completed_sessions=amounts['complete_sessions'],
        terminal=dict(schema=SCHEMA,native=deepcopy(native),effects=deepcopy(effects),audit=deepcopy(audit),
            physical=deepcopy(physical_proof),physical_observed_epoch=physical_observed_epoch))
    r.update(status=e.STATES[2],settled_epoch=now_epoch,**fields(native,effects,audit,amounts))
    v['revision']+=1;execution.validate(v)
    from scripts.pi0294.pv1.neural_ledger import check
    check(v);return v
