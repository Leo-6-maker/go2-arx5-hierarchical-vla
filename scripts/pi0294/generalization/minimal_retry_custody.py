"""Readonly custody requirements for a single zero-effect engineering retry.

This module does not issue a retry, release a reservation, credit storage, or
modify a ledger. The original settled row and its gross IO charge stay intact.
"""
import json,math
from . import minimal_execution as e,minimal_terminal as terminal,minimal_scope as scope


def inspect(global_bytes,local_bytes,*,native_readback,closed_readback):
    g=json.loads(global_bytes);l=json.loads(local_bytes);marker=e.validate(g);rows=e.rows(g)
    e.need(marker is not None and len(rows)==1,'RETRY_ONE_ORIGINAL_MAIN')
    row=rows[0];settled=marker['terminal'];actual=terminal.validate_terminal(g,row,marker)
    e.need(row['status']==e.STATES[2] and row['native_exit_status']!=0
        and actual['success'] is False and actual['updates']==actual['queries']==actual['loads']==0
        and settled['effects']['new_physics']==settled['effects']['new_episodes']==0
        and settled['effects']['pending_updates']==settled['effects']['unknown_query_intents']==[],
        'RETRY_EXACT_SETTLED_ZERO_EFFECT_FAILURE')
    v=native_readback;closed=closed_readback
    e.need(v['status']=='ACTUAL_FIRST512_NATIVE_F_EXIT1_INDEPENDENT_PARTIAL_AUDIT_READONLY'
        and all(v[k]==settled[k] for k in ('native','effects','audit'))
        and v['inventory']['worker_inventory_sha256']==settled['native']['inventory_sha256']
        and v['inventory']['allocated_bytes']>=0 and v['new_writes']==0,
        'RETRY_ORIGINAL_INDEPENDENT_ACTUAL_INVENTORY')
    e.need(closed['status']=='INDEPENDENT_ORIGINAL_LEDGER_AND_OWN_LEASE_CLOSED'
        and closed['amounts']==actual and closed['unknown_effects']==closed['pending_updates']==[]
        and closed['new_writes']==0 and len(closed['lease_sha256'])==64,
        'RETRY_SAME_TERMINAL_ONCE_AND_OWN_LEASE_CLOSED')
    e.need(g['unknown_effects']==g['pending_updates']==[] and l['unknown_effects']==l['pending_updates']==l['pending_queries']==[],
        'RETRY_NEVER_CLEAR_OR_IGNORE_LATCHES')
    now=scope.first512_forecast(global_bytes,local_bytes)
    main=e.native_resources();replacement=scope.components()['ONE_KNOWN_ZERO_EFFECT_COLD_ENGINEERING_REPLACEMENT']
    gross={};gaps={}
    for k,dimension in now['dimensions'].items():
        projected=dimension['projected']+main[k]-replacement[k]
        e.need(math.isfinite(projected) and projected>=0,'RETRY_GROSS_NONNEGATIVE')
        gross[k]=dict(projected=projected,cap=dimension['cap'],shortfall=max(0,projected-dimension['cap']))
        if gross[k]['shortfall']:gaps[k]=gross[k]['shortfall']
    return dict(status='ZERO_EFFECT_CUSTODY_ONLY_NO_RETRY_PERMISSION',request_id=row['request_id'],
        job_id=row['job_id'],inventory_sha256=v['inventory']['worker_inventory_sha256'],
        retained_worker_allocated_bytes=v['inventory']['allocated_bytes'],
        settled_original_output_gross_bytes=main['Hopper_storage_bytes'],
        gross_duplicate_charge_forecast=gross,gross_duplicate_charge_shortfalls=gaps,
        exact_reused_IO_credit_granted=False,ledger_changed=False,native_dispatch_admitted=False,
        independent_physical_output_limit_required=True,original_costs_and_peak_preserved=True)
