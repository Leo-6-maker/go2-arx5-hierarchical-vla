"""Strict original-row history for one zero-effect main replacement.

This constructs and validates a cold provenance packet only. It cannot alter
the one-row ledger, grant a storage credit, register, bind or launch a job.
"""
import json
from copy import deepcopy
from . import minimal_execution as e,minimal_retry_custody as custody,minimal_terminal as terminal

SCHEMA='ARX5_MINIMAL_FIRST512_CLOSED_PREDECESSOR_HISTORY_V1'


def capture(global_bytes,local_bytes,*,native_readback,closed_readback):
    g=json.loads(global_bytes);proof=custody.inspect(global_bytes,local_bytes,native_readback=native_readback,closed_readback=closed_readback)
    marker=e.validate(g);row=e.rows(g)[0]
    value=dict(schema=SCHEMA,predecessor_GLOBAL_sha256=e.sha(global_bytes),
        original_marker=deepcopy(marker),original_row=deepcopy(row),
        original_row_sha256=e.sha(e.raw(row)),original_marker_sha256=e.sha(e.raw(marker)),
        independent_audit_sha256=e.sha(e.raw(native_readback['audit'])),
        stable_worker_inventory_sha256=native_readback['inventory']['worker_inventory_sha256'],
        original_closed_lease_sha256=closed_readback['lease_sha256'],
        original_paid_output_bytes=proof['settled_original_output_gross_bytes'],
        original_costs_preserved=True,new_GPU_or_model_or_update=False)
    verify(g,value);return value


def verify(g,packet):
    e.need(packet.get('schema')==SCHEMA and packet.get('original_costs_preserved') is True
        and packet.get('new_GPU_or_model_or_update') is False,'RETRY_HISTORY_EXACT_COLD_CONTEXT')
    original=packet['original_marker'];row=packet['original_row']
    e.need(e.sha(e.raw(original))==packet['original_marker_sha256']
        and e.sha(e.raw(row))==packet['original_row_sha256']
        and sum(x==row for x in g['PBS_submissions'])==1
        and row['status']==e.STATES[2] and row['native_exit_status']!=0,
        'RETRY_HISTORY_EXACT_ORIGINAL_ROW_AND_MARKER')
    saved=original['terminal'];amount=terminal.terminal_values(row,saved['native'],saved['effects'],saved['audit'])
    e.need(not amount['success'] and amount['updates']==amount['queries']==amount['loads']==0
        and e.sha(e.raw(saved['audit']))==packet['independent_audit_sha256']
        and saved['native']['inventory_sha256']==packet['stable_worker_inventory_sha256']
        and packet['original_paid_output_bytes']==e.native_resources()['Hopper_storage_bytes']
        and len(packet['original_closed_lease_sha256'])==64,
        'RETRY_HISTORY_TRUE_ZERO_EFFECT_AND_GROSS_COST')
    # Audit the old terminal with a local virtual view of its original package
    # status; this never changes the authoritative ledger or a retry status.
    virtual=deepcopy(g);virtual[e.package.FIELD]['status']='MAIN_NATIVE_TERMINAL_INCOMPLETE'
    virtual[e.package.FIELD]['completed_sessions']=[]
    terminal.validate_terminal(virtual,row,original)
    return dict(original_row_sha256=packet['original_row_sha256'],failed_request_id=row['request_id'],
        paid_output_bytes=packet['original_paid_output_bytes'],history_mutated=False,native_admission=False)
