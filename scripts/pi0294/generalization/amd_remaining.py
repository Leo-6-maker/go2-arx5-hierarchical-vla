"""Original ten never-attempted GEN1 cells after saved T1 re-audit closes."""
from copy import deepcopy
from . import amd_adoption as a

PROFILE='amd-gen1-teacher-capture-r2'
STATUS='RESERVED_TEN_UNATTEMPTED_ORIGINAL_CONDITIONS'

def cards(packet_bytes,index):
    from .amd_capture import cards as original_cards
    values=original_cards(packet_bytes,index)
    a.need(values[0]['task']=='T1'and len(values)==6,'REMAINING_FIXED_FIRST_SAVED_T1')
    return values[1:]

def allocation(value,*,packet_bytes,workers,closed_reaudit_lease,now_epoch):
    from .amd_capture import COUNTER_POLICY
    mark=a.validate(value);old=mark['teacher_pilot']['engineering_recovery'];audit=old['saved_data_reaudit']
    a.need(old['status']=='NATIVE_AUDIT_ENGINEERING_TERMINAL'
        and audit['status']=='KNOWN_SAVED_RAW_REAUDIT_SETTLED'and 'remaining_conditions'not in old
        and set(audit['raw_admitted_cases'])=={'0','1'}and audit['actual']['unknown_native_effects']==[],
        'REMAINING_ONLY_AFTER_KNOWN_SAVED_DATA_AUDIT')
    a.need(closed_reaudit_lease['status']=='CLOSED'
        and closed_reaudit_lease['manifest_sha256']==audit['manifest_sha256']
        and closed_reaudit_lease['independent_terminal_audit_sha256']==audit['independent_terminal_audit_sha256'],
        'REMAINING_REAL_CLOSED_REAUDIT_ROLE')
    a.need(mark['T_CD']<=now_epoch<mark['stop_new_training_epoch']-18.25*3600,
        'REMAINING_EXPORT_TRAINING_TIME')
    a.need(set(workers)=={'0','1'},'REMAINING_TWO_DISJOINT_FINITE_WORKERS')
    ids=[]
    for index in (0,1):
        row=workers[str(index)];run='gen1-amd-teacher-r2-'+str(index).zfill(2)
        expected=[c['id']for c in cards(packet_bytes,index)];saved=audit['raw_admitted_cases'][str(index)]
        a.need(row['run_uuid']==run and row['root'].endswith('/children/'+run)
            and row['root']!=saved['raw_root']and row['attempt_ids']==expected
            and saved['attempt_id']not in expected and row['counter_policy']==COUNTER_POLICY
            and type(row['manifest_sha256'])is str and len(row['manifest_sha256'])==64,
            'REMAINING_NO_FULL_SAVED_CASE_OR_VALID_FAILURE_REPLAY')
        ids+=expected
    a.need(ids==old['unattempted_condition_ids']and len(set(ids))==10,'REMAINING_EXACT_ORIGINAL_TEN')
    out=deepcopy(value);row=dict(status=STATUS,workers=deepcopy(workers),logical_conditions=10,
        episode_starts=10,physics_ticks=320000,engine_starts=880,Core_CPU_s=36000,
        complete_saved_cases_not_recaptured=2,previous_partial_attempts_retained=2,
        new_GPU=0,new_PBS=0,valid_scientific_failure_replay=False,
        independent_saved_audit_sha256=audit['independent_terminal_audit_sha256'],
        closed_reaudit_lease_sha256=a.sha(a.raw(closed_reaudit_lease)))
    out[a.FIELD]['teacher_pilot']['engineering_recovery']['remaining_conditions']=row;out['revision']+=1
    return dict(proposed_global=out,row=row,new_GPU=0,new_PBS=0)
