"""One cold B24 replacement after the exact known zero-effect startup failure.

The original two-row accepted queue, A24 result, failed B row and all costs
remain in their original indexes. This is not an automatic retry pool. Native
registration, current owner identity, full resources and qualification remain
separate. No IO is performed by the proposal/index helpers.
"""
from copy import deepcopy
from . import evaluation_core_B24_1000 as late
import hashlib
import json
import re

FIELD='Core_B_zero_start_cold_recovery'
SCHEMA='PV1_CORE_B_ZERO_START_COLD_REPLACEMENT_V1'
FAILED='f5426fad-a5e1-4157-a761-9a562f821770'
PREVIOUS_A='88ef30eb-bb1f-459d-bbb4-095647098ce4'
FAILED_REQUEST_SHA='fe99ace64be53bee2f36f82798cd159466ef2a93f9d2724e9cf159004535e257'
FAILED_FINAL_SHA='2e86eb5fcaffead46002865f991df9a742bc626256caac5211e7e948566f0fa5'


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_ZERO_START_'+why)


def validated(value):
    """Validate the one exact failure and leave every paid row in place."""
    from . import evaluation_core_accounting as c
    rows=value.get(c.FIELD,{})
    marker=value.get(FIELD)
    if marker is None:
        need(not any('startup_recovery_sha256' in r for r in rows.values()),'NO_ORPHAN_REPLACEMENT')
        return None
    need(type(marker) is dict and set(marker)=={'schema','failed_request_id','failed_request_sha256',
        'failed_terminal_native','failed_effects','failed_dual_closure','independent_closure_readback_sha256',
        'A_dual_closure','replacement_request_id','replacement_submission_nonce','replacement_run_uuid',
        'replacement_source_sha','control_id','original_stage_GPU_upper_s','automatic_retry',
        'maximum_replacements','same_original_B24_only','created_epoch'},'EXACT_SINGLE_COLD_BOUNDARY_METADATA')
    need(marker['schema']==SCHEMA and marker['failed_request_id']==FAILED
         and marker['failed_request_sha256']==FAILED_REQUEST_SHA
         and marker['original_stage_GPU_upper_s']==57720 and marker['automatic_retry'] is False
         and type(marker['maximum_replacements']) is int and marker['maximum_replacements']==1
         and marker['same_original_B24_only'] is True
         and c.old.T0<=marker['created_epoch']<c.old.STOP_NEW,'ONE_ORIGINAL_PANEL_NO_CLOCK_OR_BUDGET_EXTENSION')
    need(FAILED in rows and PREVIOUS_A in rows,'PRESERVE_BOTH_ORIGINAL_QUEUE_ROWS')
    old,first=rows[FAILED],rows[PREVIOUS_A]
    need(old['request_sha256']==FAILED_REQUEST_SHA and old['status']==c.STATES[3]
         and old['native_exit_status']==1 and old['query_attempts']==old['model_load_attempts']==0
         and c.progress(old)==(0,0,0) and old['unknown_query_intents']==[]
         and old['conservative_GPU_s']==70 and old['conservative_CPU_s']==1,
         'EXACT_FAILED_BOUND_NATIVE_ALREADY_SETTLED_ZERO_MODEL_QUERY_EFFECTS')
    proof=marker['failed_terminal_native'];effects=marker['failed_effects'];closed=marker['failed_dual_closure']
    need(c.digest(proof)==old['terminal_native_sha256'] and c.digest(effects)==old['terminal_effect_summary_sha256']
         and proof['actual_NATIVE_PROCESS_FINAL_sha256']==FAILED_FINAL_SHA
         and proof['request_id']==effects['request_id']==closed['request_id']==FAILED
         and proof['native_handle']==effects['native_handle']==old['native_handle']
         and effects['effect_journal_complete'] is True and effects['unknown_query_intents']==[]
         and all(type(effects[k]) is int and effects[k]==0 for k in ('optimizer_updates_committed',
             'new_physics','new_episodes','query_attempts','model_load_attempts'))
         and closed['terminal_native_sha256']==old['terminal_native_sha256']
         and closed['native_exit_status']==1 and closed['GPU_lease_status']==closed['physical_lease_status']=='CLOSED'
         and all(type(closed[k]) is int and closed[k]==0 for k in ('active_native_actors','remaining_holds',
             'actual_valid_conditions','actual_model_loads','actual_queries','actual_physics','actual_training_updates')),
         'FULL_KNOWN_ZERO_EFFECTS_AND_BOTH_ACTUAL_CLOSED_LEASES')
    a=marker['A_dual_closure']
    need(first['status']==c.STATES[3] and first['native_exit_status']==0
         and a['request_id']==PREVIOUS_A and a['terminal_native_sha256']==first['terminal_native_sha256']
         and a['GPU_lease_status']==a['physical_lease_status']=='CLOSED' and a['actual_valid_conditions']==24
         and a['active_native_actors']==a['remaining_holds']==0,'A24_AUDITED_AND_CLOSED_NO_REPLAY')
    rid=marker['replacement_request_id'];nonce=marker['replacement_submission_nonce']
    need(c.canonical_uuid(rid) and c.canonical_uuid(nonce) and c.canonical_uuid(marker['control_id'])
         and rid not in (FAILED,PREVIOUS_A) and nonce not in [r['submission_nonce'] for r in rows.values() if r['request_id']!=rid]
         and re.fullmatch('[0-9a-f]{40}',str(marker['replacement_source_sha'])) is not None
         and re.fullmatch('[0-9a-f]{64}',str(marker['independent_closure_readback_sha256'])) is not None
         and marker['replacement_run_uuid']=='pv1-v4a-b4096-n294101-zero-start-recovery-01',
         'NEW_UNIQUE_COLD_IDENTITIES_NOT_OLD_REQUEST_OR_RUN_REBIND')
    from .evaluation_core_read_failure_recovery import FIELD as read_field,validate as read_validate
    extra=value.get(read_field)
    allowed=({PREVIOUS_A,FAILED},{PREVIOUS_A,FAILED,rid}) if extra is None else ({PREVIOUS_A,FAILED,rid},{PREVIOUS_A,FAILED,rid,extra.get('replacement_request_id')})
    from . import evaluation_core_shared_B24 as shared
    if shared.FIELD in value:
        shared.validate(value);allowed=(*allowed,set(allowed[-1])|{shared.RID})
    need(set(rows) in allowed,'NO_UNCLASSIFIED_PROCESS_OR_REPLACEMENT_POOL')
    if extra is not None:read_validate(value,marker)
    if rid in rows:
        r=rows[rid]
        need(r['evaluation_candidate']=='B_COVERAGE' and r['noise_bank']==294101
             and r['source_sha']==marker['replacement_source_sha'] and r['submission_nonce']==nonce
             and r['child_run_uuid']==marker['replacement_run_uuid']
             and r.get('startup_recovery_sha256')==c.digest(marker)
             and not any(k.startswith('queue_') for k in r),'NEW_STANDALONE_B_NOT_REPLAYED_ACCEPTED_QUEUE')
    need(sum(r.get('conservative_GPU_s',0) if r['status']==c.STATES[3] else r['reserved_GPU_s']
             for r in rows.values())<=57720,'FAILED_START_REMAINS_INSIDE_ORIGINAL_STAGE_GPU_BOUND')
    return marker


def effective_rows(value):
    """Scientific panel count excludes only this paid zero-effect startup."""
    from .evaluation_core_accounting import FIELD as rows_field
    marker=validated(value)
    from .evaluation_core_read_failure_recovery import FIELD as read_field,FAILED as read_failed
    excluded=set() if marker is None else {FAILED}
    if read_field in value:excluded.add(read_failed)
    from . import evaluation_core_shared_B24 as shared
    if shared.FIELD in value:shared.validate(value);excluded.add(shared.failed.RID)
    return [r for key,r in value.get(rows_field,{}).items() if key not in excluded]


def install(global_value,LOCAL_value,marker):
    """Two proposal bytes only, before any new request reserve or start."""
    from . import evaluation_core_accounting as c
    g,l=deepcopy(global_value),deepcopy(LOCAL_value);c.checked(g);c.scope.validate_pair(g,l)
    need(FIELD not in g and FIELD not in l and not g['unknown_effects'] and not g['pending_updates'],
         'ONE_NEW_INACTIVE_KNOWN_COLD_BOUNDARY')
    need(l['Core_native_GPU_reservations'][FAILED]['status']=='SETTLED_KNOWN_NATIVE_TERMINAL'
         and g['V4_DEV_control_reservations'][g[c.FIELD][FAILED]['child_run_uuid']]['status']=='CLOSED'
         and g['V4_DEV_control_reservations']==l['V4_DEV_control_reservations'],
         'DUAL_FINANCIAL_AND_CONTROL_CLOSURE_ALREADY_REAL')
    for v in (g,l):v[FIELD]=deepcopy(marker)
    validated(g);c.checked(g);c.scope.validate_pair(g,l)
    return g,l


def current_owner(root,request):
    """Fresh actual cold owner; this never claims the old queue ACK was reused."""
    import os
    from pathlib import Path
    import subprocess
    import time
    from . import evaluation_core_accounting as c
    from .future_admission import _file
    root=Path(root)
    raw=_file(root,'plans/CORE_B_ZERO_START_COLD_RECOVERY.json');marker=json.loads(raw)
    need(c.digest(marker)==request.get('startup_recovery_sha256')
         and request['request_id']==marker['replacement_request_id']
         and request['source_sha']==marker['replacement_source_sha'], 'SEALED_CURRENT_RECOVERY_IDENTITY')
    binding=json.loads(_file(root,'plans/CORE_RESIDENT_QUEUE_BINDINGS.private.json'))
    control=Path(binding['queue_root']);need(control.parent==root.parent.parent/'control'
         and control.name=='v4a-core-resident-'+marker['control_id'],'REGISTERED_CONTROL_CUSTODY')
    owner=json.loads(_file(control,'RECOVERY_NATIVE_OWNER_STARTED.private.json'))
    need(owner['request_id']==request['request_id'] and owner['request_sha256']==hashlib.sha256(_file(root,
         'pbs/alloc-'+str(request['allocation_number']).zfill(2)+'/REQUEST.json')).hexdigest()
         and owner['source_sha']==request['source_sha'] and owner['uid']==os.getuid()==2009
         and owner['hostname']==os.uname().nodename=='corelab-4080s'
         and owner['native_unit']=='pv1-core-'+marker['control_id']+'.service'
         and c.old.T0<=owner['started_epoch']<late.start_limit(request) and time.time()<c.old.STOP_PHYSICS,
         'FRESH_BEFORE_CUTOFF_OWN_REGISTERED_OWNER')
    proc=Path('/proc',str(owner['pid']));fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    need(proc.stat().st_uid==2009 and fields[0] not in ('Z','X') and int(fields[19])==owner['proc_start']
         and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id'],
         'ACTUAL_LIVE_BOOT_PID_PROCSTART')
    q=subprocess.run(['systemctl','--user','show',owner['native_unit'],'-p','MainPID','-p','ActiveState','-p','ControlGroup'],
         capture_output=True,text=True,timeout=8)
    props=dict(line.split('=',1) for line in q.stdout.splitlines() if '=' in line)
    need(q.returncode==0 and props['MainPID']==str(owner['pid']) and props['ActiveState']=='active'
         and (proc/'cgroup').read_text().strip().split('::',1)[1]==props['ControlGroup'],
         'ACTUAL_NATIVE_OWNER_UNIT_CGROUP')
    return owner


def current_boundary(value,request_id):
    from . import evaluation_core_shared_B24 as shared
    if shared.FIELD in value and request_id==shared.RID:return shared.validate(value)
    marker=validated(value)
    from .evaluation_core_read_failure_recovery import FIELD as read_field,validate as read_validate
    if read_field in value:
        latest=read_validate(value,marker)
        if latest['replacement_request_id']==request_id:return latest
    need(marker is not None and marker['replacement_request_id']==request_id,'EXACT_CURRENT_COLD_BOUNDARY_REQUEST')
    return marker
