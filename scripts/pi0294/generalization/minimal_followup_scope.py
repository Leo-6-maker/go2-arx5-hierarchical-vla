"""One exact zero-effect continuation under the latest direct user steering.

The executor records its bounded interpretation explicitly: PBS39->40 only,
not an unbounded grant. The old seven-cap marker and both failed rows remain
unchanged. Pure original-ledger proposals; no IO, scheduler or model release.
"""
import base64,json,math
from copy import deepcopy
from . import minimal_execution as e, minimal_retry_execution as retry

FIELD='ARX5_first512_second_zero_effect_continuation_authority'
INDEX='ARX5_minimal_first512_second_zero_effect_continuation'
GRANT_SHA='da2e9d4e62e361cee2c2d690abfaf8bafa677034cea873a65a697f26a567ec7d'
FAILED='b2773e0d-24e6-4d0b-93db-62cccf3e7ef3'
JOB='1502754.stdct-mgmt-02'
LEASE='39724a10dbcf6c510ed45f8fb2bde5170f977616b13d1296c96b47955be71b53'
BEFORE=39;AFTER=40
RUN='arx5-minimal-raw-parent-first512-retry-02'
PROFILE='vanda-arx5-minimal-raw-parent-first512-zero-effect-followup'


def grant(body):
    e.need(type(body)is bytes and e.sha(body)==GRANT_SHA,'FOLLOWUP_EXACT_EXECUTOR_SCOPE_BYTES')
    v=json.loads(body)
    e.need(v['failed_request_id']==FAILED and v['failed_job_id']==JOB
        and v['actual_CLOSED_lease_sha256']==LEASE and v['caps_delta']=={'PBS_submissions':{'before':BEFORE,'after':AFTER}}
        and v['maximum_additional_GPU_jobs']==1 and v['maximum_experiment_GPU_concurrency']==1
        and v['authority_basis']['direct_human_messages'][-1]=='继续推进，无需批准'
        and v['other_caps_changed']is False and v['clocks_reset']is False
        and v['original_matrix3_and_final20GPUh40GiB_protected']is True
        and v['native_admission']is False,'FOLLOWUP_ONLY_THE_SAME_UNSTARTED_EXPERIMENT')
    return v


def validate(g):
    m=g.get(FIELD)
    if m is None:return None
    h=grant(base64.b64decode(m['grant_base64'],validate=True))
    # Validate the first retry independently, even after a later row exists.
    old=retry.validate(g,generation=1);row=retry._selected(g,generation=1)[1]
    t=old['terminal'];effects=t['effects']
    e.need(m.get('schema')=='ARX5_FIRST512_SECOND_ZERO_EFFECT_SCOPE_ADOPTION_V1'
        and row['request_id']==FAILED and row['job_id']==JOB and row['status']==e.STATES[2]
        and row['native_exit_status']==1 and row['committed_optimizer_updates']==row['query_attempts']==row['model_load_attempts']==0
        and t['native']['conservative_GPU_s']==135 and t['native']['conservative_CPU_s']==242
        and effects['optimizer_updates_committed']==effects['query_attempts']==effects['model_load_attempts']==0
        and effects['pending_updates']==effects['unknown_query_intents']==[]
        and m['old_retry_marker_sha256']==e.sha(e.raw(old))and m['old_retry_row_sha256']==e.sha(e.raw(row))
        and m['closed_lease_sha256']==LEASE and m['after_PBS_cap']==AFTER
        and m['grant_sha256']==GRANT_SHA and m['control_paid_once']==retry.CONTROL
        and m['native_hold']==retry.native_resources() and m['native_admission']is False,
        'FOLLOWUP_ACTUAL_AUDITED_ZERO_EFFECT_PREDECESSOR_UNCHANGED')
    clock=e.authority.validate(g)
    e.need(all(m[k]==h[k]==clock[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch'))
        and clock['T_CD']<=m['adopted_epoch']<clock['stop_new_training_epoch']
        and m['adopted_epoch']+59520<clock['stop_new_training_epoch'],
        'FOLLOWUP_ORIGINAL_UNRESET_FULL_WINDOW')
    return m


def planned_unregistered(g):
    if validate(g)is None or INDEX in g:return e._vector({})
    return retry.native_resources()


def budget_LOCAL(g,l):
    m=validate(g)
    if m is None:return l
    key='ARX5_first512_second_continuation_control_projection'
    p=dict(grant_sha256=GRANT_SHA,costs=retry.CONTROL,canonical_LOCAL_write_proven=False)
    if key in l:
        e.need(l[key]==p,'FOLLOWUP_ONCE_LOCAL_PROJECTION');return l
    v=deepcopy(l);v['CPU_s']['local_PC']+=retry.CONTROL['local_CPU_s']
    v['network_transfer_accounted_bytes']+=retry.CONTROL['transfer_bytes'];v[key]=p
    return v


def propose(global_bytes,local_bytes,scope_bytes,*,closed_readback,now_epoch):
    from scripts.pi0294.pv1.neural_ledger import check
    from scripts.pi0294.pv1 import mix_accounting as a
    h=grant(scope_bytes);g=json.loads(global_bytes);check(g)
    e.need(FIELD not in g and INDEX not in g and g['unknown_effects']==g['pending_updates']==[]
        and all(r['status']in e.original.TERMINAL for r in g['PBS_submissions'])
        and all(r['status']=='NATIVE_TERMINAL'for r in g.get(e.original.CORE_ROWS,{}).values()),
        'FOLLOWUP_ONCE_COLD_BOUNDARY_NO_ACTIVE_OR_UNKNOWN')
    old=retry.validate(g,generation=1);row=retry._selected(g,generation=1)[1]
    proof=closed_readback
    e.need(proof['GLOBAL_sha256']==e.sha(global_bytes)and 0<=now_epoch-proof['observed_epoch']<=120
        and proof['closed_lease_sha256']==LEASE and proof['owner_OS_exit']==0
        and proof['native']==old['terminal']['native']and proof['effects']==old['terminal']['effects']
        and proof['audit']==old['terminal']['audit']and proof['all_original_workers_terminal']is True,
        'FOLLOWUP_FRESH_REAL_CANONICAL_TERMINAL_AND_OWN_CLOSED_LEASE')
    v=deepcopy(g);before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes','optimizer_updates','generation_queries','model_loads')}
    before.update(H200_s=g['used']['H200_s'],PBS_submissions=g['used']['PBS_submissions'])
    v['CPU_s']+=retry.CONTROL['Hopper_CPU_s'];v['new_storage_current_bytes']+=retry.CONTROL['Hopper_storage_bytes']
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    v[FIELD]=dict(schema='ARX5_FIRST512_SECOND_ZERO_EFFECT_SCOPE_ADOPTION_V1',grant_sha256=GRANT_SHA,
        grant_base64=base64.b64encode(scope_bytes).decode(),old_retry_marker_sha256=e.sha(e.raw(old)),
        old_retry_row_sha256=e.sha(e.raw(row)),closed_lease_sha256=LEASE,after_PBS_cap=AFTER,
        before_GLOBAL_sha256=e.sha(global_bytes),historical_LOCAL_sha256=e.sha(local_bytes),
        closed_readback_sha256=e.sha(e.raw(proof)),control_paid_once=deepcopy(retry.CONTROL),
        native_hold=retry.native_resources(),before=before,adopted_epoch=now_epoch,native_admission=False,
        **{k:h[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    v['revision']+=1;validate(v);check(v)
    future=e.scope.first512_forecast(e.raw(v),local_bytes)
    e.need(future['cap_changes_required']=={},'FOLLOWUP_COMPLETE_ORIGINAL_MAIN_ALL14_FUTURE_AND_PROTECTED_HOLDS')
    return dict(proposed_global=v,full_future_proof=future,original_onceIO_required=True,
                new_requests=0,new_PBS=0,native_admission=False)
