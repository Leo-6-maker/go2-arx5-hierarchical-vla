"""Two original C/D panels in the existing PBS ledger and package escrow.

Pure proposals only. The original onceIO, live site/queue admission, same-job
transport, model and physical gates remain required. Historical Core holds and
fees remain retained; the new physical-site CPU is explicitly Vanda CPU.
"""
from copy import deepcopy
import math
import re

from . import mix_accounting as a, mix_vanda_cohost as cohost, successor_cd as s
from . import mix_vanda_eval_identity as identity, mix_core_accounting as core
from . import evaluation_accounting as old

VERSION=identity.VERSION
STATES=identity.STATES
CPU_UPPER=173900


def need(ok, why):
    if not ok: raise PermissionError('V5_VANDA_EVAL_ACCOUNTING_'+why)


def validate_item(r):
    arm=r.get('evaluation_candidate'); core.key(arm)
    expected=dict(platform_evaluation_version=VERSION,native_platform='VANDA_A40',
        authority_release=s.TASK_SOURCE,scope_grant_sha256=s.GRANT_SHA,
        campaign_run_uuid='20261018T182956Z-localpc-pv1',owner_user='e1192548',owner_uid=22182,
        stage='V5_MATCHED_DEV24',project='CFP04-CF-007',job_name=identity.JOB_NAME,
        noise_bank=294101,remaining_conditions=24,checkpoint_step=1024,
        parent_checkpoint_manifest_sha256=a.PARENT,statistics_sha256=a.STATS,
        reserved_H200_s=28800,accounting_pad_s=60,CPU_upper_s=CPU_UPPER,
        neural_CPU_upper_s=80000,physical_CPU_upper_s=93900,active_cpu_limit=8,
        physical_GPU_count=1,physical_GPU_kind='NVIDIA_A40',query_limit=2097,
        optimizer_update_limit=0,reserved_optimizer_updates=0,
        additional_output_upper_bytes=cohost.storage_contract()['original_gross_upper_bytes'],
        legacy_GPU_seconds_field_is_projection=True,charges_DEC1_subcaps=False)
    need(all(r.get(k)==v and type(r[k]) is type(v) for k,v in expected.items()),'EXACT_JOINT_SITE_ENVELOPE')
    for key in ('request_id','submission_nonce'):
        need(core.canonical_uuid(r.get(key)),'CANONICAL_ONCE_IDENTITY')
    for key,n in [('source_sha',40),('request_sha256',64),('run_manifest_sha256',64),
                  ('physical_manifest_sha256',64),('checkpoint_manifest_sha256',64),
                  ('training_handoff_sha256',64),('cold_input_audit_sha256',64),('planned_panel_sha256',64)]:
        need(type(r.get(key)) is str and re.fullmatch('[0-9a-f]{'+str(n)+'}',r[key]),'SEALED_'+key)
    need(type(r.get('child_run_uuid')) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',r['child_run_uuid'])
         and r.get('status') in STATES,'CHILD_AND_KNOWN_LIFECYCLE')
    need(r.get('job_id') is None and r['status']==STATES[0]
         or type(r.get('job_id')) is str and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',r['job_id']), 'ACTUAL_PBS_OR_PRESERVED_UNKNOWN')
    q,loads,cpu=old.progress(r)
    need(type(r.get('reserved_neural_queries')) is int
         and r['reserved_neural_queries']==(len(r['unknown_query_intents']) if r['status']==STATES[-1] else 2097-q),
         'ALL_NEURAL_QUERIES_CHARGED_OR_HELD')
    need(s.scope(r['T_CD'])['deadline_epoch']==r['deadline_epoch']
         and r['T_CD']<=a.number(r['created_epoch'])<r['deadline_epoch']-28860,'ORIGINAL_EFFECTIVE_CLOCK')
    if r['status']==STATES[-1]:
        need(type(r['native_exit_status']) is int and q<=r['query_attempts']<=2097 and loads<=r['model_load_attempts']<=2
             and cpu<=a.number(r['conservative_CPU_s'])<=CPU_UPPER
             and a.number(r['conservative_H200_s'])<=28860
             and r['committed_optimizer_updates']==0,'ACTUAL_TERMINAL_EFFECTS_AND_COST')
        for key in ('terminal_native_sha256','terminal_effect_summary_sha256','physical_effects_sha256'):
            need(type(r.get(key)) is str and re.fullmatch('[0-9a-f]{64}',r[key]),'FULL_TERMINAL_'+key)
    return r


def rows(g):
    return [r for r in g['PBS_submissions'] if r.get('platform_evaluation_version')==VERSION]


def credit(g,r):
    return cohost.repair.output_credit(g,r)


def validate_global(g, records=None):
    records=rows(g) if records is None else records
    marker=cohost.validated(g); need(marker is not None,'ACTUALLY_ADOPTED_COLOCATION')
    need(len(records)<=2 and len({r['evaluation_candidate'] for r in records})==len(records),'ONLY_ORIGINAL_PAIR_ONE_EACH')
    for r in records:
        validate_item(r)
        need(r['T_CD']==marker['T_CD'] and r['deadline_epoch']==marker['deadline_epoch']
             and r['checkpoint_manifest_sha256']==g[a.PACKAGE]['endpoints'][r['evaluation_candidate']]['checkpoint_manifest_sha256'],
             'REAL_CLOSED_ENDPOINT_AND_SCOPE')
        credit(g,r)
        if r['status']==STATES[-1]:
            physical_summary(r['actual_physical_effects'],r['child_run_uuid'])
            need(a.sha(a.raw(r['actual_physical_effects']))==r['physical_effects_sha256'],'TERMINAL_PHYSICAL_COUNTER_BYTES_UNCHANGED')
    for key in ('request_id','submission_nonce','job_id'):
        handles=[r[key] for r in g['PBS_submissions'] if r.get(key) is not None]
        need(len(handles)==len(set(handles)),'NO_CROSS_SCOPE_REUSE_'+key)
    active=[r for r in records if r['status']!=STATES[-1]]
    need(sum(r['status'] not in a.TERMINAL for r in g['PBS_submissions'])<=1,'ONE_NATIVE_TASK_GPU')
    p=g[a.PACKAGE]['remaining_resources']
    need(g['reserved']['H200_s']>=p['H200_s']+sum(28860 for r in active)
         and g['reserved']['neural_queries']>=p['neural_queries']+sum(r['reserved_neural_queries'] for r in active)
         and g['CPU_reserved_s']>=p['Hopper_CPU_s']+sum(CPU_UPPER-old.progress(r)[2] for r in active)
         and g['storage_reserved_bytes']>=p['Hopper_storage_bytes']+sum(r['additional_output_upper_bytes']-credit(g,r) for r in active),
         'ACTIVE_AND_WHOLE_FUTURE_ESCROWS_RETAINED')
    need(sum(r['query_attempts'] if r['status']==STATES[-1] else old.progress(r)[0] for r in records)<=g['generation_queries']
         and sum(r['model_load_attempts'] if r['status']==STATES[-1] else old.progress(r)[1] for r in records)<=g['model_loads']
         and sum(r['conservative_CPU_s'] if r['status']==STATES[-1] else old.progress(r)[2] for r in records)<=g['CPU_s']
         and sum(r['conservative_H200_s'] for r in records if r['status']==STATES[-1])<=g['used']['H200_s'],
         'ACTUAL_DEBITS_AND_TERMINAL_COSTS_EXIST_IN_SAME_GLOBAL_TOTALS')
    return g


def checked(g):
    from .neural_ledger import check
    return check(g,allow_unknown=True)


def item(g, request_id):
    found=[r for r in rows(g) if r['request_id']==request_id]
    need(len(found)==1,'ONE_EXACT_EXISTING_REQUEST')
    return validate_item(found[0])


def control_global(g, *, arm, now_epoch):
    """D's original control slice; Core/LOCAL portions stay held, not paid twice.

    C already paid its full original control. This function refuses C/replay.
    No unavailable LOCAL byte is called a native mirror or overwritten.
    """
    checked(g);need(arm=='D25' and cohost.validated(g) is not None,'ONLY_UNPAID_ORIGINAL_D_CONTROL')
    v=deepcopy(g);p=v[a.PACKAGE];key=core.key(arm)
    need(key not in p.get('prepaid_control',{}),'ONCE_ORIGINAL_CONTROL')
    need(p['completed_stages']==core.before_stage(arm,g),'C_PANEL_ACTUALLY_COMPLETE_FIRST')
    costs=dict(Hopper_CPU_s=7200,Hopper_storage_bytes=64*1024**2)
    for k,x in costs.items():
        need(p['remaining_resources'][k]>=x,'EXISTING_CONTROL_'+k);p['remaining_resources'][k]-=x
    p.setdefault('prepaid_control',{})[key]=dict(costs=costs,epoch=now_epoch,
        deferred_original_LOCAL_Core_transfer_holds_retained=True,financial_upper_not_actual_process_CPU=True)
    v['CPU_reserved_s']-=7200;v['CPU_s']+=7200
    v['storage_reserved_bytes']-=costs['Hopper_storage_bytes'];v['new_storage_current_bytes']+=costs['Hopper_storage_bytes']
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    return checked(v)


def complete_stage(g,*,request_id,population_bytes,evidence,GPU_lease_bytes,physical_lease_bytes,now_epoch):
    """Original independent causal history and both actual CLOSED leases.

    A valid failure counts normally; no success-rate gate or seed replacement.
    The unavailable Core LOCAL remains explicitly pending, never fabricated.
    """
    from .baseline import partial_history
    from .dataset import POPULATION
    import json
    checked(g);r=item(g,request_id);arm=r['evaluation_candidate'];key=core.key(arm)
    need(r['status']==STATES[-1] and r['native_exit_status']==0 and r['unknown_query_intents']==[]
         and not g['unknown_effects'] and not g['pending_updates']
         and g[a.PACKAGE]['completed_stages']==core.before_stage(arm,g),'KNOWN_REAL_CLOSE_BOUNDARY_AND_ORIGINAL_ORDER')
    need(a.sha(population_bytes)==POPULATION,'FULL_ORIGINAL_POPULATION')
    history=partial_history(population_bytes,candidate=arm,evidence=evidence,
        checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'])
    need(history['valid_conditions']==24,'ALL_ORIGINAL_VALID_CONDITIONS_NOT_PARTIAL_OR_SUCCESS_ONLY')
    physical_summary(r['actual_physical_effects'],r['child_run_uuid'])
    for raw in (GPU_lease_bytes,physical_lease_bytes):
        need(type(raw) is bytes,'ACTUAL_LEASE_BYTES')
        lease=json.loads(raw)
        need(lease.get('status')=='CLOSED' and lease.get('run_uuid')==r['child_run_uuid']
             and lease.get('authority_release')==s.TASK_SOURCE
             and lease.get('terminal_native_sha256')==r['terminal_native_sha256']
             and lease.get('all_actual_workers_terminal') is True
             and r['settled_epoch']<=a.number(lease['closed_epoch'])<=now_epoch,'ACTUAL_DUAL_CLOSE_AFTER_ONCE_SETTLEMENT')
    proof=dict(schema='V5_CD_VANDA_COMPLETE_ORIGINAL_DEV24_AUDITED_DUAL_CLOSED_V1',request_id=request_id,
        candidate=arm,checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],history=history,
        terminal_native_sha256=r['terminal_native_sha256'],GPU_lease_sha256=a.sha(GPU_lease_bytes),
        physical_lease_sha256=a.sha(physical_lease_bytes),all_valid_failures_preserved=True,
        canonical_LOCAL_sync='PENDING',observed_epoch=now_epoch)
    v=deepcopy(g);v[a.PACKAGE]['completed_stages'].append(key)
    v[a.PACKAGE].setdefault('closed_Eval_stages',{})[key]=proof
    return dict(proposed_global=checked(v),receipt=proof)


def register(g,record,*,request,capture,physical_manifest_bytes,now_epoch):
    checked(g);r=deepcopy(record);validate_item(r)
    need(r['status']==STATES[0] and r['job_id'] is None and all(x['status'] in a.TERMINAL for x in g['PBS_submissions'])
         and all(x['status']=='NATIVE_TERMINAL' for x in g.get(a.CORE_ROWS,{}).values())
         and not g['unknown_effects'] and not g['pending_updates'],'COLD_INACTIVE_KNOWN_BOUNDARY')
    need(not any(x.get('request_id')==r['request_id'] or x.get('submission_nonce')==r['submission_nonce'] for x in g['PBS_submissions']),
         'FRESH_UNREGISTERED_IDENTITY_NO_RETRY')
    proof=identity.require_preparation_at_current_global(a.raw(g),request,capture)
    need(all(request[k]==r[k] and type(request[k]) is type(r[k]) for k in request if k in r),
         'ALL_REQUEST_ROW_FIELDS_AGREE')
    need(r['training_handoff_sha256']==request['training_handoff_sha256']
         and r['evaluation_candidate']==request['evaluation_candidate']==proof['arm']
         and r['child_run_uuid']==request['run_uuid'] and r['request_sha256']==a.sha(a.raw(request))
         and r['source_sha']==request['source_sha'] and r['submission_nonce']==request['submission_nonce']
         and r['run_manifest_sha256']==request['run_manifest_sha256']
         and r['physical_manifest_sha256']==a.sha(physical_manifest_bytes)
         and 0<=a.number(now_epoch)-r['created_epoch']<=120,'EXACT_FRESH_REQUEST_AND_PEER')
    physical=__import__('json').loads(physical_manifest_bytes)
    need(physical['scope']==identity.PHYSICAL_ROLE and physical['run_uuid']==r['child_run_uuid']
         and physical['candidate']==r['evaluation_candidate'] and physical['code']['vla_commit']==r['source_sha']
         and physical['execution']['profile']==identity.PHYSICAL_PROFILE
         and physical['authority']['protocol_ref']==s.TASK_SOURCE
         and physical['authority']['T_CD']==r['T_CD'] and physical['authority']['deadline_epoch']==r['deadline_epoch']
         and all(type(physical['limits'].get(k)) is int and physical['limits'][k]==v for k,v in
             dict(CPU_s=93900,physics_ticks=832000,policy_episodes=26,engine_starts=2108).items()),'EXACT_ORIGINAL_PHYSICAL_ENVELOPE')
    stage=core.key(r['evaluation_candidate']);p=g[a.PACKAGE]
    need(p['completed_stages']==core.before_stage(r['evaluation_candidate'],g)
         and stage in p.get('prepaid_control',{}),'ORIGINAL_ORDER_PAID_CONTROL')
    allocation=dict(H200_s=28860,neural_queries=2097,Hopper_CPU_s=CPU_UPPER,PBS_submissions=1,
        episode_starts=26,physics_ticks=832000,engine_starts=2108,
        Hopper_storage_bytes=r['additional_output_upper_bytes']-credit(g,r))
    v=deepcopy(g);p=v[a.PACKAGE]
    for k,x in allocation.items():
        need(p['remaining_resources'][k]>=x,'ORIGINAL_PACKAGE_SLICE_'+k);p['remaining_resources'][k]-=x
    v['PBS_submissions'].append(r);v['used']['PBS_submissions']+=1
    checked(v);validate_global(v)
    return v


def bind_job(g,*,request_id,job_id,now_epoch):
    v=deepcopy(checked(g));r=item(v,request_id)
    need(r['status']==STATES[0] and r['job_id'] is None
         and type(job_id) is str and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job_id)
         and all(x.get('job_id')!=job_id for x in v['PBS_submissions']) and now_epoch>=r['created_epoch'],
         'ONE_SUCCESSFUL_QSUB_REPLY_NO_REBIND')
    r.update(job_id=job_id,status=STATES[1],native_reply_epoch=now_epoch)
    return checked(v)


def bind_neural_process(g,*,request_id,process,now_epoch):
    v=deepcopy(checked(g));r=item(v,request_id)
    need(r['status']==STATES[1] and 'neural_process' not in r,'ONE_NATIVE_PROCESS_BIND_NOT_A_RESTART')
    exact=dict(schema='V5_CD_VANDA_NEURAL_PROCESS_STARTED_V1',uid=22182,job_id=r['job_id'],native_handle=r['job_id'],
        request_id=request_id,request_sha256=r['request_sha256'],submission_nonce=r['submission_nonce'],
        source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],native_model_permission=False,physics_authorized=False)
    need(all(process.get(k)==x and type(process[k]) is type(x) for k,x in exact.items())
         and type(process.get('pid')) is int and process['pid']>0
         and type(process.get('proc_start')) is int and process['proc_start']>0
         and type(process.get('boot_id')) is str and core.canonical_uuid(process['boot_id'])
         and type(process.get('cgroup')) is str and process['cgroup']
         and re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',process.get('GPU_uuid',''))
         and r['created_epoch']<=process['started_epoch']<=process['observed_epoch']<=now_epoch
         and now_epoch-process['observed_epoch']<=120,'FRESH_REAL_KERNEL_PROCESS_BEFORE_MODEL')
    native=identity.native_allocation(process['actual_PBS'],r,job_id=r['job_id'],hostname=process['hostname'],now_epoch=now_epoch)
    need(a.sha(a.raw(process['actual_PBS']))==process['actual_PBS_sha256']
         and native['allocation_start_epoch']==process['allocation_start_epoch']
         and native['allocation_end_epoch']==process['allocation_end_epoch']
         and process['raw_GPU_processes']==[],'REAL_NATIVE_START_NO_EXISTING_MODEL_OR_FOREIGN_GPU_PROCESS')
    r['neural_process']=deepcopy(process);r['native_process_bind_sha256']=a.sha(a.raw(process))
    return checked(v)


def charge(g,*,request_id,job_id,kind,count,effect_id):
    v=deepcopy(checked(g));r=item(v,request_id)
    need(r['status'] in STATES[1:3] and r['job_id']==job_id,'SAME_BOUND_ACTIVE_JOB')
    need(kind in ('generation_queries','model_loads','CPU_s') and type(effect_id) is str
         and re.fullmatch(r'[A-Za-z0-9._:/-]{1,200}',effect_id) and effect_id not in r.get('charged_effects',{}),'UNIQUE_EFFECT_NO_REPLAY')
    amount=a.number(count,integer=kind!='CPU_s');q,loads,cpu=old.progress(r)
    need(kind=='CPU_s' or amount==1,'ONE_ATTEMPT')
    if kind!='CPU_s':need(not v['unknown_effects'] and not v['pending_updates'],'UNKNOWN_FENCE')
    if kind=='generation_queries':
        need(loads>0 and q+amount<=2097,'ONLY_RESERVED_QUERIES_AFTER_LOAD')
        r['charged_query_attempts']=q+amount;r['reserved_neural_queries']-=amount
        v['generation_queries']+=amount;v['used']['neural_queries']+=amount;v['reserved']['neural_queries']-=amount
    elif kind=='model_loads':
        need(loads+amount<=2,'BOUNDED_LOADS');r['charged_model_loads']=loads+amount;v['model_loads']+=amount
    else:
        need(cpu+amount<=CPU_UPPER,'JOINT_CPU_ACTUAL_DEBIT');r['charged_CPU_s']=cpu+amount
        v['CPU_s']+=amount;v['CPU_reserved_s']-=amount
    r.setdefault('charged_effects',{})[effect_id]=dict(kind=kind,count=amount);r['status']=STATES[2]
    return checked(v)


def physical_summary(value, run_uuid):
    need(value.get('schema')=='V5_CD_VANDA_POLICY_EFFECT_LEDGER_V1'
         and value.get('run_uuid')==run_uuid and value.get('physical_site')=='VANDA_A40'
         and value.get('authority_release')==s.TASK_SOURCE
         and value.get('reservations')=={} and value.get('unknown_effects')==[],
         'REAL_KNOWN_PHYSICAL_TERMINAL_NOT_ABSENT_OR_UNRESOLVED_COUNTERS')
    counters={k:a.number(value.get(k),integer=True) for k in
        ('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations')}
    need(counters['main_starts']<=26 and counters['engine_starts']<=2108
         and counters['live_and_recovery_integrations']+counters['shadow_integrations']<=832000
         and a.number(value.get('Core_CPU_s'))<=93900,'ORIGINAL_PHYSICAL_UPPERS')
    return counters


def resource_projection(g,used,held):
    """Original resource dimensions, with Vanda physical counters made explicit.

    The old LOCAL witness does not contain these effects. Never report it as a
    fresh Core mirror; add the uniquely represented physical site here only.
    """
    for r in rows(g):
        if r['status']==STATES[-1]:
            p=physical_summary(r['actual_physical_effects'],r['child_run_uuid'])
            used['episode_starts']+=p['main_starts'];used['engine_starts']+=p['engine_starts']
            used['physics_ticks']+=p['live_and_recovery_integrations']+p['shadow_integrations']
        else:
            held['episode_starts']+=26;held['engine_starts']+=2108;held['physics_ticks']+=832000


def classify_PBS_CPU(*,raw_PBS_CPU_s,neural_CPU_s,physical_counter_CPU_s,physical_RPC_CPU_s):
    """Original prepaid control covers the local owner/RPC/wrapper CPU.

    PBS includes those children too. Bill the neural + non-RPC physical CPU
    once, keep the already paid 7200s upper, and retain the complete raw PBS
    figure. Two seconds cover PBS whole-second/tick reporting, not a refund.
    """
    raw,neural,physical,rpc=map(a.number,(raw_PBS_CPU_s,neural_CPU_s,physical_counter_CPU_s,physical_RPC_CPU_s))
    need(rpc<=physical<=93900 and neural<=80000,'REAL_COMPONENT_CPU_WITHIN_ORIGINAL_UPPERS')
    payload=neural+physical-rpc
    need(payload<=raw+2,'RAW_PBS_CPU_COVERS_ALL_MEASURED_PAYLOAD_CHILDREN')
    prepaid=max(0,raw-payload)
    need(prepaid+600<=7200,'COMPUTE_CONTROL_AND_ORIGINAL_COLD_CONTROL_WITHIN_SAME_PREPAY')
    return dict(schema='V5_CD_PBS_CPU_WITH_EXISTING_PREPAID_CONTROL_CLASSIFICATION_V1',raw_PBS_CPU_s=raw,
        neural_CPU_s=neural,physical_counter_CPU_s=physical,physical_RPC_CPU_s=rpc,
        payload_CPU_s=payload,conservative_payload_CPU_s=math.ceil(payload),
        compute_control_CPU_s_covered_by_existing_prepaid_upper=prepaid,
        existing_prepaid_upper_s=7200,original_cold_control_upper_retained_s=600,
        original_raw_PBS_CPU_retained=True,old_fees_refunded=False)


def settle(g,*,request_id,native,effects,physical,now_epoch):
    """Known PBS finality, complete query intents, actual physical counter bytes.

    PBS CPU includes neural + physical + control children. Per-effect neural
    CPU is credited once; the physical counter is provenance, never another
    bill. Unknown physical effects retain the reservation and stop this join.
    """
    v=deepcopy(checked(g));r=item(v,request_id)
    need(r['status'] in STATES[1:3],'ONE_TERMINAL_NOT_SECOND_SETTLEMENT')
    exact=dict(schema='V5_CD_VANDA_COLOCATED_PBS_TERMINAL_V1',job_id=r['job_id'],job_state='F',
        project=r['project'],job_name=r['job_name'],owner='e1192548',ngpus=1,
        request_id=request_id,submission_nonce=r['submission_nonce'],
        recorded_process_terminal=True,terminal_inventory_stable=True,all_actual_workers_terminal=True)
    need(all(native.get(k)==x and type(native[k]) is type(x) for k,x in exact.items())
         and type(native.get('Exit_status')) is int
         and 0<=a.number(now_epoch)-a.number(native['observed_epoch'])<=120,'ACTUAL_FRESH_PBS_FINAL_AND_REAL_ACTOR_FINALITY')
    gpu=a.number(native.get('conservative_GPU_s'));cpu=a.number(native.get('conservative_CPU_s'))
    need(gpu<=28860 and cpu<=CPU_UPPER and native.get('new_resident_allocated_upper_bytes')==r['additional_output_upper_bytes'],
         'FULL_ORIGINAL_GROSS_IO_AND_JOINT_ACTUAL_COST')
    expected=dict(schema='V5_CD_VANDA_COLOCATED_NEURAL_EFFECT_SUMMARY_V1',job_id=r['job_id'],request_id=request_id,
        request_sha256=r['request_sha256'],source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],
        checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],statistics_sha256=a.STATS,
        evaluation_candidate=r['evaluation_candidate'],optimizer_updates_committed=0,effect_journal_complete=True)
    need(all(effects.get(k)==x and type(effects[k]) is type(x) for k,x in expected.items()),'COMPLETE_SAME_ACTUAL_NEURAL_EFFECT_JOURNAL')
    q,loads,charged_cpu=old.progress(r)
    attempts=a.number(effects['query_attempts'],integer=True);model_attempts=a.number(effects['model_load_attempts'],integer=True)
    need(q<=attempts<=2097 and loads<=model_attempts<=2 and (not attempts or model_attempts)
         and charged_cpu<=cpu,'COMPLETE_ACTUAL_INTENTS_COVER_PRE_DEBITS_NOT_RESERVATION_MAXIMUM')
    unknown=effects.get('unknown_query_intents')
    need(type(unknown) is list and len(set(unknown))==len(unknown)
         and all(type(i) is int and 0<=i<attempts for i in unknown),'PRESERVE_EXACT_UNKNOWN_QUERY_HOLDS')
    physical_summary(physical,r['child_run_uuid'])
    classification=native['CPU_classification']
    need(classification==classify_PBS_CPU(**{k:classification[k] for k in
         ('raw_PBS_CPU_s','neural_CPU_s','physical_counter_CPU_s','physical_RPC_CPU_s')})
         and classification['physical_counter_CPU_s']==physical['Core_CPU_s']
         and classification['conservative_payload_CPU_s']==cpu
         and core.key(r['evaluation_candidate']) in g[a.PACKAGE]['prepaid_control'],
         'RAW_PBS_NEURAL_PHYSICAL_AND_ORIGINAL_PREPAID_CONTROL_DEDUPLICATED_ONCE')
    v['used']['H200_s']+=gpu;v['reserved']['H200_s']-=28860
    v['CPU_s']+=cpu-charged_cpu;v['CPU_reserved_s']-=CPU_UPPER-charged_cpu
    v['generation_queries']+=attempts-q;v['used']['neural_queries']+=attempts-q;v['model_loads']+=model_attempts-loads
    v['reserved']['neural_queries']-=r['reserved_neural_queries']-len(unknown)
    output=r['additional_output_upper_bytes']-credit(g,r)
    v['storage_reserved_bytes']-=output;v['new_storage_current_bytes']+=output
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    if unknown:v['unknown_effects'].append(dict(kind='V5_CD_VANDA_UNKNOWN_QUERY_INTENTS_NO_RETRY',request_id=request_id,job_id=r['job_id'],unknown_query_intents=deepcopy(unknown)))
    r.update(status=STATES[-1],native_exit_status=native['Exit_status'],settled_epoch=now_epoch,
        conservative_H200_s=gpu,conservative_CPU_s=cpu,physical_A40_allocated_s=gpu,
        query_attempts=attempts,model_load_attempts=model_attempts,committed_optimizer_updates=0,
        reserved_neural_queries=len(unknown),unknown_query_intents=deepcopy(unknown),
        terminal_native_sha256=a.sha(a.raw(native)),terminal_effect_summary_sha256=a.sha(a.raw(effects)),
        physical_effects_sha256=a.sha(a.raw(physical)),actual_physical_effects=deepcopy(physical),
        settlement_financial_deltas=dict(GPU_s=gpu,CPU_s=cpu-charged_cpu,queries=attempts-q,model_loads=model_attempts-loads,
            output_bytes=output,paid_IO_credit_bytes=credit(g,r)),
        physical_site='VANDA_A40',canonical_LOCAL_sync='PENDING')
    return checked(v)
