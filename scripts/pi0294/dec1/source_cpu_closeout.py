"""Pure terminal-bound financial closeout of a terminal source engineering failure.

No IO, actor, permission, scientific reclassification or release is performed.
The old child ledger/UNKNOWN/live liability remain immutable evidence. An owner
must verify actual bytes/actors under a once writer before adopting either
proposal. Every unresolved live liability is charged at its complete original upper.
Only provably unstarted work in this now-terminal child can be unreserved.
"""
from copy import deepcopy
from pathlib import PurePosixPath
import hashlib
import json
import math
from . import source_cpu_intake as intake

SCHEMA='DEC1_SOURCE_CPU_TERMINAL_BOUND_FINANCIAL_CLOSEOUT_V1'
VERDICT='ENGINEERING_INCOMPLETE_UNKNOWN'


def need(value,reason):
    if not value:raise PermissionError('SOURCE_FINANCIAL_'+reason)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def encode(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def obj(raw,name):return intake._object(raw,name)


def dims(value):return intake._mapping(value,'financial dimensions')


def _hash(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _row(ledger,request,manifest_hash):
    row=ledger['source_cpu_reservations'][request]
    need(row.get('schema')=='DEC1_SOURCE_CPU_COMPONENT_NATIVE_RESERVATION_V1'
        and row.get('request_id')==request and row.get('role')==intake.ROLE
        and row.get('authority_release')==intake.AUTHORITY and row.get('unknown_effects')==[]
        and row.get('status') in ('RESERVED','RUNNING')
        and row.get('child_manifest_sha256')==manifest_hash
        and row.get('remaining')==row.get('requested'),'EXACT_UNSETTLED_FULL_ORIGINAL_RESERVATION')
    dims(row['requested']);dims(ledger['source_cpu_reserved'])
    total=dict.fromkeys(intake.DIMS,0)
    for key,item in ledger['source_cpu_reservations'].items():
        need(item.get('request_id')==key and item.get('role')==intake.ROLE
            and item.get('schema')==row['schema'] and item.get('unknown_effects')==[]
            and item.get('status') in ('RESERVED','RUNNING','SETTLED'),
            'ALL_INDEX_ROWS_TYPED_AND_KNOWN')
        original,left=dims(item['requested']),dims(item['remaining'])
        need(all(left[k]<=original[k] for k in intake.DIMS),'NO_EXPANDED_OTHER_ROW')
        if item['status']=='SETTLED':need(not any(left.values()),'SETTLED_OTHER_ROW_ZERO')
        else:
            for k in intake.DIMS:total[k]+=left[k]
    need(total==ledger['source_cpu_reserved'],'WHOLE_INDEX_AGGREGATE_EXACT')
    need(all(ledger['source_cpu_reserved'][k]>=row['requested'][k] for k in intake.DIMS),
        'AGGREGATE_CONTAINS_WHOLE_ORIGINAL_BOUND')
    need(request not in ledger.get('source_cpu_financial_closeouts',{}),'NEVER_SETTLE_SAME_REQUEST_TWICE')
    return row


def _settle_index(ledger,request,invoice):
    row=ledger['source_cpu_reservations'][request];amount=row['requested']
    row.update(status='SETTLED',remaining=dict.fromkeys(intake.DIMS,0),
        financial_closeout_sha256=sha(encode(invoice)),scientific_verdict=VERDICT,
        full_unresolved_live_bound_charged=True)
    ledger['source_cpu_reserved']={k:ledger['source_cpu_reserved'][k]-amount[k] for k in intake.DIMS}
    ledger.setdefault('source_cpu_financial_closeouts',{})[request]=deepcopy(invoice)


def propose(*,parent_bytes,Core_bytes,manifest_bytes,budget_bytes,plan_bytes,
            terminal_bytes,artifact_bytes,settlement_id,settled_epoch):
    """Two copied terminal-bound charges; never changes the original child ledger.

    terminal_bytes is a fresh actual filesystem/actor readback, not CPU fixtures
    presented as admission. The caller still verifies every supplied SHA and PID
    against the native site. This return explicitly provides no dispatch right.
    """
    p=obj(parent_bytes,'LOCAL parent');c=obj(Core_bytes,'Core current');m=obj(manifest_bytes,'manifest')
    b=obj(budget_bytes,'original source budget');f=obj(terminal_bytes,'actual terminal inputs')
    mh=sha(manifest_bytes);request=b['request_id'];plan=obj(plan_bytes,'fixed component plan')
    need(type(settlement_id) is str and settlement_id==request+'-terminal-bound-close','DISTINCT_ONCE_SETTLEMENT_ID')
    need(type(settled_epoch) in (int,float) and type(settled_epoch) is not bool
        and math.isfinite(settled_epoch) and settled_epoch>=p['T0_epoch'],'FINITE_ORIGINAL_CLOCK')
    need(m.get('scope')==intake.ROLE and m.get('owner')=='EXISTING_LOCAL_PC'
        and m.get('authority',{}).get('protocol_ref')==intake.PARENT_AUTHORITY
        and m['limits']['PV1_T0_epoch']==p['T0_epoch']==c['original_T0_epoch']
        and m['limits']['PV1_hard_close_epoch']==p['hard_closeout_epoch']==c['hard_close_epoch']==intake.HARD_CLOSE,
        'ORIGINAL_SOURCE_ROLE_AUTHORITY_AND_CLOCK')
    need(c.get('schema')=='DEC1_NONADDITIVE_RESOURCE_CLASSIFICATION_V1'
        and c.get('authority_release')==intake.AUTHORITY and c.get('unknown_effects')==[]
        and p.get('unknown_effects')==[] and p.get('Hopper_unknown_effects')==[],
        'NO_UNKNOWN_PARENT_WRITE_OR_GLOBAL_EFFECT')
    pr=_row(p,request,mh);cr=_row(c,request,mh)
    need(b.get('schema')==intake.BUDGET_SCHEMA and b.get('native_complete') is True
        and all(type(b.get(k)) is int and b[k]==0 for k in ('GR00T_queries','GPU_submissions','training_updates')),
        'ORIGINAL_TYPED_ZERO_NEURAL_BUDGET')
    need(pr==cr and cr['child_run_uuid']==m['run_uuid'] and cr['campaign_run_uuid']==p['run_uuid']==c['campaign_run_uuid']
        and cr['budget_sha256']==sha(budget_bytes) and cr['requested']==b['requested']
        and cr['component_plan_sha256']==sha(plan_bytes)==intake.PLAN_SHA256,
        'SAME_DUAL_BOUND_REQUEST_BUDGET_PLAN_AND_MANIFEST')
    need(f.get('status')=='READ_ONLY_TERMINAL_ACTORS_AND_PRESERVED_FINANCIAL_BOUND_INPUTS'
        and f.get('new_writes')==f.get('new_native_integrations')==0
        and f.get('manifest_sha256')==mh and f.get('returned_STOP_ACK') is False
        and f.get('scientific_classification')==VERDICT
        and type(f.get('observed_epoch')) in (int,float) and 0<=settled_epoch-f['observed_epoch']<900,
        'FRESH_ACTUAL_TERMINAL_READBACK_NOT_SCIENTIFIC_PASS')
    need(f['parent_Core_ledger']==c and f['parent_Core_ledger_sha256']==sha(Core_bytes),
        'ACTUAL_CORE_CURRENT_BYTES_NOT_STALE_PROJECTION')
    child=f['child_ledger'];lease=f['child_lease']
    need(child.get('schema')=='PI0294_PV1_CORE_POLICY_LEDGER_V1' and child.get('run_uuid')==m['run_uuid']
        and child.get('authority_release')==intake.PARENT_AUTHORITY and child.get('source_budget_request_id')==request
        and f['child_ledger_sha256']==sha(encode(child)) and child.get('unknown_effects')
        and lease['status']=='ACTIVE' and lease['owner']=='EXISTING_LOCAL_PC'
        and lease['authority_release']==intake.PARENT_AUTHORITY and lease['run_uuid']==m['run_uuid']
        and lease['deadline_epoch']==m['limits']['operation_deadline_epoch']
        and f['child_lease_sha256']==sha(encode(lease)),
        'ORIGINAL_UNKNOWN_CHILD_AND_ACTIVE_LEASE_BOUND_BYTES')
    handles=f['handles'];names={}
    need(type(handles) is list and len(handles)==4 and all(h.get('active') is False for h in handles),
        'BOTH_ORIGINAL_ACTORS_ACTUALLY_TERMINAL')
    for h in handles:
        n=PurePosixPath(h['relative']);v=h['value']
        need(n.parent.as_posix()=='worker_processes' and n.name not in names and _hash(h['sha256'])
            and v.get('source_sha')==m['code']['vla_commit'] and type(v.get('pid')) is int
            and type(v.get('proc_start')) is str,'ORIGINAL_UNIQUE_HANDLES_AND_SOURCE')
        names[n.name]=v
    exits=[v for n,v in names.items() if n.endswith('.exit.json')]
    need(len(exits)==2 and sorted(v.get('exit_status') for v in exits)==[0,1],
        'COUNTER_EXIT0_AND_COMPONENT_EXIT1_REQUIRED')
    for n,v in names.items():
        if n.endswith('.exit.json'):
            peer=names.get(n[:-10]+'.json')
            need(peer is not None and peer['pid']==v['pid'] and peer['proc_start']==v['proc_start']
                and v['manifest_sha256']==mh and v['run_uuid']==m['run_uuid'],
                'EXACT_PID_START_TERMINAL_PAIRS')
    files=f['native_files'];required=('RESULT.json','S1_RESULT.json','SOURCE_COMPONENT_STOP.json')
    need(set(artifact_bytes)==set(required) and all(n in files and sha(artifact_bytes[n])==files[n]['sha256'] for n in required),
        'ACTUAL_FAILED_ARTIFACT_BYTES_AND_FRESH_HASHES')
    for n,v in files.items():
        path=PurePosixPath(n)
        need(not path.is_absolute() and '..' not in path.parts and _hash(v['sha256'])
            and type(v['bytes']) is int and v['bytes']>0,'BOUNDED_ORIGINAL_FILE_IDENTITIES')
    need({'highrate.npz','SOURCE_COMPONENT_EVENTS.jsonl'}<=set(files),'RAW_PHYSICS_AND_EVENTS_PRESERVED')
    result=obj(artifact_bytes['RESULT.json'],'original result');s1=obj(artifact_bytes['S1_RESULT.json'],'original S1')
    stop=obj(artifact_bytes['SOURCE_COMPONENT_STOP.json'],'unacknowledged original STOP')
    conditions={x['condition_id']:x for x in plan['conditions']};cid=result['id']
    need(cid in conditions and result['fixture'] is False and result['source_sha']==m['code']['vla_commit']
        and result['card_sha256']==conditions[cid]['card_sha256'] and result['status']=='FAILED_PRESERVED'
        and s1['episode_id']==cid and s1['engineering_unknown'] is True and s1['actual_stop_event'] is None
        and s1['actual_stop_state']['stop_acknowledged'] is False and s1['actual_stop_state']['unknown'] is True
        and stop['episode_id']==cid and stop['stopped'] is True and stop['native_frame_inflight'] is False,
        'ACTUAL_ENGINEERING_UNKNOWN_NOT_VALID_TASK_OR_STOP_PASS')
    live=intake._number(result['physics_steps'],'actual lower bound',True)
    shadow=intake._number(s1['shadow_integrations'],'known shadow',True)
    need(live==s1['native_live_integrations']==s1['live_integrations_lower_bound']==stop['native_steps']
        and f['highrate_shape'][0]==live and child['main_starts']==1
        and child['shadow_integrations']==shadow and child['live_and_recovery_integrations']==0
        and child['reservations']=={cid:dict(kind='live_and_recovery_integrations',maximum=16000,
            started_epoch=child['reservations'][cid]['started_epoch'])},
        'KNOWN_PARTIAL_NATIVE_COUNTERS_AND_ORIGINAL_LIVE_UPPER')
    amount=dims(cr['requested'])
    need(child['Core_CPU_s']<=amount['CoreLab_CPU_s'] and live+shadow<=amount['physics_ticks']
        and child['engine_starts']<=amount['constructors'] and child['main_starts']<=amount['episode_starts'],
        'ORIGINAL_REQUEST_FULL_UPPER_COVERS_ALL_OBSERVED_EFFECTS')
    # Main/counter process CPU is independently bounded by both the original
    # child meter and the two actual process-exit records. Registration was
    # prepaid into child Core_CPU_s. Control/audit allowances remain charged in
    # full; no uncertainty in those operations creates an optimistic refund.
    components=b['components']
    process_CPU=sum(intake._number(v['process_CPU_s'],'terminal process CPU') for v in exits)
    registration=intake._number(components['registration']['CoreLab_CPU_s'],'registration upper')
    control=intake._number(components['control']['CoreLab_CPU_s'],'control upper')
    audit=intake._number(components['audit_closeout']['CoreLab_CPU_s'],'audit upper')
    need(s1['execution_state']['unknown']['shadow'] is False and s1['component_meter_unknown'] is False,
        'CLOSED_KNOWN_SHADOW_ACCOUNTING_REQUIRED')
    charged=deepcopy(amount)
    child_CPU=intake._number(child['Core_CPU_s'],'original child CPU')
    charged.update(CoreLab_CPU_s=max(child_CPU,registration+process_CPU)+control+audit,
        episode_starts=child['main_starts'],physics_ticks=16000+shadow,constructors=child['engine_starts'])
    charged=dims(charged)
    need(all(charged[k]<=amount[k] for k in intake.DIMS),'TERMINAL_BOUNDS_MUST_FIT_ORIGINAL_RESERVATION')
    invoice=dict(schema=SCHEMA,request_id=request,settlement_id=settlement_id,
        child_run_uuid=m['run_uuid'],child_manifest_sha256=mh,runtime_source=m['code']['vla_commit'],
        authority_release=intake.AUTHORITY,original_T0_epoch=p['T0_epoch'],hard_close_epoch=intake.HARD_CLOSE,
        settled_epoch=settled_epoch,terminal_inputs_sha256=sha(terminal_bytes),
        original_child_ledger_sha256=f['child_ledger_sha256'],original_child_lease_sha256=f['child_lease_sha256'],
        native_file_hashes=deepcopy(files),financial_charged=deepcopy(charged),
        unstarted_or_known_unused={k:amount[k]-charged[k] for k in intake.DIMS},
        live_financial_upper=16000,known_shadow_charged=shadow,
        observed=dict(component_attempts=1,live_lower_bound=live,known_shadow_ticks=shadow,
            constructors=child['engine_starts'],child_CPU_s=child['Core_CPU_s']),
        original_live_reservation_and_unknowns=deepcopy(dict(reservations=child['reservations'],unknown_effects=child['unknown_effects'])),
        scientific_verdict=VERDICT,actual_returned_STOP_ACK=False,source_or_target_qualified=False,
        charges_are_terminal_bounds_not_claimed_native_actual_ticks=True,native_dispatch_admitted=False)
    cp,pp=deepcopy(c),deepcopy(p)
    _settle_index(cp,request,invoice);_settle_index(pp,request,invoice)
    class_before=dims(c['source_cpu_current_classification'])
    for k,field in (('CoreLab_CPU_s','CPU_s'),('episode_starts','episode_starts'),('physics_ticks','physics_ticks'),('constructors','constructors')):
        intake._number(c[field],'original Core '+field,field!='CPU_s')
        need(class_before[k]==c[field],'EXACT_EXISTING_CORE_CLASSIFICATION:'+field)
        cp[field]+=charged[k]
    cp['source_cpu_current_classification']={k:class_before[k]+charged[k] for k in intake.DIMS}
    for site,k in (('CoreLab','CoreLab_CPU_s'),('local_PC','local_CPU_s')):
        need(pp['CPU_reserved_s'][site]>=amount[k],'ORIGINAL_FORMAL_PARENT_CPU_HOLD')
        pp['CPU_reserved_s'][site]-=amount[k];pp['CPU_s'][site]+=charged[k]
    pp['episode_starts']+=charged['episode_starts'];pp['engine_starts']+=charged['constructors']
    # A separate explicit upper bucket cannot masquerade as observed native
    # live/shadow ticks. Original budget sums all buckets exactly once.
    pp['physics_ticks']['closed_source_component_financial_upper']=p['physics_ticks'].get(
        'closed_source_component_financial_upper',0)+charged['physics_ticks']
    pp['network_transfer_accounted_bytes']+=charged['transfer_bytes']
    children=[x for x in pp['children'] if x.get('source_budget_request_id')==request]
    need(len(children)==1,'EXACT_FORMAL_PARENT_CHILD')
    child_row=children[0]
    for key,k in (('CPU_reserved_s','CoreLab_CPU_s'),('local_CPU_reserved_s','local_CPU_s'),
        ('episode_starts_reserved','episode_starts'),('physics_ticks_reserved','physics_ticks'),('engine_starts_reserved','constructors')):
        need(child_row[key]==amount[k],'FULL_FORMAL_CHILD_HOLD:'+key);child_row[key]=0
    child_row.update(status='CLOSED_SOURCE_FINANCIAL_TERMINAL_BOUND_SCIENCE_UNKNOWN',
        financial_closeout_sha256=sha(encode(invoice)),financial_charged=deepcopy(charged),
        scientific_verdict=VERDICT,original_child_ledger_unknowns_preserved=True)
    return dict(status='PROPOSAL_ONLY_TERMINAL_BOUND_FINANCIAL_CLOSEOUT_NOT_SCIENTIFIC_QUALIFICATION',
        original_parent_sha256=sha(parent_bytes),original_Core_sha256=sha(Core_bytes),invoice=invoice,
        invoice_sha256=sha(encode(invoice)),proposed_parent=pp,proposed_Core=cp,
        original_child_ledger_must_remain_unchanged=True,native_dispatch_admitted=False)
