"""Pure known-Source02 terminal settlement proposal; no IO or native GO.

The owner supplies exact independent audit, immutable child meter and fresh
all-actor terminal readback bytes. This validates their bindings, not OS truth.
Unknown results/effects use the existing unknown path, never this release path.
"""
from copy import deepcopy
from pathlib import PurePosixPath
import math
from . import source_cpu_closeout as old
from . import source_cpu_intake as intake
from . import source_cpu_history as history

SCHEMA='DEC1_SOURCE_CPU_KNOWN_TERMINAL_SETTLEMENT_V1'
TERMINAL_SCHEMA='DEC1_SOURCE_CPU_KNOWN_TERMINAL_READBACK_V1'
INDEX='source_cpu_known_terminal_settlements'
SOURCE_REQUEST='081c5299-5594-4ff6-8676-d48b9652cf52'
SOURCE_MANIFEST_SHA='14ce2f841c8db3de7e2eb9c191e95fc9b26720f628493774e4fd954012556b0f'
SOURCE_RELEASE='47dc6001227572f031ebc3cc33d56a487027d8cb'
KNOWN={'COMPONENT_COMPLETE','KNOWN_REFERENCE_OR_S1_REJECT','NATIVE_SAFETY_FAIL'}
NOT_RUN='NOT_RUN'
encode=old.encode
sha=old.sha


def need(ok,reason):
    if not ok:raise PermissionError('SOURCE_KNOWN_TERMINAL_'+reason)


def obj(raw,name):return intake._object(raw,name)
def dims(v):return intake._mapping(v,'known terminal dimensions')
def num(v,integer=False):return intake._number(v,'known terminal value',integer)


def _actor_proofs(t,files,manifest,mh,attempted):
    handles=t['handles'];pairs={}
    need(type(handles) is list and len(handles)>=2,'ALL_ACTOR_HANDLES_REQUIRED')
    for handle in handles:
        path=handle['relative'];p=PurePosixPath(path)
        need(p.parent.as_posix()=='worker_processes' and p.name not in pairs
            and p.name.endswith('.json') and path in files and type(files[path]) is bytes,
            'UNIQUE_SUPPLIED_ACTOR_BYTES')
        raw=files[path];v=obj(raw,'actor handle')
        need(sha(raw)==handle['sha256'] and v==handle['value'] and handle['active'] is False
            and v['source_sha']==manifest['code']['vla_commit']
            and type(v['pid']) is int and v['pid']>0 and type(v['proc_start']) is str and v['proc_start'],
            'EXACT_ACTOR_IDENTITY_AND_TERMINAL_READBACK')
        pairs[p.name]=v
    exits=[(n,v) for n,v in pairs.items() if n.endswith('.exit.json')]
    need(len(exits)*2==len(pairs),'EVERY_ACTOR_HAS_EXIT_ANCHOR')
    expected={'source-counter-01',*('component-'+r['condition_id'] for r in attempted)}
    need({n[:-5] for n in pairs if not n.endswith('.exit.json')}==expected,
        'COUNTER_AND_EVERY_ACTUAL_ATTEMPT_ACTOR_REQUIRED')
    need(pairs['source-counter-01.exit.json']['exit_status']==0,
        'ORIGINAL_COUNTER_QUALIFICATION_EXIT_ZERO')
    cpu=0
    for n,v in exits:
        peer=pairs.get(n[:-10]+'.json')
        need(peer is not None and peer['pid']==v['pid'] and peer['proc_start']==v['proc_start']
            and v['manifest_sha256']==mh and v['run_uuid']==manifest['run_uuid']
            and type(v['exit_status']) is int,'EXACT_PID_START_EXIT_PAIR')
        cpu+=num(v['process_CPU_s'])
    need(t['all_owned_actors_terminal'] is True and t['active_workers']==0
        and type(t['active_workers']) is int and t['unknown_effects']==[],
        'NO_ACTIVE_OR_UNKNOWN_ACTOR_MAY_RELEASE')
    return cpu


def propose(*,parent_bytes,Core_bytes,manifest_bytes,budget_bytes,plan_bytes,
            audit_bytes,child_ledger_bytes,terminal_bytes,actor_bytes,attempt_bytes,
            settlement_id,settled_epoch):
    """Return copied dual proposals only. Exact once IO remains the real owner.

    audit_bytes is the original independent terminal auditor's JSON result;
    terminal_bytes binds that audit plus the complete attempt/anchor file index.
    Root must create the readback schema from actual actors/files, not fixtures.
    No lease closure or child meter mutation is performed here.
    """
    p=obj(parent_bytes,'LOCAL parent');c=obj(Core_bytes,'Core current')
    m=obj(manifest_bytes,'source manifest');b=obj(budget_bytes,'original budget')
    plan=obj(plan_bytes,'fixed source plan');a=obj(audit_bytes,'independent source audit')
    child=obj(child_ledger_bytes,'original child meter');t=obj(terminal_bytes,'fresh terminal readback')
    request=b['request_id'];mh=sha(manifest_bytes);when=num(settled_epoch)
    need(request==SOURCE_REQUEST and mh==SOURCE_MANIFEST_SHA and m['code']['vla_commit']==SOURCE_RELEASE,
        'ONLY_FIXED_NEW_SOURCE02')
    need(settlement_id==request+'-known-terminal-close','DISTINCT_ONCE_ID')
    for ledger in (p,c):
        previous=ledger.get('source_cpu_financial_closeouts',{})
        need(set(previous)=={history.OLD_REQUEST} and sha(encode(previous[history.OLD_REQUEST]))==history.INVOICE_SHA
            and previous[history.OLD_REQUEST]['scientific_verdict']=='ENGINEERING_INCOMPLETE_UNKNOWN'
            and previous[history.OLD_REQUEST]['financial_charged']['physics_ticks']==16490,
            'EXACT_OLD_UNKNOWN_INVOICE_COST_RETAINED')
        old_row=ledger['source_cpu_reservations'][history.OLD_REQUEST]
        need(old_row['status']=='SETTLED' and old_row['financial_closeout_sha256']==history.INVOICE_SHA
            and old_row['scientific_verdict']=='ENGINEERING_INCOMPLETE_UNKNOWN'
            and old_row['full_unresolved_live_bound_charged'] is True,'EXACT_OLD_UNKNOWN_FINANCIAL_ROW')
    need(p['source_cpu_reservations'][history.OLD_REQUEST]==c['source_cpu_reservations'][history.OLD_REQUEST]
        and p['physics_ticks']['closed_source_component_financial_upper']==16490,
        'ORIGINAL_DUAL_OLD_ROW_AND_16490_UPPER')
    need(m['scope']==intake.ROLE and m['owner']=='EXISTING_LOCAL_PC'
        and m['authority']['protocol_ref']==intake.PARENT_AUTHORITY
        and m['limits']['PV1_T0_epoch']==p['T0_epoch']==c['original_T0_epoch']
        and m['limits']['PV1_hard_close_epoch']==p['hard_closeout_epoch']==c['hard_close_epoch']==intake.HARD_CLOSE
        and p['T0_epoch']<=when,'ORIGINAL_ROLE_AND_CLOCK')
    need(c['schema']=='DEC1_NONADDITIVE_RESOURCE_CLASSIFICATION_V1'
        and c['authority_release']==intake.AUTHORITY and p['unknown_effects']==c['unknown_effects']==[]
        and p.get('Hopper_unknown_effects')==[],'KNOWN_GLOBAL_WRITES_REQUIRED')
    need(request not in p.get(INDEX,{}) and request not in c.get(INDEX,{}),'NEVER_SETTLE_REQUEST_TWICE')
    pr=old._row(p,request,mh);cr=old._row(c,request,mh);amount=dims(cr['requested'])
    need(pr==cr and cr['child_run_uuid']==m['run_uuid']
        and cr['campaign_run_uuid']==p['run_uuid']==c['campaign_run_uuid']
        and cr['budget_sha256']==sha(budget_bytes) and cr['requested']==b['requested']
        and cr['component_plan_sha256']==sha(plan_bytes)==intake.PLAN_SHA256,
        'EXACT_DUAL_ORIGINAL_BUDGET_PLAN_MANIFEST')
    need(b['schema']==intake.BUDGET_SCHEMA and b['native_complete'] is True
        and all(type(b.get(k)) is int and b[k]==0 for k in ('GR00T_queries','GPU_submissions','training_updates')),
        'SOURCE_ONLY_ORIGINAL_BUDGET')
    need(t['schema']==TERMINAL_SCHEMA and t['status']=='READ_ONLY_ALL_SOURCE_ACTORS_TERMINAL'
        and t['new_writes']==t['new_native_integrations']==0
        and t['manifest_sha256']==mh and t['source_sha']==m['code']['vla_commit']
        and t['plan_sha256']==sha(plan_bytes) and t['audit_sha256']==sha(audit_bytes)
        and t['child_ledger_sha256']==sha(child_ledger_bytes)
        and t['parent_sha256']==sha(parent_bytes) and t['Core_sha256']==sha(Core_bytes)
        and 0<=when-num(t['observed_epoch'])<900,'FRESH_EXACT_READBACK_BYTES')
    need(a['schema']=='DEC1_SOURCE_COMPONENT_INDEPENDENT_TERMINAL_AUDIT_V1'
        and a['purpose']=='SOURCE_INTERFACE_COMPONENT_QUALIFICATION_ONLY'
        and a['student_neural_ACK_qualification'] is False and a['target_qualification'] is False
        and a['core64_cells']==0 and type(a['core64_cells']) is int and a['new_training_data'] is False,
        'ACTUAL_SOURCE_ONLY_INDEPENDENT_AUDIT')
    ids=[x['condition_id'] for x in plan['conditions']];rows=a['conditions']
    need(len(ids)==len(set(ids))==4 and len(rows)==4 and [x['condition_id'] for x in rows]==ids,
        'FIXED_FOUR_ORDERED_CONDITIONS_NO_DUPLICATES')
    need(all(r['status'] in KNOWN|{NOT_RUN} for r in rows),'UNKNOWN_NEVER_USES_KNOWN_PATH')
    counts={s:sum(r['status']==s for r in rows) for s in {r['status'] for r in rows}}
    need(a['component_counts']==counts,'AUDIT_COUNTS_MATCH_ALL_CONDITIONS')
    attempted=[r for r in rows if r['status']!=NOT_RUN]
    process_cpu=_actor_proofs(t,actor_bytes,m,mh,attempted)
    anchors=t['attempts']
    need(type(anchors) is list and len(anchors)==len(attempted)
        and [x['condition_id'] for x in anchors]==[r['condition_id'] for r in attempted],
        'ALL_AND_ONLY_ACTUAL_ATTEMPTS_RETAINED')
    need(type(attempt_bytes) is dict and set(attempt_bytes)=={r['condition_id'] for r in attempted},
        'EXACT_SUPPLIED_ACTUAL_ATTEMPT_BYTES')
    live=shadow=0
    for r,anchor in zip(attempted,anchors):
        need(anchor['status']==r['status'] and anchor['manifest_sha256']==mh
            and anchor['source_sha']==m['code']['vla_commit'] and anchor['plan_sha256']==sha(plan_bytes)
            and old._hash(anchor['terminal_anchor_sha256'])
            and old._hash(anchor['attempt_sha256']) and anchor['actual_stop_returned'] is True
            and anchor['audit_file_hashes_verified'] is True and anchor['unknown_effects']==[],
            'KNOWN_ATTEMPT_OWNER_ANCHOR_AND_RETURNED_STOP')
        raw=attempt_bytes[r['condition_id']]
        need(type(raw) is dict and set(raw)=={'terminal_anchor','attempt'},'EXACT_ATTEMPT_AND_ANCHOR_BYTES')
        ar=obj(raw['terminal_anchor'],'actual terminal anchor');rr=obj(raw['attempt'],'actual attempt')
        need(sha(raw['terminal_anchor'])==anchor['terminal_anchor_sha256']
            and sha(raw['attempt'])==anchor['attempt_sha256']
            and ar['schema']=='DEC1_SOURCE_COMPONENT_TERMINAL_BYTE_ANCHOR_V1'
            and ar['condition_id']==r['condition_id'] and ar['component_plan_sha256']==sha(plan_bytes)
            and ar['manifest_sha256']==mh and ar['runtime_release']==m['code']['vla_commit']
            and ar['owner_closeout_known'] is True
            and ar['engine_sha256']==intake.ORIGINAL_13['scripts/pi0291_arx5_data/engine.py']
            and rr['schema']=='DEC1_SOURCE_COMPONENT_ATTEMPT_V1'
            and rr['condition_id']==r['condition_id'] and rr['attempt_id']==r['condition_id']
            and rr['component_plan_sha256']==sha(plan_bytes)
            and rr['terminal_anchor_sha256']==sha(raw['terminal_anchor'])
            and rr['native_STOP_return_and_owner_event_confirmed'] is True
            and rr['main_integrations']==r['main_integrations']
            and rr['shadow_integrations']==r['shadow_integrations']
            and rr['student_neural_ACK_qualification'] is False and rr['GR00T_queries']==0
            and rr['new_training_data'] is False,'ACTUAL_BYTE_ANCHOR_AND_ATTEMPT_BINDING')
        live+=num(r['main_integrations'],True);shadow+=num(r['shadow_integrations'],True)
    need(child['schema']=='PI0294_PV1_CORE_POLICY_LEDGER_V1' and child['run_uuid']==m['run_uuid']
        and child['authority_release']==intake.PARENT_AUTHORITY and child['source_budget_request_id']==request
        and child['unknown_effects']==[] and child['reservations']=={}
        and child['main_starts']==len(attempted)
        and child['live_and_recovery_integrations']==live and child['shadow_integrations']==shadow,
        'ORIGINAL_METER_KNOWN_CLOSED_AND_EXACT_AUDIT_COUNTERS')
    constructors=num(child['engine_starts'],True)
    components=b['components']
    need(set(components)==intake.COMPONENTS
        and {k:sum(dims(v)[k] for v in components.values()) for k in intake.DIMS}==amount,
        'FULL_EIGHT_DIMENSION_ORIGINAL_COMPONENT_BUDGET')
    # Keep local/output/transfer full original uppers. Core CPU bounds both child
    # meter and exited process accounting; control/audit/static uppers stay full.
    charged=deepcopy(amount)
    other=sum(num(components[n]['CoreLab_CPU_s']) for n in ('control','audit_closeout','static_reconstruction'))
    charged.update(CoreLab_CPU_s=max(num(child['Core_CPU_s']),
        num(components['registration']['CoreLab_CPU_s'])+process_cpu)+other,
        episode_starts=len(attempted),physics_ticks=live+shadow,constructors=constructors)
    need(all(dims(charged)[k]<=amount[k] for k in intake.DIMS),'ALL_ACTUAL_OR_CONSERVATIVE_COSTS_WITHIN_ORIGINAL')
    receipt=dict(schema=SCHEMA,request_id=request,settlement_id=settlement_id,
        child_run_uuid=m['run_uuid'],manifest_sha256=mh,budget_sha256=sha(budget_bytes),
        plan_sha256=sha(plan_bytes),source_sha=m['code']['vla_commit'],audit_sha256=sha(audit_bytes),
        child_ledger_sha256=sha(child_ledger_bytes),terminal_readback_sha256=sha(terminal_bytes),
        settled_epoch=when,attempts=deepcopy(anchors),conditions=deepcopy(rows),
        actual_live_ticks=live,actual_shadow_ticks=shadow,financial_charged=charged,
        unused_released={k:amount[k]-charged[k] for k in intake.DIMS},
        source_component_result_counts=counts,student_neural_ACK_qualification=False,
        target_qualification=False,core64_cells=0,new_training_data=False,native_dispatch_admitted=False)
    rh=sha(encode(receipt));pp=deepcopy(p);cc=deepcopy(c)
    for ledger in (pp,cc):
        ledger['source_cpu_reservations'][request].update(status='SETTLED',remaining=dict.fromkeys(intake.DIMS,0),
            known_terminal_settlement_sha256=rh,scientific_verdict='SOURCE_COMPONENT_RESULTS_KNOWN_NOT_STUDENT_QUALIFICATION')
        ledger['source_cpu_reserved']={k:ledger['source_cpu_reserved'][k]-amount[k] for k in intake.DIMS}
        ledger.setdefault(INDEX,{})[request]=deepcopy(receipt)
    before=dims(c['source_cpu_current_classification'])
    for k,field in (('CoreLab_CPU_s','CPU_s'),('episode_starts','episode_starts'),('physics_ticks','physics_ticks'),('constructors','constructors')):
        need(before[k]==c[field],'ORIGINAL_CORE_CLASSIFICATION_COUNTERS');cc[field]+=charged[k]
    cc['source_cpu_current_classification']={k:before[k]+charged[k] for k in intake.DIMS}
    for site,k in (('CoreLab','CoreLab_CPU_s'),('local_PC','local_CPU_s')):
        need(pp['CPU_reserved_s'][site]>=amount[k],'FULL_ORIGINAL_PARENT_HOLD')
        pp['CPU_reserved_s'][site]-=amount[k];pp['CPU_s'][site]+=charged[k]
    pp['episode_starts']+=charged['episode_starts'];pp['engine_starts']+=constructors
    pp['physics_ticks']['live']+=live;pp['physics_ticks']['shadow']+=shadow
    pp['network_transfer_accounted_bytes']+=charged['transfer_bytes']
    formal=[x for x in pp['children'] if x.get('source_budget_request_id')==request]
    need(len(formal)==1,'ONE_ORIGINAL_FORMAL_CHILD')
    for field,k in (('CPU_reserved_s','CoreLab_CPU_s'),('local_CPU_reserved_s','local_CPU_s'),
        ('episode_starts_reserved','episode_starts'),('physics_ticks_reserved','physics_ticks'),('engine_starts_reserved','constructors')):
        need(formal[0][field]==amount[k],'FULL_FORMAL_CHILD_HOLD');formal[0][field]=0
    formal[0].update(status='CLOSED_SOURCE_KNOWN_TERMINAL_NOT_STUDENT_QUALIFICATION',known_terminal_settlement_sha256=rh)
    # Previous unknown invoice/index and financial-upper bucket are immutable.
    for prior,after in ((p,pp),(c,cc)):
        need(prior.get('source_cpu_financial_closeouts',{})==after.get('source_cpu_financial_closeouts',{}),
            'OLD_UNKNOWN_INVOICES_IMMUTABLE')
    need(p['physics_ticks'].get('closed_source_component_financial_upper')==
        pp['physics_ticks'].get('closed_source_component_financial_upper'),'OLD_UPPER_NEVER_MOVED_TO_LIVE')
    return dict(status='PROPOSAL_ONLY_KNOWN_TERMINAL_NO_NATIVE_GO',
        before_parent_sha256=sha(parent_bytes),before_Core_sha256=sha(Core_bytes),
        settlement=receipt,settlement_sha256=rh,proposed_parent=pp,proposed_Core=cc,
        original_child_ledger_must_remain_unchanged=True,native_dispatch_admitted=False)
