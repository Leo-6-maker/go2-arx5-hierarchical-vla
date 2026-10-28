"""Exact human-directed handoff preparation; no IO or fresh-Hopper claim.

The registered Vanda administrative entry must persist the result once, under
one owner, before any terminal accounting or fresh training admission. The old
Hopper parent remains an explicitly fenced historical origin until reconciled.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from . import b_retry2970_failure as failed

PLAN='reports/PI0294/design/parallel-science-20261022/VANDA_ONLY_TRAINING_EXECUTION_DIRECTION_20261023.json'
RELEASE='0568957e67ee6356b9eec4cd262d9f569fee6a51'
GRANT_SHA='7a33de7f976ffbd4556400478fbbe1aa7c3545a72ff1ef9b32e1b37d8ddcfe72'
GLOBAL_SHA='963cb3b81c807c33443f5988413eb8b15a99c62a50421cdaceb6cc17c642faa9'
DEC1_SHA='3a0c7502fd73e9bf76967aa7c6d527e77089a561e497fd2760f7b96d7cb6951a'
FIELD='Vanda_only_controller_handoff'
JOURNAL_CATALOG_SHA='7af6c6d8d74e24d7441e9cf99169836614bc28496c1453a7bcf59bb7a63cc8a0'
JOURNAL_FILES=1599
RUN='vap1-a40-b-recovery1-b2-retry1-01'
IDENTITY=dict(run_uuid=RUN,job_id=failed.JOB,source_sha=failed.SOURCE,request_sha256=failed.REQUEST_SHA)
RELEASE_LABELS=('source-A40-once-native-MODEL_START-release','source-A40-once-native-FIRST_UPDATE-release')
RELEASE_PHASES={'MUTATION_'+hashlib.sha256(label.encode()).hexdigest()[:20].upper() for label in RELEASE_LABELS}


def need(ok, why):
    if not ok:raise PermissionError('VANDA_CONTROL_HANDOFF_'+why)


def grant(raw):
    need(isinstance(raw,bytes) and hashlib.sha256(raw).hexdigest()==GRANT_SHA,'EXACT_PUBLISHED_HUMAN_DIRECTION_BYTES')
    value=json.loads(raw)
    need(value['status']=='APPROVED_USER_DIRECTION_COLD_IMPLEMENTATION_NOT_NATIVE_ADOPTION'
         and value['scope']['training_platform']=='VANDA_A40'
         and value['budget_and_science']['new_budget_grant'] is False,
         'VANDA_DIRECTION_IS_NOT_ADDITIONAL_BUDGET')
    return value


def read_journal(directory):
    """Read the entire stopped-owner catalog; no cherry-picked operation list."""
    directory=Path(directory)
    need(directory.is_dir() and not directory.is_symlink() and directory.stat().st_uid==os.getuid(),
         'REAL_OWN_STOPPED_CONTROLLER_JOURNAL')
    paths=sorted(directory.glob('*.json'));need(len(paths)==JOURNAL_FILES,'COMPLETE_CONTROLLER_FILE_CATALOG')
    raw={}
    for path in paths:
        value=path.lstat()
        need(stat.S_ISREG(value.st_mode) and value.st_uid==os.getuid()
             and value.st_size<2*1024**2,'BOUNDED_OWN_REGULAR_JOURNAL_JSON')
        raw[path.name]=path.read_bytes()
    catalog=[(name,hashlib.sha256(value).hexdigest()) for name,value in sorted(raw.items())]
    digest=hashlib.sha256(json.dumps(catalog,separators=(',',':')).encode()).hexdigest()
    need(digest==JOURNAL_CATALOG_SHA,'FULL_EXACT_FINISHED_CONTROLLER_CATALOG')
    need([p.name for p in paths]==[p.name for p in sorted(directory.glob('*.json'))]
         and all(path.read_bytes()==raw[path.name] for path in paths),'JOURNAL_UNCHANGED_DURING_COMPLETE_READ')
    values={k:json.loads(v) for k,v in raw.items()};rows=[]
    for name,row in sorted(values.items()):
        if not name.endswith('-INTENT.json'):continue
        intent=row['intent'];operation=intent['operation_id']
        confirmed=values.get(operation+'-CONFIRMED.json',{})
        if confirmed.get('intent')==intent and confirmed.get('event')=='CONFIRMED':
            status='CONFIRMED'
        else:
            error=values.get(operation+'-ERROR.json',{})
            need(intent['effect']=='READ_ONLY' and error.get('intent')==intent
                 and error.get('event')=='ERROR' and error.get('outcome')=='READ_FAILED',
                 'EVERY_EFFECT_BEARING_INTENT_HAS_ACTUAL_CONFIRMATION')
            status='READ_FAILED'
        rows.append(dict(intent=intent,resolved_status=status))
    return rows,values['RPC_EFFECT_STATE.json'],dict(sha256=digest,files=len(paths),intents=len(rows))


def journal(rows, current, catalog):
    """Every effect-bearing call in the stopped controller must be accounted for.

    The original qualified ColdRPC maps these two release labels to Vanda;
    every other effect-bearing phase is rejected. Read-only failures are not
    converted into success and do not stand in for a native queue observation.
    """
    need(catalog==dict(sha256=JOURNAL_CATALOG_SHA,files=JOURNAL_FILES,intents=799)
         and len(rows)==799,'SEALED_COMPLETE799_OPERATION_CATALOG')
    need(current.get('status')=='CONFIRMED' and current.get('intent',{}).get('identity')==IDENTITY
         and current['intent']['effect']=='HEARTBEAT','EXACT_LAST_CONFIRMED_HEARTBEAT')
    operations=set();releases=set();read_count=heartbeat_count=read_failed=0
    for row in rows:
        intent=row['intent'];operation=intent['operation_id']
        need(operation not in operations and intent['identity']==IDENTITY,'UNIQUE_SAME_REQUEST_OPERATIONS')
        operations.add(operation);effect=intent['effect']
        need(effect in ('READ_ONLY','HEARTBEAT','MUTATION'),'EXPLICIT_EFFECT_CLASS')
        if effect=='READ_ONLY':
            need(row.get('resolved_status') in ('CONFIRMED','READ_FAILED'),'CLASSIFIED_READ_OUTCOME_RETAINED')
            read_count+=1;read_failed+=row.get('resolved_status')=='READ_FAILED';continue
        need(row.get('resolved_status') in ('CONFIRMED','CONFIRMED_BY_READ_ONLY_READBACK'),
             'NO_UNRESOLVED_EFFECT_BEARING_CALL')
        if effect=='MUTATION':
            need(intent['phase'] in RELEASE_PHASES and intent['phase'] not in releases,
                 'ONLY_ORIGINAL_ONCE_VANDA_RELEASES_NO_UNSEEN_PARENT_WRITES')
            releases.add(intent['phase'])
        else:heartbeat_count+=1
    need(releases==RELEASE_PHASES and current['intent']['operation_id'] in operations
         and read_count>0 and heartbeat_count>0,'BOTH_RELEASES_AND_COMPLETE_FINISHED_OWNER_JOURNAL')
    return dict(read_calls=read_count,failed_reads_preserved=read_failed,heartbeat_calls=heartbeat_count,Vanda_release_calls=len(releases),
                inferred_Hopper_writes_after_snapshot=0,unresolved_effect_calls=0)


def prepare(global_bytes, dec1_bytes, *, grant_bytes, owner_state, stop, current_rpc,
            controller_journal, controller_catalog, now_epoch):
    """Return copies with an origin marker; budgets/reservations are unchanged."""
    grant(grant_bytes);failed.owner(owner_state,stop,current_rpc)
    need(type(now_epoch) in (int,float) and math.isfinite(now_epoch)
         and now_epoch<1792952996,'ACTUAL_CLOCK_BEFORE_ORIGINAL_HARD_CLOSE')
    need(hashlib.sha256(global_bytes).hexdigest()==GLOBAL_SHA
         and hashlib.sha256(dec1_bytes).hexdigest()==DEC1_SHA,'EXACT_LAST_REAL_PAIRED_PARENT_BYTES')
    proof=journal(controller_journal,current_rpc,controller_catalog)
    h,d=json.loads(global_bytes),json.loads(dec1_bytes)
    need(FIELD not in h and FIELD not in d,'NO_SECOND_HANDOFF_OR_REPLAY')
    need(h['optimizer_updates']==d['optimizer_updates']==8744 and h['generation_queries']==1197
         and h['used']['PBS_submissions']==23 and len(d['new_submissions'])==12
         and h['unknown_effects']==h['pending_updates']==d['unknown_effects']==[],
         'EXACT_PRE_FAILURE_PARENT_EFFECTS_AND_NO_UNKNOWN_WRITES')
    pending=[r for r in h['PBS_submissions'] if r['status']!='NATIVE_TERMINAL']
    need(len(pending)==1 and pending[0]['job_id']==failed.JOB
         and pending[0]['request_id']==failed.REQUEST and pending[0]['native_platform']=='VANDA_A40',
         'NO_UNRESOLVED_HOPPER_ALLOCATION_OR_OTHER_CAMPAIGN_OWNER')
    need(h['reserved']['H200_s']==22560 and h['reserved']['optimizer_updates']==2048
         and h['reserved']['neural_queries']==29 and h['CPU_reserved_s']==54000
         and h['storage_reserved_bytes']==70000000000,'OLD_RESERVATIONS_ARE_NOT_RELEASED_BY_MIGRATION')
    marker=dict(schema='WDR1_VANDA_SINGLE_CONTROLLER_HANDOFF_V1',authority_release=RELEASE,
        grant_sha256=GRANT_SHA,origin_global_sha256=GLOBAL_SHA,origin_DEC1_sha256=DEC1_SHA,
        origin_platform='HOPPER',origin_access='UNAVAILABLE_FENCED_BY_HUMAN_DIRECTION',
        origin_is_fresh_native_read=False,new_control_platform='VANDA_A40',
        old_parent_reconciliation_required_when_available=True,old_parent_mutations_permitted=False,
        one_mutation_owner=True,old_job_and_all_reservations_pending_terminal_accounting=True,
        new_GPU_PBS_training_or_budget=0,journal=proof,prepared_epoch=now_epoch)
    nh,nd=deepcopy(h),deepcopy(d);nh[FIELD]=deepcopy(marker);nd[FIELD]=deepcopy(marker)
    need({k:v for k,v in nh.items() if k!=FIELD}==h
         and {k:v for k,v in nd.items() if k!=FIELD}==d,'EVERY_HISTORY_COST_CAP_COUNTER_RESERVATION_UNCHANGED')
    return nh,nd,marker
