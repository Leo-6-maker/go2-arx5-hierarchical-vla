"""Original onceIO for exact candidate/previously prepaid Core RPC telemetry.

No resource reservation or canonical mutation. Terminal owner must merge these
confirmed records together with the finite neural effect state, exactly once.
"""
from copy import deepcopy
import json
import base64
from pathlib import Path

from scripts.pi0294.dec1 import source_cpu_once_io as once
from . import mix_accounting as a, mix_core_delegation as contract

FILE='CORE_DELEGATED_CONTROL.json'
PREFIX='DELEGATED-CONTROL-'
SCHEMA='V5_CD_CORE_DELEGATED_ORIGINAL_CONTROL_V1'


def need(ok,why):
    if not ok:raise PermissionError('CORE_DELEGATED_CONTROL_'+why)


def scope(grant_bytes,global_bytes):
    """Exact already-paid control slice, independently derived from GLOBAL."""
    from . import mix_core_accounting as c
    body=contract._body(grant_bytes);g=json.loads(global_bytes);rid=body['request']['request_id']
    from scripts.pi0294.generalization import minimal_core_accounting as first512
    if first512.typed(body['request']):
        row=g[first512.FIELD][rid];first512.validate_row(row)
        need(row['control_payment']==first512.CONTROL,'ACTUAL_NEW512_PAID_CONTROL')
        return first512.finite_control_scope(g,rid)
    if 'V5_CD_D25_exact_e686_local_successor' in g:
        from .mix_core_d25_local_successor import finite_control_scope
        finite=finite_control_scope(g,rid)
        if finite is not None:return finite
    item=g[a.PACKAGE]['prepaid_control'][c.key(body['request']['evaluation_candidate'])]
    need(item['costs']==c.CONTROL,'ORIGINAL_PAID_CONTROL_SLICE')
    return dict(schema='V5_CD_ORIGINAL_PREPAID_CONTROL_SCOPE_V1',marker_sha256=a.sha(a.raw(item)),
        request_id=rid,CPU_s=7200,RPC_calls=150000,source_preparation_CPU_s=600,
        financial_costs=deepcopy(c.CONTROL))


def initial(grant_bytes,*,adopted_global_bytes):
    body=contract._body(grant_bytes);g=json.loads(adopted_global_bytes)
    need(g.get(contract.FIELD,{}).get(body['request']['request_id'])==contract._marker(grant_bytes),
         'ACTUAL_DELEGATION_ADOPTION_BEFORE_CONTROL_INITIALIZATION')
    return dict(schema=SCHEMA,mode=contract.MODE,grant_sha256=a.sha(grant_bytes),
        request_id=body['request']['request_id'],control_scope=scope(grant_bytes,adopted_global_bytes),
        CPU_s=0,RPC_calls=0,candidates={},final_neural_CPU_s=None,events=[],
        canonical_GLOBAL_is_fresh=False,canonical_GLOBAL_is_writable=False)


def _adopted(root,grant_bytes):
    from .mix_core_delegated_ledger import ACTIVATION,verify_activation
    from . import mix_core_delegation_store as store
    bundle_bytes=once._file(Path(root)/'evidence'/ACTIVATION)
    state=store.read_confirmed(root,expected_grant_sha256=a.sha(grant_bytes))['state']
    verify_activation(grant_bytes,bundle_bytes,state)
    return base64.b64decode(json.loads(bundle_bytes)['raw_base64']['global_bytes'],validate=True)


def _fold(value,event,body):
    need(set(event)=={'token','kind','values','actor_pid','actor_start','epoch','before_sha256'},'EXACT_EVENT')
    need(event['token']==f"{event['actor_pid']}:{event['actor_start']}:{len(value['events'])}:{event['kind']}"
         and all(type(event[k]) is int and event[k]>0 for k in ('actor_pid','actor_start')),'EXACT_ONCE_ACTOR_TOKEN')
    now=a.number(event['epoch']);need(body['issued_epoch']<=now<body['accounting_not_after_epoch'],'ORIGINAL_ACCOUNTING_WINDOW')
    if value['events']:need(now>=value['events'][-1]['epoch'],'MONOTONIC_EVENT_CLOCK')
    kind,vals=event['kind'],event['values']
    if kind=='METRICS':
        need(set(vals)=={'CPU_s','RPC_calls'},'ONLY_PREPAID_METRICS')
        cpu=a.number(vals['CPU_s']);calls=a.number(vals['RPC_calls'],integer=True)
        limits=value['control_scope']
        need(value['CPU_s']+cpu+limits['source_preparation_CPU_s']<=limits['CPU_s']
             and value['RPC_calls']+calls<=limits['RPC_calls'],'ORIGINAL_SHARED_PREPAID_CONTROL_LIMIT')
        value['CPU_s']+=cpu;value['RPC_calls']+=calls
    elif kind=='CANDIDATE':
        request=body['request'];candidate=request['evaluation_candidate']
        from scripts.pi0294.generalization import minimal_core_accounting as first512
        step=512 if first512.typed(request) else 1024
        expected={candidate:dict(checkpoint_manifest_sha256=request['checkpoint_manifest_sha256'],global_step=step)}
        need((event['actor_pid'],event['actor_start'])==(body['issued_row']['pid'],body['issued_row']['proc_start'])
             and vals==expected and not value['candidates'],'ONLY_BOUND_NEURAL_EXACT_CLOSED_CANDIDATE_ONCE')
        value['candidates']=deepcopy(expected)
    elif kind=='FINAL_NEURAL_CPU':
        need(set(vals)=={'CPU_s'} and value['final_neural_CPU_s'] is None
             and (event['actor_pid'],event['actor_start'])==(body['issued_row']['pid'],body['issued_row']['proc_start']),
             'ONLY_BOUND_NEURAL_FINAL_CPU_ONCE')
        cpu=a.number(vals['CPU_s']);need(cpu<=body['limits']['CPU_s'],'ORIGINAL_NEURAL_CPU_UPPER')
        value['final_neural_CPU_s']=cpu
    else:need(False,'KNOWN_ORIGINAL_CONTROL_OPERATION')
    value['events'].append(deepcopy(event))


def read_confirmed(root,grant_bytes,*,adopted_global_bytes=None):
    path=Path(root)/'evidence'/FILE;raw=once._file(path);value=json.loads(raw)
    body=contract._body(grant_bytes)
    adopted=_adopted(root,grant_bytes) if adopted_global_bytes is None else adopted_global_bytes
    rebuilt=initial(grant_bytes,adopted_global_bytes=adopted)
    need(type(value.get('events')) is list and len(value['events'])<=256,'FINITE_ORIGINAL_CONTROL_ACTORS')
    expected=set()
    for event in value['events']:
        before=a.sha(a.raw(rebuilt));need(event['before_sha256']==before,'EXACT_BEFORE_STATE')
        _fold(rebuilt,event,body);after=a.sha(a.raw(rebuilt))
        operation=PREFIX+a.sha(event['token'].encode());ip=path.with_name(operation+'.INTENT.json');rp=path.with_name(operation+'.RECEIPT.json')
        ib,rb=once._file(ip),once._file(rp);contract._receipt(ib,rb,before=before,after=after)
        intent=json.loads(ib);receipt=json.loads(rb)
        need(intent['target']==receipt['target']==str(path.absolute())
             and intent['request_id']==receipt['request_id']==operation,'EXACT_ORIGINAL_CONTROL_CAS')
        expected.update((ip.name,rp.name))
    need(rebuilt==value,'EXACT_REPLAY_NO_COUNTER_RESET_OR_FAKE_GLOBAL')
    need({p.name for p in path.parent.glob(PREFIX+'*')}==expected
         and not any(path.parent.glob(path.name+'.source-once-'+PREFIX+'*')),'UNKNOWN_OR_ORPHAN_CONTROL_INTENT')
    return value


def commit(root,grant_bytes,*,kind,values,actor_pid,actor_start,now_epoch,expected_before_sha256,adopted_global_bytes=None):
    path=Path(root)/'evidence'/FILE;value=read_confirmed(root,grant_bytes,adopted_global_bytes=adopted_global_bytes);before=a.sha(a.raw(value))
    need(before==expected_before_sha256,'NO_STALE_TRANSACTION_OR_RETRY')
    event=dict(token=f'{actor_pid}:{actor_start}:{len(value["events"])}:{kind}',kind=kind,values=values,
               actor_pid=actor_pid,actor_start=actor_start,epoch=now_epoch,before_sha256=before)
    _fold(value,event,contract._body(grant_bytes));operation=PREFIX+a.sha(event['token'].encode())
    receipt=once.commit(path,expected_before_sha256=before,proposed_bytes=a.raw(value),request_id=operation,
        intent_path=path.with_name(operation+'.INTENT.json'),receipt_path=path.with_name(operation+'.RECEIPT.json'),
        lock_path=path.with_name('CORE_DELEGATED_CONTROL.owner.lock'))
    need(read_confirmed(root,grant_bytes,adopted_global_bytes=adopted_global_bytes)==value,'INDEPENDENT_ACTUAL_COMMIT_READBACK')
    return dict(receipt=receipt,record=value,canonical_GLOBAL_is_fresh=False,canonical_CPU_charged=False)


def merge_terminal(root,grant_bytes,state_bytes,global_bytes,local_bytes,**terminal_inputs):
    """Confirmed local receipts + original pure terminal merge, no native write.

    Only this request's prepaid telemetry/candidate is appended. The current
    canonical ledgers and all other scopes' changes are retained. The control
    digest remains in both terminal markers, preventing independent replay.
    """
    from . import mix_core_delegation_store as store, mix_core_accounting as c
    confirmed=store.read_confirmed(root,expected_grant_sha256=a.sha(grant_bytes))
    need(confirmed['state_sha256']==a.sha(state_bytes),'CURRENT_CONFIRMED_EFFECT_BYTES')
    value=read_confirmed(root,grant_bytes);body=contract._body(grant_bytes)
    from scripts.pi0294.generalization import minimal_core_accounting as first512
    if first512.typed(body['request']):c=first512
    proof=terminal_inputs['native_proof'];rid=body['request']['request_id']
    need(all(event['epoch']<=terminal_inputs['now_epoch'] for event in value['events']),
         'ALL_ACTUAL_CONTROL_TERMINAL_RECORDS_PRECEDE_MERGE')
    if value['final_neural_CPU_s'] is not None:
        need(value['final_neural_CPU_s']<=proof['conservative_CPU_s'],'INDEPENDENT_TERMINAL_COVERS_MEASURED_NEURAL_CPU')
    result=contract.merge_terminal(grant_bytes,state_bytes,global_bytes,local_bytes,**terminal_inputs)
    g,l=result['proposed_global'],result['proposed_LOCAL']
    need(scope(grant_bytes,global_bytes)==value['control_scope'],'SAME_ACTUAL_PAID_CONTROL_SCOPE_AT_TERMINAL')
    metrics=g.setdefault('V5_CD_prepaid_control_actual_metrics',{}).setdefault(rid,dict(CPU_s=0,RPC_calls=0))
    need(metrics==dict(CPU_s=0,RPC_calls=0),'NO_MIXED_CANONICAL_CONTROL_WRITER')
    limits=value['control_scope']
    need(value['CPU_s']+limits['source_preparation_CPU_s']<=limits['CPU_s']
         and value['RPC_calls']<=limits['RPC_calls'],'ORIGINAL_PREPAID_CONTROL_BOUND')
    metrics.update(CPU_s=value['CPU_s'],RPC_calls=value['RPC_calls'])
    candidates=g.setdefault(c.CANDIDATES,{})
    for key,endpoint in value['candidates'].items():
        need(key not in candidates or candidates[key]==endpoint,'NO_DIFFERENT_CANONICAL_CANDIDATE')
        candidates[key]=deepcopy(endpoint)
    need(len(candidates)<=1 if c is first512 else len(candidates)<=2,'DECLARED_CANDIDATES_ONLY')
    for ledger in (g,l):
        ledger[contract.FIELD][rid]['terminal_control_sha256']=a.sha(a.raw(value))
    c.checked(g);c.paired(g,l)
    result['receipt']['delegated_control_sha256']=a.sha(a.raw(value))
    result['receipt']['prepaid_control_actual_metrics']=deepcopy(metrics)
    return result
