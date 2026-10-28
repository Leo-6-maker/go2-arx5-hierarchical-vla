"""Pure once-paid control CPU credit in the original GLOBAL/LOCAL ledgers.

No registration, transport, physical quota or native write. A cold owner must
persist both proposals using original locks/once IO before using the credit.
All paid failed controls stay charged; only the exact current request may
reuse its own live control envelope. Shared preparation is charged once.
"""
from copy import deepcopy
import re
import uuid
from . import evaluation_scope as scope

FIELD='V4_DEV_once_control_CPU_upper_payments'
BASE=dict(Core_CPU_s=180598.97856362094,Hopper_CPU_s=156569.658597758,local_CPU_s=51058.974682509)
CPU_KEYS=tuple(BASE)
SITES=dict(Core_CPU_s='CoreLab',Hopper_CPU_s='Hopper',local_CPU_s='local_PC')


def need(ok,why):
    if not ok:raise PermissionError('PV1_V4_CONTROL_'+why)


def index(g,l):
    rows=g.get(FIELD,{})
    need(type(rows) is dict and rows==l.get(FIELD,{}) and len(rows)<=6,'SAME_ONCE_CONTROL_INDEX_BOTH_LEDGERS')
    preparers=0
    for pid,r in rows.items():
        try:valid=str(uuid.UUID(pid))==pid and uuid.UUID(pid).version==4
        except (ValueError,TypeError,AttributeError):valid=False
        need(valid and type(r) is dict and set(r)=={'payment_id','request_id','source_sha','run_uuid','work_id',
             'candidate','noise_bank','control_CPU_upper_s','shared_preparation_CPU_upper_s','before_sha256',
             'grant_sha256','created_epoch','native_dispatch_admitted','charges_DEC1_subcaps'}
             and r['payment_id']==pid and r['grant_sha256']==scope.GRANT_SHA
             and r['candidate'] in ('B0_CORR','A_CONTROL','B_COVERAGE') and r['noise_bank'] in (294101,294102)
             and r['work_id']=='DEV:'+r['candidate']+':N'+str(r['noise_bank'])
             and r['control_CPU_upper_s']==7200 and r['shared_preparation_CPU_upper_s'] in (0,600)
             and r['native_dispatch_admitted'] is False and r['charges_DEC1_subcaps'] is False
             and type(r['created_epoch']) in (int,float) and scope.APPROVED<=r['created_epoch']<scope.STOP_NEW
             and re.fullmatch('[0-9a-f]{40}',r['source_sha']) is not None
             and type(r['before_sha256']) is dict and set(r['before_sha256'])=={'GLOBAL','LOCAL'}
             and all(re.fullmatch('[0-9a-f]{64}',v) is not None for v in r['before_sha256'].values()),
             'EXACT_PAID_CURRENT_CONTROL_METADATA_NOT_ARBITRARY_CREDIT')
        preparers+=r['shared_preparation_CPU_upper_s']==600
    need(not rows or preparers==1,'PREPARATION_PAID_EXACTLY_ONCE')
    total=sum(r['control_CPU_upper_s']+r['shared_preparation_CPU_upper_s'] for r in rows.values())
    need(all(l['CPU_s'][site]>=BASE[key]+total for key,site in SITES.items())
         and g['CPU_s']>=BASE['Hopper_CPU_s']+total,'ALL_THREE_CPU_DEBITS_REMAIN_IN_ACTUAL_COUNTERS')
    return rows


def credit(g,l,request,work):
    if FIELD not in g and FIELD not in l:return dict.fromkeys(CPU_KEYS,0)
    scope.validate_pair(g,l);rows=index(g,l)
    shared=600 if rows else 0;pid=request.get('control_prepayment_id');current=0
    if pid is not None:
        need(pid in rows,'NO_UNKNOWN_OR_REUSED_OTHER_PREPAYMENT')
        r=rows[pid]
        need(all(r[k]==request[f] for k,f in (('request_id','request_id'),('source_sha','source_sha'),
             ('run_uuid','run_uuid'),('candidate','evaluation_candidate'),('noise_bank','noise_bank')))
             and r['work_id'] in {j['work_id'] for j in work['jobs']},'EXACT_CURRENT_STILL_UNFINISHED_REQUEST_CONTROL')
        current=7200
    return dict.fromkeys(CPU_KEYS,shared+current)


def prepay_proposal(global_bytes,LOCAL_bytes,*,recompute_inputs,payment_id,now_epoch):
    """Propose the actual upper payment, never a reservation/refund or new IO."""
    from . import evaluation_admission as a
    from .evaluation_accounting import checked
    from .evaluation_work import remaining
    i=dict(recompute_inputs)
    need(i['global_bytes']==global_bytes and i['local_bytes']==LOCAL_bytes and i['now_epoch']==now_epoch,
         'CURRENT_RAW_BEFORE_INPUTS_AND_REAL_CLOCK')
    g,_=a._read(global_bytes);l,_=a._read(LOCAL_bytes);request,_=a._read(i['request_bytes'])
    scope.validate_pair(g,l);old=index(g,l)
    need(payment_id not in old and request.get('control_prepayment_id')==payment_id
         and not any(r['request_id']==request['request_id'] for r in old.values())
         and not any(r.get('status')!='NATIVE_TERMINAL' for r in g['PBS_submissions'])
         and g['unknown_effects']==g['pending_updates']==l['unknown_effects']==l['pending_queries']==l['pending_updates']==[],
         'ONE_NEW_INACTIVE_CONTROL_PAYMENT_NO_UNKNOWN_REISSUE')
    # Check full future using the actual pre-payment state. The new id is not
    # yet an actual payment, so it cannot receive credit before both writes.
    r0=deepcopy(request);r0.pop('control_prepayment_id')
    check_i=dict(i,request_bytes=a.canonical(r0))
    proof=a.recompute(**check_i,purpose='PREPARE')
    need(proof['all_remaining_costs_fit'] and proof['current_job_wall_fits_current_window']
         and proof['current_stage_clock_fits'],
         'COMPLETE_RESOURCES_AND_CURRENT_STAGE_CLOCK_BEFORE_PREPAYMENT')
    g,l=deepcopy(g),deepcopy(l);shared=0 if old else 600
    record=dict(payment_id=payment_id,request_id=request['request_id'],source_sha=request['source_sha'],
        run_uuid=request['run_uuid'],work_id=proof['work_id'],candidate=request['evaluation_candidate'],noise_bank=request['noise_bank'],
        control_CPU_upper_s=7200,shared_preparation_CPU_upper_s=shared,
        before_sha256=dict(GLOBAL=a.hashlib.sha256(global_bytes).hexdigest(),LOCAL=a.hashlib.sha256(LOCAL_bytes).hexdigest()),
        grant_sha256=scope.GRANT_SHA,created_epoch=now_epoch,native_dispatch_admitted=False,charges_DEC1_subcaps=False)
    for v in (g,l):v.setdefault(FIELD,{})[payment_id]=deepcopy(record)
    g['CPU_s']+=7200+shared
    for site in SITES.values():l['CPU_s'][site]+=7200+shared
    index(g,l);checked(g)
    return dict(proposed_global=g,proposed_LOCAL=l,payment=record,
        original_remaining_work=proof['remaining_work_sha256'],native_dispatch_admitted=False,IO_performed=False)
