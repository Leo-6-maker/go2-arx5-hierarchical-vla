"""Close an unsubmitted preparation without inventing a native PBS result.

The old prepaid control and complete directory remain. Only its unused future
native forecast is retired, using the actual dead owner's closed evidence.
No transport, effects, lease changes or scheduler are implemented here.
"""
from copy import deepcopy
import re

from . import minimal_execution as e

SCHEMA='ARX5_E2_CLOSED_UNREGISTERED_PREPARATION_V1'
OLD_REQUEST='7a21e9cd-c0d5-430b-a39c-7c90e0deff10'
OLD_SOURCE='de0241051ebda953812bf69386156647e7f5c06e'
OLD_MANIFEST='20a13c6707bf7ee3fe8956cbdcd5c5062be147314c367d68656ba2a23a2275fa'
OLD_OS='9aec6a958cada94bf2b03525b39831ee604fe3a6fc1f96eff69239d224cc9b0b'
OLD_STOP='620bd19ccd678184c9a034fb210dd37a28f0b5e2575872e0d68446585154916b'
OLD_RUN='arx5-minimal-e2-only-raw-fixed512-02'
PINS=('closed_lease_sha256','snapshot_sha256','snapshot_identity_sha256')


def binding(v):
    e.need(type(v)is dict and v.get('schema')==SCHEMA
        and v.get('request_id')==OLD_REQUEST and v.get('run_uuid')==OLD_RUN
        and v.get('source_sha')==OLD_SOURCE and v.get('manifest_sha256')==OLD_MANIFEST
        and v.get('owner_OS_sha256')==OLD_OS and v.get('owner_stop_sha256')==OLD_STOP
        and v.get('native_requests')==0 and type(v['native_requests'])is int
        and v.get('all_recorded_actors_terminal')is True
        and v.get('lease_status')=='CLOSED'
        and type(v.get('allocated_bytes'))is int and 0<v['allocated_bytes']<=256*1024**2
        and all(type(v.get(k))is str and re.fullmatch('[0-9a-f]{64}',v[k])for k in PINS),
        'E2ONLY_EXACT_REAL_UNSUBMITTED_OWNER_CLOSURE')
    return v


def retired(g):
    from . import minimal_e2only_execution as x
    entries=g.get(x.RETIRED,{})
    e.need(type(entries)is dict and set(entries)<= {OLD_REQUEST},
        'E2ONLY_ONLY_IDENTIFIED_UNSUBMITTED_PREPARATION')
    for rid,p in entries.items():
        binding(p);paid=g.get(x.PREPAID,{}).get(rid,{})
        e.need(all(paid.get(k)==p[k]for k in('request_id','run_uuid','source_sha'))
            and paid.get('run_manifest_sha256')==p['manifest_sha256']
            and all(r.get('request_id')!=rid and r.get('run_uuid')!=p['run_uuid']for r in g['PBS_submissions']),
            'E2ONLY_NEVER_RETIRE_A_REGISTERED_OR_UNKNOWN_NATIVE_REQUEST')
    return entries


def propose(g,record,*,fresh,now_epoch):
    from . import minimal_e2only_execution as x
    from scripts.pi0294.pv1.neural_ledger import check
    check(g);binding(record);retired(g)
    e.need(x.RETIRED not in g and g['pending_updates']==g['unknown_effects']==[]
        and all(r['status']==e.STATES[2]for r in g['PBS_submissions'])
        and fresh['GLOBAL_sha256']==e.sha(e.raw(g))
        and fresh['request_id']==OLD_REQUEST and fresh['native_requests']==0
        and fresh['all_recorded_actors_terminal']is True
        and fresh['closed_record_sha256']==e.sha(e.raw(record))
        and 0<=now_epoch-fresh['observed_epoch']<=120,
        'E2ONLY_FRESH_EXACT_UNSUBMITTED_CLOSE_NOT_QUEUE_ABSENCE_ALONE')
    result=deepcopy(g);result[x.RETIRED]={OLD_REQUEST:deepcopy(record)};result['revision']+=1
    retired(result);check(result)
    e.need(result[x.PREPAID]==g[x.PREPAID]
        and result['PBS_submissions']==g['PBS_submissions'],
        'E2ONLY_RETAIN_ALL_PAID_CONTROLS_AND_NATIVE_HISTORY')
    return result


def validate_payment(g,m,*,allow_registered=False):
    from . import minimal_e2only_execution as x
    e.need(m.get('preparation_only')is True and m.get('schema')=='ARX5_E2_PRE_NATIVE_PREPARATION_PAYMENT_V1'
        and e._uuid(m.get('request_id'))and e._uuid(m.get('submission_nonce'))
        and m['run_uuid']==x.c.run_name(2,1)
        and m['paid_control']==x.CONTROL and m['native_upper']==x.native_resources()
        and m['new_GPU_permission']is False and m['Core_LOCAL_written']is False
        and m['new_storage_charge_bytes']==0 and g['CPU_s']>=m['before_CPU_s']+x.CONTROL['Hopper_CPU_s']
        and type(m['source_sha'])is str and re.fullmatch('[0-9a-f]{40}',m['source_sha'])
        and all(type(m.get(k))is str and re.fullmatch('[0-9a-f]{64}',m[k])for k in
            ('preparation_plan_sha256','before_GLOBAL_sha256','retired_preparation_sha256'))
        and m['retired_preparation_sha256']==e.sha(e.raw(retired(g)[OLD_REQUEST]))
        and (allow_registered or not any(r['request_id']==m['request_id']for r in g['PBS_submissions'])),
        'E2ONLY_PREPARATION_PAYMENT_IS_NO_NATIVE_MANIFEST_OR_GPU_GRANT')
    return m


def propose_payment(global_bytes,local_bytes,*,identity,plan_sha256,fresh,now_epoch):
    """Pay control before the finite CPU preflight; bind its real manifest later."""
    from . import minimal_e2only_execution as x
    from scripts.pi0294.pv1.neural_ledger import check
    g=__import__('json').loads(global_bytes);check(g);closed=retired(g)[OLD_REQUEST]
    e.need(identity['run_uuid']==x.c.run_name(2,1)
        and identity['request_id']not in x.paid_retry_controls(g)
        and g['pending_updates']==g['unknown_effects']==[]
        and all(r['status']==e.STATES[2]for r in g['PBS_submissions'])
        and fresh['GLOBAL_sha256']==e.sha(global_bytes)
        and fresh['retired_preparation_sha256']==e.sha(e.raw(closed))
        and fresh['no_existing_new_owner_or_dispatch']is True
        and 0<=now_epoch-fresh['observed_epoch']<=120,
        'E2ONLY_FRESH_ONE_CLOSED_PREPARATION_BEFORE_NEW_CONTROL')
    v=deepcopy(g);v['CPU_s']+=x.CONTROL['Hopper_CPU_s'];v['revision']+=1
    marker=dict(identity,schema='ARX5_E2_PRE_NATIVE_PREPARATION_PAYMENT_V1',preparation_only=True,
        preparation_plan_sha256=plan_sha256,retired_preparation_sha256=e.sha(e.raw(closed)),
        paid_control=deepcopy(x.CONTROL),native_upper=x.native_resources(),before_CPU_s=g['CPU_s'],
        before_GLOBAL_sha256=e.sha(global_bytes),paid_epoch=now_epoch,new_storage_charge_bytes=0,
        new_GPU_permission=False,Core_LOCAL_written=False)
    v.setdefault(x.PREPAID,{})[identity['request_id']]=marker
    validate_payment(v,marker);check(v)
    future=e.scope.first512_forecast(e.raw(v),local_bytes)
    e.need(future['cap_changes_required']=={},'E2ONLY_ORIGINAL_ALL14_BEFORE_PREPARATION_CONTROL')
    return dict(proposed_global=v,full_future_proof=future,new_native_permission=False)
