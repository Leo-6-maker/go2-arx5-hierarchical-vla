"""One cold redistribution inside the SAME paid64MiB D25 control invoice.

No model/physics/compute/transfer total is added. The original financial marker
and its old plan are immutable. A new explicit dual-canonical marker may move
exactly64KiB from unused READ_ONLY differential capacity to setup, preserving
all spent/UNKNOWN uppers and both protected terminal/closeout banks. Existing
Journal/onceIO and the original counter remain the only mutation primitives.
"""
from copy import deepcopy
import json,re
from . import mix_accounting as a,mix_core_accounting as c
from . import mix_core_d25_local_successor as paid,mix_core_d25_cold_source as source

FIELD='V5_CD_D25_same_invoice_cold_control_wire'
SCHEMA='V5_CD_D25_SAME_64MIB_COLD_CONTROL_WIRE_V1'
COUNTER_FIELD='same_invoice_cold_wire_plan'
DONOR='bounded_unknown_readonly_differential'
RECEIVER='grant_bind_and_fixed_configuration'
SHIFT=64*1024
PROTECTED=('two_source_install_envelopes_total','terminal_reconcile_merge_payload','protected_closeout')
CPU=dict(core_cold=600,core_close=600,vanda=1800,local=1800)
ORIGINAL_CONTROLLER_SHA='e162079851661984110bdb69a54720ae5f13a6721ed82242a8979d575590fcba'

def need(ok,why):
    if not ok:raise PermissionError('D25_COLD_CONTROL_WIRE_'+why)
def digest(v):return a.sha(a.raw(v))
def caps():
    v=deepcopy(paid.WIRE);v[DONOR]-=SHIFT;v[RECEIVER]+=SHIFT
    need(sum(v.values())==paid.CONTROL['transfer_bytes']==64*1024**2,'SAME_TOTAL_INVOICE')
    return v

def effective_caps(counter):
    """Validate the same growing original counter after explicit adoption."""
    m=counter.get(COUNTER_FIELD)
    if m is None:return deepcopy(paid.WIRE)
    need(m['schema']==SCHEMA and m['old_caps']==paid.WIRE and m['new_caps']==caps()
         and m['shift_bytes']==SHIFT and m['donor']==DONOR and m['receiver']==RECEIVER
         and m['financial_costs']==counter['financial_costs']==paid.CONTROL
         and m['CPU_upper']==counter['CPU_upper']==CPU
         and m['original_counter_identity']==counter['identity']
         and m['same_original_counter_and_Journal']is True
         and m['financial_marker_unchanged']is True
         and m['new_model_or_physics_permission']is False
         and m['request_id']==source.REQUEST,'EXACT_NEW_PLAN_NOT_AN_UNBOUNDED_POOL')
    n=m['old_operation_count'];need(type(n)is int and 0<=n<=len(counter['operations'])
         and digest(counter['operations'][:n])==m['old_operations_sha256'],
         'EVERY_OLD_OPERATION_AND_UNKNOWN_UPPER_RETAINED')
    need(all(counter['wire_spent'][k]>=v for k,v in m['old_wire_spent'].items())
         and all(counter['CPU_spent'][k]>=v for k,v in m['old_CPU_spent'].items())
         and all(0<=counter['wire_spent'][k]<=v for k,v in caps().items())
         and all(0<=counter['CPU_spent'][k]<=v for k,v in CPU.items()),'NO_REFUND_OR_CAP_OVERSPEND')
    for key in('before_GLOBAL_sha256','before_LOCAL_sha256','before_counter_sha256',
               'original_financial_marker_sha256','fresh_native_join_sha256'):
        need(type(m[key])is str and re.fullmatch('[0-9a-f]{64}',m[key]),'FULL_IMMUTABLE_BOUNDARY:'+key)
    need(counter['wire_caps']==m['new_caps']
         and all(m['new_caps'][k]==m['old_caps'][k]for k in PROTECTED),
         'PROTECTED_BANKS_AND_EXPLICIT_EFFECTIVE_CAPS')
    return deepcopy(m['new_caps'])

def propose(counter_bytes,global_bytes,local_bytes,*,fresh_native_join,now_epoch):
    need(all(type(b)is bytes for b in(counter_bytes,global_bytes,local_bytes)),'EXACT_ORIGINAL_BYTES')
    counter,g,l=map(json.loads,(counter_bytes,global_bytes,local_bytes))
    need(COUNTER_FIELD not in counter and FIELD not in g and FIELD not in l
         and counter['wire_caps']==paid.WIRE and counter['financial_costs']==paid.CONTROL
         and counter['CPU_upper']==CPU,'ONCE_ORIGINAL_FINANCIAL_PLAN_NOT_REPAYMENT')
    # Reject before any budget proposal if even one old mutation remains
    # unqualified. An ERROR/UNKNOWN status itself is never cleared or rewritten.
    need(not any(x.get('status')=='RESERVED_UNKNOWN_UNTIL_EXACT_REPLY'
         and not x.get('independent_exact_readback_sha256')for x in counter['operations']),
         'UNQUALIFIED_OLD_UNKNOWN_REQUIRES_EXACT_READONLY')
    need(all(0<=counter['wire_spent'][k]<=v for k,v in caps().items())
         and all(0<=counter['CPU_spent'][k]<=v for k,v in CPU.items()),
         'ONLY_UNUSED_DONOR_CAPACITY_NOT_ALREADY_SENT_BYTES')
    c.checked(g);c.paired(g,l);financial=paid.validated(g);cold=source.validated(g,financial)
    need(cold is not None and financial==paid.validated(l)
         and cold==source.validated(l,financial)
         and financial['finite_control_costs']==paid.CONTROL
         and financial['wire_phase_upper_bytes']==paid.WIRE
         and financial['control_plan']['wire_phase_upper_bytes']==paid.WIRE,
         'SAME_NATIVE_DUAL_FINANCIAL_INVOICE_AND_TRUTHFUL_SOURCE')
    row=g[c.FIELD][source.REQUEST]
    need(row==l[c.FIELD][source.REQUEST]and row['status']==c.STATES[0]
         and row['native_handle']is None and c.progress(row)==(0,0,0)
         and g['unknown_effects']==g['pending_updates']==l['unknown_effects']==l['pending_updates']==l['pending_queries']==[]
         and all(x['status']in a.TERMINAL for x in g['PBS_submissions'])
         and all(k==source.REQUEST or x['status']==c.STATES[3]for k,x in g[c.FIELD].items()),
         'REGISTERED_COLD_ZERO_EFFECT_BOUNDARY_NOT_A_LIVE_CAPSULE_CHANGE')
    f=fresh_native_join
    need(f['GLOBAL_sha256']==a.sha(global_bytes)and f['LOCAL_sha256']==a.sha(local_bytes)
         and f['request_id']==source.REQUEST and f['request_sha256']==row['request_sha256']
         and f['native_owner_and_neural_units_not_started']is True
         and f['new_mutations']==0 and 0<=now_epoch-f['observed_epoch']<=120
         and financial['T_CD']<=now_epoch<financial['deadline_epoch']-28860,
         'FRESH_ACTUAL_DUAL_BYTES_AND_NATIVE_UNITS_NOT_A_CPU_PASS')
    m=dict(schema=SCHEMA,request_id=source.REQUEST,source_sha=cold['source_sha'],
        before_GLOBAL_sha256=a.sha(global_bytes),before_LOCAL_sha256=a.sha(local_bytes),
        before_counter_sha256=a.sha(counter_bytes),original_financial_marker_sha256=digest(financial),
        fresh_native_join_sha256=digest(f),old_caps=deepcopy(paid.WIRE),new_caps=caps(),
        donor=DONOR,receiver=RECEIVER,shift_bytes=SHIFT,financial_costs=deepcopy(paid.CONTROL),
        CPU_upper=deepcopy(CPU),old_wire_spent=deepcopy(counter['wire_spent']),old_CPU_spent=deepcopy(counter['CPU_spent']),
        old_operation_count=len(counter['operations']),old_operations_sha256=digest(counter['operations']),
        original_counter_identity=deepcopy(counter['identity']),adopted_epoch=now_epoch,
        T_CD=financial['T_CD'],deadline_epoch=financial['deadline_epoch'],
        same_original_counter_and_Journal=True,financial_marker_unchanged=True,
        new_model_or_physics_permission=False,new_budget=0,new_tensor_transfer=0)
    v,w=deepcopy(g),deepcopy(l)
    v[FIELD]=deepcopy(m);w[FIELD]=deepcopy(m);v['revision']+=1
    # Financial state, rows, history, clocks and all resource vectors are exact
    # unchanged bytes; only this management plan + GLOBAL revision are new.
    need({k:x for k,x in v.items()if k not in(FIELD,'revision')}=={k:x for k,x in g.items()if k!='revision'}
         and {k:x for k,x in w.items()if k!=FIELD}==l,'NO_FINANCIAL_OR_SCIENTIFIC_DELTA')
    return dict(marker=m,proposed_global=v,proposed_LOCAL=w,
        new_financial_resources=0,original_onceIO_required=True,native_admission=False)

def counter_after_native_dual_adoption(counter,marker,*,GLOBAL_sha256,LOCAL_sha256):
    """Pure same-counter transition after the root independently verifies CAS.

    The supplied hashes identify receipts; this function is not their verifier
    and does not write or confer runtime permission. Root onceIO remains needed.
    """
    need(COUNTER_FIELD not in counter and counter['wire_caps']==paid.WIRE,
         'ORIGINAL_COUNTER_APPLIED_ONCE')
    v=deepcopy(counter);v['wire_caps']=deepcopy(marker['new_caps']);v[COUNTER_FIELD]=deepcopy(marker)
    v['same_invoice_cold_wire_dual_receipts']=dict(GLOBAL_sha256=GLOBAL_sha256,LOCAL_sha256=LOCAL_sha256,
        independent_native_CAS_readback_required=True)
    effective_caps(v);return v

def controller_source(original_body):
    """Narrow cold policy injection into the SAME original ControlIO class.

    Original transport/envelope, request identity, Journal, onceIO, CPU, timeout,
    receipt, per-operation no-replay and counter file stay exact. No source on
    an experiment host is edited. Legacy counters retain the original caps.
    """
    need(type(original_body)is str and a.sha(original_body.encode())==ORIGINAL_CONTROLLER_SHA,
         'EXACT_ORIGINAL_MANAGEMENT_CONTROLLER_NOT_A_NEW_TRANSPORT')
    replacements={
      "and v['wire_caps']==scope.WIRE":"and v['wire_caps']==wire_plan.effective_caps(v)",
      "for k,n in scope.WIRE.items()":"for k,n in v['wire_caps'].items()",
      "<=scope.WIRE[phase]":"<=value['wire_caps'][phase]",
      "x.get('status')=='RESERVED_UNKNOWN_UNTIL_EXACT_REPLY'":
      "x.get('status')=='RESERVED_UNKNOWN_UNTIL_EXACT_REPLY'and not x.get('independent_exact_readback_sha256')"}
    body=original_body
    for before,after in replacements.items():
        need(body.count(before)==1,'EXACT_SINGLE_POLICY_SEAM:'+before);body=body.replace(before,after)
    return "from scripts.pi0294.pv1 import mix_core_d25_control_wire as wire_plan\n"+body
