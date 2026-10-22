"""Pure cold projection of CLOSED Core source costs into cumulative DEC1.

No IO/native adoption. Apply only after A4096 real closure, never to active A.
Original global spend is already charged by the LOCAL parent and is NOT charged
again. GPU/native CPU, model, request, checkpoint and grants remain unchanged.
"""
from copy import deepcopy
import hashlib,json,math
AUTHORITY='a7a7ecd9a6be62a7221af25a5e32a0009b076a84'
SCHEMA='DEC1_SOURCE_CPU_TERMINAL_BOUND_FINANCIAL_CLOSEOUT_V1'
FIELDS={'episode_starts':'episode_starts','physics_ticks':'physics_ticks','constructors':'constructors',
        'transfer_bytes':'network_transfer_accounted_bytes'}
MARKER='closed_Core_source_financial_projections'


def encode(v):return (json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def need(v,r):
 if not v:raise PermissionError('SOURCE_COST_PROJECTION_'+r)
def doc(raw):
 def pairs(values):
  out={}
  for k,v in values:need(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
def number(v,integer=False):
 need(type(v) in ((int,) if integer else (int,float)) and math.isfinite(v) and v>=0,'NONNEGATIVE_TYPED_COST');return v


def propose(parent_bytes,Core_bytes,Hopper_bytes,*,dual_receipt_bytes,A_boundary_bytes):
 p,c,d=map(doc,(parent_bytes,Core_bytes,Hopper_bytes));receipt=doc(dual_receipt_bytes)
 boundary=doc(A_boundary_bytes)
 need(boundary.get('schema')=='VAP1_A4096_CLOSED_FOR_INACTIVE_DEC1_SOURCE_COST_PROJECTION_V1'
      and boundary.get('job_id')=='1477527.stdct-mgmt-02' and boundary.get('step')==4096
      and boundary.get('native_state')=='F' and type(boundary.get('Exit_status')) is int and boundary['Exit_status']==0
      and boundary.get('owner_naturally_exited') is True and boundary.get('GPU_lease')==boundary.get('CPU_lease')=='CLOSED'
      and boundary.get('full_state_audit_passed') is True and boundary.get('once_settlement_confirmed') is True,
      'ACTUAL_A4096_COMPLETE_AND_BOTH_CLOSURES_BEFORE_INACTIVE_PROJECTION')
 need(number(p.get('GPU_reserved_s'))==0 and all(number(d.get(k,0))==0 for k in ('GPU_reserved_s','CPU_reserved_s','optimizer_updates_reserved','neural_queries_reserved')),
      'NEVER_MODIFY_ACTIVE_A_OR_ANY_NATIVE_GPU_RESERVATION')
 need(c.get('authority_release')==d.get('authority_release')==AUTHORITY and c['campaign_run_uuid']==d['campaign_run_uuid']==p['run_uuid']
      and c['original_T0_epoch']==d['original_T0_epoch']==p['T0_epoch']
      and c['hard_close_epoch']==d['hard_close_epoch']==p['hard_closeout_epoch']==1792952996,'ORIGINAL_SCOPE_AND_CLOCK')
 need(not any(p['CPU_reserved_s'].values()) and p['unknown_effects']==[] and p['Hopper_unknown_effects']==[]
      and c['unknown_effects']==d['unknown_effects']==[],'NO_ACTIVE_SOURCE_HOLD_OR_UNKNOWN_PARENT_WRITE')
 need(receipt['status']=='SOURCE_DUAL_FINANCIAL_TERMINAL_BOUND_SETTLED_CORE_LEASE_CLOSED_COMPONENT_UNKNOWN'
      and receipt['Core']['status']=='CORE_FINANCIAL_BOUND_CHARGED_LEASE_CLOSED_SCIENCE_UNKNOWN'
      and receipt['LOCAL_readback']['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT',
      'ACTUAL_DUAL_FINANCIAL_CLOSURE_REQUIRED')
 invoice=receipt['invoice'];request=invoice['request_id'];ih=sha(encode(invoice));amount=invoice['financial_charged']
 need(invoice['schema']==SCHEMA and invoice['authority_release']==AUTHORITY
      and invoice['scientific_verdict']=='ENGINEERING_INCOMPLETE_UNKNOWN'
      and invoice['actual_returned_STOP_ACK'] is False and receipt['Core']['invoice_sha256']==ih,
      'KEEP_UNKNOWN_SCIENTIFIC_IDENTITY')
 for ledger in (p,c):
  row=ledger['source_cpu_reservations'][request]
  need(row['status']=='SETTLED' and not any(row['remaining'].values()) and row['financial_closeout_sha256']==ih
       and ledger['source_cpu_financial_closeouts'][request]==invoice,'EXACT_CLOSED_SOURCE_RECORDS')
 need(request not in d.get(MARKER,{}),'NO_SECOND_CUMULATIVE_PROJECTION')
 need(invoice['original_T0_epoch']==p['T0_epoch'] and invoice['hard_close_epoch']==p['hard_closeout_epoch'],'INVOICE_ORIGINAL_CLOCK')
 for k in amount:number(amount[k],k not in ('CoreLab_CPU_s','local_CPU_s'))
 after=deepcopy(d)
 for k,field in FIELDS.items():after[field]=number(d.get(field,0),True)+amount[k]
 # These scalar fields are GPU-site CPU/charges and must NEVER receive Core or
 # LOCAL CPU. Their original mature snapshot reads those CPUs from LOCAL parent.
 for field in ('CPU_s','GPU_allocated_s','GPU_reserved_s','optimizer_updates','neural_queries','new_submissions'):
  need(after[field]==d[field],'PRESERVE_ALL_GPU_COSTS_AND_IDENTITIES')
 marker=dict(schema='DEC1_CLOSED_CORE_SOURCE_NONADDITIVE_COST_PROJECTION_V1',request_id=request,
  source_invoice_sha256=ih,dual_financial_receipt_sha256=sha(dual_receipt_bytes),
  actual_parent_ledger_sha256=sha(parent_bytes),actual_Core_ledger_sha256=sha(Core_bytes),
  already_inside_global_parent_spend=True,adds_no_global_charge=True,financial_charged=deepcopy(amount),
  Core_output_addition_to_legacy_prior_peak_upper=amount['CoreLab_output_bytes'],
  local_output_already_inside_inclusive_campaign_peak_upper_requires_actual_namespace_readback=True,
  scientific_verdict='ENGINEERING_INCOMPLETE_UNKNOWN',source_or_target_qualification=False)
 after.setdefault(MARKER,{})[request]=marker
 return dict(status='PROPOSAL_ONLY_INACTIVE_GPU_SITE_DEC1_COST_CATCHUP_NOT_TRAINING_ADMISSION',
  old_Hopper_DEC1_sha256=sha(Hopper_bytes),proposed_Hopper_DEC1=after,marker=marker,
  total_closed_Core_output_upper_bytes=sum(r['financial_charged']['CoreLab_output_bytes'] for r in after[MARKER].values()),
  native_A4096_terminal_and_both_leases_closed_gate_still_required=True,new_global_charge=False,native_dispatch_admitted=False)
