"""One human-approved PBS count extension after the closed -3 startup failure.

No GPU/CPU, update, query, IO, physics, clock or scientific gate changes.
"""
from copy import deepcopy
import math
from pathlib import Path
from scripts.pi0294.pv1.future_budget import _read
from . import b_heartbeat_recovery as b,b_saved_state_reuse as saved
JOB='1480412.stdct-mgmt-02'
REQUEST='9895f812-13f5-4fb0-9dcb-0c27ebb97d41'
REQUEST_SHA='4455e91b4f49c7f627d82dd9dacab0ec939a23e11cbf1034468d73d43f4523f9'
SOURCE='07937041d7b330354dfe004ed6bf5a179588147e'

RELEASE='e48db225033ba0510230d96b43f58c0743b05912'
GRANT_SHA='98c081a33e60a7a0d98a4e7a392530fca2de77c395f1262dd0ffd8b5e5b421e0'
APPROVED=1792746048.7650247
PLAN='plans/B2_SYSTEM_STARTUP_RETRY_PBS_AUTHORIZATION.json'
FIELD='B_system_startup_retry_PBS_extension'
SCOPE_KEYS=('B_system_startup_retry_release','B_system_startup_retry_grant_sha256')
GPU_TERMINAL='6c471e0c789ddc2eafb0b03f579027387a40fbb23b70d03b5d026c69e14a83c9'
GPU_LEASE='b8f19d13cc7ae0008755173f6ae1fea24d9b0b242d9139f6a8e0fa9c3b72dd59'
CPU_TERMINAL='e1b4074b0908f256152c165dd0a747eff4c1c5e5f87a239cff817a011078568f'
CPU_LEASE='5093434a037fb45b86cab05a18239c9c44f3d57c9d7baedaf1ba579ffd6c1d7a'

def fail(why):raise PermissionError('B_SYSTEM_STARTUP_RETRY_'+why)
def scope():return dict(zip(SCOPE_KEYS,(RELEASE,GRANT_SHA)))

def grant(raw):
 v,digest=_read(raw)
 if (digest!=GRANT_SHA or v.get('schema')!='B2_SYSTEM_STARTUP_RETRY_PBS_AUTHORIZATION_V1'
     or v.get('approval_recorded_epoch')!=APPROVED or v.get('human_answer')!='批准仅这两项 PBS 上限调整'
     or v.get('scope')!=dict(DEC1_PBS_limit=dict(prior=14,approved=15),global_PBS_limit=dict(prior=25,approved=26),additional_unique_retry_submissions=1)
     or v.get('old_failed_job')!=JOB or v.get('old_failed_request_sha256')!=REQUEST_SHA
     or v.get('new_GPU_CPU_query_update_storage_transfer_physics_or_clock_allowance') is not False
     or v.get('source_chain_count_must_truthfully_include_the_one_extra_PBS') is not True):fail('EXACT_TWO_PBS_ONLY_HUMAN_GRANT')
 return v

def enabled(row):
 if not any(k in row for k in SCOPE_KEYS):return False
 if tuple(row.get(k) for k in SCOPE_KEYS)!=(RELEASE,GRANT_SHA):fail('NO_PARTIAL_OR_UNBOUND_RETRY_SCOPE')
 return True

def request(row):
 if not enabled(row):return False
 from . import b_vanda_recovery2 as vanda2
 if vanda2.enabled(row):return vanda2.request(row)
 if (not saved.request(row) or b.lineage(row)!=b.LINEAGE
     or row.get('expected_native_start_step',row.get('start_native_step'))!=2048
     or row.get('segment_stop_step',row.get('training_block',{}).get('stop_after_step'))!=4096
     or row.get('request_id')==REQUEST):fail('ONE_NEW_REQUEST_ORIGINAL_B2048_TO4096')
 return True

def marker(now):
 return dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,failed_job_id=JOB,failed_request_sha256=REQUEST_SHA,
     GPU_closed_sha256=GPU_TERMINAL,CPU_closed_sha256=CPU_TERMINAL,DEC1_PBS_limit=15,global_PBS_limit=26,
     old_costs_and_failures_preserved=True,other_caps_or_clock_changed=False,adopted_epoch=now)

def validate_global(h):
 b.validate_global(h);m=h.get(FIELD);now=m.get('adopted_epoch') if isinstance(m,dict) else None
 if type(now) not in (int,float) or not math.isfinite(now) or not APPROVED<=now<b.CUTOFF or m!=marker(now):fail('EXACT_ADOPTED_PBS_ONLY_MARKER')
 rows=[v for v in h['PBS_submissions'] if v.get('request_id')==REQUEST]
 want=dict(job_id=JOB,request_sha256=REQUEST_SHA,source_sha=SOURCE,status='NATIVE_TERMINAL',native_exit_status=-3,
     start_native_step=2048,committed_optimizer_updates=0,query_attempts=0,conservative_H200_s=484,conservative_CPU_s=15384,reserved_optimizer_updates=0,reserved_neural_queries=0)
 if (len(rows)!=1 or any(rows[0].get(k)!=z for k,z in want.items()) or h['used']['H200_s']<82257 or h['optimizer_updates']<8744):fail('ALL_FAILED_STARTUP_COSTS_PRESERVED')
 return m

def validate_adopted(h,d):
 saved.validate_adopted(h,d);m=validate_global(h)
 rows=[r for r in d['new_submissions'] if r.get('request_id')==REQUEST]
 if m!=d.get(FIELD) or len(rows)!=1 or rows[0].get('job_id')!=JOB or rows[0].get('status')!='NATIVE_TERMINAL':fail('DUAL_EXACT_MARKERS_AND_FAILED_ROW')
 return m

def adopt(h,d,*,grant_bytes,GPU_terminal_bytes,GPU_lease_bytes,CPU_terminal_bytes,CPU_lease_bytes,now_epoch):
 grant(grant_bytes);saved.validate_adopted(h,d)
 if (FIELD in h or FIELD in d or type(now_epoch) not in (int,float) or not math.isfinite(now_epoch)
     or not APPROVED<=now_epoch<b.CUTOFF or h['optimizer_updates']!=8744 or d['optimizer_updates']!=8744
     or h['training_runs'][b.LINEAGE]['last_native_step']!=2048 or h['unknown_effects'] or h['pending_updates'] or d['unknown_effects']
     or any(h['reserved'].values()) or h['CPU_reserved_s'] or h['storage_reserved_bytes']
     or any(d.get(k,0) for k in ('GPU_reserved_s','CPU_reserved_s','neural_queries_reserved','optimizer_updates_reserved','GPU_site_output_reserved_bytes','network_transfer_reserved_bytes'))
     or any(r['status']!='NATIVE_TERMINAL' for r in h['PBS_submissions'])):fail('ONCE_INACTIVE_FULL_CLOSED_FAILURE_BOUNDARY')
 for raw,lease_raw,digest,lease_digest,status in (
     (GPU_terminal_bytes,GPU_lease_bytes,GPU_TERMINAL,GPU_LEASE,'CLOSED_PREWRAPPER_STARTUP_FAILURE_NO_NEW_SCIENTIFIC_STATE'),
     (CPU_terminal_bytes,CPU_lease_bytes,CPU_TERMINAL,CPU_LEASE,'CLOSED_CPU_PREWRAPPER_STARTUP_FAILURE_SHARED_PREPAID_CONTROL')):
  c,cs=_read(raw);l,ls=_read(lease_raw)
  if (cs!=digest or ls!=lease_digest or c.get('status')!=status or c.get('job_id')!=JOB
      or c.get('new_scientific_state_created') is not False or c.get('retired_or_deleted_files')!=0
      or l.get('status')!='CLOSED' or l.get('terminal_receipt_sha256')!=cs):fail('EXACT_ACTUAL_GPU_AND_CPU_CLOSED_BYTES')
 hh,dd=deepcopy(h),deepcopy(d);hh[FIELD]=deepcopy(marker(now_epoch));dd[FIELD]=deepcopy(marker(now_epoch));validate_adopted(hh,dd)
 return hh,dd

def admitted(root,manifest,row):
 if not request(row):return False
 p=b.zero._readonly(Path(root)/PLAN);raw=p.read_bytes();grant(raw)
 if (any(not enabled(v) for v in (manifest.get('authority',{}),row))
     or manifest.get('execution',{}).get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA
     or row.get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA
     or b.zero._readonly(Path(root).parent.parent/'authority/DEC1'/Path(PLAN).name).read_bytes()!=raw):fail('EXACT_READONLY_SEALED_PAIRED_NATIVE_RETRY_GRANT')
 return True
