"""Exact PBS-only extension after the closed 1481489 pre-CUDA failure.

Pure proposal and validation. No transport or native mutation. Every old
marker, paid cost, lineage and scientific condition remains unchanged.
"""
from copy import deepcopy
import math
from pathlib import Path
from scripts.pi0294.pv1.future_budget import _read
from . import b_vanda_recovery2 as prior

RELEASE='0a7444b0a47712a0be0b0534f20a1d3753273a47'
GRANT_SHA='5b1d2c72b1e4aa2cebb9376faaaaebdd3f343756a2475c68b3dc098fe435a9de'
APPROVED=1792776527.4877334
PLAN='plans/B_PRECUDA1481489_RETRY_PBS_AUTHORIZATION.json'
FIELD='B_precuda1481489_retry_PBS_extension'
SCOPE_KEYS=('B_precuda_retry_release','B_precuda_retry_grant_sha256')
JOB='1481489.stdct-mgmt-02'
REQUEST='599a7171-c17f-42a5-8c9e-33e00dcad52c'
REQUEST_SHA='a9ba3e165352daec9348c9678c72246fba8cd8a3d7c6468adf570f9f1320af59'
SOURCE='90d357a74d3d8e1f033ca978f16c5f3191a087f9'
GPU_TERMINAL='4c07d6151d2563f9c3c83fe149d45dadaae2034c5e98604f0df2453b3c4e169a'
GPU_LEASE='383f094616e2614ff5409270ac9d149b01ae54916bb0db84985f443b8118e43b'
CPU_TERMINAL='fbafed614c8fc9f0d0d28799c900c2ca71367353a9c550b035a78b47d5ed5d46'
CPU_LEASE='fbdba46a3d3bc9dfef6204205d8bfcc3fd832f5f00eccd8cd878aa0a7d265349'
CHANGES=dict(DEC1_PBS=[16,17],global_PBS=[27,28])

def fail(why):raise PermissionError('B_PRECUDA_RETRY_'+why)
def scope():return dict(zip(SCOPE_KEYS,(RELEASE,GRANT_SHA)))

def grant(raw):
    v,digest=_read(raw)
    if (digest!=GRANT_SHA or v.get('schema')!='B_PRECUDA1481489_RETRY_PBS_AUTHORIZATION_V1'
            or v.get('status')!='HUMAN_AUTHORIZED_EXACT_TWO_PBS_COUNTERS_ONLY'
            or v.get('approval_recorded_epoch')!=APPROVED or v.get('narrow_changes')!=CHANGES
            or v.get('human_answer')!='批准仅这两项 PBS 上限调整'):
        fail('EXACT_PUBLISHED_TWO_COUNTER_GRANT')
    return v

def enabled(row):
    if not any(k in row for k in SCOPE_KEYS):return False
    if tuple(row.get(k) for k in SCOPE_KEYS)!=(RELEASE,GRANT_SHA):fail('NO_PARTIAL_OR_OTHER_SCOPE')
    return True

def request(row):
    if not enabled(row):return False
    if not prior.request(row) or row.get('request_id')==REQUEST:fail('NEW_REQUEST_SAME_B2048_TO4096_ONLY')
    return True

def marker(now):
    return dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,failed_job_id=JOB,
        failed_request_sha256=REQUEST_SHA,GPU_closed_sha256=GPU_TERMINAL,CPU_closed_sha256=CPU_TERMINAL,
        DEC1_PBS_limit=17,global_PBS_limit=28,other_caps_or_clock_changed=False,
        old_costs_failures_and_seed_preserved=True,adopted_epoch=now)

def _history(h):
    rows=[v for v in h['PBS_submissions'] if v.get('request_id')==REQUEST]
    want=dict(job_id=JOB,source_sha=SOURCE,request_sha256=REQUEST_SHA,status='NATIVE_TERMINAL',
        native_exit_status=1,start_native_step=2048,committed_optimizer_updates=0,query_attempts=0,
        conservative_H200_s=265,conservative_CPU_s=1180,reserved_optimizer_updates=0,reserved_neural_queries=0)
    if (len(rows)!=1 or any(rows[0].get(k)!=z for k,z in want.items())
            or h['used']['H200_s']<91003 or h['optimizer_updates']<9666 or h['generation_queries']<1214):
        fail('FULL_PAID_PRECUDA_FAILURE_HISTORY')

def validate_global(h):
    prior.validate_global(h);_history(h)
    m=h.get(FIELD);now=m.get('adopted_epoch') if isinstance(m,dict) else None
    if (type(now) not in (int,float) or not math.isfinite(now) or not APPROVED<=now<prior.b.CUTOFF
            or m!=marker(now)):fail('EXACT_ADOPTED_MARKER')
    return m

def validate_adopted(h,d):
    prior.validate_adopted(h,d);m=validate_global(h)
    rows=[v for v in d['new_submissions'] if v.get('request_id')==REQUEST]
    if (m!=d.get(FIELD) or len(rows)!=1 or rows[0].get('job_id')!=JOB
            or rows[0].get('status')!='NATIVE_TERMINAL'):fail('DUAL_MARKERS_AND_TERMINAL_ROW')
    return m

def adopt(h,d,*,grant_bytes,GPU_terminal_bytes,GPU_lease_bytes,CPU_terminal_bytes,CPU_lease_bytes,now_epoch):
    grant(grant_bytes);prior.validate_adopted(h,d);_history(h)
    if (FIELD in h or FIELD in d or type(now_epoch) not in (int,float) or not math.isfinite(now_epoch)
            or not APPROVED<=now_epoch<prior.b.CUTOFF or h['optimizer_updates']!=9666 or d['optimizer_updates']!=9666
            or h['training_runs'][prior.LINEAGE]['last_native_step']!=2048
            or h['unknown_effects'] or h['pending_updates'] or d['unknown_effects']
            or any(h['reserved'].values()) or h['CPU_reserved_s'] or h['storage_reserved_bytes']
            or any(d.get(k,0) for k in ('GPU_reserved_s','CPU_reserved_s','neural_queries_reserved',
                'optimizer_updates_reserved','GPU_site_output_reserved_bytes','network_transfer_reserved_bytes'))
            or any(v['status']!='NATIVE_TERMINAL' for v in h['PBS_submissions'])):
        fail('ONCE_INACTIVE_FULL_CLOSED_BOUNDARY')
    for terminal,lease,ts,ls in ((GPU_terminal_bytes,GPU_lease_bytes,GPU_TERMINAL,GPU_LEASE),
            (CPU_terminal_bytes,CPU_lease_bytes,CPU_TERMINAL,CPU_LEASE)):
        t,td=_read(terminal);l,ld=_read(lease)
        if td!=ts or ld!=ls or t.get('job_id')!=JOB or l.get('status')!='CLOSED' or l.get('terminal_receipt_sha256')!=td:
            fail('EXACT_ACTUAL_DUAL_CLOSED_BYTES')
    hh,dd=deepcopy(h),deepcopy(d)
    hh[FIELD]=deepcopy(marker(now_epoch));dd[FIELD]=deepcopy(marker(now_epoch));validate_adopted(hh,dd)
    return hh,dd

def admitted(root,manifest,row):
    if not request(row):return False
    raw=prior.b.zero._readonly(Path(root)/PLAN).read_bytes();grant(raw)
    if (not enabled(manifest.get('authority',{}))
            or manifest.get('execution',{}).get('control_execution_platform')!='VANDA_A40'
            or manifest.get('execution',{}).get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA
            or row.get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA
            or prior.b.zero._readonly(Path(root).parent.parent/'authority/DEC1'/Path(PLAN).name).read_bytes()!=raw):
        fail('EXACT_PAIRED_SEALED_VANDA_ONLY_SCOPE')
    return True
