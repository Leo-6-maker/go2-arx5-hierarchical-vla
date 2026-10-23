"""Exact five-cap authorization; preserve the failed2970 lineage and costs.

Pure cold adoption only. No transport, registration or native dispatch occurs.
The unavailable old physical CPU lease remains explicitly fenced, never closed.
"""
from copy import deepcopy
import math
from pathlib import Path
from scripts.pi0294.pv1.future_budget import _read
from . import b_heartbeat_recovery as b

RELEASE='1f700ee346a91006da6ed7e7856c9b42ec496335'
GRANT_SHA='86eb04c444d027f081fa921860a9dfdfd31d726b6abe208c1b78c7b9e43047f7'
PLAN='plans/B2970_RECOVERY2_FIVE_CAP_AUTHORIZATION.json'
FIELD='B_Vanda_recovery2_five_cap_extension'
SCOPE_KEYS=('B_Vanda_recovery2_release','B_Vanda_recovery2_grant_sha256')
LINEAGE='B_COVERAGE_HEARTBEAT_RECOVERY2'
JOB='1480817.stdct-mgmt-02'
REQUEST='28efd26a-f579-454b-b14a-a4c9847cbfa2'
REQUEST_SHA='8e437f152edf4c2f3f28c11795d31594b65fca9f6fb91666e8017edb13e3be24'
SOURCE='36b5ed27d41f71a74767c17f0e028149c34ac8a1'
GPU_TERMINAL='548b23720524f46e8008fd256be480b842d9be19536b1f628e9ba0dc507bfddc'
SEED='db2b2be84e1d2e6eeca1715107f6eb42a8b151e38fe513ddac6029e8db97bbcf'
APPROVED=1792767597.5203986
CHANGES=dict(DEC1_optimizer_updates=[10792,11714],DEC1_PBS=[15,16],global_PBS=[26,27],
    DEC1_GPU_site_storage_GiB=[549,614],global_GPU_site_storage_GiB=[615,680])

def fail(why):raise PermissionError('B_VANDA_RECOVERY2_'+why)
def scope():return dict(zip(SCOPE_KEYS,(RELEASE,GRANT_SHA)))

def grant(raw):
    v,digest=_read(raw)
    if (digest!=GRANT_SHA or v.get('schema')!='B2970_RECOVERY2_FIVE_CAP_AUTHORIZATION_V1'
            or v.get('status')!='HUMAN_AUTHORIZED_EXACT_PRESERVED_COST_RECOVERY2'
            or v.get('approval_recorded_epoch')!=APPROVED or v.get('narrow_changes')!=CHANGES
            or v.get('human_answer')!='批准仅这五项调整'):
        fail('EXACT_PUBLISHED_HUMAN_FIVE_CAP_GRANT')
    return v

def enabled(row):
    if not any(k in row for k in SCOPE_KEYS):return False
    if tuple(row.get(k) for k in SCOPE_KEYS)!=(RELEASE,GRANT_SHA):fail('NO_PARTIAL_OR_OTHER_SCOPE')
    return True

def lineage(row):
    block=row.get('training_block',{})
    if (not enabled(row) or not b.enabled(row) or row.get('arm',block.get('arm'))!='B_COVERAGE'
            or row.get('training_lineage_key')!=LINEAGE):fail('EXPLICIT_NEW_SAME_B_LINEAGE_ONLY')
    prior={k:v for k,v in row.items() if k not in SCOPE_KEYS}
    prior['training_lineage_key']=b.LINEAGE
    b.lineage(prior)
    return LINEAGE

def request(row):
    if not enabled(row):return False
    lineage(row)
    from . import b_saved_state_reuse as saved,b_system_startup_retry as retry
    if (not saved.enabled(row) or not retry.enabled(row)
            or row.get('expected_native_start_step',row.get('start_native_step'))!=2048
            or row.get('segment_stop_step',row.get('training_block',{}).get('stop_after_step'))!=4096
            or row.get('request_id') in (REQUEST,saved.REQUEST,retry.REQUEST)):
        fail('NEW_REQUEST_IMPORT2048_TO4096_NO_FAILED_REQUEST_REPLAY')
    return True

def marker(now):
    return dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,failed_job_id=JOB,
        failed_request_sha256=REQUEST_SHA,failed_lineage_key=b.LINEAGE,failed_lineage_last_native_step=2970,
        paid_lost_updates=922,replacement_lineage_key=LINEAGE,seed_native_step=2048,
        seed_checkpoint_sha256=SEED,probe_sha256=b.PROBE_SHA,GPU_closed_terminal_sha256=GPU_TERMINAL,
        old_Hopper_CPU_physical_lease='UNAVAILABLE_FENCED_NOT_CLOSED',
        original_T0_epoch=b.T0,DEC1_updates_limit=11714,DEC1_PBS_limit=16,global_PBS_limit=27,
        DEC1_storage_GiB=614,global_storage_GiB=680,adopted_epoch=now)

def _history(h):
    rows=[r for r in h['PBS_submissions'] if r.get('request_id')==REQUEST]
    want=dict(job_id=JOB,source_sha=SOURCE,request_sha256=REQUEST_SHA,status='NATIVE_TERMINAL',
        native_exit_status=1,start_native_step=2048,committed_optimizer_updates=922,query_attempts=17,
        conservative_H200_s=8481,conservative_CPU_s=11023,reserved_optimizer_updates=0,reserved_neural_queries=0)
    if (len(rows)!=1 or any(rows[0].get(k)!=v for k,v in want.items())
            or h['training_runs'].get(b.LINEAGE)!=dict(last_native_step=2970,recipe_sha256=b.RECIPE)
            or h['optimizer_updates']<9666 or h['used']['optimizer_updates']!=h['optimizer_updates']
            or h['used']['H200_s']<90738 or h['generation_queries']<1214
            or h.get('Vanda_only_controller_handoff',{}).get('old_parent_mutations_permitted') is not False):
        fail('PRESERVE_FULL_PAID_FAILED2970_HISTORY_AND_FENCED_ORIGIN')
    return rows[0]

def validate_global(h):
    from . import b_system_startup_retry as retry
    retry.validate_global(h);_history(h)
    m=h.get(FIELD);now=m.get('adopted_epoch') if isinstance(m,dict) else None
    run=h['training_runs'].get(LINEAGE,{})
    if (type(now) not in (int,float) or not math.isfinite(now) or not APPROVED<=now<b.CUTOFF
            or m!=marker(now) or set(run)!={'last_native_step','recipe_sha256'}
            or run.get('recipe_sha256')!=b.RECIPE or type(run.get('last_native_step')) is not int
            or not 2048<=run['last_native_step']<=4096):fail('EXACT_ADOPTION_NEW_PROGRESS_NO_ROLLBACK')
    return m

def validate_adopted(h,d):
    from . import b_system_startup_retry as retry
    retry.validate_adopted(h,d);m=validate_global(h)
    rows=[r for r in d['new_submissions'] if r.get('request_id')==REQUEST]
    if (m!=d.get(FIELD) or d['optimizer_updates']!=h['optimizer_updates'] or len(rows)!=1
            or rows[0].get('job_id')!=JOB or rows[0].get('status')!='NATIVE_TERMINAL'):
        fail('PAIRED_ADOPTION_AND_TERMINAL_HISTORY')
    return m

def adopt(h,d,*,grant_bytes,GPU_terminal_bytes,now_epoch):
    from . import b_system_startup_retry as retry
    grant(grant_bytes);retry.validate_adopted(h,d);_history(h)
    c,cs=_read(GPU_terminal_bytes)
    if (FIELD in h or FIELD in d or LINEAGE in h['training_runs'] or cs!=GPU_TERMINAL
            or c.get('status')!='CLOSED_B_RETRY1_HEARTBEAT_FAILURE_2970_SAVED2048_PRESERVED'
            or h['optimizer_updates']!=9666 or d['optimizer_updates']!=9666
            or h['unknown_effects'] or h['pending_updates'] or d['unknown_effects']
            or any(h['reserved'].values()) or h['CPU_reserved_s'] or h['storage_reserved_bytes']
            or any(d.get(k,0) for k in ('GPU_reserved_s','CPU_reserved_s','neural_queries_reserved',
                'optimizer_updates_reserved','GPU_site_output_reserved_bytes','network_transfer_reserved_bytes'))
            or any(r['status']!='NATIVE_TERMINAL' for r in h['PBS_submissions'])
            or type(now_epoch) not in (int,float) or not math.isfinite(now_epoch)
            or not APPROVED<=now_epoch<b.CUTOFF):fail('ONCE_INACTIVE_REAL_CLOSED_FAILED_BOUNDARY')
    hh,dd=deepcopy(h),deepcopy(d);hh[FIELD]=deepcopy(marker(now_epoch));dd[FIELD]=deepcopy(marker(now_epoch))
    hh['training_runs'][LINEAGE]=dict(last_native_step=2048,recipe_sha256=b.RECIPE)
    validate_adopted(hh,dd)
    return hh,dd

def admitted(root,manifest,row):
    if not request(row):return False
    p=b.zero._readonly(Path(root)/PLAN);raw=p.read_bytes();grant(raw)
    if (any(not enabled(v) for v in (manifest.get('authority',{}),row))
            or manifest.get('execution',{}).get('control_execution_platform')!='VANDA_A40'
            or manifest.get('execution',{}).get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA
            or row.get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA
            or b.zero._readonly(Path(root).parent.parent/'authority/DEC1'/Path(PLAN).name).read_bytes()!=raw):
        fail('NEW_SEALED_VANDA_ONLY_COLD_SCOPE')
    return True
