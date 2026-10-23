"""Exact human-approved reuse of one closed failed B2048 state.

Pure validation and marker adoption only. Failure, costs, limits, original
probes and checkpoint origins remain intact; this grants no native dispatch.
"""
from copy import deepcopy
import hashlib
import json
import math
from . import b_heartbeat_recovery as b

RELEASE='918c675a7b7f778b947005fc69777f4b6364f792'
GRANT_SHA='c8f36abdbadd362156a6cd28d5599da6e0bc72d708de47a218fa536271c2b253'
PLAN='plans/B_FAILED_SAVED2048_REUSE_AUTHORIZATION.json'
FIELD='B_failed_saved2048_reuse_adoption'
KEY='B_failed_saved2048_reuse'
SCOPE_KEYS=('B_failed_saved2048_reuse_release','B_failed_saved2048_reuse_grant_sha256')
APPROVED=1792740189.1965473
JOB='1479559.stdct-mgmt-02'
REQUEST='185541e4-2348-4a3c-bee7-1a5703369780'
REQUEST_SHA='6c6d12d14895e4e22fa7a835c7edac8579ff341825911fd0d3b5fbbec76809ca'
SOURCE='2ed8aea9f782741dfc0298438f5762c592e87149'
CHECKPOINT='db2b2be84e1d2e6eeca1715107f6eb42a8b151e38fe513ddac6029e8db97bbcf'
GPU_TERMINAL='4801d547e8d099313818422857fd29768ca2b84472d5cd5a81fc51dc3811fcf2'
GPU_LEASE='eb358f7df8f6b25d4c74a0b61374895e69b4c4ceee3e94577824f351c86d1191'
CPU_TERMINALS=('2ef7dc6f6aa951f856e38cb1290924a4d1bcaea2f3e1ef23120966237bd7ea79','c1e465807b9eb466b16c682d35db429778eebce19cdf4f50df2304a9b46626f2')
CPU_LEASES=('3bd638f24591bbb98030a472f53c9f1264a809d60f6e864fb34608e4c1b9033a','26ba734f8723a3ead6ddfc9cd4cedd9a05889d35fb6c79bb3fb9235e9f7bb03d')
LOCAL_PHASE_CPU_LIMITS=dict(prepay=5,control_register=10,control_check=10,adopt=10,
    adopt_readback=5,prepare=5,register=10,cold_check=10,preflight=5,launch=5,serve=100,terminal=5)


def control_plan(*,local_charged_s,future_local_s,original_control_upper):
    """Keep all old future ceilings; fund a finite, separately enforced owner.

    This neither refunds the earlier shared1200prepayment nor subtracts any
    of the original3600future controller ceiling. Native admission remains.
    """
    if (original_control_upper!=dict(Hopper_CPU_s=5520,local_CPU_s=3600,transfer_bytes=805306368)
            or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in (local_charged_s,future_local_s))
            or local_charged_s+future_local_s+sum(LOCAL_PHASE_CPU_LIMITS.values())>57600):
        fail('FULL_OLD_LOCAL_FUTURE_CEILING_PLUS_NEW_ENFORCED180_INSIDE16H')
    return dict(Hopper_CPU_upper_s=1800,LOCAL_CPU_upper_s=180,transfer_upper_bytes=256*1024**2,
        output_upper_bytes=64*1024**2,LOCAL_phase_CPU_limits=dict(LOCAL_PHASE_CPU_LIMITS),
        old_shared1200_charge_preserved=True,original_future_controller_upper_preserved=True,
        local_headroom_s=57600-local_charged_s-future_local_s-180,native_admission=False)


def fail(why):raise PermissionError('B_SAVED2048_REUSE_'+why)


def read(raw):
    if not isinstance(raw,bytes):fail('ACTUAL_BYTES_REQUIRED')
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def grant(raw):
    value,digest=read(raw)
    if (digest!=GRANT_SHA or value.get('schema')!='B_FAILED_SAVED2048_REUSE_AUTHORIZATION_V1'
            or value.get('human_answer')!='批准复用已审计的 B2048'
            or value.get('approval_recorded_epoch')!=APPROVED
            or value.get('native_failure_not_relabelled_success') is not True
            or value.get('new_caps_or_budget_or_clock') is not False):fail('EXACT_HUMAN_EXCEPTION_NO_NEW_BUDGET')
    return value


def scope():return dict(zip(SCOPE_KEYS,(RELEASE,GRANT_SHA)))


def enabled(row):
    if not any(k in row for k in SCOPE_KEYS):return False
    if tuple(row.get(k) for k in SCOPE_KEYS)!=(RELEASE,GRANT_SHA):fail('NO_PARTIAL_OR_OTHER_EXCEPTION')
    return True


def request(row):
    if not enabled(row):return False
    b.lineage(row);block=row.get('training_block',{})
    start=row.get('expected_native_start_step',row.get('start_native_step'))
    stop=block.get('stop_after_step',row.get('segment_stop_step'))
    if (start!=2048 or stop!=4096 or row.get('arm',block.get('arm'))!='B_COVERAGE'
            or block and (block.get('arm')!='B_COVERAGE' or block.get('stop_after_step')!=4096)
            or 'start_native_step' in row and row['start_native_step']!=2048
            or 'segment_stop_step' in row and row['segment_stop_step']!=4096
            or row.get('training_lineage_key')!=b.LINEAGE):fail('ONLY_ORIGINAL_REMAINING_B2048_TO4096')
    return True


def identity():
    return dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,native_exit_status=1,
        owner_exit_status=1,job_id=JOB,request_id=REQUEST,checkpoint2048_sha256=CHECKPOINT)


def binding(value):
    if KEY not in value:return False
    expected=dict(job_id=JOB,request_id=REQUEST,request_sha256=REQUEST_SHA,source_sha=SOURCE,
        terminal_sha256=GPU_TERMINAL,lease_sha256=GPU_LEASE,checkpoint1024_sha256=b.SEED_SHA,
        checkpoint2048_sha256=CHECKPOINT,probe_result_sha256=b.PROBE_SHA,arm='B_COVERAGE')
    if value[KEY]!=identity() or any(value.get(k)!=v for k,v in expected.items()):
        fail('ONLY_EXACT_APPROVED_FAILED_SAVED_STATE')
    return True


def terminal(value):
    e=value.get('effects',{});n=value.get('native_terminal',{})
    want=dict(job_id=JOB,request_id=REQUEST,request_sha256=REQUEST_SHA,source_sha=SOURCE,
        arm='B_COVERAGE',start_native_step=1024,last_native_step=2048,optimizer_updates_committed=1024,
        query_attempts=17,model_load_attempts=2,recipe_sha256=b.RECIPE,pending_updates=[],
        unknown_query_intents=[],effect_journal_complete=True,new_physics=0,new_episodes=0)
    if (value.get('status')!='CLOSED_B_RECOVERY1_HEARTBEAT_FAILURE_SAVED2048_PRESERVED'
            or value.get('job_id')!=JOB or value.get('recoverable_full_state') is not True
            or value.get('latest_complete_step')!=2048 or value.get('committed_updates_without_new_complete_state')!=0
            or value.get('retired_or_deleted_files')!=0 or n.get('job_state')!='F' or n.get('Exit_status')!=1
            or any(e.get(k)!=v for k,v in want.items())
            or value.get('settlement',{}).get('updates')!=8744
            or value.get('settlement',{}).get('unknown_effects')!=[]):fail('PRESERVED_ACTUAL_FAILED_TERMINAL_NOT_SUCCESS')
    cp=value.get('checkpoints',{})
    for step,digest in ((1024,b.SEED_SHA),(2048,CHECKPOINT)):
        if cp.get(str(step))!=dict(complete_sha256=digest,members=15,actual_consumed_cursor=step*32):
            fail('EXACT_AUDITED_FULL_STATES_AND_CURSORS')
    return value


def marker(now):
    return dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,job_id=JOB,request_id=REQUEST,
        checkpoint2048_sha256=CHECKPOINT,probe_result_sha256=b.PROBE_SHA,source_native_exit_status=1,
        source_owner_exit_status=1,cumulative_updates_at_adoption=8744,new_paid_lost_updates=0,
        prior_B1829_lost805_preserved=True,new_caps_or_clock=False,adopted_epoch=now)


def validate_adopted(h,d):
    b.validate_adopted(h,d);m=h.get(FIELD);now=m.get('adopted_epoch') if isinstance(m,dict) else None
    if (type(now) not in (int,float) or not math.isfinite(now) or not APPROVED<=now<1792866596
            or m!=marker(now) or d.get(FIELD)!=m):fail('DUAL_EXACT_ADOPTED_EXCEPTION')
    rows=[x for x in h['PBS_submissions'] if x.get('request_id')==REQUEST]
    if (len(rows)!=1 or any(rows[0].get(k)!=v for k,v in dict(job_id=JOB,status='NATIVE_TERMINAL',
            request_sha256=REQUEST_SHA,source_sha=SOURCE,native_exit_status=1,
            start_native_step=1024,committed_optimizer_updates=1024,query_attempts=17,
            reserved_optimizer_updates=0,reserved_neural_queries=0).items())
            or h['training_runs'][b.LINEAGE]['last_native_step']<2048
            or h['optimizer_updates']<8744):fail('FAILED_ROW_AND_ALL_PAID_PROGRESS_PRESERVED')
    return m


def adopt(h,d,*,grant_bytes,GPU_terminal_bytes,GPU_lease_bytes,CPU_terminal_bytes,CPU_lease_bytes,now_epoch):
    grant(grant_bytes);b.validate_adopted(h,d)
    if (FIELD in h or FIELD in d or type(now_epoch) not in (int,float) or not math.isfinite(now_epoch)
            or not APPROVED<=now_epoch<1792866596):
        fail('ONE_COLD_ADOPTION_ONLY')
    # Both exact equality checks are intentional: no unknown/reserved work is
    # erased to manufacture the inactive boundary.
    if (h['optimizer_updates']!=8744 or d['optimizer_updates']!=8744
            or h['training_runs'][b.LINEAGE]['last_native_step']!=2048
            or h['unknown_effects'] or h['pending_updates'] or d['unknown_effects']
            or any(h['reserved'].values()) or h['CPU_reserved_s'] or d['CPU_reserved_s']
            or h['storage_reserved_bytes'] or any(d.get(k,0) for k in
                ('GPU_reserved_s','neural_queries_reserved','optimizer_updates_reserved','GPU_site_output_reserved_bytes','network_transfer_reserved_bytes'))
            or any(x['status']!='NATIVE_TERMINAL' for x in h['PBS_submissions'])):fail('REAL_INACTIVE_NO_PENDING_OR_REPEATED_COSTS')
    g,gs=read(GPU_terminal_bytes);terminal(g);lease,ls=read(GPU_lease_bytes)
    if gs!=GPU_TERMINAL or ls!=GPU_LEASE or lease.get('status')!='CLOSED' or lease.get('terminal_receipt_sha256')!=gs:
        fail('ACTUAL_CLOSED_GPU_BYTES')
    if len(CPU_terminal_bytes)!=2 or len(CPU_lease_bytes)!=2:fail('BOTH_SHARED_PREPAID_CPU_CONTROLS')
    for raw,lease_raw,expected,expected_lease in zip(CPU_terminal_bytes,CPU_lease_bytes,CPU_TERMINALS,CPU_LEASES):
        c,cs=read(raw);l,ls=read(lease_raw)
        if (cs!=expected or ls!=expected_lease or l.get('status')!='CLOSED'
                or l.get('terminal_receipt_sha256')!=cs or c.get('job_id')!=JOB
                or c.get('new_B_step')!=2048 or c.get('cumulative_updates')!=8744
                or c.get('new_CPU_charge_s')!=0 or c.get('unknown_effects')!=[]
                or c.get('GPU_close',{}).get('terminal_receipt')!=g):fail('EXACT_BOTH_CPU_CLOSED_WITHOUT_SECOND_CHARGE')
    hh,dd=deepcopy(h),deepcopy(d);hh[FIELD]=dd[FIELD]=marker(now_epoch);validate_adopted(hh,dd)
    return hh,dd
