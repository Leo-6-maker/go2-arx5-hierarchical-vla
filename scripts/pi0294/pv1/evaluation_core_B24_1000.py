"""Exact human-authorized B24 start exception; original global clocks remain.

Only the previously unregistered B4096/N294101 request may start by 10:00 SGT.
This pure module never writes, registers, starts, grants model/physics GO or
retries an UNKNOWN. Current source/manifest/native identity and full resources
are still independently required by the existing evaluator.
"""
from copy import deepcopy
import hashlib,json,re,uuid

FIELD='Core_B24_only_1000_start_exception'
GRANT_SHA='7ac451eee07a67804039fd0df497c5d5e5b7d509d0635ade5625536286ca4251'
PLAN='plans/B24_ONLY_1000_EXACT_HUMAN_GRANT.json'
RID='4b0f26e7-dfa9-4e5e-ac05-8de1187f6653'
RUN='pv1-v4a-b4096-n294101-read-recovery-02'
LATEST=1792893600
ORIGINAL_STOP_NEW=1792866596
T0=1792348196
STOP_PHYSICS=1792924196
HARD_CLOSE=1792952996
PREP='pv1-v4a-b24-closed-preparation-before-1000'
ROW_FIELD='Core_native_GPU_allocations'


def need(ok,why):
    if not ok:raise PermissionError('PV1_B24_1000_'+why)


def grant(raw):
    need(type(raw) is bytes and hashlib.sha256(raw).hexdigest()==GRANT_SHA,'EXACT_DIRECT_HUMAN_GRANT_BYTES')
    g=json.loads(raw)
    need(g['status']=='DIRECT_HUMAN_APPROVED_EXACT_SCOPE' and g['direct_human_answer']=='仅这份 B24 最晚 10:00 启动'
         and g['request_id']==RID and g['latest_native_registration_and_start_epoch']==LATEST
         and g['automatic_retry'] is False and g['maximum_native_scientific_requests']==1
         and g['original_T0']==T0 and g['original_stop_new']==ORIGINAL_STOP_NEW
         and g['original_stop_physics']==STOP_PHYSICS and g['original_hard_close']==HARD_CLOSE,
         'ONLY_THE_APPROVED_BOUNDARY_NO_T0_OR_OTHER_DEADLINE_RESET')
    return g


def start_limit(r):
    if 'late_start_grant_sha256' not in r:return ORIGINAL_STOP_NEW
    need(r.get('late_start_grant_sha256')==GRANT_SHA and r.get('request_id')==RID
         and r.get('run_uuid',r.get('child_run_uuid'))==RUN and r.get('evaluation_candidate')=='B_COVERAGE'
         and r.get('stage')=='P4_DEV' and r.get('noise_bank')==294101
         and not any(k.startswith('queue_') for k in r),'ONE_EXACT_UNQUEUED_B24_ONLY')
    return LATEST


def validate(g):
    rows=g.get(ROW_FIELD,{})
    if FIELD not in g:
        need(not any('late_start_grant_sha256' in r for r in rows.values()),'NO_ORPHAN_LATE_ROW_WITHOUT_ADOPTED_GRANT')
        return None
    m=g[FIELD]
    need(type(m) is dict and m.get('schema')=='PV1_B24_ONLY_1000_ONCE_ADOPTED_V1'
         and m['grant_sha256']==GRANT_SHA and m['request_id']==RID
         and m['latest_native_registration_and_start_epoch']==LATEST
         and T0<=m['adopted_epoch']<LATEST and m['original_stop_new_epoch']==ORIGINAL_STOP_NEW
         and m['original_T0_epoch']==T0 and m['stop_physics_epoch']==STOP_PHYSICS
         and m['hard_close_epoch']==HARD_CLOSE and m['automatic_retry'] is False,
         'EXACT_ONCE_MARKER_NOT_A_GLOBAL_TIME_OVERRIDE')
    need(str(uuid.UUID(m['control_id']))==m['control_id']
         and re.fullmatch('[0-9a-f]{40}',m['source_sha']) is not None,'FRESH_BOUND_CONTROL_AND_SOURCE')
    p=g['V4_DEV_control_reservations'][PREP]
    need(p['status']=='CLOSED' and p['charged_GPU_site_control_CPU_s']==7200
         and p['grant_sha256']==GRANT_SHA and g['CPU_s']>=m['before_CPU_s']['GLOBAL']+7200
         and g['new_storage_current_bytes']>=m['before_GPU_site_storage_bytes']+64*1024**2,
         'CLOSED_PREPARATION_COSTS_NOT_ERASED_OR_REPAID')
    if RID in rows:
        r=rows[RID];need(start_limit(r)==LATEST and r['source_sha']==m['source_sha']
             and r.get('control_id')==m['control_id'],'SAME_ONLY_REQUEST_SOURCE_CONTROL_AND_GRANT')
    return m


def validate_pair(g,l):
    m=validate(g)
    if m is None:need(FIELD not in l,'NO_LOCAL_ORPHAN_GRANT');return None
    need(l.get(FIELD)==m and l['V4_DEV_control_reservations'].get(PREP)==g['V4_DEV_control_reservations'][PREP],
         'SAME_EXACT_ONCE_DUAL_MARKER_AND_COST_ROW')
    need(all(l['CPU_s'][site]>=m['before_CPU_s'][site]+7200 for site in ('CoreLab','Hopper','local_PC'))
         and l['network_transfer_accounted_bytes']>=m['before_transfer_bytes']+256*1024**2
         and l.get('Core_B24_preparation_output_upper_bytes')==64*1024**2
         and l.get('Core_B24_preparation_LOCAL_output_upper_bytes')==256*1024**2,
         'COMPLETE_CPU_OUTPUT_TRANSFER_PREPARATION_UPPERS')
    return m


def limit_for_marker(g,m):
    if m.get('replacement_request_id')!=RID:return ORIGINAL_STOP_NEW
    adopted=validate(g)
    if adopted is None:return ORIGINAL_STOP_NEW
    need(m['control_id']==adopted['control_id'] and m['replacement_source_sha']==adopted['source_sha'],
         'EXACT_KNOWN_ZERO_BOUNDARY_OWNS_THE_LATE_EXCEPTION')
    return LATEST


def check_request(manifest,r,g=None):
    limit=start_limit(r)
    if limit==ORIGINAL_STOP_NEW:return limit
    need(manifest['execution']['sealed_inputs'].get(PLAN)==r['sealed_inputs'].get(PLAN)==GRANT_SHA,
         'HUMAN_GRANT_SEALED_IN_BOTH_MANIFEST_AND_REQUEST')
    if g is not None:
        m=validate(g);need(m is not None and m['source_sha']==r['source_sha'] and m['control_id']==r['control_id'],
             'ACTUAL_ADOPTED_DUAL_BOUNDARY_NOT_A_PREPARED_GRANT')
    return limit


def adopt(g,l,raw_grant,*,source_sha,control_id,now_epoch):
    from . import evaluation_core_accounting as c
    grant(raw_grant);c.checked(g);c.scope.validate_pair(g,l)
    need(type(now_epoch) in (int,float) and T0<=now_epoch<LATEST and FIELD not in g and FIELD not in l,
         'ONCE_CURRENT_REAL_WINDOW_NOT_HISTORICAL_NOW')
    need(RID not in g.get(c.FIELD,{}) and all(r['status']==c.STATES[3] for r in g[c.FIELD].values())
         and all(x['status']=='CLOSED' for x in g['V4_DEV_control_reservations'].values())
         and g['unknown_effects']==l['unknown_effects']==[] and g['pending_updates']==l['pending_updates']==[],
         'COMPLETE_INACTIVE_KNOWN_ZERO_PREDECESSOR')
    before=dict(GLOBAL=g['CPU_s'],**l['CPU_s']);g,l=deepcopy(g),deepcopy(l)
    marker=dict(schema='PV1_B24_ONLY_1000_ONCE_ADOPTED_V1',grant_sha256=GRANT_SHA,request_id=RID,
        source_sha=source_sha,control_id=control_id,adopted_epoch=now_epoch,
        latest_native_registration_and_start_epoch=LATEST,original_stop_new_epoch=ORIGINAL_STOP_NEW,
        original_T0_epoch=T0,stop_physics_epoch=STOP_PHYSICS,hard_close_epoch=HARD_CLOSE,
        automatic_retry=False,before_CPU_s=before,before_GPU_site_storage_bytes=g['new_storage_current_bytes'],
        before_transfer_bytes=l['network_transfer_accounted_bytes'])
    closed=dict(run_uuid=PREP,arm='B_COVERAGE',source_sha=source_sha,control_CPU_upper_s=7200,
        Core_worker_CPU_reserved_s=0,created_epoch=now_epoch,closed_epoch=now_epoch,status='CLOSED',
        charged_GPU_site_control_CPU_s=7200,grant_sha256=GRANT_SHA,
        financial_upper_not_claimed_exact_process_CPU=True,charges_DEC1_subcaps=False)
    for v in (g,l):v[FIELD]=deepcopy(marker);need(PREP not in v['V4_DEV_control_reservations'],'NO_COST_REPLAY');v['V4_DEV_control_reservations'][PREP]=deepcopy(closed)
    g['CPU_s']+=7200
    for site in ('CoreLab','Hopper','local_PC'):l['CPU_s'][site]+=7200
    g['new_storage_current_bytes']+=64*1024**2;g['new_storage_peak_bytes']=max(g['new_storage_peak_bytes'],g['new_storage_current_bytes'])
    l['network_transfer_accounted_bytes']+=256*1024**2
    l['Core_B24_preparation_output_upper_bytes']=64*1024**2;l['Core_B24_preparation_LOCAL_output_upper_bytes']=256*1024**2
    validate_pair(g,l);c.checked(g);c.scope.validate_pair(g,l)
    return g,l
