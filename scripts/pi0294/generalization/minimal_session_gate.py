"""Finite first512 tickets consumed by the existing owner/watch and worker.

No submission or model/update occurs here. Every phase gets its own original
Journal mutation intent and fresh whole-budget/queue/project release. A13-step
ticket plan is not another scheduler or a license to replay a missed reply.
"""
import json,os,time
from pathlib import Path
from . import minimal_execution as e,minimal_effects as effects
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.pbs_owner import publish_release_record as once

VERSION='ARX5_MINIMAL_ORIGINAL_OWNER_FINITE_SESSION_TICKETS_V1'
READY='A40_MINIMAL_SESSION_READY_V1'
RELEASE='A40_MINIMAL_ACTUAL_OWNER_SESSION_RELEASE_V1'


def plan():
    rows=[]
    for spec in e.native_sessions():
        phases=['MODEL_RAW']
        if spec['optimizer_limit']:
            phases+=['FIRST_UPDATE','OFFICIAL_EXPORT'if spec['formal_candidate']else'RESTORE_FULL']
        for phase in phases:
            rows.append(dict(index=len(rows),session_id=spec['id'],arm=spec['arm'],phase=phase))
    return rows


def descriptor(session_id,phase):
    matches=[r for r in plan()if r['session_id']==session_id and r['phase']==phase]
    e.need(len(matches)==1,'GATE_EXACT_FINITE_SESSION_PHASE');return matches[0]


def directory(allocation_directory,session_id,phase):
    spec=descriptor(session_id,phase)
    return Path(allocation_directory)/'session-gates'/str(spec['index']).zfill(2)


def check_ready(value,request,*,request_sha256,job_id,effect_value):
    """Check the current persisted allocation slice, never user PASS flags."""
    e.request(request);d=descriptor(value['session_id'],value['phase'])
    e.need(value.get('schema')==READY and value.get('ticket_version')==VERSION
        and all(value.get(k)==v for k,v in d.items())
        and value.get('job_id')==job_id and value.get('request_id')==request['request_id']
        and value.get('submission_nonce')==request['submission_nonce']
        and value.get('request_sha256')==request_sha256 and value.get('source_sha')==request['source_sha']
        and value.get('raw_parent_files_identity')==e.training.RAW_PARENT_FILES_ID
        and value.get('statistics_sha256')==e.R0,
        'GATE_OWN_REQUEST_SOURCE_RAW_PARENT_AND_ORIGINAL_SCOPE')
    e.need(effect_value.get('schema')==effects.SCHEMA
        and effect_value['request_sha256']==request_sha256
        and effect_value['job_id']==job_id and effect_value['source_sha']==request['source_sha']
        and effect_value['active_session']==value['session_id']
        and effect_value['pending_updates']==effect_value['unknown_effects']==[],
        'GATE_CURRENT_OWN_LOCAL_EFFECTS_NOT_PRIOR_SESSION')
    row=effect_value['sessions'][value['session_id']];spec=next(r for r in e.native_sessions()if r['id']==value['session_id'])
    counters={k:row[k]for k in ('optimizer_updates','generation_queries','model_loads')}
    e.need(value.get('session_counters')==counters,'GATE_READY_EFFECT_COUNTERS_CHANGED')
    phase=value['phase']
    if phase=='MODEL_RAW':
        e.need(all(v==0 for v in counters.values()),'GATE_NEW_SESSION_INDEPENDENT_RAW_MODEL_ONLY')
    elif phase=='FIRST_UPDATE':
        e.need(counters['optimizer_updates']==0 and counters['model_loads']==1
            and counters['generation_queries']==(0 if spec['formal_candidate']else 15),
            'GATE_REAL_MATCHED_ZERO_STEP_BEFORE_TWO_CALLS_OR_FRESH_FORMAL512')
    elif phase=='RESTORE_FULL':
        e.need(counters==dict(optimizer_updates=2,generation_queries=16,model_loads=1),
            'GATE_ACTUAL_STEP2_REFERENCE_BEFORE_SEPARATE_FULL_RESTORE')
    else:
        e.need(counters==dict(optimizer_updates=512,generation_queries=36,model_loads=1),
            'GATE_ACTUAL_FULL512_AND_ALL36_PROBES_BEFORE_OFFICIAL_EXPORT')
    return d


def publish_ready(root,allocation_directory,allocation,ledger,session_id,phase):
    """Actual compute metadata only; no tensor import or model permission."""
    from scripts.pi0294.dec1.a40_training_runtime import native_job
    job,host,_=native_job(ledger.request);v=ledger.guard();d=descriptor(session_id,phase)
    e.need(job==allocation['job_id']and host.split('.')[0].lower().startswith('gn-a40-'),
           'GATE_ACTUAL_OWN_RUNNING_JOB_BEFORE_READY')
    slot=directory(allocation_directory,session_id,phase);slot.mkdir(parents=True,exist_ok=False,mode=0o700)
    value=dict(schema=READY,ticket_version=VERSION,**d,job_id=job,hostname=host,
        request_id=ledger.request['request_id'],submission_nonce=ledger.request['submission_nonce'],
        request_sha256=allocation['request_sha256'],source_sha=ledger.request['source_sha'],
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,
        session_counters={k:v['sessions'][session_id][k]for k in ('optimizer_updates','generation_queries','model_loads')},
        effects_sha256=sha(ledger.path),observed_epoch=time.time(),pid=os.getpid(),
        proc_start=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        no_model_or_optimizer_permission_from_READY=True)
    check_ready(value,ledger.request,request_sha256=allocation['request_sha256'],job_id=job,effect_value=v)
    once(slot/'READY.json',value);return slot


def check_release(value,ready_bytes,request,*,job_id,now_epoch):
    ready=json.loads(ready_bytes);p=value.get('native_proof',{})
    e.need(value.get('schema')==RELEASE and value.get('ticket_version')==VERSION
        and value.get('status')=='PASS_MINIMAL_SESSION_'+ready['phase']+'_NATIVE'
        and value.get('ready_sha256')==e.sha(ready_bytes)
        and value.get('session_id')==ready['session_id']and value.get('phase')==ready['phase']
        and value.get('job_id')==job_id==ready['job_id']
        and value.get('request_sha256')==ready['request_sha256']
        and value.get('scope_grant_sha256')==e.authority.GRANT_SHA
        and ready['observed_epoch']<=value.get('observed_epoch',0)<=now_epoch
        and 0<=now_epoch-value['observed_epoch']<=120
        and p.get('whole_remaining_package_shortfalls')=={}
        and p.get('complete_queue_verified')is True and p.get('project_eligible')is True
        and p.get('actual_project_quota_covers_remaining_output')is True
        and p.get('global_unknown_effects')==p.get('global_pending_updates')==[]
        and p.get('native_own_job_id')==job_id,
        'GATE_FRESH_SAME_JOB_EXACT_READY_AND_FULL_ORIGINAL_NATIVE_RELEASE')
    e.need(request['request_id']==ready['request_id']and request['submission_nonce']==ready['submission_nonce']
        and request['source_sha']==ready['source_sha'],'GATE_CURRENT_REQUEST_NOT_OLD_SOURCE')
    return value


def wait_release(root,allocation_directory,allocation,ledger,session_id,phase,*,guard,clock=time.monotonic,sleep=time.sleep):
    slot=publish_ready(root,allocation_directory,allocation,ledger,session_id,phase)
    until=clock()+180
    while not(slot/'RELEASE.json').is_file():
        guard()
        if clock()>=until:raise TimeoutError('MINIMAL_ORIGINAL_OWNER180S_PHASE_RELEASE_MISSING_NO_REPLAY')
        sleep(.2)
    guard();body=(slot/'READY.json').read_bytes();ready=json.loads(body);v=ledger.guard()
    check_ready(ready,ledger.request,request_sha256=allocation['request_sha256'],job_id=allocation['job_id'],effect_value=v)
    need_release=check_release(read(slot/'RELEASE.json'),body,ledger.request,job_id=allocation['job_id'],now_epoch=time.time())
    # Do not equate a later legitimate Core query counter change with an old
    # permission. The actual guard must recompute the CURRENT whole GLOBAL;
    # the original release retains its exact observation and receipt bytes.
    once(slot/'RELEASE_CONSUMED.json',dict(release_sha256=sha(slot/'RELEASE.json'),observed_epoch=time.time(),
        permission_consumed_not_a_model_or_update_effect=True))
    return need_release
