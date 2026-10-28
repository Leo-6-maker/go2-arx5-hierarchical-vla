"""Finite first512 tickets consumed by the existing owner/watch and worker.

No submission or model/update occurs here. Every phase gets its own original
Journal mutation intent and fresh whole-budget/queue/project release. A13-step
ticket plan is not another scheduler or a license to replay a missed reply.
"""
import json,os,subprocess,time
from pathlib import Path
from . import minimal_execution as e,minimal_effects as effects
from . import minimal_execution_adapter as adapters
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.pbs_owner import publish_release_record as once

VERSION='ARX5_MINIMAL_ORIGINAL_OWNER_FINITE_SESSION_TICKETS_V1'
READY='A40_MINIMAL_SESSION_READY_V1'
RELEASE='A40_MINIMAL_ACTUAL_OWNER_SESSION_RELEASE_V1'
REMAINING_VERSION='ARX5_MINIMAL_SAVED_PREFIX_NINE_ORIGINAL_OWNER_TICKETS_V1'


def remaining(request):
    from . import minimal_remaining_contract as r
    if request is None or request.get('platform_training_version')!=r.VERSION:return False
    e.need(request.get('profile')==r.PROFILE,'GATE_EXACT_REMAINDER_PROFILE_NOT_OLD_ALIAS')
    return True


def ticket_version(request=None):return REMAINING_VERSION if remaining(request)else VERSION


def prefix_binding(request):
    if not remaining(request):return {}
    from . import minimal_remaining_contract as r
    return dict(saved_prefix_result_sha256=r.RESULT,saved_prefix_closure_sha256=r.CLOSED,
        completion_scope_grant_sha256=r.authority.GRANT_SHA)


def plan(request=None):
    rows=[]
    if remaining(request):
        from . import minimal_remaining_contract as r
        specs=r.sessions()
    else:specs=e.native_sessions()
    for spec in specs:
        phases=['MODEL_RAW']
        if spec['optimizer_limit']:
            phases+=['FIRST_UPDATE','OFFICIAL_EXPORT'if spec['formal_candidate']else'RESTORE_FULL']
        for phase in phases:
            rows.append(dict(index=len(rows),session_id=spec['id'],arm=spec['arm'],phase=phase))
    return rows


def descriptor(session_id,phase,*,request=None):
    matches=[r for r in plan(request)if r['session_id']==session_id and r['phase']==phase]
    e.need(len(matches)==1,'GATE_EXACT_FINITE_SESSION_PHASE');return matches[0]


def directory(allocation_directory,session_id,phase,*,request=None):
    spec=descriptor(session_id,phase,request=request)
    return Path(allocation_directory)/'session-gates'/str(spec['index']).zfill(2)


def check_ready(value,request,*,request_sha256,job_id,effect_value):
    """Check the current persisted allocation slice, never user PASS flags."""
    adapter=adapters.request(request);d=descriptor(value['session_id'],value['phase'],request=request)
    e.need(value.get('schema')==READY and value.get('ticket_version')==ticket_version(request)
        and all(value.get(k)==v for k,v in prefix_binding(request).items())
        and all(value.get(k)==v for k,v in d.items())
        and value.get('job_id')==job_id and value.get('request_id')==request['request_id']
        and value.get('submission_nonce')==request['submission_nonce']
        and value.get('request_sha256')==request_sha256 and value.get('source_sha')==request['source_sha']
        and value.get('raw_parent_files_identity')==e.training.RAW_PARENT_FILES_ID
        and value.get('statistics_sha256')==e.R0,
        'GATE_OWN_REQUEST_SOURCE_RAW_PARENT_AND_ORIGINAL_SCOPE')
    if remaining(request):
        from . import minimal_remaining_contract as r
        from .minimal_remaining_effects import SCHEMA as effect_schema
        binding=effect_value.get('audited_saved_prefix',{})
        e.need(binding.get('result_sha256')==r.RESULT and binding.get('closure_sha256')==r.CLOSED
            and binding.get('completed_sessions')==[s['id']for s in e.native_sessions()[:2]]
            and binding.get('E2_saved_zero_queries')==15,'GATE_REAL_AUDITED_PREFIX_IN_OWN_EFFECTS')
    else:effect_schema=effects.SCHEMA
    e.need(effect_value.get('schema')==effect_schema
        and (adapter is e or effect_value.get('platform_training_version')==adapter.VERSION)
        and effect_value['request_sha256']==request_sha256
        and effect_value['job_id']==job_id and effect_value['source_sha']==request['source_sha']
        and effect_value['active_session']==value['session_id']
        and effect_value['pending_updates']==effect_value['unknown_effects']==[],
        'GATE_CURRENT_OWN_LOCAL_EFFECTS_NOT_PRIOR_SESSION')
    row=effect_value['sessions'][value['session_id']];spec=next(r for r in e.native_sessions()if r['id']==value['session_id'])
    counters={k:row[k]for k in ('optimizer_updates','generation_queries','model_loads')}
    e.need(value.get('session_counters')==counters,'GATE_READY_EFFECT_COUNTERS_CHANGED')
    phase=value['phase']
    if remaining(request):
        r.check_phase(value['session_id'],phase,counters)
        return d
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
    job,host,_=native_job(ledger.request);v=ledger.guard();d=descriptor(session_id,phase,request=ledger.request)
    e.need(job==allocation['job_id']and host.split('.')[0].lower().startswith('gn-a40-'),
           'GATE_ACTUAL_OWN_RUNNING_JOB_BEFORE_READY')
    slot=directory(allocation_directory,session_id,phase,request=ledger.request);slot.mkdir(parents=True,exist_ok=False,mode=0o700)
    value=dict(schema=READY,ticket_version=ticket_version(ledger.request),**prefix_binding(ledger.request),**d,job_id=job,hostname=host,
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
    adapters.request(request)
    ready=json.loads(ready_bytes);p=value.get('native_proof',{})
    e.need(value.get('schema')==RELEASE and value.get('ticket_version')==ready.get('ticket_version')==ticket_version(request)
        and all(value.get(k)==ready.get(k)==v for k,v in prefix_binding(request).items())
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


def _within_phase_reads(function,*,guard,until,clock,sleep,job_id,observations,failures):
    """At most three failed qstat reads inside the same original180s phase.

    A failed read grants no model/update permission. Identity, STOP, ledger,
    parse and other errors propagate immediately; the original qstat25s and
    owner heartbeat limits remain unchanged. The caller shares one deadline
    and failure list across READY and final release verification.
    """
    def timely():
        if clock()>=until:raise TimeoutError('MINIMAL_ORIGINAL180S_NATIVE_READ_PHASE_EXPIRED')
    while True:
        timely();guard();timely()
        try:
            value=function()
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired)as error:
            if error.cmd!=['/opt/pbs/bin/qstat','-f','-F','json',job_id]:raise
            def text(value):
                return (value.decode(errors='replace')if isinstance(value,bytes)else value or '')[-2048:]
            record=dict(job_id=job_id,error_type=type(error).__name__,returncode=getattr(error,'returncode',None),
                stdout=text(error.stdout),stderr=text(error.stderr),observed_epoch=time.time(),
                remaining_original_phase_s=max(0,until-clock()),model_or_update_permission=False)
            failures.append(record);observations.mkdir(parents=True,exist_ok=True,mode=0o700)
            once(observations/('READ_FAILURE_'+str(len(failures)).zfill(2)+'.json'),record)
            if len(failures)>=3:raise
            timely();sleep(min(2.,until-clock()));continue
        timely();return value


def wait_release(root,allocation_directory,allocation,ledger,session_id,phase,*,guard,clock=time.monotonic,sleep=time.sleep):
    until=clock()+180;failures=[]
    observations=Path(allocation_directory)/'session-read-observations'/str(descriptor(session_id,phase,request=ledger.request)['index']).zfill(2)
    def finite_read(function):
        return _within_phase_reads(function,guard=guard,until=until,clock=clock,sleep=sleep,
            job_id=allocation['job_id'],observations=observations,failures=failures)
    slot=finite_read(lambda:publish_ready(root,allocation_directory,allocation,ledger,session_id,phase))
    while not(slot/'RELEASE.json').is_file():
        guard()
        if clock()>=until:raise TimeoutError('MINIMAL_ORIGINAL_OWNER180S_PHASE_RELEASE_MISSING_NO_REPLAY')
        sleep(.2)
    guard();body=(slot/'READY.json').read_bytes();ready=json.loads(body);v=ledger.guard()
    check_ready(ready,ledger.request,request_sha256=allocation['request_sha256'],job_id=allocation['job_id'],effect_value=v)
    need_release=check_release(read(slot/'RELEASE.json'),body,ledger.request,job_id=allocation['job_id'],now_epoch=time.time())
    from .minimal_native_context import verify_release
    finite_read(lambda:verify_release(root,ledger.request_path,allocation,need_release,ready_bytes=body,effect_bytes=ledger.path.read_bytes()))
    # Time spent observing PBS must not consume a now-stale owner permission.
    check_release(need_release,body,ledger.request,job_id=allocation['job_id'],now_epoch=time.time())
    # Do not equate a later legitimate Core query counter change with an old
    # permission. The actual guard must recompute the CURRENT whole GLOBAL;
    # the original release retains its exact observation and receipt bytes.
    once(slot/'RELEASE_CONSUMED.json',dict(release_sha256=sha(slot/'RELEASE.json'),observed_epoch=time.time(),
        permission_consumed_not_a_model_or_update_effect=True))
    return need_release
