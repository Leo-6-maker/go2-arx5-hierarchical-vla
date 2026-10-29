"""First512 cold branch of the original registered A40 worker entry.

No scheduler, login, GLOBAL mutation, extra owner or task physics lives here.
The original detached owner/watch must register/bind and release its own job;
the five-session worker consumes13 distinct native science tickets.
"""
import json,os,resource,time
from pathlib import Path
from scripts.pi0292.durable import OwnerLock,read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.pbs_owner import publish_release_record as once
from scripts.pi0294.dec1.a40_training_runtime import native_job
from . import minimal_execution as e,minimal_native_context as context,minimal_native_admission as admission
from . import minimal_execution_adapter as adapters

need=e.need


def allocation_from_native(root,request_path,r,job,host,native):
    from scripts.pi0294.e0_scope import pbs_epoch
    start=pbs_epoch(native['stime']);cpus,memory=adapters.request(r).select_profile(native['Resource_List'],r)
    return dict(status=admission.STATUS,run_uuid=Path(root).name,job_id=job,hostname=host,
        request_id=r['request_id'],submission_nonce=r['submission_nonce'],request_sha256=sha(request_path),
        source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],submit_count=r['submit_count'],
        allocation_start_epoch=start,allocation_end_epoch=min(start+r['walltime_s'],r['deadline_epoch']),
        deadline_epoch=r['deadline_epoch'],native_allocated_ncpus=cpus,native_allocated_memory_bytes=memory,
        active_CPU_limit=8,observed_epoch=time.time(),models_loaded=0,model_permission=False)


def await_bound(root,r,allocation,*,clock=time.monotonic,sleep=time.sleep):
    until=clock()+180;root=Path(root)
    while True:
        g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json')
        _,row,_=adapters.registered(g,r['request_id'],version=adapters.request(r).VERSION)
        need(row['request_sha256']==allocation['request_sha256']
            and row['job_id']in(None,allocation['job_id'])
            and g['unknown_effects']==g['pending_updates']==[],'RUNTIME_EXACT_KNOWN_RESERVATION_DURING_BIND')
        if row['status']==e.STATES[1] and row['job_id']==allocation['job_id']:return
        b=read(root/'OUTPUT_BINDING.private.json')
        need(read(b['lease'])['status']=='ACTIVE' and time.time()<allocation['allocation_end_epoch']
            and not any(p.exists()for p in(root/'STOP.json',root.parent.parent/'STOP.json')),'RUNTIME_STOP_BEFORE_NATIVE_BIND')
        if clock()>=until:raise TimeoutError('MINIMAL_ORIGINAL180S_BIND_MISSING_NO_MODEL')
        sleep(.2)


def await_running_owner(guard,*,clock=time.monotonic,sleep=time.sleep):
    until=clock()+180
    while True:
        try:guard(startup=True);return
        except (FileNotFoundError,context.StartupOwnerStillQueued):
            if clock()>=until:raise TimeoutError('MINIMAL_ORIGINAL180S_FIRST_R_OWNER_MISSING_NO_MODEL')
            sleep(.2)


def initial_release(root,request_path,allocation,guard,*,clock=time.monotonic,sleep=time.sleep):
    root=Path(root);d=Path(request_path).parent;p=d/'START_NEURAL.json';until=clock()+180
    while not p.is_file():
        guard()
        if clock()>=until:raise TimeoutError('MINIMAL_ORIGINAL180S_ALLOCATION_RELEASE_MISSING_NO_CUDA')
        sleep(.2)
    guard();v=read(p)
    need(v.get('schema')=='ARX5_MINIMAL_INITIAL_NATIVE_ALLOCATION_RELEASE_V1'
        and v.get('job_id')==allocation['job_id'] and v.get('request_sha256')==allocation['request_sha256']
        and v.get('ready_sha256')==sha(d/'PBS_READY.json') and v.get('scope_grant_sha256')==e.authority.GRANT_SHA
        and 0<=time.time()-v['observed_epoch']<=120,'RUNTIME_FRESH_OWN_NATIVE_INITIAL_RELEASE')
    proof=context.verify_release(root,request_path,allocation,v)
    once(d/'INITIAL_NATIVE_RELEASE_CONSUMED.json',dict(observed_epoch=time.time(),release_sha256=sha(p),
        actual_recomputed_proof=proof,model_permission=False))
    return proof


def run(root,request_path):
    root=Path(root);request_path=Path(request_path);d=request_path.parent
    # Context may observe the exact registered pre-bind row while qsub starts
    # its own worker. No model or permission follows until actual bind and R.
    manifest,r=context.root_context(root,request_path,full=True,require_bound=False)
    need(not(d/'PBS_PROCESS_FINAL.json').exists(),'RUNTIME_NO_NATIVE_RESTART_OR_MODEL_REPLAY')
    lock=OwnerLock(root/'A40_NATIVE_ALLOCATION_OWNER.lock');began=time.process_time()
    ledger=None;worker=None;result=None;error=None;phase='NATIVE_IDENTITY';allocation=None
    try:
        job,host,native=native_job(r);allocation=allocation_from_native(root,request_path,r,job,host,native)
        need(time.time()<allocation['allocation_end_epoch']-2700,'RUNTIME_FULL_SAVE_RESERVE_BEFORE_START')
        grant=sorted(os.sched_getaffinity(0));need(len(grant)>=8,'RUNTIME_EIGHT_ACTUAL_GRANTED_CPU_SLOTS')
        os.sched_setaffinity(0,set(grant[:8]));limits=adapters.request(r).process_CPU_limits(r)
        resource.setrlimit(resource.RLIMIT_CPU,(limits['parent_RLIMIT_CPU_s'],limits['parent_RLIMIT_CPU_s']))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        once(d/'ALLOCATION_ADMISSION.json',allocation)
        guard=context.Guard(root,allocation)
        adapter=adapters.request(r)
        if adapter is adapters.saved512:
            from .minimal_saved512_physical import Guard as SavedGuard
            guard=SavedGuard(root,r,manifest,guard)
        elif adapter is adapters.remaining:
            from .minimal_remaining_physical import Guard as RemainingGuard
            guard=RemainingGuard(root,r,manifest,guard)
        elif adapter is adapters.retry or adapters.PHYSICAL in manifest['execution']['sealed_inputs']:
            from .minimal_retry_runtime_guard import Guard as RetryGuard
            guard=RetryGuard(root,r,manifest,guard)
        if adapter is adapters.saved512:
            from .minimal_saved512_effects import Saved512Effects as Effects
        elif adapter is adapters.remaining:
            from .minimal_remaining_effects import RemainingEffects as Effects
        else:
            from .minimal_effects import Effects
        ledger=Effects(root,allocation,request_path=request_path,guard=guard);ledger.initialize()
        # An authoritative zero-effect journal precedes the original180s bind
        # and same-owner Q->R wait. Neither wait permits model/update effects.
        phase='AWAIT_REAL_BIND_AND_RUNNING_OWNER';await_bound(root,r,allocation);await_running_owner(guard)
        context.root_context(root,request_path,full=False)
        stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        once(d/'PBS_READY.json',dict(run_uuid=root.name,job_id=job,hostname=host,pid=os.getpid(),
            proc_start=int(stat[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            cgroup=Path('/proc/self/cgroup').read_text(),observed_epoch=time.time(),request_sha256=sha(request_path),
            source_sha=r['source_sha'],models_loaded=0,original_kernel_CPU_grant=grant,
            active_CPU_slots=grant[:8],amgr_login_in_compute=False))
        phase='INITIAL_NATIVE_RELEASE';initial_release(root,request_path,allocation,guard)
        # The same actual cluster git/module pin used by successful GEN1.
        # The official reader checks its upstream tree with git; compute
        # nodes do not inherit a usable login PATH. No loader/environment
        # installation or change to the parent LD path is introduced.
        from .amd_export_repair_runtime import tools
        native_tools=tools(run_git=True)
        from scripts.pi0294.numerics import MODE,prepare_environment,configure_torch
        need(r.get('numerical_mode')==MODE,'RUNTIME_EXPLICIT_ORIGINAL_DETERMINISTIC_MODE')
        prepare_environment(r['numerical_mode']);import torch
        need(torch.cuda.device_count()==1 and 'A40'in torch.cuda.get_device_name(0),'RUNTIME_ONE_ACTUAL_VISIBLE_A40')
        numeric=configure_torch(torch,r['numerical_mode']);torch.set_num_threads(4)
        once(d/'A40_PREMODEL_QUALIFICATION.json',dict(status='PASS_MINIMAL_ACTUAL_A40_CONTEXT_NO_MODEL',
            request_sha256=sha(request_path),job_id=job,numerics=numeric,original_cluster_tools=native_tools,
            models_loaded=0,queries=0,updates=0,new_physics=0))
        if adapter is adapters.saved512:
            from .minimal_saved512_worker import Saved512Worker as Worker
            phase='SAVED_E1_INFERENCE_AND_FRESH_RAW_E2_512'
        elif adapter is adapters.remaining:
            from .minimal_remaining_worker import RemainingWorker as Worker
            phase='THREE_REMAINING_INDEPENDENT_RAW_SESSIONS'
        else:
            from .minimal_worker import Worker
            phase='FIVE_INDEPENDENT_RAW_SESSIONS'
        worker=Worker(root,request_path,allocation,ledger,guard=guard)
        result=worker.run();phase='TERMINAL';return result
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc)[-2000:]);raise
    finally:
        if worker is not None:
            try:worker.unload()
            except BaseException as exc:
                if error is None:error=dict(type=type(exc).__name__,message=str(exc)[-2000:])
        final=dict(schema='ARX5_MINIMAL_ORIGINAL_A40_PROCESS_FINAL_V1',phase=phase,error=error,
            status=('SAVED512_SESSIONS_SAVED_NOT_INDEPENDENTLY_AUDITED'if r.get('platform_training_version')==adapters.saved512.VERSION else'REMAINING_SESSIONS_SAVED_NOT_INDEPENDENTLY_AUDITED'if r.get('platform_training_version')==adapters.remaining.VERSION else'FIVE_SESSIONS_SAVED_NOT_INDEPENDENTLY_AUDITED')if error is None and result else'FAILED_OR_PARTIAL_PRESERVED',
            job_id=os.environ.get('PBS_JOBID'),request_id=r['request_id'],source_sha=r['source_sha'],
            request_sha256=sha(request_path),CPU_actual_s=time.process_time()-began,
            new_physics=0,new_episodes=0,result=result)
        if ledger is not None:
            try:final['effects']=ledger.summary()
            except BaseException as exc:final['effect_summary_error']=dict(type=type(exc).__name__,message=str(exc)[-1000:])
        try:once(d/'PBS_PROCESS_FINAL.json',final)
        finally:lock.close()
