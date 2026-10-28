"""Cold Core hook into the existing model service and original DEV evaluator.

No training implementation, new controller or changed scientific gate. The
server owner must hold the exact registered request and bind its real unit,
process and fresh resources before this hook proceeds. C/D keeps its own scope.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.e1 import model
from scripts.pi0294.e1.neural import checkpoint_identity,serve
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.native_train import parameter_inventory
from scripts.pi0294.numerics import prepare_environment,configure_torch
from scripts.pi0294.mp1.owner_heartbeat import read_live_owner
from . import evaluation_core_accounting as c,evaluation_core_identity as identity
from .evaluation_core_scope import validate
from .evaluation_core_ledger import Ledger
from . import evaluation_core_native as native
from .native_transport_qualification import run as qualify_transport
from .pre_model_qualification import require_current
from .evaluation_plan import session_spec,validate_registry,remaining_panel
from .visibility import refresh


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_EVALUATOR_'+why)


def put(path,value):
    need(not path.exists() and not path.is_symlink(),'PRESERVE_EACH_NATIVE_RECEIPT_NO_RESTART')
    write(path,value);path.chmod(0o400)


def phase_deadline(request):
    from . import mix_core_identity as cd,successor_cd as successor
    from . import mix_vanda_eval_identity as colocated
    if colocated.typed(request):
        need(request.get('stage')=='V5_MATCHED_DEV24' and request.get('evaluation_candidate') in ('C50','D25')
             and request.get('deadline_epoch')==successor.scope(request.get('T_CD'))['deadline_epoch'],
             'EXPLICIT_ADOPTED_COLOCATED_WINDOW')
        return request['deadline_epoch']
    if cd.typed(request):
        need(request.get('platform_evaluation_version')==cd.c.VERSION
             and request.get('stage')=='V5_MATCHED_DEV24' and request.get('evaluation_candidate')in ('C50','D25')
             and request.get('deadline_epoch')==successor.scope(request.get('T_CD'))['deadline_epoch'],
             'EXPLICIT_ADOPTED_SUCCESSOR_PHASE_NOT_OLD_AB_CLOCK')
        return request['deadline_epoch']
    return c.old.STOP_PHYSICS


def release(root,directory,request,process,name,*,phase_started,guard):
    """One actual owner's immutable reply; read failure never resends a write."""
    path=directory/name;end=min(time.time()+180,process['started_epoch']+request['walltime_s']-2700,phase_deadline(request))
    while time.time()<end:
        guard()
        if path.exists():
            need(not path.is_symlink() and path.stat().st_uid==os.getuid() and not path.stat().st_mode&0o222,
                 'IMMUTABLE_OWNER_RELEASE')
            value=read(path)
            exact=dict(schema='PV1_CORE_EXISTING_NATIVE_PHASE_RELEASE_V1',run_uuid=root.name,
                request_id=request['request_id'],request_sha256=sha(directory/'REQUEST.json'),
                source_sha=request['source_sha'],job_id=process['native_handle'],phase=name,
                physics_authorized=False,new_optimizer_updates=0)
            need(all(value.get(k)==v and type(value[k]) is type(v) for k,v in exact.items())
                 and phase_started<=value['observed_epoch']<=time.time()
                 and time.time()-value['observed_epoch']<=120,'SAME_CURRENT_OWNER_EXACT_BOUND_PHASE')
            return value
        time.sleep(.2)
    raise TimeoutError('PV1_CORE_NATIVE_RELEASE_NOT_OBSERVED_NO_RESTART')


def run(root,request_path):
    root,request_path=Path(root),Path(request_path);request=read(request_path)
    from . import mix_vanda_eval_identity as colocated
    cohost=colocated.typed(request)
    # Ledger's explicit factory also covers original transport/RPC actors.
    # Preserve the original qualification, two releases, S1 and model service.
    from .mix_core_delegated_ledger import selected as finite_selected
    finite=finite_selected(read(root/'RUN_MANIFEST.json'),request)
    need(not (cohost and finite),'NO_CROSS_PLATFORM_DELEGATION_ALIAS')
    validator,process_identity,backend,ledger_type=validate,identity,native,Ledger
    if cohost:
        from . import mix_vanda_eval_context as process_identity,mix_vanda_eval_native as backend
        from .mix_vanda_eval_ledger import Ledger as ledger_type
        validator=process_identity.validate
    manifest,*_=validator(root)
    from . import mix_core_identity as cd
    successor=cohost or cd.typed(request)
    if cohost:request_identity=colocated.request_identity
    elif successor:request_identity=cd.request_identity
    else:
        need(request.get('stage')=='P4_DEV' and request.get('evaluation_candidate')in c.stage.CANDIDATES
             and request.get('platform_evaluation_version')==c.VERSION,'ORIGINAL_AB_FIRST_STAGE_ONLY_ZERO_TRAINING')
        from .evaluation_core_admission import request_identity
    request_identity(manifest,request,sha(root/'RUN_MANIFEST.json'))
    end=phase_deadline(request)
    directory=request_path.parent
    need(directory.parent.resolve()==(root/'pbs').resolve() and request_path.name=='REQUEST.json'
         and directory.name=='alloc-'+str(request['allocation_number']).zfill(2)
         and not request_path.stat().st_mode&0o222,'EXACT_IMMUTABLE_REGISTERED_CORE_REQUEST')
    lock=OwnerLock(root/'NEURAL_OWNER.lock');began=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    ledger=None;instance=None;status='FAILED_PRESERVED';phase='NATIVE_CORE_ENTRY';error=None
    process=None;records=[]
    def guard():
        now=time.time()
        need(now<end and not any(p.exists() for p in (root/'STOP.json',root.parent.parent/'STOP.json',
            root/'SOURCE_PHYSICS_STOP.json',root/'SOURCE_SAFETY_STOP_FROM_CORE.json')),'ORIGINAL_STOP_FENCES')
        if process is not None:need(now<min(process['started_epoch']+request['walltime_s'],end)-2700,
            'ORIGINAL_45_MINUTE_NATIVE_CLOSEOUT_RESERVE')
        if ledger is None:
            # A just-started unit announces its real PID before the owner can
            # bind it. During this bounded zero-model wait, observe the actual
            # already accepted resident owner rather than inventing a live
            # GPU heartbeat before a native handle has been bound.
            if cohost:process_identity.owner_current(root,request,request_sha256=sha(request_path))
            elif successor:cd.current_owner(root,request,request_sha256=sha(request_path))
            elif 'startup_recovery_sha256' in request:
                from .evaluation_core_zero_start_recovery import current_owner
                current_owner(root,request)
            else:identity.current_queue(root,request,request_sha256=sha(request_path))
        else:
            read_live_owner(root,root.name);ledger.guard()
    try:
        # The actual process is observed before any Torch/CUDA import or model.
        process=process_identity.observe_current(root,dict(request,request_sha256=sha(request_path)))
        if not cohost:os.environ[identity.HANDLE_ENV]=process['native_handle']
        resource.setrlimit(resource.RLIMIT_CPU,(request['declared_process_CPU_limit_s'],request['declared_process_CPU_limit_s']));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        put(directory/'NATIVE_CORE_PROCESS_STARTED.json',process)
        phase='WAIT_ACTUAL_GLOBAL_BIND_AND_PREMODEL_RELEASE'
        release(root,directory,request,process,'START_NEURAL.json',phase_started=process['observed_epoch'],guard=guard)
        allocation=backend.admit(root,request_path);ledger=ledger_type(root,request_path)
        phase='ACTUAL_NATIVE_600S_80_CYCLE_QUALIFICATION'
        qualification=qualify_transport(root,directory,allocation,guard)
        put(directory/'PV1_PREMODEL_QUALIFICATION_READBACK.json',qualification)
        phase_started=time.time();phase='WAIT_FRESH_POST_QUALIFICATION_MODEL_RELEASE'
        put(directory/'CORE_FUTURE_PHASE_READY.json',dict(schema='PV1_CORE_NATIVE_PHASE_READY_V1',
            request_id=request['request_id'],request_sha256=sha(request_path),job_id=allocation['job_id'],
            source_sha=request['source_sha'],run_uuid=root.name,phase_started_epoch=phase_started,
            qualification_receipt_sha256=qualification['receipt_sha256'],new_models=0,new_queries=0,new_physics=0))
        release(root,directory,request,process,'CORE_MODEL_RELEASE.json',phase_started=phase_started,guard=guard)
        proof,_=backend.budget_now(root,request_path,ledger)
        put(directory/'POST_QUALIFICATION_NATIVE_RESOURCE_PROOF.json',proof)
        require_current(root,allocation);guard();prepare_environment(request['numerical_mode'])
        import torch
        from .neural import require_native_device,evaluation_labels,record_candidate
        require_native_device(request,allocation,torch)
        configure_torch(torch,request['numerical_mode']);torch.set_num_threads(4)
        metadata=read(root/'plans/EVALUATION_CHECKPOINT.private.json');candidate=request['evaluation_candidate']
        checkpoint=Path(metadata['checkpoint']);phase=candidate+'-MODEL_LOAD'
        expected_checkpoint=request['checkpoint_manifest_sha256'] if successor else c.old.ENDPOINTS[candidate]
        need(metadata.get('full_resume_files_present') is False
             and metadata['checkpoint_manifest_sha256']==expected_checkpoint
             and sha(checkpoint/'CHECKPOINT_COMPLETE.json')==expected_checkpoint,
             'EXACT_INFERENCE_VIEW_OF_CLOSED_FULL_STATE_NOT_NEW_TRAINING')
        model_identity=checkpoint_identity(checkpoint,candidate,request['source_sha'])
        intent=directory/'MODEL_LOAD_INTENTS'/ (candidate+'.json');intent.parent.mkdir(mode=0o700)
        put(intent,dict(job_id=allocation['job_id'],candidate=candidate,request_sha256=sha(request_path),
            checkpoint_manifest_sha256=model_identity['checkpoint_manifest_sha256'],epoch=time.time()))
        ledger.charge('model_loads',1)
        instance=model.load(root,checkpoint,model_identity,trainable=False)
        inventory=parameter_inventory(instance.model);original=read(root/'plans/INITIAL_CORR_PARAMETER_INVENTORY.json')
        need(set(original)==set(inventory) and all(v['sha256']==inventory[n]['sha256']
             for n,v in original.items() if not v['requires_grad']),'ORIGINAL_FROZEN_VLM_PARAMETER_HASHES')
        put(directory/'FROZEN_VLM_PROOF.json',dict(status='PASS_ORIGINAL_FROZEN_VLM_NATIVE_PARAMETER_HASHES'))
        record_candidate(ledger,candidate,model_identity)
        plan=read(root/'plans/SOURCE_EVALUATION_PLAN.json');prior=read(root/'plans/PRIOR_EVALUATION_OUTCOMES.json')
        labels=evaluation_labels(request,plan,prior);need(len(labels)==1,'ONE_ORIGINAL_WHOLE_FIRST_NOISE_PANEL')
        for label in labels:
            phase=label;spec=session_spec(root,label)
            need(len(remaining_panel(prior,spec))==request['remaining_conditions'] and spec['candidate']==candidate,'EXACT_ORIGINAL_24_CONDITIONS')
            peer=read(root/'plans'/('PV1-PREMODEL-ALLOCATION-'+str(request['allocation_number']).zfill(2))/'REGISTRY.json')
            registry_path=root/'plans'/label/'REGISTRY.json';sealed_registry=[None]
            def panel_guard():
                guard()
                if registry_path.exists():
                    current=sha(registry_path)
                    if sealed_registry[0] is None:
                        validate_registry(read(registry_path),spec,model_identity,run_uuid=root.name,
                            job_id=allocation['job_id'],peer=peer,prior_rows=prior)
                        sealed_registry[0]=current
                    else:need(sealed_registry[0]==current,'PRESERVE_ORIGINAL_MODEL_REGISTRY')
            serve(root,label,instance,allocation,ledger,panel_guard,max_queries=1920,
                ready_status='PV1_NATIVE_MODEL_READY',noise_namespace=spec['noise_namespace'],
                terminal_status='PV1_NATIVE_PANEL_TERMINAL_OUTCOMES_AUDITED_SEPARATELY',
                native_pre_model_qualification=require_current(root,allocation),visibility_refresh=refresh)
            records.append(dict(panel=label,status='NATIVE_PANEL_TERMINAL_NOT_TASK_SUCCESS'))
        status='PV1_DEV_NATIVE_PANELS_TERMINAL_ACTUAL_OUTCOMES_AUDITED_SEPARATELY'
        return dict(status=status,job_id=allocation['job_id'])
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc);raise
    finally:
        try:
            if instance is not None:model.unload(instance)
        except BaseException as exc:
            status='FAILED_PRESERVED';error=error or 'UNLOAD:'+type(exc).__name__+': '+str(exc);raise
        finally:
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu=time.process_time()-began+after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime
            try:
                if ledger is not None:ledger.cpu(cpu)
            finally:
                try:
                    put(directory/'NATIVE_PROCESS_FINAL.json',dict(status=status,phase=phase,error=error,CPU_s=cpu,
                        job_id=process['native_handle'] if process is not None else None,records=records,
                        completed_epoch=time.time(),selected='NONE',deployment='NONE',new_optimizer_updates=0))
                finally:lock.close()
