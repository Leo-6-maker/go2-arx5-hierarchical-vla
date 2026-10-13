"""Sequential subrequests in one allocation; the local-PC owner owns all stages."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.pi0292.durable import OwnerLock,read,write
from .native_dataset import sha
from .neural import verify_allocation,run_smoke,smoke_output_root,retire_redundant_smoke
from .run_admission import CUTOFF,DEADLINE


def validate_stage(root,original,stage,*,job_id,sequence):
    if set(stage)!={'run_uuid','job_id','sequence','mode','request_relative','request_sha256'}:
        raise PermissionError('EXACT_NEXT_STAGE_FIELDS_REQUIRED')
    if stage['run_uuid']!=Path(root).name or stage['job_id']!=job_id or stage['sequence']!=sequence:
        raise PermissionError('EXACT_OWNER_JOB_AND_NEXT_SEQUENCE_REQUIRED')
    if stage['mode']=='finish':
        if stage['request_relative'] is not None or stage['request_sha256'] is not None:
            raise PermissionError('FINISH_DOES_NOT_RUN_A_SUBREQUEST')
        return None
    if stage['mode'] not in ('main','resident','checkpoint'):
        raise PermissionError('ONLY_PREREGISTERED_MAIN_OR_RESIDENT_SUBSTAGES')
    relative=Path(stage['request_relative']);path=Path(root)/relative
    if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()) or sha(path)!=stage['request_sha256']:
        raise PermissionError('SEALED_SUBREQUEST_INSIDE_OWN_RUN_REQUIRED')
    request=read(path)
    for key in ('root','source_sha','run_manifest_sha256','code_seal_sha256','bindings','bindings_sha256',
                'walltime_s','submit_count','prior_GPU_allocated_s','numerical_mode','microbatch','workers'):
        if request[key]!=original[key]:raise PermissionError('NIGHT_BOUNDARY_CANNOT_CHANGE:'+key)
    if request['mode']!=stage['mode'] or request['stage']!=stage['mode']:
        raise PermissionError('EXACT_SUBSTAGE_ADMISSION_REQUIRED')
    if request['allowed_modes']!=[stage['mode']]:
        raise PermissionError('SUBREQUEST_CANNOT_REOPEN_ENGINEERING_UPDATES')
    if stage['mode']=='main' and (request['max_steps'] not in (4096,6144) or
        request['stop_at'] not in (512,2048,4096,6144) or request['stop_at']>request['max_steps']):
        raise PermissionError('ONLY_PREREGISTERED_MAIN_CHECKPOINT_BOUNDARIES')
    return path


def seal_single_measured_profile(root,request):
    """Only microbatch2 was measured; do not claim unrun 4/8 benchmarks."""
    smoke=smoke_output_root(root,request)
    gate=read(root/'evidence/MODEL_SMOKE.json')
    run_path=smoke/'training/smoke_continuous/session-start/TRAINING_RUN.json'
    actual=read(run_path)
    if gate['status']!='TRAINER_MODEL_INTERFACE_QUALIFIED' or actual['global_step']!=4:
        raise PermissionError('ACTUAL_FOUR_UPDATE_RATE_AND_COMPLETE_SMOKE_REQUIRED')
    if actual['recipe']['per_device_train_batch_size']!=2 or len(actual['actual_synchronized_optimizer_step_s'])!=4:
        raise PermissionError('ONLY_ACTUAL_MEASURED_MICROBATCH_TWO_PROFILE')
    path=root/'evidence/TRAINING_PROFILE_SEAL.json'
    if path.exists():raise FileExistsError('Never replace an already frozen profile')
    write(path,dict(status='SEALED_ACTUAL_MICROBATCH_AND_WORKER_BENCHMARK',microbatch=2,
        workers=request['workers'],actual_candidates=[dict(microbatch=2,workers=request['workers'],
            receipt=str(run_path.relative_to(root)),receipt_sha256=sha(run_path),
            steps_per_s=actual['actual_optimizer_steps_per_s'],
            peak_gpu_allocated_bytes=actual['peak_gpu_allocated_bytes'])],
        untested_microbatches=[4,8],selection='ONLY_MEASURED_STABLE_MICROBATCH_WITHIN_REMAINING_ENGINEERING_CAP',
        benchmark_updates=0,engineering_updates_total=16,model_candidate=False,
        numerical_configuration=actual['recipe']['numerical_configuration'],
        scheduler_horizon_steps=6144,warmup_steps=308,normal_stop=4096,
        manifest_sha256=actual['manifest_sha256'],statistics_sha256=actual['statistics_sha256'],
        selected='NONE',deployment='NONE'))


def run(args):
    root=Path(args.root);request=read(args.request)
    allocation=verify_allocation(root,'night-final',args.request)
    if request.get('prior_engineering_updates')!=8 or request.get('numerical_mode')!='DETERMINISTIC_CUDA_MATH_SDP_V1':
        raise PermissionError('EXACT_REMAINING_EIGHT_UPDATE_REQUALIFICATION_REQUIRED')
    owner=OwnerLock(root/'NIGHT_SESSION_OWNER.lock')
    directory=Path(args.request).parent/'night-session';directory.mkdir(exist_ok=False)
    state=dict(status='ACTUAL_FINAL_ALLOCATION_REQUALIFICATION',job_id=allocation['job_id'],
        run_uuid=root.name,sequence=0,pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        parent_request_sha256=sha(args.request),stages=[],main_started=False,selected='NONE',deployment='NONE')
    write(directory/'STATE.json',state)
    try:
        run_smoke(args)
        ledger=read(root/'HOPPER_RESOURCE_LEDGER.json')
        if ledger['model_smoke_updates']!=16 or ledger.get('pending_updates',0):
            raise PermissionError('EXACT_CUMULATIVE_SIXTEEN_ENGINEERING_UPDATES_REQUIRED')
        seal_single_measured_profile(root,request)
        if request.get('minimal_engineering_checkpoints'):
            smoke=smoke_output_root(root,request)
            retire_redundant_smoke(root,smoke,smoke/'evidence/RESUME_EQUIVALENCE.json',minimal=True)
        end=min(DEADLINE,allocation['allocation_end_epoch']-120)
        while time.time()<end:
            sequence=state['sequence']
            directive=directory/f'NEXT-{sequence:03d}.json'
            state['status']='WAIT_OWNER_NEXT_SEALED_STAGE';state['wait_started_epoch']=time.time()
            write(directory/'STATE.json',state)
            stop=min(end,time.time()+60)
            while not directive.exists() and time.time()<stop:time.sleep(.2)
            if not directive.exists():
                state['status']='OWNER_NEXT_STAGE_TIMEOUT_FULL_STATE_PRESERVED';return state
            stage=read(directive)
            path=validate_stage(root,request,stage,job_id=state['job_id'],sequence=sequence)
            if path is None:
                state['status']='OWNER_FINISHED_OWN_ALLOCATION';return state
            if stage['mode']=='main' and time.time()>=CUTOFF:
                raise PermissionError('NO_NEW_MAIN_OPTIMIZATION_AFTER_ORIGINAL_CUTOFF')
            state['status']='RUNNING_'+stage['mode'].upper();state['active_request_sha256']=sha(path)
            write(directory/'STATE.json',state)
            command=[sys.executable,'-m','scripts.pi0294.allocation_admission',str(path)]
            result=subprocess.run(command,timeout=max(1,end-time.time()))
            state['stages'].append(dict(sequence=sequence,mode=stage['mode'],request_sha256=sha(path),exit=result.returncode))
            if result.returncode:raise RuntimeError('KNOWN_SUBSTAGE_FAILED_NO_AUTOMATIC_RETRY')
            if stage['mode']=='main':state['main_started']=True
            state['sequence']+=1;write(directory/'STATE.json',state)
        state['status']='ORIGINAL_OWN_ALLOCATION_OR_DEADLINE_BOUNDARY';return state
    except BaseException as error:
        state['status']='FAILED_PRESERVED_NO_AUTOMATIC_RETRY'
        state['failure']=dict(type=type(error).__name__,error=str(error))
        raise
    finally:
        write(directory/'STATE.json',state);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','request','bindings'):p.add_argument('--'+key,required=True)
    p.add_argument('--microbatch',type=int,choices=[2],default=2)
    p.add_argument('--workers',type=int,choices=[0,2,4],default=0)
    run(p.parse_args())
