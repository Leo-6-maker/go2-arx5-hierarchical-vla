"""Same-allocation stage owner; remote archive failure is not a model gate."""
import argparse
import os
from pathlib import Path
import subprocess
import resource
import time

from scripts.pi0292.durable import OwnerLock, read, write
from .continuation import RUN, CUTOFF
from .native_dataset import sha
from .neural import verify_allocation


def validate_stage(root, original, stage, *, job_id, sequence):
    if set(stage) != {'run_uuid','job_id','sequence','mode','request_relative','request_sha256'}:
        raise PermissionError('EXACT_CONTINUATION_DIRECTIVE_REQUIRED')
    if stage['run_uuid'] != RUN or stage['job_id'] != job_id or stage['sequence'] != sequence:
        raise PermissionError('CONTINUATION_JOB_AND_NEXT_SEQUENCE_REQUIRED')
    if stage['mode'] == 'finish':
        if stage['request_relative'] is not None or stage['request_sha256'] is not None:
            raise PermissionError('FINISH_CANNOT_EXECUTE_A_REQUEST')
        return None
    if stage['mode'] not in ('resident','main'):
        raise PermissionError('ARCHIVE_AND_RESMOKE_NOT_MODEL_STAGE_DEPENDENCIES')
    from .continuation import within
    path = within(Path(root)/stage['request_relative'], root)
    if Path(stage['request_relative']).is_absolute() or sha(path) != stage['request_sha256']:
        raise PermissionError('EXACT_SEALED_CONTINUATION_SUBREQUEST')
    request = read(path)
    for k in ('root','source_sha','run_manifest_sha256','code_seal_sha256','bindings','bindings_sha256',
              'walltime_s','submit_count','prior_GPU_allocated_s','numerical_mode','microbatch','workers',
              'request_nonce','project_admission_sha256','allowed_peer_ids'):
        if request[k] != original[k]:
            raise PermissionError('CONTINUATION_BOUNDARY_CHANGED:'+k)
    if request['stage'] != stage['mode'] or request['mode'] != stage['mode'] or request['allowed_modes'] != [stage['mode']]:
        raise PermissionError('EXACT_CONTINUATION_MODE_REQUIRED')
    if stage['mode'] == 'main':
        if request['max_steps'] != 4096 or request['stop_at'] not in (2048,4096) or not request.get('resume'):
            raise PermissionError('RESUME_ORIGINAL_CHAIN_ONLY')
        if not original.get('P1_behavior_gate_sha256') and not request.get('P1_behavior_gate_sha256'):
            raise PermissionError('ACTUAL_P1_MODEL_INTERFACE_EVIDENCE_BEFORE_NEW_UPDATES')
        gate_path = within(Path(root)/request['P1_behavior_gate_relative'], root)
        if sha(gate_path) != request['P1_behavior_gate_sha256']:
            raise PermissionError('P1_BEHAVIOR_GATE_CHANGED')
        gate = read(gate_path)
        if (gate['status'] != 'PASS_ACTUAL_M512_MODEL_INTERFACE_NO_UNHANDLED_REAL_SAFETY_ERROR'
                or gate['run_uuid'] != RUN or gate['completed_development_attempts'] != 4
                or gate['generation_queries'] <= 0 or gate['real_safety_violations'] != 0
                or gate['expert_positive_control'] != 'PASS'):
            raise PermissionError('ACTUAL_MODEL_AND_EXPERT_CONTROL_REQUIRED_NOT_SUCCESS_THRESHOLD')
    return path


def run(args):
    root, request_path = Path(args.root), Path(args.request)
    original = read(request_path)
    allocation = verify_allocation(root,'continuation',request_path)
    owner = OwnerLock(root/'CONTINUATION_OWNER.lock')
    directory = request_path.parent/'continuation-session'
    directory.mkdir(exist_ok=False)
    state = dict(status='WAIT_OWNER_NEXT_SEALED_STAGE',run_uuid=RUN,job_id=allocation['job_id'],
        pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        parent_request_sha256=sha(request_path),sequence=0,stages=[],selected='NONE',deployment='NONE')
    try:
        end = min(CUTOFF,allocation['allocation_end_epoch']-120)
        while time.time() < end:
            state.update(status='WAIT_OWNER_NEXT_SEALED_STAGE',wait_started_epoch=time.time())
            write(directory/'STATE.json',state)
            path = directory/f"NEXT-{state['sequence']:03d}.json"
            # A bounded wait permits owner-side CPU inventory/route preparation.
            until = min(end,time.time()+300)
            while not path.exists() and time.time() < until:
                time.sleep(.2)
            if not path.exists():
                state['status'] = 'OWNER_NEXT_STAGE_TIMEOUT_PRESERVED'
                return state
            directive = read(path)
            child = validate_stage(root,original,directive,job_id=state['job_id'],sequence=state['sequence'])
            if child is None:
                state['status'] = 'OWNER_FINISHED_CONTINUATION_ALLOCATION'
                return state
            state.update(status='RUNNING_'+directive['mode'].upper(),active_request_sha256=sha(child))
            write(directory/'STATE.json',state)
            cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
            charged = read(root/'HOPPER_RESOURCE_LEDGER.json')['CPU_s']
            remaining_cpu = int(57600-charged-cpu.ru_utime-cpu.ru_stime-120)
            if remaining_cpu <= 0:
                raise PermissionError('CONTINUATION_INCREMENTAL_CPU_CAP')
            def cpu_limit():
                resource.setrlimit(resource.RLIMIT_CPU,(remaining_cpu,remaining_cpu))
            result = subprocess.run([original['python'],'-m','scripts.pi0294.allocation_admission',str(child)],
                                    timeout=max(1,end-time.time()),preexec_fn=cpu_limit)
            state['stages'].append(dict(sequence=state['sequence'],mode=directive['mode'],
                                       request_sha256=sha(child),exit=result.returncode))
            if result.returncode:
                raise RuntimeError('KNOWN_CONTINUATION_STAGE_FAILED_NO_AUTOMATIC_RETRY')
            state['sequence'] += 1
        state['status'] = 'CONTINUATION_ALLOCATION_OR_SCIENCE_CUTOFF'
        return state
    except BaseException as error:
        state.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',failure=dict(type=type(error).__name__,error=str(error)))
        raise
    finally:
        write(directory/'STATE.json',state)
        owner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root','request','bindings'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
