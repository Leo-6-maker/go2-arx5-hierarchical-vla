"""Bound a frozen export/qualification entry without editing its source."""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import runpy
import signal
import sys
import time

from deadline_guard import atomic_write, install_deadline, resource_bounds


def cpu():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime


def run(root, stage, arguments, stage_receipt=None):
    root = Path(root)
    began = time.monotonic()
    seal = json.loads((root/'TEACHER_AND_LAYOUT_SEAL.json').read_text())
    relative = 'scripts/pi0291_arx5_data/'+stage+'.py'
    script = root/'code'/relative
    assert hashlib.sha256(script.read_bytes()).hexdigest() == seal['code_files'][relative]
    receipt = Path(stage_receipt) if stage_receipt else root/('native_dataset/EXPORT_RECEIPT.json' if stage == 'export_dataset' else 'evidence/DATALOADER_SMOKE.json')
    assert stage != 'audit_data' or stage_receipt
    assert receipt.resolve().is_relative_to(root.resolve())
    assert not receipt.exists(), 'Preserve any prior invocation and receipt'
    usage_receipt = root/'evidence'/f'CPU_AUX_NATIVE_WRAPPER_{stage}_{int(time.time())}_{os.getpid()}.json'
    status = 'FAILED_PRESERVED'
    recorded = False
    cleanup = []
    per_process_limit = None

    def save_usage(outcome):
        nonlocal recorded
        if recorded:
            return
        covered = 0
        if receipt.is_file():
            value = json.loads(receipt.read_text())
            covered = value.get('CPU_s',value.get('CPU_total_s',0))
        total = cpu()
        unknown_upper = sum(3*(time.monotonic()-began) for row in cleanup if row['alive_after_cleanup'])
        atomic_write(usage_receipt,dict(status=outcome,CPU_s=max(0,total-covered)+unknown_upper,
            measured_uncovered_CPU_s=max(0,total-covered),unreaped_worker_CPU_s_upper=unknown_upper,
            own_worker_cleanup=cleanup,
            per_process_CPU_limit_s=per_process_limit,auxiliary_CPU_reserve_s=600,
            invocation_CPU_s=total,already_covered_by_stage_receipt_CPU_s=covered,
            stage=stage,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_sha=seal['source_sha'],physics_steps=0,render_calls=0,model_loads=0))
        recorded = True

    try:
        install_deadline(root,stage)
        ledger = json.loads((root/'RESOURCE_ACCOUNTING.json').read_text())
        _, remaining = resource_bounds(root)
        # Parent plus the frozen qualifier's two workers share a conservative reserve.
        per_process_limit = int((remaining-600)/3)
        assert per_process_limit > cpu()+1
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU,(per_process_limit,hard))
        for signum in (signal.SIGALRM,signal.SIGXCPU):
            previous = signal.getsignal(signum)
            def stop(number,frame,handler=previous):
                # Only workers created by this invocation are eligible for cleanup.
                for child in multiprocessing.active_children():
                    child.terminate()
                    child.join(timeout=1)
                    if child.is_alive():
                        child.kill()
                        child.join(timeout=2)
                    cleanup.append(dict(pid=child.pid,exit_code=child.exitcode,
                        alive_after_cleanup=child.is_alive(),scope='THIS_INVOCATION_CHILD_ONLY'))
                save_usage('STOP_ORIGINAL_WALL_OR_CPU_BOUND')
                handler(number,frame)
            signal.signal(signum,stop)
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-3:]))
        sys.path.insert(0,str(script.parent))
        sys.argv = [str(script)]+arguments
        runpy.run_path(str(script),run_name='__main__')
        status = 'FROZEN_STAGE_FINISHED_CHECK_ACTUAL_RECEIPT'
    finally:
        save_usage(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--stage',choices=('audit_data','export_dataset','qualify_loader'),required=True)
    parser.add_argument('--stage-receipt')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    run(args.root,args.stage,arguments,args.stage_receipt)
