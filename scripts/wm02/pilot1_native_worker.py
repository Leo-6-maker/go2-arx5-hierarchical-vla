"""Run one admitted native PILOT1 episode; preserve the original safety/close path."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);args=parser.parse_args(argv)
    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.storage import digest
    spec=json.loads(args.spec.read_text());root=Path(spec['registered_root']);lease_path=Path(spec['lease']);lease=json.loads(lease_path.read_text())
    if lease.get('status')!='ACTIVE'or lease.get('run_uuid')!=root.name or lease.get('spec_sha256')!=digest(args.spec):raise PermissionError('EXACT_FRESH_PILOT1_NATIVE_WORKER_REQUIRED')
    if(root/'SUPERVISOR.private.json').exists():raise FileExistsError('EXISTING_OR_UNKNOWN_NATIVE_WORKER_MUST_BE_RECONCILED')
    if spec.get('corelab_world_binding_v3') or spec.get('enforce_cpu_slot_v5') is True:
        # Apply the registered CPU slot to this process and its native children;
        # a user service's cgroup CPU property alone may not be delegated.
        cpus=set(spec['cpus'])
        if not cpus or not cpus<=os.sched_getaffinity(0):raise PermissionError('REGISTERED_WORLD_CPU_SLOT_REQUIRED')
        os.sched_setaffinity(0,cpus)
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),source_sha=spec['source_sha'],started_epoch=time.time(),actual_cpu_affinity=sorted(os.sched_getaffinity(0))))
    def interrupted(signum,frame):raise KeyboardInterrupt('OWNED_NATIVE_WORKER_SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    start=time.monotonic();cpu=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN);code=1;failure=None
    try:
        from scripts.wm02.commission_target import main as commission
        code=commission(['--spec',str(args.spec),'--output',str(root/'native-01'),'--run-id','native-01'])
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        write(root/'WORKER_EXIT.private.json',dict(exit_code=code,failure=failure,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,child_CPU_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
        current=json.loads(lease_path.read_text())
        if current.get('status')!='ACTIVE'or current.get('spec_sha256')!=digest(args.spec):raise PermissionError('NATIVE_WORKER_LEASE_CHANGED')
        current.update(status='CLOSED',closed_epoch=time.time(),worker_exit_code=code);write(lease_path,current)
    return code


if __name__=='__main__':raise SystemExit(main())
