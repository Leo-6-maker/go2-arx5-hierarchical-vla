"""Actual finite A40 target GR00T entry; caller owns PBS/registration/lease."""
import argparse,json,os,time,traceback
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    a=p.parse_args(argv);s=json.loads(a.spec.read_bytes())
    from go2arx5_vla.world_model.target_gr00t_training_v1 import site,run
    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.storage import digest
    root=Path(s['registered_root']);result=None;failure=None;exit_code=1
    lease_path=root/'control/LEASE.json';original=json.loads(lease_path.read_bytes())
    if (os.getuid()!=22182 or not os.environ.get('PBS_JOBID')
            or a.spec.resolve()!=root/'EXECUTION_SPEC.json'
            or original.get('status')!='ACTIVE' or original.get('spec_sha256')!=digest(a.spec)
            or original.get('job_id')!=os.environ['PBS_JOBID']):
        raise PermissionError('ACTUAL_TARGET_WORKER_ENVELOPE_REQUIRED')
    if (root/'WORKER_STARTED.private.json').exists():
        raise FileExistsError('RECONCILE_UNKNOWN_TARGET_WORKER_NEVER_RELAUNCH')
    write(root/'WORKER_STARTED.private.json',dict(pid=os.getpid(),job_id=os.environ['PBS_JOBID'],
        source_sha=s['source_sha'],started_epoch=time.time(),automatic_retry=False))
    started=time.monotonic()
    try:
        result=run(s);exit_code=0
    except BaseException:
        failure=traceback.format_exc();raise
    finally:
        progress=json.loads((root/'PROGRESS.json').read_bytes())if(root/'PROGRESS.json').exists()else{}
        write(root/'RESULT.json',result or dict(status='QUALIFICATION_FAILED',failure=failure,
            actual_progress=progress,GR00T_updates=progress.get('GR00T_updates',0),WM_updates=0,Actor_updates=0,
            VLA_query_attempts=len(list(root.glob('*/journal/Q*.json')))))
        write(root/'WORKER_EXIT.private.json',dict(exit_code=exit_code,wall_s=time.monotonic()-started,
            source_sha=s['source_sha'],job_id=os.environ['PBS_JOBID'],failure=failure))
        lease=json.loads(lease_path.read_bytes())
        if lease.get('status')!='ACTIVE'or lease.get('spec_sha256')!=digest(a.spec):
            raise PermissionError('TARGET_GR00T_ACTUAL_OWNER_LEASE_CHANGED')
        lease.update(status='CLOSED',worker_exit_code=exit_code,closed_epoch=time.time(),
            result_sha256=digest(root/'RESULT.json'))
        write(lease_path,lease)
    return exit_code

if __name__=='__main__':raise SystemExit(main())
