"""Check the complete delivered runtime and real parent state inside a CPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.sla.hopper_cold_restore import restore
from scripts.sla.hopper_parent_custody import PLAN, write_once


def run(root):
    cfg=json.loads((root/'MOUNT.private.json').read_bytes())
    job=os.environ.get('PBS_JOBID')
    if job!=cfg['job'] or cfg['plan']!=PLAN or time.time()>=cfg['deadline_epoch'] or (root/'STOP').exists():
        raise PermissionError('Wrong CPU allocation')
    jobs=json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',job],text=True,timeout=15))['Jobs']
    active=jobs[job]
    if (active['job_state']!='R' or active['project']!=cfg['project']
            or int(active['Resource_List'].get('ngpus',0))!=0 or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
        raise PermissionError('Expected a live CPU-only VANDA allocation for this project')
    import jax
    if jax.default_backend()!='cpu' or len(jax.devices())!=1:
        raise PermissionError('Unexpected compute backend')
    start=time.monotonic()
    readback=restore(root/'inputs',cfg['bank_sha256'],check_device_arrays=True)[-1]
    if readback['restored_arrays']!=140 or readback['device_readback_exact']!=140:
        raise ValueError('Incomplete source-side restore')
    result=dict(status='SOURCE_CPU_COLD_RESTORE_EXACT',scope='VANDA CPU using sealed runtime and parent',
                receiver_device_restore=False,training_ready=False,job=job,project=active['project'],
                source_revision=cfg['source_revision'],runtime_archive_sha256=cfg['runtime_archive_sha256'],
                wall_s=time.monotonic()-start,**readback)
    write_once(root/'RESULT.private.json',result)


if __name__=='__main__':
    import fcntl
    root=Path.cwd()
    with (root/'OWNER.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(root)
        except Exception as error:
            write_once(root/'FAILURE.private.json',dict(job=os.environ.get('PBS_JOBID'),error=str(error)))
            raise
