"""Settle a completed auxiliary H200 job beside the training ledger without blocking it."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import tempfile

from hopper_pbs import charged, native
from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_parent_custody import digest, sync_dir, write_once


def settle(root, config_sha):
    if digest(root/'CONFIG.json') != config_sha:
        raise ValueError('Unrecognized diagnostic configuration')
    cfg = json.loads((root/'CONFIG.json').read_bytes())
    ticket = json.loads((root/'PBS_JOB.json').read_bytes())
    intent = json.loads((root/'INTENT.json').read_bytes())
    if (cfg['site'] != 'hopper' or cfg['uid'] != os.getuid() or intent['config'] != config_sha
            or ticket['config'] != config_sha or intent['seconds'] != cfg['wall_seconds']):
        raise ValueError('Unbound project allocation')
    job = ticket['job']
    line = Path(cfg['lineage'])
    with closing(OwnerLock(line/'submit.lock')):
        row = native(job)
        if row['job_state'] != 'F':
            return dict(job=job,state=row['job_state'],settled=False)
        reservation = dict(job=job,group='deployment_diagnostic',reserved_seconds=cfg['wall_seconds'])
        seconds = charged(row,os.environ['USER'],reservation)
        if row.get('stime') and row.get('Exit_status') is None:
            raise ValueError('Final exit status pending')
        receipt = dict(job=job,group=reservation['group'],exit=row.get('Exit_status'),
                       GPU_seconds=seconds,pbs_used_wall=row.get('resources_used',{}).get('walltime'),
                       diagnostic_config=config_sha,no_allocation_start=not bool(row.get('stime')))
        dst = line/'allocations'/('diagnostic-'+job.split('.')[0])
        if dst.exists():
            if json.loads((dst/'PBS_FINISHED.json').read_bytes()) != receipt:
                raise ValueError('Conflicting site cost')
        else:
            # Stage outside allocations; the production budget reader only sees a complete settlement.
            tmp = Path(tempfile.mkdtemp(prefix='.numeric-settlement-',dir=line))
            write_once(tmp/'INTENT.json',intent)
            write_once(tmp/'RESERVATION.json',reservation)
            write_once(tmp/'PBS_FINISHED.json',receipt)
            os.rename(tmp,dst)
            sync_dir(dst.parent)
        return dict(job=job,settled=True,GPU_seconds=seconds,sha256=digest(dst/'PBS_FINISHED.json'))


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('diagnostic_root',type=Path)
    cli.add_argument('config_sha')
    args=cli.parse_args()
    print(json.dumps(settle(args.diagnostic_root,args.config_sha)))
