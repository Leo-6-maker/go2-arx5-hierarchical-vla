"""Continue the original private snapshot inbox on succeeding VANDA CPU allocations."""
from contextlib import closing
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.sla.hopper_parent_custody import PLAN, digest, write_once
from scripts.sla.hopper_site import DEADLINE, pinned
from scripts.sla.hopper_snapshot_relay import transfer


def tick(root, number):
    config = read(root/'CONFIG.json')
    link = config['local_relay']
    cfg = pinned(link['mount'])
    slot = root/f'{number:02d}'
    start = read(slot/'START.json')
    job = os.environ['PBS_JOBID']
    if (not start['job']==read(slot/'JOB.json')['job']==job or cfg['plan']!=PLAN
            or cfg['job']!=link['job'] or cfg['T0_epoch']+14*86400!=cfg['deadline_epoch']
            or cfg['project']!='CFP04-CF-007' or cfg['user']!=os.environ['USER']):
        raise PermissionError('Wrong snapshot grant, project or CPU owner')
    source = Path(cfg['vanda_source'])
    if digest(source/'code.sha256')!=cfg['code_sha256']:
        raise ValueError('Original private receiver source changed')
    subprocess.run(['sha256sum','--status','-c','code.sha256'], cwd=source, check=True)
    subprocess.run(['sha256sum','--status','-c','SOURCE.sha256'], cwd=root, check=True)
    previous = Path(link['mount']['path']).parent
    end = min(DEADLINE,config['grant_end'],cfg['deadline_epoch'],start['site_end']-120)
    if (previous/'STOP.json').exists() or (root/'STOP').exists() or time.time()>=end:
        return dict(status='STOP_LIMIT_OR_DEADLINE')
    receipts = Path(cfg['receipt_root'])
    with closing(OwnerLock(receipts/'relay.lock')):
        for folder in sorted(Path(cfg['vanda_inbox']).glob('TRAIN-[0-9][0-9][0-9]')):
            seal = digest(folder/'SEALED.private.json')
            path = receipts/(folder.name+'.json')
            if path.exists():
                if read(path)['seal_sha256']!=seal:
                    raise ValueError('Previously transferred Core seal changed')
                continue
            result = transfer(folder,source/'population.json',dict(cfg,deadline_epoch=min(end,time.time()+3000)),seal)
            write_once(path,result)
            return dict(status='COPIED',count=result['count'],windows=result['windows'],
                        bytes=result['bytes'],seal_sha256=seal)
    return dict(status='WAIT_SEAL',received=len(list(receipts.glob('TRAIN-*.json'))))


if __name__=='__main__':
    import json
    print(json.dumps(tick(Path(sys.argv[1]),int(sys.argv[2]))))
