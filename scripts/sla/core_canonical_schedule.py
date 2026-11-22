"""Release each registered Core IPC panel only after its own cold proof."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.sla.core_canonical_cold import INITIAL
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import pinned


def coordinate(root,sha):
    cfg=pinned(dict(path=str(root/'CONFIG.json'),sha256=sha))
    if not os.environ.get('INVOCATION_ID') or cfg['uid']!=os.getuid():
        raise PermissionError('Canonical IPC requires the original managed owner')
    cold=Path(cfg['cold_root'])
    specs=cfg['ipc']
    write_once(root/'OWNER.json',dict(pid=os.getpid(),unit=os.environ['INVOCATION_ID'],epoch=time.time(),spec=sha))
    status={}
    for arm in INITIAL:
        proof=cold/arm/'COLD_COMPARISON.json'
        term=cold/'COLD_TERMINAL.json'
        while not term.is_file() and time.time()<cfg['deadline']-660:
            if (root/'STOP').exists():
                break
            time.sleep(10)
        if ((root/'STOP').exists() or not term.is_file() or not proof.is_file()
                or json.loads(proof.read_bytes())['status']!='COLD_MATCH'):
            status[arm]='NOT_RUN_NO_SEALED_MATCH'
            continue
        pin=specs[arm]
        spec=pinned(pin)
        if (spec['deadline']!=cfg['deadline'] or Path(spec['cold_config']['path']).parent!=cold):
            raise ValueError('IPC specification differs from the registered stage')
        work=Path(pin['path']).parent
        with (work/'supervisor.log').open('xb') as log:
            command=[sys.executable,'-B','-m','scripts.sla.core_canonical_ipc',str(work),pin['sha256'],arm]
            ran=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,cfg['deadline']-time.time()))
        status[arm]=dict(exit=ran.returncode,attempts=len(list((work/'worker').glob('*/IPC_INTENT.json'))),
                         result=digest(work/'worker/IPC_COMPARISON.json') if ran.returncode==0 else None)
    write_once(root/'TERMINAL.json',dict(arms=status,updates=0,native_requests=0,epoch=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('config_sha')
    args=parser.parse_args()
    os.umask(0o077)
    coordinate(args.root,args.config_sha)
