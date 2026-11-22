"""Queue the fixed Corelab A/B event evaluation behind verified private model delivery."""
from argparse import ArgumentParser
from contextlib import closing
import os
from pathlib import Path
import sys
import time

from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.hopper_parent_custody import write_once
from scripts.sla.hopper_site import pinned
from scripts.sla.main3800_auxiliary_transfer import pair_receipt
from scripts.sla.main3800_cross_layout_proof import pin


def release(root, wait):
    delivery = Path(wait['delivery'])
    original = pinned(wait['event_config'])
    native = pinned(original['native_config'])
    site = pinned(native['expert'])
    if (root != Path(wait['root']) or not os.environ.get('INVOCATION_ID')
            or os.getuid() != site['uid'] or os.uname().nodename != site['host']
            or set(wait['jobs']) != {'A','B'}):
        raise PermissionError('Expected the Corelab service for the registered Hopper pair')
    export = delivery/'EXPORT.json'
    receipt = delivery/'DELIVERY.private.json'
    with closing(OwnerLock(root/'delivery-owner.lock')):
        while time.time()+1800 < original['deadline'] and not (root/'STOP').exists():
            if delivery.exists():
                verified = pinned(pin(receipt))
                export_pin = pin(export)
                arms = pair_receipt(delivery, export_pin['sha256'])
                if (verified['status'] != 'CORE_GO2_PAIR_VERIFIED' or verified['export_sha256'] != export_pin['sha256']
                        or verified['inference_arrays'] != 58 or verified['files'] != 61 or verified['updates']
                        or verified['generations'] or verified['native_requests']
                        or {k:v['job'] for k,v in arms.items()} != wait['jobs']
                        or {k:v['manifest_sha256'] for k,v in arms.items()} != verified['arms']):
                    raise ValueError('Private delivery does not match the registered sealed pair')
                cfg = root/'CONFIG.json'
                write_once(cfg,dict(original,auxiliary_export=export_pin))
                return pin(cfg)
            write(root/'HEARTBEAT.json',dict(epoch=time.time(),pid=os.getpid(),state='WAIT_FOR_VERIFIED_AB_DELIVERY'))
            time.sleep(30)
        write_once(root/'TERMINAL.json',dict(status='NO_CAPACITY_OR_STOP',reason='Verified pair delivery not released',
                   attempts=0,generations=0,native_requests=0,updates=0,visits=0,epoch=time.time()))
    return None


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('wait_sha256')
    args = cli.parse_args()
    os.umask(0o077)
    try:
        cfg = release(args.root,pinned(dict(path=str(args.root/'WAIT.json'),sha256=args.wait_sha256)))
    except Exception as exc:
        write_once(args.root/'REVIEW_REQUIRED.json',dict(reason=f'{type(exc).__name__}: {exc}',attempts=0,
                   generations=0,native_requests=0,updates=0,visits=0,epoch=time.time()))
        raise
    if cfg:
        os.execv(sys.executable,[sys.executable,'-B','-m','scripts.sla.main3800_event_eval',str(args.root),cfg['sha256']])
