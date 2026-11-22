"""Stream the matched Go2-only pair between the authorized private site connections."""
from argparse import ArgumentParser
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
from tempfile import TemporaryDirectory

from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_parent_custody import digest, sync_dir, write_once
from scripts.sla.hopper_site import pinned
from scripts.sla.main3800_auxiliary_readback import ARMS, MAIN3800, verify_package


MAX_BYTES = 4 * 1024**3
PARAMETER_NAMES = {f'parameters/{n:06d}.npy' for n in range(140)}


def pair_receipt(root, sha):
    value = pinned(dict(path=str(root/'EXPORT.json'), sha256=sha))
    if (value['status'] != 'MATCHED_PAIR_RECONCILED' or set(value['arms']) != set(ARMS)
            or value['comparison_parent'] != MAIN3800 or value['matched_FM_updates'] != 500):
        raise ValueError('Expected the sealed 3.8K matched pair')
    return value['arms']


def verified_arm(root, arm, receipt):
    manifest = verify_package(root/arm, receipt['manifest_sha256'])
    if (manifest['arm'] != arm or manifest['checkpoint_sha256'] != receipt['training_state_sha256']
            or manifest['training_result_sha256'] != receipt['result_sha256']):
        raise ValueError(f'Inference package does not match the terminal training receipt for {arm}')
    return manifest


def pack(root, sha, output):
    receipts = pair_receipt(root, sha)
    files = ['EXPORT.json']
    for arm in ARMS:
        manifest = verified_arm(root, arm, receipts[arm])
        files.extend([f'{arm}/CHECKPOINT.json', *(f'{arm}/{row["file"]}' for row in manifest['arrays'])])
    if sum((root/name).stat().st_size for name in files) > MAX_BYTES:
        raise ValueError('The Go2 inference pair exceeds the transfer size limit')
    with tarfile.open(fileobj=output, mode='w|', dereference=True) as stream:
        for name in files:
            stream.add(root/name, arcname=name, recursive=False)


def receive(destination, sha, source):
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    total_bytes = 0
    members = 0
    with closing(OwnerLock(destination.with_suffix('.lock'))):
        if destination.exists():
            raise FileExistsError('The transfer already has a destination to reconcile')
        with TemporaryDirectory(prefix='.auxiliary-pair-', dir=destination.parent) as incoming:
            stage = Path(incoming)
            with tarfile.open(fileobj=source, mode='r|') as stream:
                def copy(name):
                    nonlocal total_bytes, members
                    item = stream.next()
                    if (item is None or item.name != name or not item.isfile()
                            or item.size < 1 or total_bytes + item.size > MAX_BYTES
                            or (name.endswith('.json') and item.size > 1024**2)):
                        raise ValueError(f'Missing or unexpected private transfer member: {name}')
                    with stream.extractfile(item) as src, (stage/name).open('xb') as dst:
                        shutil.copyfileobj(src, dst, length=4 * 1024**2)
                        dst.flush()
                        os.fsync(dst.fileno())
                    total_bytes += item.size
                    members += 1

                copy('EXPORT.json')
                receipts = pair_receipt(stage, sha)
                for arm in ARMS:
                    (stage/arm/'parameters').mkdir(parents=True)
                    copy(f'{arm}/CHECKPOINT.json')
                    manifest = pinned(dict(path=str(stage/arm/'CHECKPOINT.json'),
                                           sha256=receipts[arm]['manifest_sha256']))
                    names = [row['file'] for row in manifest['arrays']]
                    if len(names) != 29 or len(set(names)) != 29 or not set(names) <= PARAMETER_NAMES:
                        raise ValueError(f'Unexpected private parameter inventory for {arm}')
                    for name in names:
                        copy(f'{arm}/{name}')
                    verified_arm(stage, arm, receipts[arm])
                    sync_dir(stage/arm/'parameters')
                    sync_dir(stage/arm)
                if stream.next() is not None:
                    raise ValueError('Unexpected additional transfer members')
            result = dict(status='CORE_GO2_PAIR_VERIFIED', export_sha256=sha, inference_arrays=58,
                          files=members, bytes=total_bytes, updates=0, generations=0, native_requests=0,
                          arms={arm:receipts[arm]['manifest_sha256'] for arm in ARMS})
            write_once(stage/'DELIVERY.private.json', result)
            stage.rename(destination)
            sync_dir(destination.parent)
    return result


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=('pack','receive'))
    cli.add_argument('directory', type=Path)
    cli.add_argument('export_sha256')
    args = cli.parse_args()
    os.umask(0o077)
    if args.mode == 'pack':
        pack(args.directory, args.export_sha256, sys.stdout.buffer)
    else:
        print(json.dumps(receive(args.directory, args.export_sha256, sys.stdin.buffer), sort_keys=True))
