"""Reconcile the first pre-optimizer path failure and reproduce the verified VANDA directory binding."""
import argparse
import os
from pathlib import Path
import tempfile

from scripts.pi0292.durable import read
from scripts.sla.hopper_parent_custody import digest, write_once

FOUNDATION = Path('sla-d4-vanda-20261118-02/foundation')
WEIGHTS = Path('sla-d4-vanda-20261118-01/foundation/model')
MANIFEST = '232442c8a8c934118cbcdb890c6f307d7fc857819d7e45c2286ed8e8e5d5f3a3'
VIEW = 'foundation-layout-v2'


def project_tree(source, dest, branch):
    dest.mkdir()
    for original in source.iterdir():
        target = dest/original.name
        if branch and original.name == branch[0]:
            project_tree(original, target, branch[1:])
        elif original.is_dir():
            target.symlink_to(original.resolve(strict=True), target_is_directory=True)
        elif original.is_file() and not original.is_symlink():
            os.link(original, target)
        else:
            raise ValueError('Unexpected custody entry '+str(original))


def prepare(base, line):
    from go2arx5_vla.sla_adapter.scale20k_model import parent_asset
    from scripts.sla.hopper_inputs import inputs
    base = base.resolve(strict=True)
    old = base/'assets.verified'
    run = line/'allocations/001'
    group = line/'engineering/E1'
    log = run/'E1/train.log'
    original = read(run/'MOUNT.private.json')
    error = "FileNotFoundError: [Errno 2] No such file or directory: "+repr(str(old/FOUNDATION/'model'))
    if (original['custody'] != str(base) or read(run/'JOB.json')['job'] != original['job']
            or read(run/'E1/train.EXIT.json')['status'] == 0 or error != log.read_text().splitlines()[-1]
            or set(group.iterdir()) != {group/'train'} or any((group/'train').iterdir())
            or (run/'E1/ordinary.START.json').exists() or (run/'E1/restore.START.json').exists()
            or (line/'engineering/E1.outcome.json').exists()
            or read(run/'INTERRUPTION.private.json')['job'] != original['job']):
        raise ValueError('Recovery only covers the recorded empty E1 failure before model load')
    if digest(old/FOUNDATION/'BYTE_MANIFEST.private.json') != MANIFEST:
        raise ValueError('Original manifest changed')
    view = base/VIEW
    if not view.exists():
        candidate = Path(tempfile.mkdtemp(prefix='.layout-', dir=base))
        for name in ('parent.verified','banks.verified','sla-core-hopper-v1-runtime-20261121-01',
                     'sla-core-hopper-v1-eval-inputs-20261121-01','train-ready','STOP.json','STOP'):
            (candidate/name).symlink_to(base/name)
        project_tree(old, candidate/'assets.verified', FOUNDATION.parts)
        (candidate/'assets.verified'/FOUNDATION/'model').symlink_to((old/WEIGHTS).resolve(strict=True))
        candidate.rename(view)
    spec = inputs(view/'assets.verified')
    asset = parent_asset(spec['parent_manifest'], spec['parent_params'])
    if (digest(Path(spec['parent_manifest']['path'])) != MANIFEST or
            Path(spec['parent_params']).resolve(strict=True) != (old/WEIGHTS/'params').resolve(strict=True)):
        raise ValueError('New foundation view points to different weights')
    record = dict(job=original['job'], plan=original['plan'], source=original['source_revision'],
                  error=error, train_log_sha256=digest(log), manifest=MANIFEST, revision=asset['revision'],
                  resolved_params=spec['parent_params'], inventory=read(base/'TRAIN_ASSET_BYTES_VERIFIED.private.json')['inventory_sha256'],
                  updates=0, optimization_visits=0, generations=0, native=0, technical_reserve_groups=1,
                  custody=str(view), engineering_root=str(line/'engineering-recovery-01'))
    write_once(view/'TECHNICAL_RECOVERY.json',record)
    return view/'TECHNICAL_RECOVERY.json'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('base',type=Path)
    p.add_argument('line',type=Path)
    args = p.parse_args()
    print(prepare(args.base,args.line))
