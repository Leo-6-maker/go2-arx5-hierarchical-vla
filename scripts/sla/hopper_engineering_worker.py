"""One isolated training, ordinary, or independent-restore process under shared PBS custody."""
import argparse
import os
from pathlib import Path
import signal
import time

from scripts.pi0292.durable import OwnerLock
from scripts.sla import hopper_checkpoint as checkpoint, hopper_engineering as engineering
from scripts.sla.hopper_cold_restore import restore
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import live_h200, pinned


def worker(run, mount_sha, candidate, action):
    mount = pinned(dict(path=str(run/'MOUNT.private.json'), sha256=mount_sha))
    group = pinned(mount['engineering_group'])
    if (group['candidate'] != candidate or group['plan'] != checkpoint.PLAN or group['parent'] != checkpoint.PARENT
            or group['job'] != mount['job'] or group['max_updates'] != 16 or group['max_panel_per_pass'] != 48
            or group['source'] != mount['source_revision']):
        raise PermissionError('Not the pinned engineering group or phase')
    site = live_h200(mount, 'engineering')
    gpu = {k: site[k] for k in ('host', 'gpu_uuid', 'gpu_name')}
    root, data = Path(mount['custody']), Path(mount['eval_assets'])
    fm, aux = engineering.backend(candidate, group.get('fm_outcomes', {}))
    folder = Path(group['root'])
    if folder.resolve(strict=True).parent != Path(mount['engineering_root']).resolve(strict=True):
        raise ValueError('Candidate storage lies outside its engineering scope')
    flag = False
    def stop(*_):
        nonlocal flag
        flag = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    def guard():
        if flag or (run/'STOP.json').exists() or (root/'STOP.json').exists() or time.time() >= site['deadline'] - 900:
            raise TimeoutError('Engineering stop or allocation save margin')
    guard()
    lock = OwnerLock(Path(mount['engineering_root'])/'GPU_OWNER.lock')
    try:
        guard()
        out = folder/action
        out.mkdir()
        import jax
        jax.config.update('jax_compilation_cache_dir', mount['compilation_cache_dir'])
        if action == 'train':
            restored = restore(root, checkpoint.BANKS, check_device_arrays=True)
            bound = checkpoint.binding(restored, role=candidate, source=mount['source_revision'], fm_micro=fm, aux_micro=aux)
            write_once(folder/'SITE.private.json', dict(site=site, mount_sha256=mount_sha, source=bound['source'],
                       parent=checkpoint.PARENT, role=candidate, readback_arrays=restored[-1]['device_readback_exact']))
            reference = None
            if candidate != 'E1':
                pin = group['first_step_reference']
                if not pinned(pin)['passed'] or Path(pin['path']).name != 'COMPLETE.json':
                    raise ValueError('Missing successful E1 reference pass')
                reference = Path(pin['path']).parent
            return engineering.train(out, restored, bound, guard, reference)
        train = folder/'train/TRAIN.private.json'
        train_pin = dict(path=str(train), sha256=digest(train))
        prior = engineering.load_pin(train_pin)
        orig_site = pinned(dict(path=str(folder/'SITE.private.json'), sha256=digest(folder/'SITE.private.json')))
        if (prior['role'] != candidate or prior['binding']['source'] != mount['source_revision'] or
                any(orig_site['site'][k] != gpu[k] for k in gpu)):
            raise ValueError('This candidate was trained on a different device or source')
        if action == 'ordinary':
            return engineering.outputs(out, train_pin, root, data, site=gpu, guard=guard)
        if action != 'restore':
            raise ValueError('Unknown engineering action')
        ref = folder/'ordinary/RESULT.private.json'
        return engineering.outputs(out, train_pin, root, data, site=gpu, guard=guard,
                                   independent=dict(path=str(ref), sha256=digest(ref)))
    finally:
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('mount_sha256')
    p.add_argument('candidate', choices=sorted(checkpoint.ENGINEERING))
    p.add_argument('action', choices=('train', 'ordinary', 'restore'))
    args = p.parse_args()
    os.umask(0o077)
    worker(args.run, args.mount_sha256, args.candidate, args.action)
