"""Run A/B event evaluation in the frozen, qualified Core numerical import order."""
from argparse import ArgumentParser
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def fingerprint(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_verified(ref):
    path = Path(ref['path'])
    if fingerprint(path) != ref:
        raise ValueError(f'Frozen runtime identity changed: {path.name}')
    return json.loads(path.read_bytes())


def bind_runtime(binding):
    worker = read_verified(binding['worker'])
    expected = worker['gpu_environment']
    keys = ('PYTHONPATH','CUDA_VISIBLE_DEVICES','JAX_PLATFORMS','XLA_FLAGS',
            'XLA_PYTHON_CLIENT_MEM_FRACTION','LD_LIBRARY_PATH','LD_PRELOAD')
    if (sys.executable != worker['python']
            or any(os.environ.get(key,'') != expected.get(key,'') for key in keys)):
        raise PermissionError('Expected the qualified Core numerical interpreter and environment')
    source = Path(worker['source_root'])
    relative = 'scripts/sla/core_hopper_native_entry.py'
    entry = source/relative
    if fingerprint(entry)['sha256'] != worker['source_files'][relative]:
        raise ValueError('The frozen numerical bootstrap changed')
    runpy.run_path(str(entry))['bind_imports'](source)
    wm = import_module('scripts.wm02')
    wm.__path__ = [str(source/'scripts/wm02'),*wm.__path__]
    scope = import_module('scripts.sla')
    scope.__path__ = [str(Path(__file__).resolve().parent),str(source/'scripts/sla'),*scope.__path__]
    modules = {}
    for name, ref in binding['modules'].items():
        resolved = Path(import_module(name).__file__).resolve(strict=True)
        modules[name] = fingerprint(resolved)
        if modules[name] != ref:
            raise ValueError(f'Qualified numerical dependency changed: {name}')
    return modules


def main(root, wait_pin, preflight):
    wait = read_verified(wait_pin)
    expected = read_verified(wait['source_members'])
    source_root = Path(__file__).resolve().parents[2]
    if any(fingerprint(source_root/name)['sha256'] != sha for name,sha in expected['members'].items()):
        raise ValueError('The sealed evaluation source changed')
    modules = bind_runtime(read_verified(wait['numeric_binding']))
    from scripts.sla.hopper_parent_custody import write_once
    from scripts.sla.hopper_site import pinned
    from scripts.sla.main3800_event_eval import evaluate, preflight_inputs
    from scripts.sla.main3800_pair_eval_queue import release

    if preflight:
        config = pinned(wait['event_config'])
        panel = pinned(config['panel'])
        _,_,source,_,inputs = preflight_inputs(config,panel)
        proof = dict(status='FROZEN_IMPORTS_AND_ACTUAL_INPUTS_VERIFIED',
                     runtime=wait['numeric_binding']['sha256'],wait=wait_pin['sha256'],
                     panel=config['panel']['sha256'],statistics=source['statistics']['sha256'],
                     modules=len(modules),windows=len(inputs),updates=0,generations=0,native_requests=0)
        write_once(root/'PREFLIGHT.private.json',proof)
        print(json.dumps(proof,sort_keys=True))
        return
    ready = pinned(wait['preflight'])
    if (ready['status'] != 'FROZEN_IMPORTS_AND_ACTUAL_INPUTS_VERIFIED'
            or ready['runtime'] != wait['numeric_binding']['sha256']
            or ready['panel'] != pinned(wait['event_config'])['panel']['sha256']
            or ready['windows'] != 40 or ready['modules'] != len(modules)
            or ready['generations'] or ready['updates'] or ready['native_requests']):
        raise ValueError('Missing the exact Core dependency and input preflight')
    cfg_pin = release(root,wait)
    if cfg_pin is not None:
        cfg = pinned(cfg_pin)
        evaluate(root,cfg,pinned(cfg['panel']))


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('wait_sha256')
    cli.add_argument('--preflight',action='store_true')
    args = cli.parse_args()
    os.umask(0o077)
    try:
        name = 'PREFLIGHT_SPEC.json' if args.preflight else 'WAIT.json'
        main(args.root,dict(path=str(args.root/name),sha256=args.wait_sha256),args.preflight)
    except Exception as exc:
        # The failure receipt remains available even for an import failure in
        # the numerical closure itself.
        if not args.preflight:
            result = dict(reason=f'{type(exc).__name__}: {exc}',
                attempts=len(list(args.root.glob('*/*.intent.json'))),
                generations=len(list(args.root.glob('*/*.generated.json'))),
                native_requests=0,updates=0,visits=0,epoch=time.time())
            path = args.root/'REVIEW_REQUIRED.json'
            with path.open('x') as handle:
                json.dump(result,handle,sort_keys=True,indent=2)
                handle.flush()
                os.fsync(handle.fileno())
        raise
