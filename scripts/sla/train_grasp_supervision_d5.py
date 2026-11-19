"""Run a fixed D5 arm or its independent ordinary-RGB restore."""
import argparse
from importlib.metadata import version
from pathlib import Path
import traceback

from scripts.pi0292.durable import read, write
from scripts.sla.train_multisource_d4 import restore_rgb, train
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.grasp_supervision_d5 import Engine, Provider, export, verify_history
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.site_v2 import validate

DEPENDENCY_LOCK = '31888931603422e1451e25b909dd615db459f6d2d998632444345933f6dcc82f'


def preflight(spec, site):
    verify_history(spec)
    if site['GPU_name'] != 'NVIDIA A40':
        raise ValueError('D5_REQUIRES_MATCHED_HISTORICAL_A40_OPERATOR_ENVIRONMENT')
    dependency = spec['historical_dependency_lock']
    if dependency['sha256'] != DEPENDENCY_LOCK or sha(dependency['path']) != DEPENDENCY_LOCK:
        raise ValueError('D5_HISTORICAL_OPERATOR_DEPENDENCY_LOCK_CHANGED')
    versions = dict(line.split('==', 1) for line in Path(dependency['path']).read_text().splitlines() if line)
    versions['torch'] = '2.7.1+cpu'
    if any(version(name) != value for name, value in versions.items()):
        raise ValueError('D5_DEPENDENCY_VERSION_DIFFERS_FROM_HISTORICAL_B')
    from scripts.wm02.sla_followup_population_v1 import validate_frozen

    validate_frozen(read_pin(spec['sealed_followup_population']))
    return dict(runtime_packages_verified=len(versions), GPU=site['GPU_name'])


def run(spec, *, restore):
    root = Path(spec['registered_root'])
    output = root/('RESTORE.private.json' if restore else 'RESULT.private.json')
    if output.exists():
        raise FileExistsError('D5_PRESERVE_EXISTING_TERMINAL')
    try:
        site = validate(spec)
        proof = preflight(spec, site)
        actual = (restore_rgb(spec, provider_type=Provider) if restore
                  else train(spec, engine_type=Engine, export_package=export))
        actual.update(operator_environment=proof, site=site, arm=spec['arm'])
    except Exception:
        actual = dict(exit_code=1, arm=spec['arm'], status='D5_TECHNICAL_TERMINAL_NO_AUTO_REPLAY',
                      failure=traceback.format_exc(), new_native_requests=0)
    write(output, actual)
    return actual['exit_code']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'restore'))
    parser.add_argument('execution_spec', type=Path)
    args = parser.parse_args()
    raise SystemExit(run(read(args.execution_spec), restore=args.mode == 'restore'))
