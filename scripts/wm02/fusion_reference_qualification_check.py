"""Check or execute ONE newly registered reference qualification via the old worker.

Default is a read-only recipe/ancestor check. --execute reuses the actual
PILOT1 worker, AUTO1/P1 admission, original locks and complete native closeout.
This command does not create a scope, registry entry, lease or site permission.
"""
import argparse
import json
from pathlib import Path


def check(spec_path):
    from go2arx5_vla.world_model.fusion_reference_qualification_contract import (
        episode_config, episode_limit, END_STEP)
    from go2arx5_vla.world_model.storage import digest
    spec = json.loads(Path(spec_path).read_text())
    path = Path(spec['previous_stage'])
    if digest(path) != spec['previous_stage_sha256']:
        raise ValueError('REFERENCE_QUALIFICATION_PREVIOUS_NATIVE_BYTES_CHANGED')
    previous = json.loads(path.read_text())
    episode = episode_config(spec)
    episode_limit(spec, previous)
    return dict(status='RECIPE_AND_ANCESTOR_CHECKED_ACTUAL_ADMISSION_STILL_REQUIRED',
        episode=episode, max_ticks=END_STEP, new_integrations=0, VLA_inferences=0,
        native_execution_authorized=False, existing_worker='scripts.wm02.pilot1_native_worker',
        required_original_admission='ACTIVE_AUTO1_AND_PILOT1_OWNER_REGISTRY_LEASE_AND_LOCKS',
        old_FUSION_fixed_XY_binding_reused=False, native_qualification_claim=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    result = check(args.spec)
    if not args.execute:
        print(json.dumps(result, sort_keys=True)); return 0
    from scripts.wm02.pilot1_native_worker import main as original_worker
    return original_worker(['--spec', str(args.spec)])


if __name__ == '__main__':
    raise SystemExit(main())
