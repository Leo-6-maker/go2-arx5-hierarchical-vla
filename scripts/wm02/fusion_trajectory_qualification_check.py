"""Check one new finite trajectory recipe; --execute uses the original worker.

Default is read-only and creates no scope, registry entry, lease or grant.
"""
import argparse
import json
from pathlib import Path


def check(spec_path):
    from go2arx5_vla.world_model.fusion_trajectory_qualification_contract import (
        episode_config, episode_limit)
    spec = json.loads(Path(spec_path).read_text())
    previous = json.loads(Path(spec['previous_stage']).read_text())
    episode, _ = episode_config(spec)
    limit = episode_limit(spec, previous)
    return dict(status='RECIPE_AND_ANCESTOR_CHECKED_ACTUAL_ADMISSION_STILL_REQUIRED',
        episode=episode, max_ticks=limit, new_integrations=0, VLA_inferences=0,
        native_execution_authorized=False, native_qualification_claim=False,
        existing_worker='scripts.wm02.pilot1_native_worker',
        required_original_admission='ACTIVE_AUTO1_AND_PILOT1_OWNER_REGISTRY_LEASE_AND_LOCKS')


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
