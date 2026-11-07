"""Read-only check for one response cell; --execute uses the original P1 worker."""
import argparse
import json
from pathlib import Path


def check(spec_path):
    from go2arx5_vla.world_model.fusion_trajectory_response_native_contract import episode_config, episode_limit
    spec = json.loads(Path(spec_path).read_text())
    previous = json.loads(Path(spec['previous_stage']).read_text())
    episode, _, _ = episode_config(spec)
    limit = episode_limit(spec, previous)
    return dict(status='RESPONSE_CELL_RECIPE_CHECKED_ACTUAL_ORIGINAL_ADMISSION_STILL_REQUIRED',
        episode=episode, max_ticks=limit, native_execution_authorized=False,
        native_qualification_claim=False, new_integrations=0, VLA_queries=0,
        collection=spec['trajectory_response_collection'],
        pilot_state_theta_identifiability='REQUIRES_ACTUAL_DATA_NOT_ESTABLISHED_BY_RECIPE',
        existing_worker='scripts.wm02.pilot1_native_worker')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv); result = check(args.spec)
    if not args.execute:
        print(json.dumps(result, sort_keys=True)); return 0
    from scripts.wm02.pilot1_native_worker import main as original_worker
    return original_worker(['--spec', str(args.spec)])


if __name__ == '__main__':
    raise SystemExit(main())
