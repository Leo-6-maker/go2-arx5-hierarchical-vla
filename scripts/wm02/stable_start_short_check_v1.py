"""Read-only stable-start contract check; execute uses the original worker."""
import argparse
import json
from pathlib import Path


def check(path):
    from go2arx5_vla.world_model.stable_start_short_task_v1 import episode_limit, B
    spec = json.loads(Path(path).read_bytes())
    end = episode_limit(spec,json.loads(Path(spec['previous_stage']).read_bytes()))
    return dict(status='STABLE_START_CONTRACT_CHECKED_NOT_EXECUTED',
        initialization_boundary=B,registered_final_step=end,max_ticks=end,
        native_execution_authorized=False,new_integrations=0,new_queries=0,new_updates=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a = p.parse_args(argv); value = check(a.spec)
    if not a.execute:
        print(json.dumps(value));return 0
    from scripts.wm02.pilot1_native_worker import main as worker
    return worker(['--spec',str(a.spec)])


if __name__ == '__main__':
    raise SystemExit(main())
