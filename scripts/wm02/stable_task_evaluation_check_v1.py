"""Read-only learned-candidate new-scene DEV admission; original native worker executes."""
import argparse
import json
from pathlib import Path


def check(path):
    from go2arx5_vla.world_model.stable_task_evaluation_v1 import episode_limit
    spec=json.loads(Path(path).read_bytes())
    end=episode_limit(spec,json.loads(Path(spec['previous_stage']).read_bytes()))
    return dict(status='NEW_STABLE_TASK_ADAPTER_EVALUATION_CHECKED_NOT_EXECUTED',max_ticks=end,
        registered_final_step=end,new_integrations=0,new_queries=0,new_updates=0,
        native_execution_authorized=False,data_role=spec['data_role'],data_parent=spec['data_parent'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv);result=check(a.spec)
    if not a.execute:
        print(json.dumps(result));return 0
    from scripts.wm02.pilot1_native_worker import main as worker
    return worker(['--spec',str(a.spec)])


if __name__=='__main__':raise SystemExit(main())
