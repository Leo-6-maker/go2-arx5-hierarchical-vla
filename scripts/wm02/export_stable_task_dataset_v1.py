"""CPU-only export of sealed actual task data; no physics or model is started."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.stable_task_dataset_v1 import build
    v=build(a.panel,a.runs_root,a.output)
    print(json.dumps(dict(status=v['status'],eligible=v['actual_eligible_episodes'],
        failed_or_unlabelled=v['actual_failed_or_unlabelled_episodes'],missing=len(v['missing']),
        new_physics=0,new_updates=0)))
    return 0


if __name__=='__main__':raise SystemExit(main())
