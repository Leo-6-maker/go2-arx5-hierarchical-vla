"""Train one actually admitted independent target TASK_ADAPTER candidate."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    from go2arx5_vla.world_model.stable_task_training_v1 import PROFILE,run
    with admitted_run(a.spec,a.output.name,PROFILE,a.output,physics=False,gpu=True)as spec:
        result=run(spec,a.output)
    print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
