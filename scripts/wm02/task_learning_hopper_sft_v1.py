"""An already registered single-H200 SFT allocation; no qsub or physics."""
import argparse,json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.task_learning_hopper_bootstrap_v1 import run
    run(json.loads(a.config.read_text()))
    return 0


if __name__=='__main__':raise SystemExit(main())
