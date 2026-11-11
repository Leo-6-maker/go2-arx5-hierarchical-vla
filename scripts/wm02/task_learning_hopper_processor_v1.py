"""Run original-Processor verification in one site-supported single-H200 PBS allocation; data stays on CPU."""
import argparse,json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    args=p.parse_args(argv)
    from go2arx5_vla.world_model.task_learning_hopper_processor_v1 import run
    return run(json.loads(args.config.read_bytes()))


if __name__=='__main__':raise SystemExit(main())
