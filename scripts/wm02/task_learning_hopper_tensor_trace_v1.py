"""Run one already registered PBS-owned H200 inference trace."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.task_learning_hopper_tensor_trace_v1 import run
    return run(json.loads(args.config.read_bytes()))


if __name__ == '__main__': raise SystemExit(main())
