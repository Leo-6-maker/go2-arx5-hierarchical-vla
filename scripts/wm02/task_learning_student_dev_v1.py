"""Run one already registered serial B0/SFT native DEV group."""
import argparse
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import run
    return run(p.parse_args(argv).input)


if __name__=='__main__':raise SystemExit(main())
