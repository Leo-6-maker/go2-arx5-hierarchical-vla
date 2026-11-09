"""Actual ordinary-parent Core reconstruction on forty fixed real decisions."""
import argparse
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    from go2arx5_vla.world_model.task_learning_core_inference_v1 import PROFILE
    from go2arx5_vla.world_model.target_gr00t_inference_v1 import infer
    with admitted_run(a.spec,a.output.name,PROFILE,a.output,gpu=True)as s:infer(s,a.output)
    return 0


if __name__=='__main__':raise SystemExit(main())
