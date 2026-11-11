"""Finite current-observation queries for one registered original native world."""
import argparse
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    from go2arx5_vla.world_model.task_learning_live_pair_v1 import PROFILE,infer
    with admitted_run(args.spec,args.output.name,PROFILE,args.output,gpu=True)as spec:
        infer(spec,args.output)
    return 0


if __name__=='__main__':raise SystemExit(main())
