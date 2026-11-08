"""Two fresh observations through one registered frozen resident GR00T model."""
import argparse
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    from go2arx5_vla.world_model.target_gr00t_live_pair_v1 import PROFILE,infer
    with admitted_run(a.spec,a.output.name,PROFILE,a.output,gpu=True)as s:infer(s,a.output)
    return 0


if __name__=='__main__':raise SystemExit(main())
