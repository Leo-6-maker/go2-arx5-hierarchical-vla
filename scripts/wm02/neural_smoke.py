"""WM02 finite GPU200 + H16 synthetic engineering smoke; help starts no model or physics."""
import argparse
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--microbatch',type=int,choices=(8,16,32),default=32)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,a.profile,a.output,gpu=True)as spec:
        from go2arx5_vla.world_model.neural_training import run_smoke
        result=run_smoke(a.output,spec,a.microbatch)
    print(result['status']);return 0


if __name__=='__main__':raise SystemExit(main())
