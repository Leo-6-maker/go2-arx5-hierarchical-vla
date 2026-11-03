"""Bounded native S-domain1/4/16 process profile with actual RAM/CPU limits."""
import argparse
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in('spec','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in('run-id','profile'):p.add_argument('--'+name,required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    from go2arx5_vla.world_model.parallel_source import profile
    with admitted_run(a.spec,a.run_id,a.profile,a.output,physics=True)as spec:profile(spec,a.output)
    return 0


if __name__=='__main__':raise SystemExit(main())
