"""Finite model-only real S replay diagnostic on the explicitly bound Core GPU1."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'wm02-gpu1-real-s',a.output,gpu=True)as spec:
        from go2arx5_vla.world_model.real_training import run
        print(json.dumps(run(a.output,spec)))
    return 0

if __name__=='__main__':raise SystemExit(main())
