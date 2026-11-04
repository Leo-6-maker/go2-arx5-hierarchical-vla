"""Audit pinned original S records and build numeric causal replay windows (no physics)."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prediction-atoms',type=int,choices=(8,16),default=8);a=p.parse_args(argv)
    from go2arx5_vla.world_model.replay import build
    print(json.dumps(build(a.manifest,a.output,prediction_atoms=a.prediction_atoms)));return 0

if __name__=='__main__':raise SystemExit(main())
