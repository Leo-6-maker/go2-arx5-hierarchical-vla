"""Seal one actual full-population endpoint decision through the original owner."""
import argparse,json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.task_learning_endpoint_evidence_v2 import seal
    print(json.dumps(seal(json.loads(a.request.read_bytes()))));return 0


if __name__=='__main__':raise SystemExit(main())
