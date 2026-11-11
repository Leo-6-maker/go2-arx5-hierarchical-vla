"""Seal the actual common DEV population from original closed serial groups."""
import argparse
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--panel-sha256',required=True)
    parser.add_argument('--group',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    from go2arx5_vla.world_model.task_learning_student_population_v2 import aggregate
    from go2arx5_vla.world_model.storage import write_json
    result=aggregate(dict(path=str(args.panel),sha256=args.panel_sha256),args.group)
    write_json(args.output,result)
    return 0


if __name__=='__main__':raise SystemExit(main())
