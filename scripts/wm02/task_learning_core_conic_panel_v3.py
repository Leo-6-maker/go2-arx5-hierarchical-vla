"""Run the independently registered four-condition C500 explicit-conic diagnostic."""
import argparse
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(argv)
    import json
    from go2arx5_vla.world_model.task_learning_conic_native_v3 import INPUT_SCHEMA
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import run
    if json.loads(args.input.read_bytes()).get('schema')!=INPUT_SCHEMA:
        raise ValueError('Independent C500 conic input required; original DEV is a separate entry')
    return run(args.input)


if __name__=='__main__':raise SystemExit(main())
