"""Run a separately admitted contiguous group of the fixed128 development panel."""
import argparse,json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(argv)
    from go2arx5_vla.world_model.task_learning_matched_native_v3_1 import INPUT_SCHEMA
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import run
    if json.loads(args.input.read_bytes()).get('schema')!=INPUT_SCHEMA:
        raise ValueError('Separate matched128 input required; original DEV and pilots unchanged')
    return run(args.input)


if __name__=='__main__':raise SystemExit(main())
