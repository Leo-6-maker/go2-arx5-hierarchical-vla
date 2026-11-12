"""Run only the separately registered four-TRAIN native-body diagnostic."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args()
    from go2arx5_vla.world_model.native_arx5_diagnostic_v1 import INPUT_SCHEMA
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import run
    if json.loads(args.input.read_text()).get('schema')!=INPUT_SCHEMA:
        raise ValueError('Independent native-body TRAIN4 input required')
    return run(args.input)


if __name__=='__main__':
    raise SystemExit(main())
