"""Run a single complete, independently registered existing GR00T engineering reference on the new SLA DEV16."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    a = p.parse_args()
    from go2arx5_vla.world_model.sla_g_native_v1 import INPUT_SCHEMA
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import run
    if json.loads(a.input.read_bytes()).get('schema') != INPUT_SCHEMA:
        raise ValueError('SLA_SEPARATE_NATIVE_POPULATION_INPUT_REQUIRED')
    return run(a.input)


if __name__ == '__main__':
    raise SystemExit(main())
