"""Run a single complete, independently registered SLA native population."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    a = p.parse_args()
    from go2arx5_vla.world_model.sla_native_v1 import INPUT_SCHEMA
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import run
    from go2arx5_vla.world_model.sla_native_continuation_v2 import INPUT_SCHEMA as COLD_SCHEMA
    from go2arx5_vla.world_model.sla_dev_continuation_v3 import SCHEMA as DEV_SUFFIX
    from go2arx5_vla.world_model.sla_r1_suffix_v5 import SCHEMA as R1_SUFFIX
    if json.loads(a.input.read_bytes()).get('schema') not in (INPUT_SCHEMA,COLD_SCHEMA,DEV_SUFFIX,R1_SUFFIX):
        raise ValueError('SLA_SEPARATE_NATIVE_POPULATION_INPUT_REQUIRED')
    return run(a.input)


if __name__ == '__main__':
    raise SystemExit(main())
