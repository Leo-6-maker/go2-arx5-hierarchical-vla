"""Finite CPU export of actual continuous trajectories; no training admission.

The existing site owner registers and launches this entry. Neither operation
starts physics, loads a policy model, submits PBS jobs, or retries a native case.
"""
import argparse
import time
from pathlib import Path


def run(args):
    from go2arx5_vla.world_model import task_learning_grasp_v1 as task
    from go2arx5_vla.world_model.storage import digest, write_json
    receipt = args.output / 'EXPORT_RESULT.private.json'
    task.need(not receipt.exists(), 'PRESERVE_EXISTING_DATA_EXPORT_RECEIPT')
    started, cpu = time.monotonic(), time.process_time()
    if args.operation == 'train-base':
        from go2arx5_vla.world_model.task_learning_export_v1 import build_train_base
        result = build_train_base(args.panel, args.group, args.output)
        status = 'CONTINUOUS_TRAIN_BASE_CREATED_WAIT_SEPARATE_DEV_REFERENCES'
    else:
        from go2arx5_vla.world_model.task_learning_bundle_v1 import finalize
        task.need(digest(args.train_base / 'DATASET_MANIFEST.json') == args.train_base_sha256,
                  'DECLARED_SEALED_TRAIN_BASE')
        result = finalize(args.train_base, args.train_base_sha256,
                          args.dev_reference_group, args.output)
        status = 'FORMAL_CONTINUOUS_BUNDLE_CREATED_WAIT_ACTUAL_PROCESSOR'
    result.update(status=status, dataset_root=str(args.output),
                  wall_s=time.monotonic() - started, CPU_s=time.process_time() - cpu,
                  policy_models_loaded=0, optimizer_updates=0, PBS_submissions=0)
    write_json(receipt, result)
    return result


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    base = commands.add_parser('train-base')
    base.add_argument('--panel', type=Path, required=True)
    base.add_argument('--group', type=Path, action='append', required=True)
    formal = commands.add_parser('formal')
    formal.add_argument('--train-base', type=Path, required=True)
    formal.add_argument('--train-base-sha256', required=True)
    formal.add_argument('--dev-reference-group', type=Path, required=True)
    for command in (base, formal):
        command.add_argument('--output', type=Path, required=True)
    print(json.dumps(run(parser.parse_args(argv))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
