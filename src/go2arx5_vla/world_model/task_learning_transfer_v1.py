"""An explicit private-data inventory after the real Core Processor round trip."""
import json
from pathlib import Path
import numpy as np
from . import task_learning_grasp_v1 as task
from .task_learning_bundle_v1 import member
from .task_learning_native_dataset_v1 import TaskLearningNativeDataset
from .storage import digest


def inventory(root):
    root = Path(root)
    task.need(root.is_absolute() and root.resolve() == root, 'CANONICAL_PRIVATE_TRANSFER_SOURCE')
    data = TaskLearningNativeDataset(root, 'unused-before-Processor', 'unused-before-Processor')
    manifest = data.manifest
    task.need(manifest['bundle_kind'] == 'FORMAL_TASK_LEARNING_V1',
              'FORMAL_TRAIN_AND_SEPARATE_DEV_BEFORE_PRIVATE_TRANSFER')
    exported = json.loads((root / 'EXPORT_RESULT.private.json').read_bytes())
    processor = json.loads((root / 'PROCESSOR_TRAINING_ADMISSION.json').read_bytes())
    task.need(exported.get('status') == 'FORMAL_CONTINUOUS_BUNDLE_CREATED_WAIT_ACTUAL_PROCESSOR'
              and exported.get('manifest_sha256') == data.manifest_sha
              and exported.get('statistics_sha256') == data.statistics_sha,
              'ACTUAL_CORE_FORMAL_EXPORT')
    task.need(processor.get('status') == 'PASS_ACTUAL_OFFICIAL_PROCESSOR_ROUNDTRIP'
              and processor.get('dataset_family') == 'TASK_LEARNING_CONTINUOUS_V1'
              and processor.get('manifest_sha256') == data.manifest_sha
              and processor.get('statistics_sha256') == data.statistics_sha
              and processor.get('processed_TRAIN_windows') == len(data)
              and len(processor.get('records', [])) == len(data)
              and processor.get('fixed_component_tolerance') == 1e-6
              and np.isfinite(processor.get('maximum_abs_error', np.nan))
              and 0 <= processor['maximum_abs_error'] <= 1e-6
              and processor.get('TRAIN_only_statistics') is True,
              'ACTUAL_ALL_UNIQUE_TRAIN_DECISIONS_ORIGINAL_CORE_PROCESSOR')
    task.need([(r.get('episode'), r.get('anchor')) for r in processor['records']]
              == [(row['id'], anchor) for row, anchor in data.rows],
              'SAME_ORDERED_UNIQUE_TRAIN_PROCESSOR_DECISIONS')
    files = {}

    def add(name, pin):
        member(root, name, pin)
        task.need(name not in files or files[name] == pin, 'ONE_IDENTITY_PER_TRANSFER_MEMBER')
        files[name] = pin

    add('DATASET_MANIFEST.json', data.manifest_sha)
    add('statistics.json', data.statistics_sha)
    for row in manifest['episodes']:
        add(row['file'], row['sha256'])
        with np.load(root / row['file'], allow_pickle=False) as packet:
            paths, pins = packet['rgb_paths'], packet['rgb_sha256']
            task.need(paths.shape == pins.shape == (15, 2), 'ALL_REAL_DUAL_RGB_CAPTURES')
            for name, pin in zip(paths.flat, pins.flat):
                add(str(name), str(pin))
    for name in ('EXPORT_RESULT.private.json', 'PROCESSOR_TRAINING_ADMISSION.json'):
        add(name, digest(root / name))
    task.need(bool(processor.get('processor_files')), 'SAVED_ORIGINAL_CORE_PROCESSOR')
    for name, pin in processor['processor_files'].items():
        add('target-processor-v1/' + name, pin)
    return dict(schema='TASK_LEARNING_PRIVATE_DATA_TRANSFER_V1',
                manifest_sha256=data.manifest_sha, statistics_sha256=data.statistics_sha,
                files=files, bytes=sum((root / name).stat().st_size for name in files),
                source_dataset_root=str(root), source_processor_admission_sha256=
                files['PROCESSOR_TRAINING_ADMISSION.json'],
                new_policy_models=0, new_optimizer_updates=0, new_native_integrations=0)
