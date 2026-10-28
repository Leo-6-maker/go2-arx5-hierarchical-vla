"""Read-only inventory of a complete local native Trainer checkpoint.

No tensor data is materialized and no pickle is executed. This verifies saved
bytes and structural bindings; the separate actual smoke proves resumability.
"""
from __future__ import annotations

from pathlib import Path
import re

from .durable import read
from .train_adapter import CyclicSampler, sha

REQUIRED = ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json',
            'statistics.json', 'SAMPLER_STATE.json')
PROCESSOR_FILES = ('processor/processor_config.json', 'processor/statistics.json',
                   'processor/embodiment_id.json')
ALLOWED_FILES = set(REQUIRED) | set(PROCESSOR_FILES) | {
    'CHECKPOINT_COMPLETE.json', 'config.json', 'training_args.bin',
    'generation_config.json', 'model.safetensors.index.json', 'model.safetensors'}


def inventory(checkpoint, *, expected_step, recipe, statistics_sha256, manifest_sha256):
    """Original PI0292 inventory retains its registered steps and cyclic sampler."""
    if expected_step not in (2, 4, 2048, 4096):
        raise ValueError('REGISTERED_CHECKPOINT_STEP_REQUIRED')
    checkpoint=Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError('REGULAR_CHECKPOINT_DIRECTORY_REQUIRED')
    if any(path.is_symlink() for path in checkpoint.rglob('*')):
        raise ValueError('CHECKPOINT_SYMLINK_REFUSED')
    saved=read(checkpoint/'SAMPLER_STATE.json')
    expected=CyclicSampler(saved['sampler']['size'],stop=recipe['max_steps']*32).state_dict(expected_step*32)
    return inventory_bound_sampler(checkpoint,expected_step=expected_step,recipe=recipe,
        statistics_sha256=statistics_sha256,manifest_sha256=manifest_sha256,expected_sampler=expected)


def inventory_bound_sampler(checkpoint, *, expected_step, recipe, statistics_sha256,
                            manifest_sha256, expected_sampler):
    """Shared byte verifier; caller reconstructs the registered sampler independently.

    This function grants no execution, resume qualification, or policy selection.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError('REGULAR_CHECKPOINT_DIRECTORY_REQUIRED')
    if any(path.is_symlink() for path in checkpoint.rglob('*')):
        raise ValueError('CHECKPOINT_SYMLINK_REFUSED')
    if (type(expected_step) is not int or not 0<expected_step<=6144 or
        expected_sampler['cursor']!=expected_step*32 or expected_sampler['stop']!=recipe['max_steps']*32):
        raise ValueError('EXACT_INDEPENDENT_CONSUMED_SAMPLER_REQUIRED')
    complete = read(checkpoint / 'CHECKPOINT_COMPLETE.json')
    if complete['status'] != 'FULL_STATE_SAVED_MODEL_ARCHIVE_HASH_PENDING':
        raise ValueError('NATIVE_FULL_STATE_SAVE_RECEIPT_REQUIRED')
    if complete['global_step'] != expected_step:
        raise ValueError('WRONG_COMPLETE_STEP')
    if set(complete['required_files']) != set(REQUIRED):
        raise ValueError('INCOMPLETE_RESUME_FILE_LIST')
    state = read(checkpoint / 'trainer_state.json')
    sampler = read(checkpoint / 'SAMPLER_STATE.json')
    if state['global_step'] != expected_step or sampler['recipe'] != recipe:
        raise ValueError('CHECKPOINT_STEP_OR_RECIPE_CHANGED')
    if sampler['statistics_sha256'] != statistics_sha256 or sampler['manifest_sha256'] != manifest_sha256:
        raise ValueError('CHECKPOINT_DATA_IDENTITY_CHANGED')
    if sampler['sampler'] != expected_sampler:
        raise ValueError('CONSUMED_SAMPLE_CURSOR_CHANGED')
    trace = sampler['worker_trace']
    if (trace['status'] != 'PASS_ACTUAL_CONSUMED_WORKER_IDENTITIES'
            or trace['stop'] != expected_step * 32 or not trace['prefetch_excluded']
            or trace['count'] != trace['stop'] - trace['start']):
        raise ValueError('CONSUMED_WORKER_TRACE_MISSING')
    if sha(checkpoint / 'statistics.json') != statistics_sha256:
        raise ValueError('FROZEN_STATISTICS_BYTES_CHANGED')
    files = {}
    for path in sorted(checkpoint.rglob('*')):
        if path.is_symlink():
            raise ValueError('CHECKPOINT_SYMLINK_REFUSED')
        if path.is_file():
            relative = path.relative_to(checkpoint).as_posix()
            if relative not in ALLOWED_FILES and not re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors', relative):
                raise ValueError('UNREGISTERED_CHECKPOINT_FILE_REFUSED')
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('CHECKPOINT_IS_STILL_CHANGING')
            files[relative] = dict(bytes=before.st_size, sha256=digest)
    for name, digest in complete['required_files'].items():
        if name not in files or files[name]['sha256'] != digest:
            raise ValueError('FULL_STATE_FILE_MISSING_OR_CHANGED')
    for name in ('config.json', *PROCESSOR_FILES):
        if name not in files or not files[name]['bytes']:
            raise ValueError('MODEL_OR_PROCESSOR_CONFIG_MISSING')
        read(checkpoint / name)
    if read(checkpoint / 'processor/statistics.json') != read(checkpoint / 'statistics.json'):
        raise ValueError('SAVED_PROCESSOR_STATISTICS_CHANGED')
    if read(checkpoint / 'processor/embodiment_id.json').get('new_embodiment') != 10:
        raise ValueError('SAVED_PROCESSOR_EMBODIMENT_ROUTE_CHANGED')
    processor = read(checkpoint / 'processor/processor_config.json')['processor_kwargs']
    expected_processor = dict(max_state_dim=132, max_action_dim=132, max_action_horizon=40,
        use_percentiles=False, use_mean_std=False, clip_outliers=False, use_relative_action=True,
        state_dropout_prob=0., exclude_state=False, use_albumentations=False,
        image_crop_size=[256, 256], image_target_size=[256, 256], crop_fraction=1.)
    if any(processor.get(name) != value for name, value in expected_processor.items()):
        raise ValueError('SAVED_PROCESSOR_INTERFACE_CHANGED')
    actual_weight_files = sorted(name for name in files if '/' not in name and name.endswith('.safetensors'))
    index = checkpoint / 'model.safetensors.index.json'
    mapping = read(index)['weight_map'] if index.exists() else None
    expected_files = sorted(set(mapping.values())) if mapping is not None else ['model.safetensors']
    if not expected_files or actual_weight_files != expected_files:
        raise ValueError('MODEL_SHARD_SET_INCOMPLETE_OR_EXTRA')
    if any(Path(name).name != name for name in expected_files):
        raise ValueError('WEIGHT_INDEX_PATH_ESCAPE')
    from safetensors import safe_open
    tensors = {}
    for name in expected_files:
        with safe_open(checkpoint / name, framework='numpy', device='cpu') as handle:
            for key in handle.keys():
                if key in tensors or (mapping is not None and mapping.get(key) != name):
                    raise ValueError('WEIGHT_KEY_OR_SHARD_MAPPING_CHANGED')
                tensors[key] = dict(file=name, shape=handle.get_slice(key).get_shape())
    if not tensors or (mapping is not None and set(tensors) != set(mapping)):
        raise ValueError('MODEL_TENSOR_SET_INCOMPLETE')
    return dict(schema='PI0292_COMPLETE_CHECKPOINT_INVENTORY_V1',
        status='FULL_STATE_BYTES_INVENTORIED_NOT_INDEPENDENT_RESUME_PROOF',
        global_step=expected_step, consumed_samples=expected_step * 32,
        statistics_sha256=statistics_sha256, manifest_sha256=manifest_sha256,
        recipe=recipe, files=files, total_bytes=sum(row['bytes'] for row in files.values()),
        model_tensors=tensors, optimizer_pickle_loaded=False, model_loaded=False,
        processor_saved=True, original_bytes_changed=False)
