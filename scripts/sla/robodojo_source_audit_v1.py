"""Bounded, CPU-only inspection of immutable RoboDojo real ARX files.

This module creates no policy input, expert label, native request or optimizer.
Source semantics remain unknown until the separate producer/codec review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import urllib.request

PLAN = 'be37da06fd007b24da324ae1cce6ba4ba441687e'
PREFIX = 'data/RoboDojo_real/'
RAW_CAP = 32 * 1024**3
TOTAL_CAP = 64 * 1024**3


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def selection(inventory, seed=2026111801):
    """Select before reading raw data; first two/task are part of the same60."""
    tasks = sorted((r for r in inventory['tasks'] if r.get('episodes')), key=lambda r: r['task'])
    if not 0 < len(tasks) <= 6:
        raise ValueError('EXPLICIT_ACTUAL_TASK_POPULATION_REQUIRED_WITHIN_SIX_TASK_CAP')
    rows, first = [], []
    for task in tasks:
        ranked = sorted(task['episodes'], key=lambda r: hashlib.sha256(
            (str(seed)+'/'+r['path']).encode()).hexdigest())[:10]
        for index, item in enumerate(ranked):
            path = PurePosixPath(item['path'])
            if (not str(path).startswith(PREFIX) or '..' in path.parts or
                    '/arx_x5/data/' not in str(path) or path.suffix != '.hdf5'):
                raise ValueError('ONLY_PINNED_REAL_ARX_HDF5_SOURCE_FILES')
            if not item.get('lfs') or len(item['lfs']['oid']) != 64 or item['size'] != item['lfs']['size']:
                raise ValueError('ACTUAL_LFS_BYTE_SHA_AND_SIZE_REQUIRED')
            row = dict(task=task['task'], path=str(path), sha256=item['lfs']['oid'],
                git_blob_oid=item['oid'], size=item['size'], selection_rank=index,
                proposed_split='TRAIN' if index < 6 else 'SOURCE_DEV' if index < 8 else 'SOURCE_TEST',
                split_status='PROVISIONAL_PENDING_SESSION_SCENE_NEAR_DUPLICATE_AUDIT')
            rows.append(row)
            if index < 2:
                first.append(row['path'])
    if len(rows) > 60 or len(first) > 12 or sum(r['size'] for r in rows) > RAW_CAP:
        raise ValueError('PREDECLARED_EPISODE_OR_RAW_BYTE_BUDGET_EXCEEDED')
    return dict(repository=inventory['repository'], revision=inventory['revision'], seed=seed,
        candidate_count=len(rows), first_count=len(first), raw_declared_bytes=sum(r['size'] for r in rows),
        candidates=rows, first_paths=first, source_semantics_qualified=False)


def download(row, root, repository, revision, *, until):
    """One attempt; preserve incomplete bytes on failure and never replay it."""
    path = root/'raw'/row['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('EXISTING_RAW_FILE_REQUIRES_INDEPENDENT_RECONCILIATION')
    partial = path.with_name(path.name + '.part')
    url = 'https://huggingface.co/datasets/'+repository+'/resolve/'+revision+'/'+row['path']
    count = 0
    value = hashlib.sha256()
    started = time.monotonic()
    with partial.open('xb') as stream:
        with urllib.request.urlopen(urllib.request.Request(url,
                headers={'User-Agent': 'SLA-bounded-source-audit/1'}), timeout=60) as response:
            while block := response.read(1024**2):
                if time.time() >= until or count + len(block) > row['size']:
                    raise RuntimeError('FINITE_DOWNLOAD_DEADLINE_OR_PINNED_SIZE')
                stream.write(block)
                count += len(block)
                value.update(block)
        stream.flush()
        os.fsync(stream.fileno())
    if count != row['size'] or value.hexdigest() != row['sha256']:
        raise ValueError('RAW_SOURCE_SIZE_OR_SHA_MISMATCH')
    os.replace(partial, path)
    return dict(path=str(path), sha256=value.hexdigest(), bytes=count,
        transfer_wall_s=time.monotonic()-started)


def profile(path):
    """Read small numeric fields; do not load a full compressed image array."""
    import h5py
    import numpy as np
    schema, values, attributes = {}, {}, {}
    with h5py.File(path, 'r') as source:
        def attribute(value):
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='replace')[:10000]
            if isinstance(value, str):
                return value[:10000]
            array = np.asarray(value)
            if array.size > 128:
                return dict(shape=list(array.shape))
            if array.ndim:
                return [attribute(v) for v in array]
            scalar = array.item()
            if isinstance(scalar, bytes):
                return scalar.decode('utf-8', errors='replace')[:10000]
            if isinstance(scalar, float) and not np.isfinite(scalar):
                return dict(nonfinite=str(scalar))
            return scalar
        attributes['/'] = {k: attribute(v) for k, v in source.attrs.items()}
        def visit(name, item):
            if item.attrs:
                attributes[name] = {k: attribute(v) for k, v in item.attrs.items()}
            if not isinstance(item, h5py.Dataset):
                return
            row = dict(shape=list(item.shape), dtype=str(item.dtype), compression=item.compression)
            schema[name] = row
            if item.external:
                row['external_storage'] = True
                return
            if name.startswith(('vision/', 'observation/images')) and not any(
                    key in name.rsplit('/', 1)[-1].lower() for key in
                    ('timestamp', 'time_stamp', 'frequency', 'intrinsic', 'extrinsic', 'camera_matrix')):
                return
            if item.size > 5_000_000:
                row['not_read_reason'] = 'BOUNDED_NUMERIC_FIELD_SIZE'
                return
            if item.dtype.kind in 'biuf':
                array = np.asarray(item[()])
                values[name] = array
                finite = np.isfinite(array)
                row.update(finite=bool(finite.all()), nonfinite_values=int((~finite).sum()))
                if array.size:
                    row.update(minimum=float(array[finite].min()) if finite.any() else None,
                        maximum=float(array[finite].max()) if finite.any() else None,
                        sha256=hashlib.sha256(array.tobytes()).hexdigest())
            elif item.size <= 128:
                row['value'] = attribute(item[()])
        source.visititems(visit)
    arms = {}
    for arm in ('left', 'right'):
        state, action = (values.get(g+'/'+arm+'_ee_poses') for g in ('state', 'action'))
        grip_state, grip_action = (values.get(g+'/'+arm+'_ee_joint_states') for g in ('state', 'action'))
        row = dict(pose_fields_present=state is not None and action is not None,
            gripper_fields_present=grip_state is not None and grip_action is not None)
        if row['pose_fields_present']:
            row.update(state_shape=list(state.shape), action_shape=list(action.shape),
                action_equal_state=bool(np.array_equal(state, action)))
            if (state.shape == action.shape and state.ndim == 2 and state.shape[1] == 7
                    and np.isfinite(state).all() and np.isfinite(action).all()):
                delta = np.asarray(action, float)-np.asarray(state, float)
                row.update(state_action_position_RMS_native_units=float(np.sqrt(np.mean(delta[:, :3]**2))),
                    state_quaternion_norm_max_error=float(np.max(abs(np.linalg.norm(state[:, 3:7], axis=1)-1))),
                    action_quaternion_norm_max_error=float(np.max(abs(np.linalg.norm(action[:, 3:7], axis=1)-1))))
        if row['gripper_fields_present']:
            row['gripper_action_shape'] = list(grip_action.shape)
            row['gripper_state_shape'] = list(grip_state.shape)
            row['gripper_action_equal_state'] = bool(np.array_equal(grip_state, grip_action))
            g = np.asarray(grip_action, float).reshape(-1)
            if g.size and np.isfinite(g).all():
                row.update(gripper_action_min=float(g.min()), gripper_action_max=float(g.max()),
                    gripper_action_unique=int(len(np.unique(g))),
                    gripper_nonzero_frame_changes=int(np.count_nonzero(abs(np.diff(g)) > 1e-8)),
                    gripper_max_frame_change=float(np.max(abs(np.diff(g)))) if len(g) > 1 else 0.)
        arms[arm] = row
    frequency = values.get('additional_info/frequency')
    return dict(schema=schema, attributes=attributes, arms=arms,
        frequency=float(frequency) if frequency is not None and frequency.size == 1 and np.isfinite(frequency).all() else None,
        command_provenance='UNKNOWN_PENDING_PRIMARY_PRODUCER_AUDIT',
        command_consumption_evidence='UNKNOWN_NOT_INFERRED_FROM_FIELD_NAMES',
        source_native_retained=True, canonical_qualified=False,
        previous_actual_emitted_context_created=False, Go2_P10_evidence_created=False,
        h5py_version=h5py.__version__, numpy_version=np.__version__,
        HDF5_version=h5py.version.hdf5_version)


def main(spec_path):
    spec = read(spec_path)
    root = Path(spec['registered_root'])
    if (spec['plan_commit'] != PLAN or spec['phase'] != 'R1_FIRST12' or
            spec['first_episode_cap'] != 12 or spec['episode_cap'] != 60 or
            spec['raw_byte_cap'] != RAW_CAP or spec['all_byte_cap'] != TOTAL_CAP or
            spec['new_model_generations_budget'] or spec['new_physical_requests_budget'] or
            spec['optimizer_updates'] or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise PermissionError('EXPLICIT_BOUNDED_CPU_SOURCE_AUDIT_ONLY')
    runtime = spec['trusted_runtime']
    if sha(runtime['path']) != runtime['sha256']:
        raise ValueError('TRUSTED_EXECUTION_PARENT_SPEC_CHANGED')
    previous = read(runtime['path'])
    parent = previous['trusted_source_parent']
    if sha(parent['spec_path']) != parent['spec_sha256']:
        raise ValueError('TRUSTED_NATIVE_BASE_IDENTITY_CHANGED')
    source = Path(spec['source_root'])
    if Path(__file__).resolve() != source/'scripts/sla/robodojo_source_audit_v1.py':
        raise PermissionError('EXACT_NEW_ISOLATED_AUDIT_SOURCE_REQUIRED')
    for name, expected in spec['source_files'].items():
        if sha(source/name) != expected:
            raise ValueError('SEALED_CPU_SOURCE_CHANGED')
    runtime_root, base = Path(previous['source_root']), Path(parent['source_root'])
    sys.path[:0] = [str(r/p) for r in (source, runtime_root, base) for p in ('', 'src')]
    import go2arx5_vla.world_model
    import scripts.wm02
    for module, suffix in ((go2arx5_vla.world_model, 'src/go2arx5_vla/world_model'),
            (scripts.wm02, 'scripts/wm02')):
        module.__path__ = [str(r/suffix) for r in (source, runtime_root, base)] + list(module.__path__)
    from scripts.wm02.mobile_boundary_runner_v1 import registered
    registered(spec, spec_path)
    selection_path = Path(spec['selection']['path'])
    if sha(selection_path) != spec['selection']['sha256']:
        raise ValueError('SOURCE_CANDIDATE_SELECTION_CHANGED')
    population = read(selection_path)
    if (population['repository'] != 'RoboDojo-Benchmark/RoboDojo' or
            len(population['revision']) != 40 or population['revision'] != spec['HF_revision']):
        raise ValueError('EXACT_FIXED_PUBLIC_DATASET_REVISION_REQUIRED')
    wanted = set(population['first_paths'])
    rows = [r for r in population['candidates'] if r['path'] in wanted]
    if len(rows) > 12 or sum(r['size'] for r in rows) > RAW_CAP:
        raise PermissionError('FIRST12_INCLUDED_IN_TOTAL60')
    if (root/'RESULT.private.json').exists() or (root/'SUPERVISOR.private.json').exists():
        raise FileExistsError('NEW_ONCE_ONLY_AUDIT_IDENTITY_REQUIRED')
    write(root/'SUPERVISOR.private.json', dict(pid=os.getpid(), epoch=time.time(),
        source_sha=spec['source_sha'], plan_commit=PLAN))
    from scripts.pi0294.e1.journal import Journal
    journal = Journal(root/'control/journal', root.name)
    start, cpu = time.monotonic(), time.process_time()
    results, failure = [], None
    try:
        for index, row in enumerate(rows):
            if time.time() >= spec['deadline_epoch']-60 or (root/'STOP.json').exists():
                raise TimeoutError('FINITE_SOURCE_AUDIT_STOP_OR_DEADLINE')
            write(root/'PROGRESS.private.json', dict(epoch=time.time(), phase='PINNED_DOWNLOAD_AND_PROFILE',
                completed=len(results), current=row['path'], new_generations=0, new_native_requests=0))
            transfer = journal.once('FILE-%02d' % index, row,
                lambda: download(row, root, population['repository'], population['revision'], until=spec['deadline_epoch']-60))
            detail = profile(transfer['path'])
            record = dict(source=row, transfer=transfer, profile=detail)
            write(root/('PROFILE-%02d.private.json' % index), record)
            results.append(record)
            write(root/'RECORDS.private.json', results)
        code = 0
    except BaseException as error:
        code = 1
        # Do not expose a signed redirect URL in the inspection summary.
        failure = dict(type=type(error).__name__, reason=str(error).split('?')[0][:1000])
    value = dict(schema='SLA_ROBODOJO_R1_CPU_ACTUAL', exit_code=code,
        status='FIRST12_PROFILED_SEMANTICS_NOT_YET_ADMITTED' if code == 0 else 'PRESERVED_NO_AUTOMATIC_RETRY',
        source_sha=spec['source_sha'], plan_commit=PLAN, revision=population['revision'],
        selected_total=population['candidate_count'], selected_first=len(rows),
        profiled_episodes=len(results), raw_verified_bytes=sum(r['transfer']['bytes'] for r in results),
        failure=failure, results=[dict(path=str(root/('PROFILE-%02d.private.json' % i)),
            sha256=sha(root/('PROFILE-%02d.private.json' % i))) for i in range(len(results))],
        source_semantics_qualified=False, codec_compatibility_qualified=False,
        native_requests=0, model_generations=0, optimizer_updates=0, optimization_visits=0,
        wall_s=time.monotonic()-start, CPU_s=time.process_time()-cpu)
    write(root/'RESULT.private.json', value)
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(spec, code)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.spec))
