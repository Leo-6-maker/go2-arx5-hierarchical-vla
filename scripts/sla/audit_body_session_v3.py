"""Compare fresh and persistent body/wrist capture on32 actual saved states."""
import json
from pathlib import Path
import sys
import time
import numpy as np

from scripts.pi0292.durable import write
from scripts.pi0294.task_state import ActualState, parse_state
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.world_model.native_body_camera_capture_v1 import BodyBaseCameraRecorder, BodyBaseCameraSession
from go2arx5_vla.world_model.native_collection_performance_v1 import PerformanceTrace
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest


def run(spec):
    import mujoco
    if mujoco.__version__ != '3.2.7':
        raise ValueError('QUALIFIED_NATIVE_MUJOCO327_REQUIRED')
    root = Path(spec['registered_root'])
    dataset = Path(spec['dataset'])
    audit = read_pin(spec['target_audit'])
    if digest(dataset / 'DATASET_MANIFEST.json') != audit['source_dataset_sha256']:
        raise ValueError('UNCHANGED_TARGET_DATA_REQUIRED')
    manifest = json.loads((dataset / 'DATASET_MANIFEST.json').read_bytes())
    frames = {}
    for episode in manifest['episodes']:
        capture = read_pin(episode['proof']['capture_index'])
        native = read_pin(episode['proof']['spec'])
        for entry in capture['records']:
            descriptor = entry if 'path' in entry else entry['observation']
            file = Path(descriptor['path']).parent / 'STATE.private.bin'
            frames.setdefault((digest(native['model_path']), str(file)), (native['model_path'], file))
            if len(frames) == 32:
                break
        if len(frames) == 32:
            break
    if len(frames) != 32:
        raise ValueError('EXACT32_DISTINCT_SAVED_STATE_IDENTITIES_REQUIRED')
    sessions, states = {}, {}
    records, traces = [], {name: PerformanceTrace() for name in ('fresh', 'persistent')}
    begin = time.monotonic()
    try:
        with forbid_integration(mujoco) as integrations:
            for index, ((model_sha, _), (model_path, file)) in enumerate(frames.items()):
                if time.time() >= spec['deadline_epoch']:
                    raise TimeoutError('FINITE_BODY_SESSION_REPLAY')
                packet = parse_state(file.read_bytes())
                if model_sha not in sessions:
                    sessions[model_sha] = BodyBaseCameraSession(model_path)
                    states[model_sha] = ActualState(mujoco.MjModel.from_binary_path(model_path))
                state = states[model_sha]
                state.update(packet)
                cases = {}
                for mode in ('fresh', 'persistent'):
                    output = root / f'{index:02d}-{mode}'
                    (output / 'images').mkdir(parents=True)
                    recorder = BodyBaseCameraRecorder(model_path, output,
                        session=sessions[model_sha] if mode == 'persistent' else None,
                        performance=traces[mode])
                    began = time.monotonic()
                    try:
                        capture = recorder.save(0, packet)
                        calibration = json.loads((output / 'images/000000-calibration.json').read_bytes())
                        media = {key: (output / 'images' / f'000000-{key}.png').read_bytes()
                                 for key in ('body_rgb', 'wrist_rgb')}
                        recorder.camera_adapter.validate(actual_state=state, packet=packet,
                                                         calibration=calibration, image_bytes=media)
                        cases[mode] = dict(files={str(path.relative_to(output)): digest(path)
                                                 for path in output.rglob('*') if path.is_file()},
                                           wall_s=time.monotonic() - began, capture=capture)
                    finally:
                        recorder.close()
                if cases['fresh']['files'] != cases['persistent']['files']:
                    raise ValueError('SAVED_BODY_SESSION_PIXELS_OR_CALIBRATION_DIFFER:' + str(index))
                records.append(dict(index=index, native_step=packet['step'], model_sha256=model_sha,
                                    saved_state_sha256=digest(file), **cases))
                write(root / 'PROGRESS.private.json', dict(completed=len(records), requested=32, integrations=0))
            if integrations:
                raise AssertionError('SAVED_STATE_AUDIT_MUST_NOT_INTEGRATE')
    finally:
        for session in sessions.values():
            session.close()
    return dict(schema='SLA_BODY_SESSION_SAVED32_ACTUAL_V3', passes=True, saved_states=32,
                all_fresh_persistent_PNG_and_calibration_bytes_exact=True,
                source_dataset_sha256=audit['source_dataset_sha256'], records=records,
                sessions=len(sessions), renderer_contexts_per_session=2, fresh_capture_count=32,
                profiles={key: value.report() for key, value in traces.items()},
                fresh_capture_wall_s=sum(row['fresh']['wall_s'] for row in records),
                persistent_capture_wall_s=sum(row['persistent']['wall_s'] for row in records),
                wall_s=time.monotonic()-begin, physics_integrations=0, model_queries=0, optimizer_updates=0,
                original_native_state_unchanged=True, PNG_file_ABI_retained=True,
                memory_transport_admitted=False, native_task_success=False)


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    write(Path(spec['registered_root']) / 'RESULT.private.json', run(spec))
