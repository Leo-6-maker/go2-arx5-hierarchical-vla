"""Qualify one source-rig CPU render from a sealed actual target packet.

Saved-state reconstruction only: never restore a controller or integrate, query
GR00T, edit the scene, replace source pixels, or admit a dynamic target job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import uuid

import numpy as np

from scripts.pi0294.task_state import ActualState, parse_state
from scripts.wm02.capture_fusion_observation import software_renderer_identity, count_forward, write_bytes
from go2arx5_vla.world_model.fusion_observation import TargetShadowBinding, shadow_inference_payload
from go2arx5_vla.world_model.fusion_source_camera_capture import SourceCameraCapture
from go2arx5_vla.world_model.fusion_source_camera_rig import SOURCE_IDENTITY_SHA256, FRAME_MAP_SHA256, RIG
from go2arx5_vla.world_model.native_replay import forbid_integration, validate_qualified_recording
from go2arx5_vla.world_model.pilot1_storage import local_path
from go2arx5_vla.world_model.storage import digest, write_json

PROFILE = 'wm02-fusion-source-camera-qualification-cpu'
KIND = 'FUSION_SOURCE_CAMERA_QUALIFICATION'
SITE = 'CORE_FUSION_SOURCE_CAMERA_QUALIFICATION_CPU'
RECIPE = 'WM02_FUSION_ONE_SEALED_TARGET_SOURCE_CAMERA_CPU_V1'
C02_RECIPE = 'WM02_FUSION_C02_SEALED_SOURCE_CAMERA_CPU_V1'
STABLE_RECIPE = 'WM02_STABLE_START_SEALED_DECISION_CAMERA_CPU_V1'
C02_SCENE = 'f27d7026ef8f182a169d7c5b4a7ed066de414196ce59d0dd97a605819050fdbf'
C02_PHASE = 'collect-target-fusion-c02-source-state-feedback-v2'
ENTRY = 'scripts/wm02/qualify_fusion_source_camera.py'
SCENE = '40408bbcbef2469396187c741750905a6ba9ed7161dc031d14bfab5fed62ddd2'
DESCRIPTORS = ('execution_spec', 'episode_manifest', 'native_result', 'worker_exit', 'lease', 'journal', 'scene')
_ALLOWED = {('collect-target-fusion-paused-query-v2', 3703),
            ('collect-target-fusion-source-posture-qualification-v1', 8503)}


def need(ok, reason):
    if not ok:
        raise ValueError('FUSION_SOURCE_CAMERA_QUALIFICATION_' + reason)


def _finite(spec):
    if (time.time() >= spec['deadline_epoch']
            or (Path(spec['registered_root']) / 'STOP.json').exists()
            or (Path(spec['campaign_root']) / 'STOP.json').exists()
            or ('_camera_started' in spec and time.monotonic() - spec['_camera_started'] >= spec['wall_limit_s'])):
        raise TimeoutError('FINITE_SOURCE_CAMERA_CPU_ENDPOINT_OR_STOP')


def _scope(spec, output):
    from go2arx5_vla.world_model.fusion_scope import validate_source_camera_cpu
    validate_source_camera_cpu(spec)
    root = Path(spec['registered_root']).resolve(strict=True)
    need(Path(output).resolve(strict=True).parent == root and Path(output).name == 'fit-01'
         and not Path(output).is_symlink(), 'EXACT_ADMITTED_FRESH_FIT_01')
    entry = Path(spec['source_root']) / ENTRY
    need(Path(__file__).resolve() == entry.resolve(strict=True)
         and spec['source_files'].get(ENTRY) == digest(entry), 'EXACT_ENTRY_CAPSULE_PIN')
    _finite(spec)
    local_path(spec, output)


def _read(spec, descriptor):
    need(type(descriptor) is dict and set(descriptor) == {'path', 'sha256'}, 'EXACT_FILE_DESCRIPTOR')
    path = Path(descriptor['path'])
    need(path.is_absolute() and not path.is_symlink(), 'ABSOLUTE_ORIGINAL_REGULAR_FILE')
    local_path(spec, path)
    _finite(spec)
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == descriptor['sha256'], 'PINNED_INPUT_BYTES:' + str(path))
    return path, raw


def endpoint_contract(spec, saved, native_spec, episode, native):
    """C02 is a distinct diagnostic course; original T06 rules are unchanged."""
    ticks = saved['expected_native_ticks']
    if spec.get('recipe') == STABLE_RECIPE:
        from go2arx5_vla.world_model.stable_start_short_task_v1 import PHASE, B, E, validate
        validate(native_spec)
        q = native.get('reference_qualification') or {}
        need(type(ticks) is int and saved['native_step'] == B
             and native_spec.get('max_ticks') == ticks == E
             and native_spec.get('phase') == PHASE
             and saved['scene']['sha256'] == SCENE
             and episode.get('scene_sha256') == SCENE
             and q.get('hold_stability_pass') is True
             and q.get('capability_reference_all40_consumed') is True,
             'EXACT_NEW_STABLE_START_DECISION_WITH_COMPLETE_NATIVE_LIFECYCLE')
        return
    need(type(ticks) is int and type(saved['native_step']) is int
         and saved['native_step'] == native_spec.get('max_ticks') == ticks,
         'EXACT_SAVED_ENDPOINT_CLOCK')
    if spec.get('recipe') == C02_RECIPE:
        need((native_spec['phase'], ticks) == (C02_PHASE, 44703)
             and saved['scene']['sha256'] == C02_SCENE
             and episode.get('scene_id') == 'FUSION_SOURCE_STATE_C02_V1'
             and episode.get('parent') == 'FUSION_SOURCE_STATE_C02_PRIVILEGED_STATIC_DESIGN'
             and episode.get('split') == 'DIAGNOSTIC', 'EXACT_SEPARATE_C02_ENDPOINT')
        q = native.get('reference_qualification') or {}
        need(q.get('all_40_knots_consumed') is True
             and q.get('actual_initialization_feedback_decisions') == 16
             and q.get('C02_initialization_qualification', {}).get('qualified') is True,
             'ACTUAL_C02_SOURCE_STATE_QUALIFIED_NOT_STATIC_OR_FAILED_INITIALIZATION')
        return
    need(spec.get('recipe') == RECIPE and (native_spec['phase'], ticks) in _ALLOWED
         and saved['scene']['sha256'] == SCENE
         and episode.get('scene_id') == 'F2T06' and episode.get('split') == 'TRAIN'
         and episode.get('parent') == 'FUSION_F2_T06_ROOT_XY_V1', 'ONE_ALLOWED_ACTUAL_T06_SEALED_ENDPOINT')


def validate_inputs(spec):
    saved = spec['saved_native']
    need(type(saved) is dict and set(saved) == set(DESCRIPTORS) | {
        'native_step', 'expected_native_ticks', 'packet_sha256'}, 'EXACT_SEALED_NATIVE_INPUT_FIELDS')
    paths, raw = {}, {}
    for name in DESCRIPTORS:
        paths[name], raw[name] = _read(spec, saved[name])
    native_spec, episode, native, worker, lease = [json.loads(raw[name]) for name in
        ('execution_spec', 'episode_manifest', 'native_result', 'worker_exit', 'lease')]
    root = paths['execution_spec'].parent
    expected = dict(execution_spec=root/'EXECUTION_SPEC.json', episode_manifest=root/'EPISODE_MANIFEST.private.json',
        native_result=root/'native-01/RESULT.json', worker_exit=root/'WORKER_EXIT.private.json',
        journal=root/'native-01/state.bin.journal', lease=Path(native_spec['lease']), scene=Path(native_spec['model_path']))
    need(all(paths[name].resolve() == path.resolve() for name, path in expected.items())
         and root.resolve() == Path(native_spec['registered_root']).resolve()
         and paths['lease'].resolve().is_relative_to(root.resolve())
         and Path(native_spec['episode_manifest']).resolve() == paths['episode_manifest'].resolve()
         and native_spec['episode_manifest_sha256'] == saved['episode_manifest']['sha256'],
         'EXACT_SEALED_RUN_MEMBER_PATH_JOINS')
    ticks = saved['expected_native_ticks']
    endpoint_contract(spec, saved, native_spec, episode, native)
    need(lease.get('status') == 'CLOSED' and lease.get('worker_exit_code') == 0
         and lease.get('run_uuid') == root.name
         and lease.get('spec_sha256') == saved['execution_spec']['sha256']
         and worker.get('exit_code') == 0 and worker.get('failure') is None,
         'CLOSED0_HISTORICAL_WORKER_IS_DATA_NOT_A_NEW_GRANT')
    need(native.get('failure') is None and native.get('reached_exact_pause') is True
         and native.get('actual_target_integrations') == ticks
         and native.get('native_outer_guard_rejection_count') == 0
         and native.get('trace_audit_error') is None
         and len(native.get('owned_process_exits', [])) == 4
         and {x['name'] for x in native['owned_process_exits']} == {'sim', 'planner', 'controller', 'solver'}
         and native.get('phase') == native_spec['phase'], 'ACTUAL_CLEAN_NATIVE_ENDPOINT')
    trace = native.get('trace_audit', {})
    need(trace.get('writer', {}).get('dropped') == 0
         and trace.get('writer', {}).get('events') == trace.get('writer', {}).get('written')
         and trace.get('native_step_records') == ticks, 'COMPLETE_ZERO_DROP_ORIGINAL_TRACE')
    rows, selected = [], None
    blob, offset = raw['journal'], 0
    while offset < len(blob):
        _finite(spec)
        need(offset + 4 <= len(blob), 'COMPLETE_JOURNAL_LENGTH_HEADER')
        size, = struct.unpack_from('<I', blob, offset); offset += 4
        need(40 <= size <= 2*1024**2 and offset + size <= len(blob), 'COMPLETE_BOUNDED_JOURNAL_PACKET')
        packet_raw = blob[offset:offset+size]; offset += size
        packet = parse_state(packet_raw)
        need(packet['step'] == len(rows) + 1
             and abs(packet['time_s'] - packet['step']*.002) <= 1e-8, 'ORIGINAL_CONTIGUOUS_2MS_JOURNAL')
        for contact in packet['contacts']:
            need(all(np.isfinite(np.asarray(contact[k])).all() for k in
                ('distance_m', 'wrench', 'normal', 'frame', 'point')), 'FINITE_ORIGINAL_CONTACTS')
        rows.append(packet)
        if packet['step'] == saved['native_step']:
            selected = packet_raw
    validate_qualified_recording(dict(scene=saved['scene'], original_native_source_sha=native_spec['source_sha']), native, rows)
    need(len(rows) == ticks and selected is not None
         and hashlib.sha256(selected).hexdigest() == saved['packet_sha256'], 'EXACT_SELECTED_ACTUAL_PACKET_BYTES')
    identity_path, identity = _read(spec, dict(path=spec['source_camera_identity'], sha256=spec['source_camera_identity_sha256']))
    need(spec['source_camera_identity_sha256'] == SOURCE_IDENTITY_SHA256
         and spec['frame_map']['sha256'] == FRAME_MAP_SHA256, 'PINNED_SOURCE_CAMERA_AUDIT_AND_FRAME')
    recipe = SourceCameraCapture(identity, spec['frame_map']['T_source_from_target'])
    return paths, native_spec, selected, recipe


def run(spec, output):
    output = Path(output)
    start, cpu = time.monotonic(), time.process_time()
    spec = dict(spec, _camera_started=start)
    paths, native_spec, raw, recipe = validate_inputs(spec)
    import mujoco
    need(mujoco.__version__ == '3.2.7' and mujoco.get_mjcb_control() is None,
         'ORIGINAL_MUJOCO_WITHOUT_CONTROL_CALLBACK')
    recorder = None; failure = None; renderer = None; saved = None; retained = {}; forwards = [0]; calls = []
    try:
        (output/'images').mkdir(exist_ok=False)
        with forbid_integration(mujoco) as calls, count_forward(mujoco) as forwards:
            packet = parse_state(raw)
            model = mujoco.MjModel.from_binary_path(str(paths['scene']))
            need((model.nq, model.nv, model.nu, model.na, model.nplugin) == (34, 32, 20, 0, 0), 'EXACT_NATIVE_MODEL_ABI')
            need(all(contact[key] < model.ngeom for contact in packet['contacts'] for key in ('geom0', 'geom1')),
                 'ACTUAL_PACKET_CONTACT_MODEL_IDENTITIES')
            state = ActualState(model)
            binding = TargetShadowBinding('ACTUAL_TARGET_WORLD', spec['saved_native']['scene']['sha256'],
                spec['saved_native']['execution_spec']['sha256'], spec['frame_map']['T_source_from_target'], FRAME_MAP_SHA256)
            recorder = recipe.recorder(paths['scene'], output)
            saved = recorder.save(0, packet)
            renderer = software_renderer_identity()
            need(saved == dict(index=0, native_step=packet['step'], time_s=packet['time_s']), 'SAME_ACTUAL_SEALED_PACKET')
            state_path = output/'STATE.private.bin'; write_bytes(state_path, raw)
            calibration_path = output/'images/000000-calibration.json'
            images = {name: output/'images'/f'000000-{name}.png' for name in ('body_rgb', 'wrist_rgb')}
            language = 'Source camera geometry qualification on a sealed actual target packet.'
            if spec.get('recipe') == STABLE_RECIPE:
                from go2arx5_vla.world_model.fusion_short_skill import LANGUAGE
                language = LANGUAGE
            if spec.get('recipe') == C02_RECIPE:
                from scripts.pi0292.relay_protocol import unseal
                _, source_raw = _read(spec, spec['source_observation'])
                need(spec['source_observation']['sha256'] == '71558ff5bcbc03e83eef9fbe79c868e75a2fece5c712f708682c8f23a090a594',
                     'UNCHANGED_SOURCE_INSTRUCTION_DOCUMENT')
                language = unseal(json.loads(source_raw))['payload']['language']
            bound = binding.bind(actual_state=state, native_state_bytes=state_path.read_bytes(),
                expected_state_sha256=digest(state_path), capture=saved,
                calibration_bytes=calibration_path.read_bytes(), expected_calibration_sha256=digest(calibration_path),
                image_bytes={k: p.read_bytes() for k, p in images.items()},
                expected_image_sha256={k: digest(p) for k, p in images.items()},
                language=language,
                snapshot_uuid=str(uuid.uuid4()), camera_adapter=recorder.camera_adapter)
            shadow_inference_payload(bound)
            bound_path = output/'BOUND_OBSERVATION.private.json'; write_json(bound_path, bound)
            retained = {p.relative_to(output).as_posix(): digest(p) for p in
                [state_path, calibration_path, *images.values(), bound_path, *recorder.retained_files(0)]}
            need(len(retained) == 8 and all(digest(output/name) == pin for name, pin in retained.items()) and not calls,
                 'COMPLETE_EIGHT_RETAINED_MEMBERS_ZERO_INTEGRATIONS')
        # Recheck exactly the immutable inputs used above, including closed proof.
        validate_inputs(spec)
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        close_failure = None
        if recorder is not None:
            try:
                recorder.close()
            except BaseException as error:
                close_failure = type(error).__name__ + ': ' + str(error)
        result = dict(schema='WM02_FUSION_SOURCE_CAMERA_QUALIFICATION_RESULT_V1',
            status='SINGLE_SEALED_TARGET_SOURCE_RIG_CPU_RENDER_PASS' if failure is None and close_failure is None
                   else 'SOURCE_CAMERA_QUALIFICATION_FAILED_PRESERVED',
            failure=failure, recorder_close_failure=close_failure, source_sha=spec['source_sha'],
            saved_native=spec['saved_native'], source_camera_identity_sha256=spec['source_camera_identity_sha256'],
            frame_map_sha256=FRAME_MAP_SHA256, camera_recipe=RIG, renderer=renderer, capture=saved,
            actual_snapshot_render_saves=int(saved is not None), bound_observations=int(bool(retained)),
            retained_files=retained, actual_python_forward_calls=forwards[0],
            native_integrations=0, prohibited_integration_calls=len(calls), VLA_inferences=0,
            GR00T_model_loads=0, WM_model_loads=0, GR00T_updates=0, WM_updates=0, Actor_updates=0,
            online_closed_loop=False, exact_controller_restore=False, source_images_supplied=False,
            native_camera_equivalence=False, image_domain_equivalence=False,
            target_task_success=False, native_execution_authorized=False,
            capture_recipe=spec['recipe'],
            initialization_qualification_is_not_GR00T_task_success=True,
            wall_s=time.monotonic()-start, CPU_s=time.process_time()-cpu)
        write_json(output/'RESULT.private.json', result)
        if close_failure is not None:
            raise RuntimeError('SOURCE_CAMERA_RECORDER_CLOSE_FAILED:' + close_failure)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec, args.output.name, PROFILE, args.output, physics=False, gpu=False) as spec:
        _scope(spec, args.output)
        result = run(spec, args.output)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
