"""Finite offline TRAIN100 body re-render. No model calls, steps, or admission.

The output remains unadmitted until the actual final VLM image processing audit
has been joined. A raw coverage pass alone never qualifies training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

from go2arx5_vla.world_model.native_camera_transfer_v1 import (
    CAMERA_ID, POSITION, LOOK_AT, FOVY, FixedBodyRenderer, coverage, unchanged_packet)
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.task_state import parse_state

MANIFEST = 'ff94fc77c5a8309dc1460032b308c2f213c96c088449ee1821b3e46e045107b0'
STATS = '0faba237b8922ed9bbd53bd6732a9f449f46dcb6fc19017eae8f3cffa3afd9f2'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, pin=None):
    raw = Path(path).read_bytes()
    if pin is not None and sha(raw) != pin:
        raise ValueError('ORIGINAL_FILE_HASH_CHANGED_' + str(path))
    return raw


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')


def export(source, output, *, deadline):
    source, output = Path(source), Path(output)
    if output.exists() or source.resolve() == output.resolve():
        raise FileExistsError('FRESH_INDEPENDENT_OUTPUT_REQUIRED')
    manifest = json.loads(read(source/'DATASET_MANIFEST.json', MANIFEST))
    stats = read(source/'statistics.json', STATS)
    episodes = [e for e in manifest['episodes'] if e['split'] == 'TRAIN']
    if len(episodes) != 100 or sum(e['supervised_policy_decisions'] for e in episodes) != 1450:
        raise ValueError('ORIGINAL_TRAIN100_DENOMINATOR_CHANGED')
    output.mkdir()
    (output/'episodes').mkdir()
    (output/'audit').mkdir()
    (output/'statistics.json').write_bytes(stats)
    rows, exported = [], []
    start = time.monotonic()
    renderer = None
    model_pin = None
    try:
        with (output/'audit/windows.private.jsonl').open('x') as audit:
            for ep in episodes:
                if time.time() >= deadline:
                    raise TimeoutError('FINITE_OFFLINE_RENDER_DEADLINE')
                proof = ep['proof']
                spec = json.loads(read(proof['spec']['path'], proof['spec']['sha256']))
                captures = json.loads(read(proof['capture_index']['path'], proof['capture_index']['sha256']))['records']
                model_raw = read(spec['model_path'])
                mp = sha(model_raw)
                if mp != model_pin:
                    if renderer is not None:
                        renderer.close()
                    renderer = FixedBodyRenderer(spec['model_path'])
                    model_pin = mp
                with np.load(source/ep['file'], allow_pickle=False) as npz:
                    original = {k: npz[k].copy() for k in npz.files}
                if sha(read(source/ep['file'])) != ep['sha256'] or len(captures) != 15:
                    raise ValueError('ORIGINAL_EPISODE_HASH_OR_CAPTURE_COUNT')
                packet = {k: v.copy() for k, v in original.items()}
                for i, descriptor in enumerate(captures):
                    if time.time() >= deadline:
                        raise TimeoutError('FINITE_OFFLINE_RENDER_DEADLINE')
                    path = Path(descriptor['path'])
                    body = unseal(json.loads(read(path, descriptor['sha256'])))
                    state = parse_state(read(path.parent/'STATE.private.bin', body['files']['STATE.private.bin']))
                    if state['step'] != descriptor['native_step'] or state['time_s'] != body['native_time_s']:
                        raise ValueError('EXACT_ORIGINAL_CAPTURE_CLOCK')
                    if body['world']['scene_sha256'] != mp:
                        raise ValueError('ORIGINAL_CAPTURE_SCENE_CHANGED')
                    rgb, masks, calibration, box = renderer.render(state)
                    native_cal = json.loads(read(path.parent/'native-camera/images/000000-calibration.json',
                        body['files']['native-camera/images/000000-calibration.json']))
                    if calibration['wrist_rgb'] != native_cal['wrist_rgb']:
                        raise ValueError('ORIGINAL_WRIST_CALIBRATION_CHANGED')
                    for camera_index, camera in enumerate(('body_rgb', 'wrist_rgb')):
                        dest = output/str(packet['rgb_paths'][i, camera_index])
                        if not dest.resolve().is_relative_to(output.resolve()):
                            raise ValueError('RELATIVE_MEDIA_PATH_REQUIRED')
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if camera_index == 0:
                            Image.fromarray(rgb).save(dest)
                        else:
                            raw = read(path.parent/'native-camera/images/000000-wrist_rgb.png',
                                body['files']['native-camera/images/000000-wrist_rgb.png'])
                            if sha(raw) != original['rgb_sha256'][i, 1]:
                                raise ValueError('ORIGINAL_EXPORT_WRIST_BYTES_DIFFER')
                            dest.write_bytes(raw)
                        packet['rgb_sha256'][i, camera_index] = sha(dest.read_bytes())
                    stem = output/'audit'/f"{ep['id']}-{i:02d}"
                    np.savez_compressed(str(stem)+'-masks.npz', **masks)
                    write(str(stem)+'-calibration.json', calibration)
                    row = dict(episode=ep['id'], anchor=i, supervised=bool(packet['valid_window'][i]),
                        phase=str(packet['expert_stage_audit_only'][i]), state_sha256=body['files']['STATE.private.bin'],
                        scene_sha256=mp, native_step=state['step'], timestamp_s=state['time_s'],
                        body_projection=box, visible_pixels={k: int(v.sum()) for k, v in masks.items()},
                        masks_file=stem.name+'-masks.npz', rgb_sha256=packet['rgb_sha256'][i].tolist())
                    rows.append(row)
                    audit.write(json.dumps(row, allow_nan=False)+'\n')
                    audit.flush()
                unchanged_packet(original, packet)
                dest = output/ep['file']
                np.savez_compressed(dest, **packet)
                with np.load(dest, allow_pickle=False) as saved:
                    unchanged_packet(original, saved)
                new_ep = dict(ep, sha256=sha(dest.read_bytes()))
                exported.append(new_ep)
                write(output/'PROGRESS.private.json', dict(completed_episodes=len(exported),
                      captures=len(rows), wall_s=time.monotonic()-start))
        read(source/'DATASET_MANIFEST.json', MANIFEST)
        result = dict(schema='NATIVE_CAMERA_TRANSFER_RAW_COVERAGE_V1', camera_id=CAMERA_ID,
            source_manifest_sha256=MANIFEST, statistics_sha256=STATS, training_admitted=False,
            final_Processor_coverage='PENDING_ACTUAL_VLM_TRANSFORM_AUDIT', coverage=coverage(rows),
            model_calls=0, optimizer_updates=0, physics_steps=0, episodes=len(exported),
            camera=dict(position_BASE=POSITION.tolist(), look_at_BASE=LOOK_AT.tolist(), up_BASE=[0, 0, 1],
                        vertical_fovy_deg=FOVY, resolution=[256, 256]), wall_s=time.monotonic()-start)
        write(output/'RAW_COVERAGE.private.json', result)
        new_manifest = dict(manifest, schema='NATIVE_ARX5_TARGET_OBSERVATION_DATA_V1',
            profile=CAMERA_ID, training_admitted=False, episodes=exported,
            source_manifest_sha256=MANIFEST, source_statistics_sha256=STATS,
            old_DEV_references_copied=False, new_observation_only=True,
            coverage_report='RAW_COVERAGE.private.json',
            camera_role_map={'external_rgb': CAMERA_ID, 'wrist_rgb': 'ORIGINAL_NATIVE_WRIST'},
            requires=['ACTUAL_FINAL_PROCESSOR_COVERAGE', 'OFFICIAL_PARENT_TRAINING_QUALIFICATION'])
        # Historical export also contains four DEV references and its own
        # eligibility claims. Preserve that document by hash, not by inheriting
        # its population/admission metadata into this TRAIN-only version.
        for key in ('population', 'attempts', 'groups', 'panel', 'fixed_DEV_reference_group',
                    'forty_condition_native_VLA_DEV', 'original_Processor_route'):
            new_manifest.pop(key, None)
        new_manifest['population'] = dict(TRAIN_episodes=100, supervised_windows=1450,
                                          original_captures=1500, excluded_interventions=50)
        write(output/'DATASET_MANIFEST.json', new_manifest)
        return result
    finally:
        if renderer is not None:
            renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, deadline=args.deadline)))


if __name__ == '__main__':
    main()
